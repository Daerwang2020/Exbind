from __future__ import annotations

import csv
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from exbind.evaluation.cross_format import map_prediction_to_canonical
from exbind.schema import BindingRecord, Candidate
from vlm_diag.experiments.common import apply_reproducibility_seed, load_model
from vlm_diag.model.tokenizer_utils import extract_json


DEFAULT_STAGE_ORDER = ("identity", "structure", "role", "executable_exact")


@dataclass
class PilotConfig:
    records_path: Path
    groups_path: Path | None
    out_dir: Path
    model: str
    model_name: str
    model_loader: str = "qwen"
    device: str | None = None
    max_new_tokens: int = 96
    limit: int | None = None
    seed: int = 42
    write_prompts: bool = False
    strict_contract: bool = False


def run_frozen_model_pilot(config: PilotConfig) -> dict[str, Any]:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    records = _read_records(config.records_path, config.limit)
    groups = _read_jsonl(config.groups_path) if config.groups_path and config.groups_path.exists() else []
    model_config = {
        "model": config.model,
        "model_name": config.model_name,
        "model_loader": config.model_loader,
        "device": config.device,
        "seed": config.seed,
    }
    apply_reproducibility_seed(model_config)
    vlm = load_model(model_config)
    rows: list[dict[str, Any]] = []
    for record in records:
        prompt = build_cross_format_prompt(record)
        start = time.time()
        raw_output = ""
        error = None
        try:
            raw_output = vlm.generate_text(_image_path(record), prompt, max_new_tokens=config.max_new_tokens)
        except Exception as exc:
            error = repr(exc)
        parsed = parse_prediction(raw_output)
        prediction = normalize_prediction(
            parsed,
            raw_output,
            record,
            allow_mentioned_candidate_fallback=not config.strict_contract,
        )
        scored = score_prediction(prediction, record)
        row = {
            "case_id": record.case_id,
            "latent_id": str(record.metadata.get("latent_id", "")),
            "format": str(record.metadata.get("format") or record.source_family),
            "source_family": record.source_family,
            "model": config.model_name or config.model,
            "instruction": record.instruction,
            "gold_executable_target": record.gold.executable_target,
            "gold_semantic_target": record.gold.semantic_target,
            "prediction_raw": raw_output,
            "prediction_parsed": parsed,
            "prediction_normalized": prediction,
            "parser_version": (
                "json_strict_no_candidate_mention_fallback_v1"
                if config.strict_contract
                else "legacy_json_with_candidate_mention_fallback_v1"
            ),
            "canonical_prediction": prediction.get("canonical_target"),
            "is_valid": bool(scored["candidate_valid"]),
            "exact": bool(scored["executable_exact"]),
            "reward_vector": scored["reward_vector"],
            "failure_type": scored["failure_type"],
            "seconds": round(time.time() - start, 3),
            "error": error,
        }
        if config.write_prompts:
            row["prompt"] = prompt
        rows.append(row)
    outputs = summarize_pilot(rows, records, groups)
    _write_jsonl(config.out_dir / "predictions.jsonl", rows)
    _write_jsonl(config.out_dir / "intermediate_rewards.jsonl", _reward_rows(rows))
    _write_csv(config.out_dir / "format_accuracy.csv", outputs["format_accuracy"])
    _write_csv(config.out_dir / "capability_accuracy.csv", outputs["capability_accuracy"])
    _write_csv(config.out_dir / "survival_curve.csv", outputs["survival_curve"])
    _write_csv(config.out_dir / "failure_distribution.csv", outputs["failure_distribution"])
    _write_csv(config.out_dir / "failure_by_format.csv", outputs["failure_by_format"])
    _write_csv(config.out_dir / "cross_format_consistency.csv", outputs["cross_format_consistency"])
    _write_csv(config.out_dir / "cross_format_pair_outcomes.csv", outputs["cross_format_pair_outcomes"])
    _write_json(config.out_dir / "pilot_summary.json", outputs["summary"])
    _write_json(config.out_dir / "run_config.json", {**asdict(config), "records_path": str(config.records_path), "groups_path": str(config.groups_path) if config.groups_path else None, "out_dir": str(config.out_dir)})
    return {"rows": rows, **outputs}


def build_cross_format_prompt(record: BindingRecord) -> str:
    fmt = str(record.metadata.get("format") or record.source_family)
    source = record.observation.source or ""
    candidates = [_candidate_prompt_row(candidate, record) for candidate in record.candidate_set]
    schema = _schema_hint(record)
    graph_constraint = ""
    if fmt == "graph" and record.metadata.get("endpoint_explicit"):
        graph_constraint = "For graph, source_id and target_id must be endpoint candidate IDs, never displayed node IDs.\n"
    return (
        "You are evaluating a binding benchmark. Select only from the candidate IDs.\n"
        f"Format: {fmt}\n"
        f"Instruction: {record.instruction}\n\n"
        f"Observation ({record.observation.source_type or 'source'}):\n{source}\n\n"
        "Candidates:\n"
        f"{json.dumps(candidates, ensure_ascii=False, indent=2)}\n\n"
        f"{graph_constraint}"
        f"Return strict JSON only with this schema: {schema}\n"
    )


def parse_prediction(raw_output: str) -> Any:
    parsed = _parse_prediction_strict(raw_output)
    if parsed is not None:
        return parsed
    if r"\_" in raw_output:
        parsed = _parse_prediction_strict(raw_output.replace(r"\_", "_"))
        if parsed is not None:
            return parsed
    return None


def _parse_prediction_strict(raw_output: str) -> Any:
    try:
        parsed = extract_json(raw_output)
    except Exception:
        parsed = None
    if parsed is not None:
        return parsed
    match = re.search(r"\{.*\}", raw_output, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def normalize_prediction(
    parsed: Any,
    raw_output: str,
    record: BindingRecord,
    *,
    allow_mentioned_candidate_fallback: bool = True,
) -> dict[str, Any]:
    payload = dict(parsed) if isinstance(parsed, dict) else {}
    executable = payload.get("executable_target")
    if isinstance(executable, dict):
        payload.update(executable)
    candidate_ids = {candidate.candidate_id for candidate in record.candidate_set}
    mentioned = _mentioned_candidate_ids(raw_output, candidate_ids) if allow_mentioned_candidate_fallback else []
    fmt = str(record.metadata.get("format") or record.source_family)
    if "candidate_id" in payload and "target_id" not in payload and fmt not in {"html_dom", "table"}:
        payload["target_id"] = payload["candidate_id"]
    if fmt == "html_dom" and "dom_id" not in payload:
        payload["dom_id"] = payload.get("target_id") or payload.get("candidate_id") or (mentioned[0] if len(mentioned) == 1 else None)
    elif fmt == "table" and "cell_id" not in payload:
        payload["cell_id"] = payload.get("target_id") or payload.get("candidate_id") or (mentioned[0] if len(mentioned) == 1 else None)
    elif fmt == "graph":
        if "source_id" not in payload and "target_id" not in payload and len(mentioned) >= 2:
            payload["source_id"], payload["target_id"] = mentioned[0], mentioned[1]
    elif "target_id" not in payload:
        payload["target_id"] = payload.get("candidate_id") or (mentioned[0] if len(mentioned) == 1 else None)
    payload = {key: value for key, value in payload.items() if value is not None}
    payload["canonical_target"] = map_prediction_to_canonical(payload, record)
    payload["executable_target"] = _executable_prediction(payload, record)
    return payload


def score_prediction(prediction: dict[str, Any], record: BindingRecord) -> dict[str, Any]:
    candidate_valid = _candidate_valid(prediction, record)
    exact = _exact(prediction.get("executable_target"), record.gold.executable_target)
    identity = 1 if prediction.get("canonical_target") == record.gold.semantic_target else 0
    attribute = _score_attribute(prediction, record)
    anchor = _score_anchor(prediction, record)
    relation = _score_relation(prediction, record)
    hierarchy = _score_hierarchy(prediction, record)
    role = _score_role(prediction, record)
    table_row = _score_table_row(prediction, record)
    table_column = _score_table_column(prediction, record)
    table_cell = _score_table_cell(prediction, record)
    vector = {
        "identity": identity,
        "attribute": attribute,
        "anchor": anchor,
        "relation": relation,
        "hierarchy": hierarchy,
        "role": role,
        "candidate_valid": candidate_valid,
        "executable_exact": exact,
        "table_row": table_row,
        "table_column": table_column,
        "table_cell": table_cell,
    }
    return {
        "candidate_valid": candidate_valid,
        "executable_exact": exact,
        "reward_vector": vector,
        "failure_type": classify_cross_format_failure(prediction, record, vector),
    }


def classify_cross_format_failure(prediction: dict[str, Any], record: BindingRecord, vector: dict[str, int | None]) -> str:
    if vector["executable_exact"] == 1:
        return "correct"
    if vector["candidate_valid"] != 1:
        return "invalid/unresolved"
    # Task-specific stages take precedence over the generic latent identity
    # check. A table cell encodes both row identity and column attribute, so a
    # wrong column must not be mislabeled as a wrong entity.
    fmt = str(record.metadata.get("format") or record.source_family)
    if fmt == "table":
        if vector.get("table_row") == 0:
            return "wrong identity"
        if vector.get("table_column") == 0:
            return "wrong attribute"
        if vector.get("table_cell") == 0:
            return "valid-but-wrong exact reference"
    if _is_swapped_graph_pair(prediction, record):
        return "wrong role/order"
    for key, label in (
        ("identity", "wrong identity"),
        ("attribute", "wrong attribute"),
        ("anchor", "wrong anchor"),
        ("relation", "wrong relation"),
        ("hierarchy", "wrong hierarchy"),
        ("role", "wrong role/order"),
    ):
        if vector.get(key) == 0:
            return label
    return "valid-but-wrong exact reference"


def summarize_pilot(rows: list[dict[str, Any]], records: list[BindingRecord], groups: list[dict[str, Any]]) -> dict[str, Any]:
    format_accuracy = _accuracy_by(rows, "format", "exact")
    capability_accuracy = _capability_accuracy(rows, records)
    survival_curve = _survival_curve(rows)
    failure_distribution = _distribution(rows, "failure_type")
    failure_by_format = _group_distribution(rows, "format", "failure_type")
    pair_outcomes = _cross_format_pair_outcomes(rows, groups)
    consistency = _cross_format_summary(pair_outcomes)
    summary = {
        "n": len(rows),
        "models": sorted({row["model"] for row in rows}),
        "overall_exact_accuracy": _rate(row["exact"] for row in rows),
        "overall_validity": _rate(row["is_valid"] for row in rows),
        "formats": {row["group"]: row for row in format_accuracy},
        "cross_format_groups_scored": len(pair_outcomes),
        "outputs": {
            "predictions": "predictions.jsonl",
            "format_accuracy": "format_accuracy.csv",
            "capability_accuracy": "capability_accuracy.csv",
            "survival_curve": "survival_curve.csv",
            "failure_distribution": "failure_distribution.csv",
            "cross_format_consistency": "cross_format_consistency.csv",
        },
    }
    return {
        "format_accuracy": format_accuracy,
        "capability_accuracy": capability_accuracy,
        "survival_curve": survival_curve,
        "failure_distribution": failure_distribution,
        "failure_by_format": failure_by_format,
        "cross_format_pair_outcomes": pair_outcomes,
        "cross_format_consistency": consistency,
        "summary": summary,
    }


def _candidate_prompt_row(candidate: Candidate, record: BindingRecord) -> dict[str, Any]:
    row: dict[str, Any] = {
        "candidate_id": candidate.candidate_id,
        "candidate_type": candidate.candidate_type,
        "bbox": candidate.bbox,
    }
    fmt = str(record.metadata.get("format") or record.source_family)
    if fmt == "graph":
        row["role"] = candidate.attributes.get("role")
        row["node"] = _surface_for_latent(record, candidate.attributes.get("node"))
        if record.metadata.get("endpoint_explicit"):
            row["endpoint_candidate"] = True
    if fmt == "table" and record.metadata.get("candidate_interface_variant") == "explicit_column_metadata_v1":
        row["column"] = candidate.attributes.get("column")
    return {key: value for key, value in row.items() if value not in (None, [], {})}


def _schema_hint(record: BindingRecord) -> str:
    fmt = str(record.metadata.get("format") or record.source_family)
    if fmt == "graph":
        return '{"source_id": "...", "target_id": "..."}'
    if fmt == "html_dom":
        return '{"dom_id": "..."}'
    if fmt == "table":
        return '{"cell_id": "..."}'
    return '{"target_id": "..."}'


def _image_path(record: BindingRecord) -> Path | None:
    value = record.observation.image
    if not value:
        return None
    path = Path(value)
    return path if path.exists() else None


def _mentioned_candidate_ids(text: str, candidate_ids: set[str]) -> list[str]:
    positions = []
    for candidate_id in candidate_ids:
        idx = text.find(candidate_id)
        if idx >= 0:
            positions.append((idx, candidate_id))
    return [candidate_id for _, candidate_id in sorted(positions)]


def _executable_prediction(prediction: dict[str, Any], record: BindingRecord) -> Any:
    gold = record.gold.executable_target
    if isinstance(gold, dict):
        return {key: prediction.get(key) for key in gold if prediction.get(key) is not None}
    return prediction.get("target_id") or prediction.get("candidate_id")


def _candidate_valid(prediction: dict[str, Any], record: BindingRecord) -> int:
    ids = {value for key, value in prediction.items() if key.endswith("_id") and isinstance(value, str)}
    executable = prediction.get("executable_target")
    if isinstance(executable, dict):
        ids.update(value for value in executable.values() if isinstance(value, str))
    if not ids:
        return 0
    candidate_ids = {candidate.candidate_id for candidate in record.candidate_set}
    return 1 if ids <= candidate_ids else 0


def _exact(prediction: Any, gold: Any) -> int:
    return 1 if prediction == gold else 0


def _score_attribute(prediction: dict[str, Any], record: BindingRecord) -> int | None:
    gold_attrs = record.binding_structure.attributes or None
    if gold_attrs is None:
        return None
    candidates = _predicted_candidates(prediction, record)
    if not candidates:
        return 0
    if str(record.metadata.get("format")) == "table":
        return _score_table_attributes(candidates[0], record)
    target = candidates[-1]
    return 1 if all(target.attributes.get(key) == value for key, value in gold_attrs.items()) else 0


def _score_table_attributes(candidate: Candidate, record: BindingRecord) -> int:
    gold_attrs = record.binding_structure.attributes
    if candidate.attributes.get("column") != gold_attrs.get("column"):
        return 0
    owner = gold_attrs.get("owner")
    row = candidate.attributes.get("row")
    owner_cells = [
        item
        for item in record.candidate_set
        if item.attributes.get("row") == row and item.attributes.get("column") == "owner"
    ]
    return 1 if len(owner_cells) == 1 and owner_cells[0].attributes.get("value") == owner else 0


def _table_prediction_candidate(prediction: dict[str, Any], record: BindingRecord) -> Candidate | None:
    candidates = _predicted_candidates(prediction, record)
    return candidates[0] if candidates else None


def _table_gold_candidate(record: BindingRecord) -> Candidate | None:
    gold = record.gold.executable_target
    cell_id = gold.get("cell_id") if isinstance(gold, dict) else None
    return next((candidate for candidate in record.candidate_set if candidate.candidate_id == cell_id), None)


def _score_table_row(prediction: dict[str, Any], record: BindingRecord) -> int | None:
    if str(record.metadata.get("format") or record.source_family) != "table":
        return None
    predicted = _table_prediction_candidate(prediction, record)
    gold = _table_gold_candidate(record)
    if predicted is None or gold is None:
        return 0
    return int(predicted.attributes.get("row") == gold.attributes.get("row"))


def _score_table_column(prediction: dict[str, Any], record: BindingRecord) -> int | None:
    if str(record.metadata.get("format") or record.source_family) != "table":
        return None
    predicted = _table_prediction_candidate(prediction, record)
    gold = _table_gold_candidate(record)
    if predicted is None or gold is None:
        return 0
    return int(predicted.attributes.get("column") == gold.attributes.get("column"))


def _score_table_cell(prediction: dict[str, Any], record: BindingRecord) -> int | None:
    if str(record.metadata.get("format") or record.source_family) != "table":
        return None
    predicted = _table_prediction_candidate(prediction, record)
    gold = _table_gold_candidate(record)
    if predicted is None or gold is None:
        return 0
    return int(
        predicted.attributes.get("row") == gold.attributes.get("row")
        and predicted.attributes.get("column") == gold.attributes.get("column")
    )


def _score_anchor(prediction: dict[str, Any], record: BindingRecord) -> int | None:
    if record.binding_structure.anchor is None:
        return None
    return _score_relation(prediction, record)


def _score_relation(prediction: dict[str, Any], record: BindingRecord) -> int | None:
    relation = record.binding_structure.relation
    if relation is None:
        return None
    fmt = str(record.metadata.get("format") or record.source_family)
    if fmt in {"canvas2d", "svg"}:
        candidates = _predicted_candidates(prediction, record)
        if not candidates:
            return 0
        anchor = _candidate_for_latent(record, str(relation.get("anchor")))
        return 1 if _relation_holds(candidates[-1], anchor, relation.get("type")) else 0
    if fmt == "graph":
        return 1 if prediction.get("canonical_target") == record.gold.semantic_target else 0
    return None


def _score_hierarchy(prediction: dict[str, Any], record: BindingRecord) -> int | None:
    gold_path = record.binding_structure.hierarchy_path
    if gold_path is None:
        return None
    candidates = _predicted_candidates(prediction, record)
    if not candidates:
        return 0
    attrs = candidates[-1].attributes
    candidate_path = attrs.get("hierarchy_path")
    if candidate_path is not None:
        return 1 if list(candidate_path) == list(gold_path) else 0
    pred_path = [attrs.get("group"), f"{attrs.get('module')} module" if attrs.get("module") else None, attrs.get("part")]
    pred_path = [str(item) for item in pred_path if item]
    return 1 if pred_path == list(gold_path) else 0


def _score_role(prediction: dict[str, Any], record: BindingRecord) -> int | None:
    gold = record.gold.executable_target
    if isinstance(gold, dict) and {"source_id", "target_id"} <= set(gold):
        return 1 if prediction.get("source_id") == gold.get("source_id") and prediction.get("target_id") == gold.get("target_id") else 0
    if record.binding_structure.part is None:
        return None
    candidates = _predicted_candidates(prediction, record)
    if not candidates:
        return 0
    role = candidates[-1].attributes.get("role") or candidates[-1].attributes.get("part")
    return 1 if role == record.binding_structure.part else 0


def _predicted_candidates(prediction: dict[str, Any], record: BindingRecord) -> list[Candidate]:
    by_id = {candidate.candidate_id: candidate for candidate in record.candidate_set}
    ids = []
    for key in ("source_id", "target_id", "dom_id", "cell_id", "candidate_id"):
        value = prediction.get(key)
        if isinstance(value, str):
            ids.append(value)
    return [by_id[item] for item in ids if item in by_id]


def _candidate_for_latent(record: BindingRecord, latent_id: str) -> Candidate | None:
    surface = _surface_for_latent(record, latent_id)
    return next((candidate for candidate in record.candidate_set if candidate.candidate_id == surface), None)


def _surface_for_latent(record: BindingRecord, latent_id: Any) -> str | None:
    if latent_id is None:
        return None
    return dict(record.metadata.get("mapping_latent_to_surface") or {}).get(str(latent_id))


def _relation_holds(candidate: Candidate, anchor: Candidate | None, relation: str | None) -> bool:
    if anchor is None:
        return False
    cx, cy = float(candidate.attributes.get("x", 0)), float(candidate.attributes.get("y", 0))
    ax, ay = float(anchor.attributes.get("x", 0)), float(anchor.attributes.get("y", 0))
    if relation == "left_of":
        return cx < ax
    if relation == "right_of":
        return cx > ax
    if relation == "above":
        return cy < ay
    if relation == "below":
        return cy > ay
    if relation == "nearest":
        return (cx - ax) ** 2 + (cy - ay) ** 2 <= 30**2
    return False


def _is_swapped_graph_pair(prediction: dict[str, Any], record: BindingRecord) -> bool:
    gold = record.gold.executable_target
    return (
        isinstance(gold, dict)
        and prediction.get("source_id") == gold.get("target_id")
        and prediction.get("target_id") == gold.get("source_id")
    )


def _reward_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": row["case_id"],
            "latent_id": row["latent_id"],
            "format": row["format"],
            "model": row["model"],
            **row["reward_vector"],
            "failure_type": row["failure_type"],
        }
        for row in rows
    ]


def _capability_accuracy(rows: list[dict[str, Any]], records: list[BindingRecord]) -> list[dict[str, Any]]:
    tags_by_case = {record.case_id: set(record.metadata.get("capability_tags") or []) for record in records}
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    tag_to_metric = {
        "identity": "identity",
        "attribute": "attribute",
        "anchor": "anchor",
        "relation": "relation",
        "hierarchy": "hierarchy",
        "role": "role",
        "ordered_role": "role",
        "type": "candidate_valid",
        "executable": "executable_exact",
    }
    for row in rows:
        vector = row["reward_vector"]
        for tag in sorted(tags_by_case.get(row["case_id"], set())):
            metric = tag_to_metric.get(tag)
            value = vector.get(metric) if metric else None
            if value is None:
                continue
            grouped[(row["format"], tag)].append(int(value))
    return [
        {"format": fmt, "capability": cap, "correct": sum(values), "total": len(values), "accuracy": sum(values) / len(values)}
        for (fmt, cap), values in sorted(grouped.items())
    ]


def _survival_curve(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["format"]].append(row)
    out = []
    for fmt, items in sorted(grouped.items()):
        for stage in DEFAULT_STAGE_ORDER:
            total = 0
            survived = 0
            for row in items:
                vector = row["reward_vector"]
                ok = True
                seen = False
                for sub_stage in DEFAULT_STAGE_ORDER:
                    value = _stage_value(vector, sub_stage)
                    if value is not None:
                        seen = True
                        ok = ok and bool(value)
                    if sub_stage == stage:
                        break
                if seen:
                    total += 1
                    survived += int(ok)
            if total:
                out.append({"format": fmt, "stage": stage, "survived": survived, "total": total, "survival": survived / total})
    return out


def _stage_value(vector: dict[str, int | None], stage: str) -> int | None:
    if stage == "structure":
        values = [vector.get(key) for key in ("relation", "hierarchy") if vector.get(key) is not None]
        return min(values) if values else None
    return vector.get(stage)


def _cross_format_pair_outcomes(rows: list[dict[str, Any]], groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case = {row["case_id"]: row for row in rows}
    out = []
    for group in groups:
        case_ids = list((group.get("case_ids") or {}).values())
        items = [by_case[case_id] for case_id in case_ids if case_id in by_case]
        if len(items) < 2:
            continue
        values = [item.get("canonical_prediction") for item in items]
        exacts = [bool(item.get("exact")) for item in items]
        out.append(
            {
                "latent_id": group.get("latent_id"),
                "formats": "|".join(item["format"] for item in items),
                "canonical_gold": group.get("canonical_target"),
                "canonical_predictions": "|".join(str(value) for value in values),
                "cross_format_semantic_consistent": len(set(values)) == 1 and None not in values,
                "both_correct": all(exacts),
                "at_least_one_wrong": not all(exacts),
                "one_correct_one_wrong": sum(exacts) == 1,
                "format_sensitivity": sum(exacts) == 1,
            }
        )
    return out


def _cross_format_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    both = [row for row in rows if row["both_correct"]]
    wrong = [row for row in rows if row["at_least_one_wrong"]]
    return [
        {
            "n_groups": len(rows),
            "C_semantic": _rate(row["cross_format_semantic_consistent"] for row in rows),
            "C_given_both_correct": _rate(row["cross_format_semantic_consistent"] for row in both),
            "C_given_at_least_one_wrong": _rate(row["cross_format_semantic_consistent"] for row in wrong),
            "P_one_format_correct_one_wrong": _rate(row["one_correct_one_wrong"] for row in rows),
        }
    ]


def _accuracy_by(rows: list[dict[str, Any]], group_field: str, value_field: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_field))].append(bool(row.get(value_field)))
    return [
        {"group": group, "correct": sum(values), "total": len(values), "accuracy": sum(values) / len(values)}
        for group, values in sorted(grouped.items())
    ]


def _distribution(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    counts = Counter(str(row.get(field)) for row in rows)
    total = sum(counts.values())
    return [{"category": key, "n": n, "rate": n / total if total else None} for key, n in sorted(counts.items())]


def _group_distribution(rows: list[dict[str, Any]], group_field: str, value_field: str) -> list[dict[str, Any]]:
    groups: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        groups[str(row.get(group_field))][str(row.get(value_field))] += 1
    out = []
    for group, counts in sorted(groups.items()):
        total = sum(counts.values())
        for category, n in sorted(counts.items()):
            out.append({group_field: group, "category": category, "n": n, "rate": n / total if total else None})
    return out


def _rate(values: Iterable[Any]) -> float | None:
    values = list(values)
    return sum(int(bool(value)) for value in values) / len(values) if values else None


def _read_records(path: Path, limit: int | None = None) -> list[BindingRecord]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(BindingRecord.from_dict(json.loads(line)))
            if limit is not None and len(records) >= limit:
                break
    return records


def _read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
