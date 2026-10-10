from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from exbind.converters.dom import convert_dom_row
from exbind.converters.evg import convert_evg_dataset
from exbind.schema import BindingRecord, Candidate


CAPABILITY_NAMES = ("identity", "type", "relation", "hierarchy", "role", "state_conditioned")

DATASET_ROOTS = {
    "EVG-Bench": Path("vlm_diag/dataset/evg_bench"),
    "EVG-Bench-Perturbed": Path("vlm_diag/dataset/evg_bench_perturbed"),
    "Novel-SVG": Path("vlm_diag/dataset/evg_external_50"),
    "SEB-hard": Path("vlm_diag/dataset/svgeditbench_v2_subset"),
    "SEB-random": Path("vlm_diag/dataset/svgeditbench_v2_random"),
    "SEB-perturbed": Path("vlm_diag/dataset/svgeditbench_v2_perturbed"),
}


def annotate_record(record: BindingRecord) -> dict[str, Any]:
    gold_ids = _gold_ids(record.gold.executable_target)
    candidate_ids = {candidate.candidate_id for candidate in record.candidate_set}
    gold_candidates = [candidate for candidate in record.candidate_set if candidate.candidate_id in gold_ids]
    gold_unique = bool(gold_ids) and gold_ids <= candidate_ids
    if len(gold_ids) == 1:
        gold_unique = sum(candidate.candidate_id in gold_ids for candidate in record.candidate_set) == 1

    lexical_shortcut = _lexical_shortcut(record, gold_candidates)
    type_shortcut = _type_shortcut(record, gold_candidates)
    relation_required = _relation_required(record, lexical_shortcut, type_shortcut)
    hierarchy_required = _hierarchy_required(record, lexical_shortcut, type_shortcut)
    role_required = _role_required(record)
    type_required = _type_required(record, gold_candidates)
    capabilities = {
        "identity": True,
        "type": type_required,
        "relation": relation_required,
        "hierarchy": hierarchy_required,
        "role": role_required,
        "state_conditioned": bool(record.metadata.get("state_conditioned")),
    }
    primary = _primary_capability(capabilities)
    secondary = [name for name in CAPABILITY_NAMES if name != primary and capabilities[name]]
    return {
        "case_id": record.case_id,
        "source_family": record.source_family,
        "split": record.split or "",
        "operation": record.operation or "",
        "capabilities": capabilities,
        "primary_capability": primary,
        "secondary_capabilities": secondary,
        "candidate_count": len(record.candidate_set),
        "same_type_distractor_count": _same_type_distractor_count(record, gold_candidates),
        "structurally_plausible_distractor_count": _structurally_plausible_distractor_count(record, gold_candidates),
        "lexical_shortcut_available": lexical_shortcut,
        "type_shortcut_available": type_shortcut,
        "gold_executable": record.gold.executable_target,
        "gold_unique": gold_unique,
        "metadata": {
            "target_type": record.binding_structure.target_type,
            "executable_type": record.binding_structure.executable_type,
            "relation": record.binding_structure.relation,
            "hierarchy_path": record.binding_structure.hierarchy_path,
        },
    }


def run_capability_annotation(
    *,
    out_dir: Path,
    limit_per_dataset: int | None = None,
    dom_rows_path: Path | None = Path("data/dom_transfer/miniwob_binding_smoke.jsonl"),
    dry_run: bool = False,
) -> dict[str, Any]:
    records: list[BindingRecord] = []
    dataset_counts: dict[str, int] = {}
    for family, root in DATASET_ROOTS.items():
        if not root.exists():
            dataset_counts[family] = 0
            continue
        converted = convert_evg_dataset(root, limit=limit_per_dataset, source_family=family)
        records.extend(converted)
        dataset_counts[family] = len(converted)
    dom_count = 0
    if dom_rows_path and dom_rows_path.exists():
        for row in _read_jsonl(dom_rows_path):
            records.append(convert_dom_row(row))
            dom_count += 1
            if limit_per_dataset is not None and dom_count >= limit_per_dataset:
                break
    dataset_counts["DOM/MiniWoB"] = dom_count

    annotations = [annotate_record(record) for record in records]
    summary = _summary(annotations)
    validation = _validation(annotations, dataset_counts)
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(out_dir / "capability_annotations.jsonl", annotations)
        _write_csv(out_dir / "capability_summary.csv", summary)
        (out_dir / "capability_validation.json").write_text(
            json.dumps(validation, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
    return {"annotations": annotations, "summary": summary, "validation": validation}


def _relation_required(record: BindingRecord, lexical_shortcut: bool, type_shortcut: bool) -> bool:
    if record.source_family == "relational_synth":
        return not lexical_shortcut and not type_shortcut and record.binding_structure.relation is not None
    if lexical_shortcut or type_shortcut:
        return False
    return str(record.binding_structure.target_type) == "edge_rejection"


def _hierarchy_required(record: BindingRecord, lexical_shortcut: bool, type_shortcut: bool) -> bool:
    if record.source_family == "hierarchical_synth":
        return not lexical_shortcut and not type_shortcut and bool(record.binding_structure.hierarchy_path)
    if lexical_shortcut or type_shortcut:
        return False
    return str(record.binding_structure.target_type) == "group_grounding"


def _role_required(record: BindingRecord) -> bool:
    target_type = str(record.binding_structure.target_type or "")
    executable_type = str(record.binding_structure.executable_type or "")
    return target_type == "endpoint_grounding" or executable_type == "ordered_endpoint_pair"


def _type_required(record: BindingRecord, gold_candidates: list[Candidate]) -> bool:
    target_type = str(record.binding_structure.target_type or "")
    if target_type in {"endpoint_grounding", "edge_rejection", "group_grounding", "dom_reference_selection"}:
        return True
    if not gold_candidates:
        return False
    gold_types = {candidate.candidate_type for candidate in gold_candidates}
    all_types = {candidate.candidate_type for candidate in record.candidate_set}
    return len(all_types) > 1 and bool(gold_types)


def _lexical_shortcut(record: BindingRecord, gold_candidates: list[Candidate]) -> bool:
    if not gold_candidates:
        return False
    query = _tokens(record.instruction or "")
    if not query:
        return False
    scores = []
    for candidate in record.candidate_set:
        text = " ".join([candidate.label or "", str(candidate.attributes.get("visible_text") or ""), candidate.candidate_id])
        tokens = _tokens(text)
        score = len(query & tokens)
        scores.append((score, candidate.candidate_id))
    max_score = max(score for score, _candidate_id in scores)
    if max_score <= 0:
        return False
    winners = {candidate_id for score, candidate_id in scores if score == max_score}
    gold_ids = {candidate.candidate_id for candidate in gold_candidates}
    return winners == gold_ids


def _type_shortcut(record: BindingRecord, gold_candidates: list[Candidate]) -> bool:
    if len(gold_candidates) != 1:
        return False
    gold_type = gold_candidates[0].candidate_type
    if not gold_type:
        return False
    return sum(candidate.candidate_type == gold_type for candidate in record.candidate_set) == 1


def _primary_capability(capabilities: dict[str, bool]) -> str:
    for name in ("state_conditioned", "role", "relation", "hierarchy", "type", "identity"):
        if capabilities.get(name):
            return name
    return "identity"


def _same_type_distractor_count(record: BindingRecord, gold_candidates: list[Candidate]) -> int:
    gold_types = {candidate.candidate_type for candidate in gold_candidates if candidate.candidate_type}
    gold_ids = {candidate.candidate_id for candidate in gold_candidates}
    return sum(candidate.candidate_id not in gold_ids and candidate.candidate_type in gold_types for candidate in record.candidate_set)


def _structurally_plausible_distractor_count(record: BindingRecord, gold_candidates: list[Candidate]) -> int:
    gold_ids = {candidate.candidate_id for candidate in gold_candidates}
    gold_attrs = record.binding_structure.attributes or {}
    count = 0
    for candidate in record.candidate_set:
        if candidate.candidate_id in gold_ids:
            continue
        overlap = sum(1 for key, value in gold_attrs.items() if candidate.attributes.get(key) == value)
        if overlap:
            count += 1
    return count


def _gold_ids(target: Any) -> set[str]:
    if isinstance(target, str):
        return {target}
    if isinstance(target, dict):
        return {value for value in target.values() if isinstance(value, str)}
    return set()


def _tokens(text: str) -> set[str]:
    import re

    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _summary(annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        groups[annotation["source_family"]].append(annotation)
    rows = []
    for family, items in sorted(groups.items()):
        n = len(items)
        row = {"source_family": family, "n": n}
        for capability in CAPABILITY_NAMES:
            row[f"{capability}_n"] = sum(int(item["capabilities"][capability]) for item in items)
            row[f"{capability}_rate"] = row[f"{capability}_n"] / n if n else None
        row["gold_unique_rate"] = sum(int(item["gold_unique"]) for item in items) / n if n else None
        row["lexical_shortcut_rate"] = sum(int(item["lexical_shortcut_available"]) for item in items) / n if n else None
        row["mean_candidate_count"] = sum(int(item["candidate_count"]) for item in items) / n if n else None
        rows.append(row)
    return rows


def _validation(annotations: list[dict[str, Any]], dataset_counts: dict[str, int]) -> dict[str, Any]:
    counts = Counter(item["primary_capability"] for item in annotations)
    relation_not_necessary = sum(
        int(item["capabilities"]["relation"] and item["lexical_shortcut_available"]) for item in annotations
    )
    hierarchy_not_necessary = sum(
        int(item["capabilities"]["hierarchy"] and item["lexical_shortcut_available"]) for item in annotations
    )
    return {
        "n_annotations": len(annotations),
        "dataset_counts": dataset_counts,
        "primary_capability_counts": dict(sorted(counts.items())),
        "relation_marked_despite_lexical_shortcut": relation_not_necessary,
        "hierarchy_marked_despite_lexical_shortcut": hierarchy_not_necessary,
        "gold_unique_failures": sum(int(not item["gold_unique"]) for item in annotations),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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
