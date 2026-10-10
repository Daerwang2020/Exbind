from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from exbind.generators.cross_format import (
    make_graph_latent,
    make_hierarchy_latent,
    make_relational_latent,
    make_table_latent,
)
from exbind.latent import LatentBindingInstance, RenderedBindingInstance
from exbind.representations import compile_canvas2d, compile_graph, compile_html_dom, compile_svg, compile_table, compile_tree
from exbind.representations.base import rendered_to_record
from exbind.schema import BindingRecord


COMPILERS = {
    "svg": compile_svg,
    "canvas2d": compile_canvas2d,
    "tree": compile_tree,
    "html_dom": compile_html_dom,
    "graph": compile_graph,
    "table": compile_table,
}


def run_cross_format_benchmark(
    *,
    out_dir: Path,
    n_per_family: int = 50,
    n_cross_format_groups: int = 50,
    seed: int = 20260823,
    dry_run: bool = False,
    surface_variant: str = "v1",
) -> dict[str, Any]:
    if surface_variant not in {"v1", "interface_gate_v1"}:
        raise ValueError(f"Unsupported surface variant: {surface_variant}")
    out_dir.mkdir(parents=True, exist_ok=True)
    latents: list[LatentBindingInstance] = []
    rendered: list[RenderedBindingInstance] = []
    records: list[BindingRecord] = []
    groups: list[dict[str, Any]] = []

    relational_latents = [make_relational_latent(index + seed, distractors=4) for index in range(n_per_family)]
    varied = surface_variant == "interface_gate_v1"
    hierarchy_latents = [make_hierarchy_latent(index + seed, varied=varied) for index in range(n_per_family)]
    graph_latents = [make_graph_latent(index + seed, varied=varied) for index in range(n_per_family)]
    table_latents = [make_table_latent(index + seed, varied=varied) for index in range(n_per_family)]
    latents.extend(relational_latents + hierarchy_latents + graph_latents + table_latents)

    for latent in relational_latents:
        records.extend(_compile_records(latent, ["canvas2d"], surface_variant=surface_variant, seed=seed))
    for latent in hierarchy_latents:
        records.extend(_compile_records(latent, ["tree"], surface_variant=surface_variant, seed=seed))
    for latent in graph_latents:
        records.extend(_compile_records(latent, ["graph"], surface_variant=surface_variant, seed=seed))
    for latent in table_latents:
        records.extend(_compile_records(latent, ["table"], surface_variant=surface_variant, seed=seed))

    records_by_case_id = {record.case_id: record for record in records}
    paired_latents = []
    rel_pairs = min(n_cross_format_groups // 2, len(relational_latents))
    hier_pairs = min(n_cross_format_groups - rel_pairs, len(hierarchy_latents))
    paired_latents.extend((latent, ["svg", "canvas2d"]) for latent in relational_latents[:rel_pairs])
    paired_latents.extend((latent, ["tree", "html_dom"]) for latent in hierarchy_latents[:hier_pairs])
    for latent, formats in paired_latents:
        pair_records = []
        for fmt in formats:
            case_id = f"{latent.latent_id}__{fmt}"
            record = records_by_case_id.get(case_id)
            if record is None:
                record = _compile_records(latent, [fmt], surface_variant=surface_variant, seed=seed)[0]
                records.append(record)
                records_by_case_id[case_id] = record
            pair_records.append(record)
        rendered.extend(COMPILERS[fmt](latent) for fmt in formats)
        groups.append(_cross_format_group(latent, pair_records))

    # Keep one rendered copy for the single-format records as well.
    for record in records:
        if not any(item.latent_id == record.metadata["latent_id"] and item.format == record.metadata["format"] for item in rendered):
            latent = next(item for item in latents if item.latent_id == record.metadata["latent_id"])
            rendered.append(COMPILERS[record.metadata["format"]](latent))

    gates = [generator_gate_row(record) for record in records]
    gate_summary = summarize_generator_gates(gates)
    interface_gate_summary = summarize_interface_gates(gates)
    validation = [_validate_group(group, records) for group in groups]
    consistency_rows = gold_cross_format_consistency(groups, records)
    consistency_summary = summarize_cross_format_consistency(consistency_rows)
    manifest = benchmark_manifest(records, groups, gate_summary)
    manifest["surface_variant"] = surface_variant
    manifest["interface_gate"] = interface_gate_summary
    model_pilot_status = [
        {
            "status": "not_run",
            "model": "not_run",
            "reason": "no frozen-model inference launched; deterministic generator gates only",
            "overall": "",
            "cross_format_consistency": "",
        }
    ]

    if not dry_run:
        _write_jsonl(out_dir / "latent_instances.jsonl", [latent.to_dict() for latent in latents])
        _write_jsonl(out_dir / "rendered_instances.jsonl", [item.to_dict() for item in rendered])
        _write_jsonl(out_dir / "binding_records.jsonl", [record.to_dict() for record in records])
        _write_jsonl(out_dir / "cross_format_groups.jsonl", groups)
        _write_csv(out_dir / "generator_gates.csv", gates)
        _write_csv(out_dir / "generator_gate_summary.csv", gate_summary)
        _write_csv(out_dir / "interface_gate_summary.csv", interface_gate_summary)
        _write_csv(out_dir / "cross_format_validation.csv", validation)
        _write_csv(out_dir / "cross_format_consistency_gold.csv", consistency_rows)
        _write_csv(out_dir / "cross_format_consistency_summary.csv", consistency_summary)
        _write_csv(out_dir / "model_pilot_status.csv", model_pilot_status)
        (out_dir / "benchmark_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
    return {
        "latents": latents,
        "records": records,
        "groups": groups,
        "gates": gates,
        "gate_summary": gate_summary,
        "interface_gate_summary": interface_gate_summary,
        "validation": validation,
        "consistency_rows": consistency_rows,
        "consistency_summary": consistency_summary,
        "manifest": manifest,
        "model_pilot_status": model_pilot_status,
    }


def _compile_records(
    latent: LatentBindingInstance,
    formats: list[str],
    *,
    surface_variant: str,
    seed: int,
) -> list[BindingRecord]:
    generator_version = "cross_format.v1" if surface_variant == "v1" else "cross_format.interface_gate.v1"
    records = [rendered_to_record(COMPILERS[fmt](latent), latent, generator_version=generator_version) for fmt in formats]
    if surface_variant == "interface_gate_v1":
        for record in records:
            order_seed = _candidate_order_seed(seed, record)
            random.Random(order_seed).shuffle(record.candidate_set)
            record.metadata["surface_variant"] = surface_variant
            record.metadata["candidate_order_seed"] = order_seed
    return records


def _cross_format_group(latent: LatentBindingInstance, records: list[BindingRecord]) -> dict[str, Any]:
    return {
        "latent_id": latent.latent_id,
        "formats": [record.metadata["format"] for record in records],
        "canonical_target": latent.canonical_target,
        "surface_targets": {record.metadata["format"]: _surface_target(record.gold.executable_target) for record in records},
        "case_ids": {record.metadata["format"]: record.case_id for record in records},
    }


def map_prediction_to_canonical(prediction: dict[str, Any], record: BindingRecord) -> str | None:
    surface_to_latent = dict(record.metadata.get("surface_to_latent") or {})
    executable = prediction.get("executable_target")
    if isinstance(executable, dict):
        prediction = {**prediction, **executable}
    if "cell_id" in prediction:
        return surface_to_latent.get(str(prediction["cell_id"]))
    if "dom_id" in prediction:
        return surface_to_latent.get(str(prediction["dom_id"]))
    if "target_id" in prediction and "source_id" not in prediction:
        return surface_to_latent.get(str(prediction["target_id"]))
    if "source_id" in prediction and "target_id" in prediction:
        source = surface_to_latent.get(str(prediction["source_id"]))
        target = surface_to_latent.get(str(prediction["target_id"]))
        return f"{source}->{target}" if source and target else None
    return None


def cross_format_binding_consistency(predictions: list[dict[str, Any]], records_by_case_id: dict[str, BindingRecord]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        record = records_by_case_id[str(prediction["case_id"])]
        grouped[str(record.metadata["latent_id"])].append(
            {
                "case_id": record.case_id,
                "format": record.metadata["format"],
                "canonical_prediction": map_prediction_to_canonical(prediction, record),
                "canonical_gold": record.gold.semantic_target,
                "exact": prediction.get("exact"),
            }
        )
    rows = []
    for latent_id, items in sorted(grouped.items()):
        if len(items) < 2:
            continue
        canonical_values = {item["canonical_prediction"] for item in items}
        exact_values = [bool(item["exact"]) for item in items if item.get("exact") is not None]
        rows.append(
            {
                "latent_id": latent_id,
                "formats": "|".join(item["format"] for item in items),
                "cross_format_semantic_consistency": len(canonical_values) == 1 and None not in canonical_values,
                "all_exact": all(exact_values) if exact_values else "",
                "any_exact": any(exact_values) if exact_values else "",
                "format_sensitivity": len(canonical_values) > 1,
            }
        )
    return rows


def gold_cross_format_consistency(groups: list[dict[str, Any]], records: list[BindingRecord]) -> list[dict[str, Any]]:
    by_case = {record.case_id: record for record in records}
    predictions = []
    for group in groups:
        for case_id in group["case_ids"].values():
            record = by_case[case_id]
            predictions.append({"case_id": case_id, "executable_target": record.gold.executable_target, "exact": True})
    return cross_format_binding_consistency(predictions, by_case)


def summarize_cross_format_consistency(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    n = len(rows)
    return [
        {
            "n_groups": n,
            "semantic_consistency": _rate(rows, "cross_format_semantic_consistency"),
            "all_exact": _rate(rows, "all_exact"),
            "format_sensitivity": _rate(rows, "format_sensitivity"),
        }
    ]


def generator_gate_row(record: BindingRecord) -> dict[str, Any]:
    gold = record.gold.executable_target
    oracle_prediction = structural_oracle(record)
    lexical_pred = lexical_baseline(record)
    type_pred = type_baseline(record)
    random_expected = 1 / len(record.candidate_set) if record.candidate_set else 0.0
    first_candidate = record.candidate_set[0].candidate_id if record.candidate_set else None
    return {
        "case_id": record.case_id,
        "latent_id": record.metadata.get("latent_id"),
        "format": record.metadata.get("format"),
        "candidate_count": len(record.candidate_set),
        "difficulty": (record.metadata.get("difficulty") or {}).get("difficulty"),
        "gold_unique": _gold_unique(record),
        "target_id_leakage": _target_leaks(record),
        "candidate_order_shortcut": _prediction_exact({"target_id": first_candidate}, gold),
        "gold_candidate_rank": _gold_candidate_rank(record),
        "interface_fingerprint": _interface_fingerprint(record),
        "lexical_baseline_correct": _prediction_exact(lexical_pred, gold),
        "type_baseline_correct": _prediction_exact(type_pred, gold),
        "random_expected_accuracy": random_expected,
        "structural_oracle_correct": _prediction_exact(oracle_prediction, gold),
        "accepted": _gold_unique(record)
        and not _target_leaks(record)
        and not _prediction_exact(lexical_pred, gold)
        and not _prediction_exact(type_pred, gold)
        and _prediction_exact(oracle_prediction, gold),
    }


def summarize_generator_gates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["format"])].append(row)
    out = []
    for fmt, items in sorted(groups.items()):
        out.append(
            {
                "format": fmt,
                "n": len(items),
                "accepted_rate": _rate(items, "accepted"),
                "gold_unique_rate": _rate(items, "gold_unique"),
                "target_id_leakage_rate": _rate(items, "target_id_leakage"),
                "lexical_baseline_accuracy": _rate(items, "lexical_baseline_correct"),
                "type_baseline_accuracy": _rate(items, "type_baseline_correct"),
                "structural_oracle_accuracy": _rate(items, "structural_oracle_correct"),
                "mean_random_expected_accuracy": sum(float(item["random_expected_accuracy"]) for item in items) / len(items),
            }
        )
    return out


def summarize_interface_gates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_format: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_format[str(row["format"])].append(row)
    summary = []
    for fmt, items in sorted(by_format.items()):
        prompt_count = len({str(item["interface_fingerprint"]) for item in items})
        rank_count = len({str(item["gold_candidate_rank"]) for item in items})
        summary.append(
            {
                "format": fmt,
                "n": len(items),
                "unique_interface_prompts": prompt_count,
                "interface_prompt_unique_rate": prompt_count / len(items) if items else 0.0,
                "unique_gold_candidate_ranks": rank_count,
                "candidate_rank_varies": rank_count > 1,
                "all_structural_oracle_correct": all(bool(item["structural_oracle_correct"]) for item in items),
                "passed": prompt_count / len(items) >= 0.5
                and rank_count > 1
                and all(bool(item["structural_oracle_correct"]) for item in items),
            }
        )
    return summary


def structural_oracle(record: BindingRecord) -> dict[str, Any] | None:
    fmt = record.metadata.get("format")
    mapping = dict(record.metadata.get("mapping_latent_to_surface") or {})
    spec = (record.gold.intermediate or {}).get("target_spec") or {}
    if fmt in {"svg", "canvas2d"}:
        anchor_surface = mapping.get(str(spec.get("anchor")))
        anchor = next((candidate for candidate in record.candidate_set if candidate.candidate_id == anchor_surface), None)
        relation = (spec.get("relation") or {}).get("type")
        attrs = spec.get("target_attributes") or {}
        matches = [
            candidate
            for candidate in record.candidate_set
            if candidate.candidate_id != anchor_surface
            and candidate.candidate_type == spec.get("target_type")
            and all(candidate.attributes.get(key) == value for key, value in attrs.items())
            and _relation_holds(candidate, anchor, relation)
        ]
        return {"target_id": matches[0].candidate_id} if len(matches) == 1 else None
    if fmt in {"tree", "html_dom"}:
        attrs = spec.get("target_attributes") or {}
        matches = [
            candidate
            for candidate in record.candidate_set
            if (
                (spec.get("hierarchy_path") is not None and candidate.attributes.get("hierarchy_path") == spec.get("hierarchy_path"))
                or (
                    spec.get("hierarchy_path") is None
                    and all(candidate.attributes.get(key) == value for key, value in attrs.items())
                )
            )
        ]
        if len(matches) != 1:
            return None
        key = "dom_id" if fmt == "html_dom" else "target_id"
        return {key: matches[0].candidate_id}
    if fmt == "graph":
        source = mapping.get(str(spec.get("source_endpoint")))
        target = mapping.get(str(spec.get("target_endpoint")))
        return {"source_id": source, "target_id": target} if source and target else None
    if fmt == "table":
        attribute = spec.get("attribute")
        conditions = dict(spec.get("row_conditions") or {})
        if spec.get("owner") is not None:
            conditions.setdefault("owner", spec.get("owner"))
        row_ids = sorted({candidate.attributes.get("row") for candidate in record.candidate_set})
        matching_rows = [
            row_id
            for row_id in row_ids
            if all(
                any(
                    candidate.attributes.get("row") == row_id
                    and candidate.attributes.get("column") == column
                    and candidate.attributes.get("value") == value
                    for candidate in record.candidate_set
                )
                for column, value in conditions.items()
            )
        ]
        if len(matching_rows) != 1:
            return None
        row_id = matching_rows[0]
        cells = [candidate for candidate in record.candidate_set if candidate.attributes.get("row") == row_id and candidate.attributes.get("column") == attribute]
        return {"cell_id": cells[0].candidate_id} if len(cells) == 1 else None
    return None


def lexical_baseline(record: BindingRecord) -> dict[str, Any] | None:
    query = set(str(record.instruction or "").lower().replace(".", "").split())
    scored = []
    for candidate in record.candidate_set:
        tokens = set(str(candidate.label or "").lower().split()) | {candidate.candidate_id.lower()}
        scored.append((len(query & tokens), candidate.candidate_id))
    if not scored:
        return None
    scored.sort(reverse=True)
    if scored[0][0] == 0:
        return None
    candidate_id = scored[0][1]
    return _single_prediction(record, candidate_id)


def type_baseline(record: BindingRecord) -> dict[str, Any] | None:
    target_type = record.binding_structure.target_type
    matches = [candidate.candidate_id for candidate in record.candidate_set if candidate.candidate_type == target_type]
    if len(matches) != 1:
        return None
    return _single_prediction(record, matches[0])


def benchmark_manifest(records: list[BindingRecord], groups: list[dict[str, Any]], gate_summary: list[dict[str, Any]]) -> dict[str, Any]:
    formats = {}
    for row in gate_summary:
        fmt = row["format"]
        formats[fmt] = {
            "status": "generated",
            "n_cases": row["n"],
            "accepted_rate": row["accepted_rate"],
            "capabilities": _format_capabilities(fmt),
        }
    return {
        "representations": formats,
        "cross_format_groups": len(groups),
        "model_pilot": "not_run",
        "vicob_dependency": False,
    }


def _validate_group(group: dict[str, Any], records: list[BindingRecord]) -> dict[str, Any]:
    by_case = {record.case_id: record for record in records}
    case_ids = list(group["case_ids"].values())
    canonical = {by_case[case_id].gold.semantic_target for case_id in case_ids}
    surface_targets = [group["surface_targets"][fmt] for fmt in group["formats"]]
    return {
        "latent_id": group["latent_id"],
        "formats": "|".join(group["formats"]),
        "canonical_target": group["canonical_target"],
        "shared_canonical_target": len(canonical) == 1,
        "surface_targets_distinct": len(set(surface_targets)) == len(surface_targets),
        "valid_group": len(canonical) == 1,
    }


def _single_prediction(record: BindingRecord, candidate_id: str) -> dict[str, Any]:
    fmt = record.metadata.get("format")
    if fmt == "html_dom":
        return {"dom_id": candidate_id}
    if fmt == "table":
        return {"cell_id": candidate_id}
    return {"target_id": candidate_id}


def _prediction_exact(prediction: dict[str, Any] | None, gold: Any) -> bool:
    if prediction is None:
        return False
    if isinstance(gold, dict):
        return all(prediction.get(key) == value for key, value in gold.items())
    return prediction.get("target_id") == gold


def _gold_unique(record: BindingRecord) -> bool:
    oracle = structural_oracle(record)
    return oracle is not None and _prediction_exact(oracle, record.gold.executable_target)


def _target_leaks(record: BindingRecord) -> bool:
    text = " ".join([record.instruction or "", *(candidate.label or "" for candidate in record.candidate_set)])
    target = _surface_target(record.gold.executable_target)
    return bool(target and target in text)


def _surface_target(target: Any) -> str | None:
    if isinstance(target, dict):
        for value in target.values():
            if isinstance(value, str):
                return value
    return target if isinstance(target, str) else None


def _gold_candidate_rank(record: BindingRecord) -> str:
    candidate_ids = [candidate.candidate_id for candidate in record.candidate_set]
    target = record.gold.executable_target
    values = list(target.values()) if isinstance(target, dict) else [target]
    ranks = [candidate_ids.index(value) if value in candidate_ids else -1 for value in values]
    return "|".join(str(rank) for rank in ranks)


def _interface_fingerprint(record: BindingRecord) -> str:
    fmt = str(record.metadata.get("format") or record.source_family)
    candidates = []
    for candidate in record.candidate_set:
        row: dict[str, Any] = {
            "candidate_id": candidate.candidate_id,
            "candidate_type": candidate.candidate_type,
            "bbox": candidate.bbox,
        }
        if fmt == "graph":
            row["role"] = candidate.attributes.get("role")
            row["node"] = _surface_for_latent(record, candidate.attributes.get("node"))
            if record.metadata.get("endpoint_explicit"):
                row["endpoint_candidate"] = True
        candidates.append({key: value for key, value in row.items() if value not in (None, [], {})})
    payload = {
        "instruction": record.instruction,
        "source": record.observation.source,
        "candidates": candidates,
        "endpoint_constraint": bool(fmt == "graph" and record.metadata.get("endpoint_explicit")),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _surface_for_latent(record: BindingRecord, latent_id: Any) -> str | None:
    if latent_id is None:
        return None
    return (record.metadata.get("mapping_latent_to_surface") or {}).get(str(latent_id))


def _candidate_order_seed(seed: int, record: BindingRecord) -> int:
    token = f"{seed}:{record.case_id}:{record.metadata.get('format')}"
    return sum((index + 1) * ord(char) for index, char in enumerate(token))


def _relation_holds(candidate, anchor, relation: str | None) -> bool:
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
        distance = (cx - ax) ** 2 + (cy - ay) ** 2
        return distance <= 30**2
    return False


def _format_capabilities(fmt: str) -> list[str]:
    return {
        "svg": ["identity", "attribute", "relation", "executable", "cross_format"],
        "canvas2d": ["identity", "attribute", "relation", "executable", "cross_format"],
        "tree": ["identity", "hierarchy", "role", "executable", "cross_format"],
        "html_dom": ["identity", "hierarchy", "role", "executable", "cross_format"],
        "graph": ["identity", "relation", "ordered_role", "executable"],
        "table": ["identity", "attribute", "structural_reference", "executable"],
    }.get(fmt, [])


def _rate(rows: list[dict[str, Any]], field: str) -> float | None:
    return sum(int(bool(row[field])) for row in rows) / len(rows) if rows else None


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
