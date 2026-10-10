from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from exbind.converters.evg import convert_evg_case
from exbind.evaluation.legacy_adapter import normalize_stage_a_row


DATASET_ROOTS = {
    "EVG-Bench": Path("vlm_diag/dataset/evg_bench"),
    "EVG-Bench-Perturbed": Path("vlm_diag/dataset/evg_bench_perturbed"),
    "Novel-SVG": Path("vlm_diag/dataset/evg_external_50"),
    "SEB-hard": Path("vlm_diag/dataset/svgeditbench_v2_subset"),
    "SEB-random": Path("vlm_diag/dataset/svgeditbench_v2_random"),
    "SEB-perturbed": Path("vlm_diag/dataset/svgeditbench_v2_perturbed"),
}


def run_failure_taxonomy(*, predictions_path: Path, out_dir: Path, dry_run: bool = False) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(predictions_path)
    joined: list[dict[str, Any]] = []
    tax_rows: list[dict[str, Any]] = []
    audit = {
        "input_rows": len(rows),
        "matched_rows": 0,
        "unmatched_rows": 0,
        "duplicate_keys": {},
        "ambiguous_joins": 0,
        "excluded_rows": [],
    }
    seen = Counter((row.get("sample_id"), row.get("model_name"), row.get("dataset"), row.get("method")) for row in rows)
    audit["duplicate_keys"] = {str(key): count for key, count in seen.items() if count > 1}
    record_cache: dict[tuple[str, str], Any] = {}
    for row in rows:
        suite = str(row.get("dataset") or "")
        case_id = str(row.get("sample_id") or "")
        try:
            record = _record_for(suite, case_id, record_cache)
        except FileNotFoundError:
            audit["unmatched_rows"] += 1
            audit["excluded_rows"].append({"sample_id": case_id, "suite": suite, "reason": "missing_converted_record"})
            continue
        audit["matched_rows"] += 1
        norm = normalize_stage_a_row(row)
        raw_failure = classify_failure(row.get("raw_prediction"), row.get("gold"), row.get("task"), record, bool(row.get("valid_raw")), bool(row.get("exact_raw")))
        vicob_failure = classify_failure(row.get("final_prediction"), row.get("gold"), row.get("task"), record, bool(row.get("valid_final")), bool(row.get("exact_final")))
        transition = _transition(bool(row.get("exact_raw")), bool(row.get("exact_final")))
        joined_row = {
            **norm.to_dict(),
            "raw_failure_stage": raw_failure,
            "vicob_failure_stage": vicob_failure,
            "transition": transition,
        }
        joined.append(joined_row)
        tax_rows.append(
            {
                "case_id": case_id,
                "parent_case_id": norm.parent_case_id,
                "suite": norm.suite,
                "model": norm.model,
                "operation": norm.operation,
                "identifier_regime": norm.identifier_regime,
                "perturbation": norm.perturbation or "clean",
                "task": norm.role,
                "raw_failure_stage": raw_failure,
                "vicob_failure_stage": vicob_failure,
                "raw_valid": norm.raw_valid,
                "vicob_valid": norm.pred_valid,
                "raw_correct": norm.raw_correct,
                "vicob_correct": norm.pred_correct,
                "transition": transition,
            }
        )
    outputs = {
        "raw_failure_distribution": _distribution(tax_rows, "raw_failure_stage"),
        "vicob_residual_failure_distribution": _distribution(tax_rows, "vicob_failure_stage"),
        "repair_by_failure_type": _repair_by_failure(tax_rows),
        "transition_matrix": _transition_matrix(tax_rows),
        "failure_by_operation": _group_distribution(tax_rows, "operation", "raw_failure_stage"),
        "failure_by_model": _group_distribution(tax_rows, "model", "raw_failure_stage"),
        "failure_by_suite": _group_distribution(tax_rows, "suite", "raw_failure_stage"),
    }
    if not dry_run:
        _write_jsonl(out_dir / "stage_a_joined_rows.jsonl", joined)
        _write_csv(out_dir / "failure_taxonomy_rows.csv", tax_rows)
        _write_csv(out_dir / "raw_failure_distribution.csv", outputs["raw_failure_distribution"])
        _write_csv(out_dir / "vicob_residual_failure_distribution.csv", outputs["vicob_residual_failure_distribution"])
        _write_csv(out_dir / "repair_by_failure_type.csv", outputs["repair_by_failure_type"])
        _write_csv(out_dir / "transition_matrix.csv", outputs["transition_matrix"])
        _write_csv(out_dir / "failure_by_operation.csv", outputs["failure_by_operation"])
        _write_csv(out_dir / "failure_by_model.csv", outputs["failure_by_model"])
        _write_csv(out_dir / "failure_by_suite.csv", outputs["failure_by_suite"])
        (out_dir / "join_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return {"rows": tax_rows, "audit": audit, **outputs}


def classify_failure(prediction: Any, gold: Any, task: Any, record: Any, valid: bool, correct: bool) -> str:
    if correct:
        return "correct_binding"
    if not isinstance(prediction, dict):
        return "invalid_reference"
    task = str(task or "")
    pred_ids = _binding_ids(prediction)
    candidate_ids = {candidate.candidate_id for candidate in record.candidate_set}
    if not valid:
        if pred_ids and candidate_ids and not pred_ids <= candidate_ids:
            return "candidate_absent_or_unresolved"
        if _wrong_shape(prediction, task):
            return "wrong_executable_type"
        return "invalid_reference"
    if task == "endpoint_grounding" and isinstance(gold, dict):
        if prediction.get("source_id") == gold.get("target_id") and prediction.get("target_id") == gold.get("source_id"):
            return "wrong_endpoint_order"
        if prediction.get("source_id") != gold.get("source_id") or prediction.get("target_id") != gold.get("target_id"):
            return "wrong_endpoint_role"
    if task == "group_grounding":
        return "wrong_group_or_hierarchy"
    if task == "edge_rejection":
        return "existing_edge_or_illegal_pair"
    return "valid_but_wrong_binding"


def _record_for(suite: str, case_id: str, cache: dict[tuple[str, str], Any]) -> Any:
    key = (suite, case_id)
    if key in cache:
        return cache[key]
    root = DATASET_ROOTS.get(suite)
    if root is None:
        raise FileNotFoundError(suite)
    sample_dir = root / case_id
    if not (sample_dir / "meta.json").exists():
        raise FileNotFoundError(str(sample_dir))
    cache[key] = convert_evg_case(sample_dir, source_family=suite)
    return cache[key]


def _binding_ids(binding: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for key in ("target_id", "source_id", "wrong_edge", "group_id", "dom_id"):
        value = binding.get(key)
        if isinstance(value, str) and value:
            ids.add(value)
    return ids


def _wrong_shape(prediction: dict[str, Any], task: str) -> bool:
    expected = {
        "object_grounding": {"target_id"},
        "endpoint_grounding": {"source_id", "target_id"},
        "edge_rejection": {"wrong_edge"},
        "group_grounding": {"group_id"},
    }.get(task)
    return bool(expected and not expected <= set(prediction))


def _transition(raw_correct: bool, vicob_correct: bool) -> str:
    if raw_correct and vicob_correct:
        return "correct_retained"
    if not raw_correct and vicob_correct:
        return "wrong_repaired"
    if not raw_correct and not vicob_correct:
        return "wrong_remains_wrong"
    return "correct_regressed"


def _distribution(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    counts = Counter(str(row.get(field)) for row in rows)
    total = sum(counts.values())
    return [{"category": key, "n": value, "rate": value / total if total else None} for key, value in sorted(counts.items())]


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


def _repair_by_failure(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["raw_failure_stage"] != "correct_binding":
            groups[row["raw_failure_stage"]].append(row)
    out = []
    for category, items in sorted(groups.items()):
        repaired = sum(1 for row in items if row["transition"] == "wrong_repaired")
        regressed = sum(1 for row in items if row["transition"] == "correct_regressed")
        out.append(
            {
                "raw_failure_stage": category,
                "n_raw_failures": len(items),
                "repaired": repaired,
                "repair_rate": repaired / len(items) if items else None,
                "regressed": regressed,
                "regression_rate": regressed / len(items) if items else None,
            }
        )
    return out


def _transition_matrix(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter((row["raw_failure_stage"], row["vicob_failure_stage"], row["transition"]) for row in rows)
    return [
        {"raw_failure_stage": raw, "vicob_failure_stage": vicob, "transition": transition, "n": n}
        for (raw, vicob, transition), n in sorted(counts.items())
    ]


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
