from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from exbind.evaluation.legacy_adapter import normalize_candidate_ids


RECONCILIATION_FIELDS = [
    "case_id",
    "model",
    "suite",
    "operation",
    "gold_binding",
    "historical_raw_binding",
    "historical_vicob_binding",
    "normalized_raw_binding",
    "normalized_vicob_binding",
    "historical_raw_correct",
    "normalized_raw_correct",
    "historical_vicob_correct",
    "normalized_vicob_correct",
    "candidate_ids",
    "difference_reason",
]


def run_stage_a_reconciliation(
    *,
    historical_path: Path,
    normalized_path: Path,
    out_dir: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    historical = _read_jsonl(historical_path)
    normalized = _read_jsonl(normalized_path)
    historical_by_key = {_key(row): row for row in historical}
    normalized_by_key = {_key(row): row for row in normalized}

    rows: list[dict[str, Any]] = []
    missing_from_normalized = sorted(set(historical_by_key) - set(normalized_by_key))
    missing_from_historical = sorted(set(normalized_by_key) - set(historical_by_key))
    for key in sorted(set(historical_by_key) | set(normalized_by_key)):
        old = historical_by_key.get(key)
        new = normalized_by_key.get(key)
        if old is None:
            rows.append(_missing_row(new, "missing_historical_row"))
            continue
        if new is None:
            rows.append(_missing_row(old, "missing_normalized_row"))
            continue
        rows.append(_reconcile_row(old, new))

    summary = _summary(rows, historical_path, normalized_path, missing_from_historical, missing_from_normalized)
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(out_dir / "stage_a_reconciliation.csv", rows)
        (out_dir / "stage_a_reconciliation_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
    return {"rows": rows, "summary": summary}


def _reconcile_row(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    candidate_ids = _candidate_ids(old) or _candidate_ids(new)
    reason = _difference_reason(old, new, candidate_ids)
    return {
        "case_id": str(old.get("sample_id") or new.get("sample_id") or ""),
        "model": str(old.get("model_name") or new.get("model_name") or ""),
        "suite": str(old.get("dataset") or new.get("dataset") or ""),
        "operation": str(old.get("operation") or new.get("operation") or ""),
        "gold_binding": _json_cell(old.get("gold", new.get("gold"))),
        "historical_raw_binding": _json_cell(old.get("raw_prediction")),
        "historical_vicob_binding": _json_cell(old.get("final_prediction")),
        "normalized_raw_binding": _json_cell(new.get("raw_prediction")),
        "normalized_vicob_binding": _json_cell(new.get("final_prediction")),
        "historical_raw_correct": _bool_cell(old.get("exact_raw")),
        "normalized_raw_correct": _bool_cell(new.get("exact_raw")),
        "historical_vicob_correct": _bool_cell(old.get("exact_final")),
        "normalized_vicob_correct": _bool_cell(new.get("exact_final")),
        "candidate_ids": _json_cell(candidate_ids),
        "difference_reason": reason,
    }


def _difference_reason(old: dict[str, Any], new: dict[str, Any], candidate_ids: list[str]) -> str:
    if old.get("gold") != new.get("gold"):
        return "gold_canonicalization_difference"
    if old.get("raw_prediction") != new.get("raw_prediction") or bool(old.get("exact_raw")) != bool(new.get("exact_raw")):
        return "legacy_prediction_field_selection_or_raw_normalization"
    if _candidate_ids(old) != _candidate_ids(new):
        return "candidate_projection_difference"
    if old.get("final_prediction") == new.get("final_prediction") and bool(old.get("exact_final")) == bool(new.get("exact_final")):
        return "no_difference"

    task = str(old.get("task") or new.get("task") or "")
    old_correct = bool(old.get("exact_final"))
    new_correct = bool(new.get("exact_final"))
    old_final = old.get("final_prediction")
    new_final = new.get("final_prediction")
    gold = old.get("gold")
    raw = old.get("raw_prediction")
    ids_are_generic = bool(candidate_ids) and all(_looks_generic(candidate_id) for candidate_id in candidate_ids)

    if (
        task == "object_grounding"
        and ids_are_generic
        and old_correct
        and not new_correct
        and old_final == gold
        and new_final == raw
        and bool(old.get("valid_final"))
        and bool(new.get("valid_final"))
    ):
        margin = _winner_minus_raw(new, raw, old_final)
        if margin is not None and margin <= 0.1:
            return "valid_raw_retention_margin_delta_vicob_zero_vs_vicob_v2"
        return "generic_object_rerank_policy_difference"

    if task == "endpoint_grounding":
        return "ordered_endpoint_or_valid_endpoint_retention"
    if task == "group_grounding":
        return "group_set_or_alias_normalization"
    if old_final is None or new_final is None:
        return "missing_invalid_legacy_values"
    return "scoring_or_gate_route_difference"


def _winner_minus_raw(new: dict[str, Any], raw: Any, historical_final: Any) -> float | None:
    if not isinstance(raw, dict) or not isinstance(historical_final, dict):
        return None
    raw_id = raw.get("target_id") or raw.get("group_id") or raw.get("wrong_edge")
    winner_id = historical_final.get("target_id") or historical_final.get("group_id") or historical_final.get("wrong_edge")
    scores = new.get("scores")
    if not isinstance(scores, dict) or raw_id not in scores or winner_id not in scores:
        return None
    try:
        return float(scores[winner_id]) - float(scores[raw_id])
    except (TypeError, ValueError):
        return None


def _summary(
    rows: list[dict[str, Any]],
    historical_path: Path,
    normalized_path: Path,
    missing_from_historical: list[tuple[Any, Any, Any]],
    missing_from_normalized: list[tuple[Any, Any, Any]],
) -> dict[str, Any]:
    reason_counts = Counter(row["difference_reason"] for row in rows)
    divergent = [
        row
        for row in rows
        if row["historical_vicob_correct"] != row["normalized_vicob_correct"]
        or row["historical_vicob_binding"] != row["normalized_vicob_binding"]
        or row["difference_reason"] not in {"no_difference"}
    ]
    historical_correct = sum(row["historical_vicob_correct"] == "true" for row in rows)
    normalized_correct = sum(row["normalized_vicob_correct"] == "true" for row in rows)
    return {
        "historical_path": str(historical_path),
        "normalized_path": str(normalized_path),
        "n_rows": len(rows),
        "historical_vicob_correct": historical_correct,
        "normalized_vicob_correct": normalized_correct,
        "correct_delta_historical_minus_normalized": historical_correct - normalized_correct,
        "divergent_rows": len(divergent),
        "difference_reason_counts": dict(sorted(reason_counts.items())),
        "missing_from_historical": len(missing_from_historical),
        "missing_from_normalized": len(missing_from_normalized),
    }


def _missing_row(row: dict[str, Any] | None, reason: str) -> dict[str, Any]:
    row = row or {}
    return {
        "case_id": str(row.get("sample_id") or ""),
        "model": str(row.get("model_name") or ""),
        "suite": str(row.get("dataset") or ""),
        "operation": str(row.get("operation") or ""),
        "gold_binding": _json_cell(row.get("gold")),
        "historical_raw_binding": "",
        "historical_vicob_binding": "",
        "normalized_raw_binding": "",
        "normalized_vicob_binding": "",
        "historical_raw_correct": "",
        "normalized_raw_correct": "",
        "historical_vicob_correct": "",
        "normalized_vicob_correct": "",
        "candidate_ids": _json_cell(_candidate_ids(row)),
        "difference_reason": reason,
    }


def _candidate_ids(row: dict[str, Any]) -> list[str]:
    return normalize_candidate_ids(
        row.get("candidate_ids")
        or row.get("candidate_id")
        or row.get("scores")
        or row.get("logit_scores")
        or row.get("candidate_scores")
    )


def _key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("sample_id") or ""),
        str(row.get("model_name") or row.get("model") or ""),
        str(row.get("dataset") or row.get("split") or ""),
    )


def _looks_generic(candidate_id: str) -> bool:
    tail = candidate_id.rsplit("_", 1)[-1]
    return tail.isdigit() or candidate_id.startswith(("path_", "shape_", "dom_"))


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _bool_cell(value: Any) -> str:
    if value is None:
        return ""
    return "true" if bool(value) else "false"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RECONCILIATION_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
