from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from exbind.evaluation.counterfactual import transform_binding
from exbind.evaluation.structural_pilot import structural_oracle
from exbind.generators.hierarchical import generate_hierarchical_batch
from exbind.generators.relational import generate_relational_batch
from exbind.generators.transforms import symbolic_id_rename
from exbind.schema import BindingRecord


def run_strict_counterfactual(
    *,
    out_dir: Path,
    n_pairs: int = 100,
    seed: int = 20260823,
    dry_run: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    half = n_pairs // 2
    records = generate_relational_batch(half, seed=seed, distractors=4)
    records.extend(generate_hierarchical_batch(n_pairs - half, seed=seed))
    pairs = []
    validations = []
    for index, record in enumerate(records):
        child = symbolic_id_rename(record, prefix=f"cf_{index:04d}")
        pair = _pair_row(record, child)
        pairs.append(pair)
        validations.append(validate_pair(record, child))
    summary = _summary(validations)
    model_rows = [
        {
            "method": "frozen_model",
            "model": "not_run",
            "status": "blocked_until_validator_passes_and_inference_is_approved",
            "n_pairs": "",
            "cf_consistency": "",
            "cf_consistency_correct": "",
            "cf_consistency_wrong": "",
            "exact_accuracy": "",
        }
    ]
    if not dry_run:
        _write_jsonl(out_dir / "strict_pairs.jsonl", pairs)
        _write_csv(out_dir / "validation_report.csv", validations)
        _write_csv(out_dir / "model_results.csv", model_rows)
        _write_csv(out_dir / "counterfactual_summary.csv", summary)
    return {"pairs": pairs, "validation": validations, "model_results": model_rows, "summary": summary}


def validate_pair(parent: BindingRecord, child: BindingRecord) -> dict[str, Any]:
    mapping = (child.transformation.metadata or {}).get("id_map") if child.transformation else {}
    parent_ids = [candidate.candidate_id for candidate in parent.candidate_set]
    child_ids = [candidate.candidate_id for candidate in child.candidate_set]
    mapped_parent_ids = [mapping.get(candidate_id) for candidate_id in parent_ids]
    bijection_valid = bool(mapping) and set(mapping) == set(parent_ids) and len(set(mapping.values())) == len(parent_ids)
    candidate_inventory_preserved = sorted(mapped_parent_ids) == sorted(child_ids)
    operation_preserved = parent.operation == child.operation
    gold_remapping_correct = transform_binding(parent.gold.executable_target, mapping) == child.gold.executable_target
    semantic_preserved = parent.gold.semantic_target == child.gold.semantic_target
    oracle_prediction = structural_oracle(child)
    child_gold_id = _target_id(child)
    oracle_correct = oracle_prediction == child_gold_id
    valid = all(
        [
            bijection_valid,
            candidate_inventory_preserved,
            operation_preserved,
            gold_remapping_correct,
            semantic_preserved,
            oracle_correct,
        ]
    )
    return {
        "parent_case_id": parent.case_id,
        "child_case_id": child.case_id,
        "family": parent.source_family,
        "bijection_valid": bijection_valid,
        "candidate_inventory_preserved": candidate_inventory_preserved,
        "operation_preserved": operation_preserved,
        "gold_remapping_correct": gold_remapping_correct,
        "semantic_preserved": semantic_preserved,
        "structural_oracle_correct": oracle_correct,
        "valid_pair": valid,
        "id_map_size": len(mapping),
        "invalid_reason": "" if valid else _invalid_reason(
            bijection_valid,
            candidate_inventory_preserved,
            operation_preserved,
            gold_remapping_correct,
            semantic_preserved,
            oracle_correct,
        ),
    }


def _pair_row(parent: BindingRecord, child: BindingRecord) -> dict[str, Any]:
    mapping = (child.transformation.metadata or {}).get("id_map") if child.transformation else {}
    return {
        "parent_case_id": parent.case_id,
        "child_case_id": child.case_id,
        "family": parent.source_family,
        "transformation": "symbolic_id_rename",
        "id_map": mapping,
        "parent_gold": parent.gold.executable_target,
        "child_gold": child.gold.executable_target,
        "parent_record": parent.to_dict(),
        "child_record": child.to_dict(),
    }


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for family in sorted({row["family"] for row in rows}):
        items = [row for row in rows if row["family"] == family]
        n = len(items)
        out.append(
            {
                "family": family,
                "n_pairs": n,
                "valid_pair_rate": _rate(items, "valid_pair"),
                "bijection_valid_rate": _rate(items, "bijection_valid"),
                "candidate_inventory_preserved_rate": _rate(items, "candidate_inventory_preserved"),
                "gold_remapping_correct_rate": _rate(items, "gold_remapping_correct"),
                "semantic_preserved_rate": _rate(items, "semantic_preserved"),
                "structural_oracle_accuracy": _rate(items, "structural_oracle_correct"),
            }
        )
    all_n = len(rows)
    out.append(
        {
            "family": "all",
            "n_pairs": all_n,
            "valid_pair_rate": _rate(rows, "valid_pair"),
            "bijection_valid_rate": _rate(rows, "bijection_valid"),
            "candidate_inventory_preserved_rate": _rate(rows, "candidate_inventory_preserved"),
            "gold_remapping_correct_rate": _rate(rows, "gold_remapping_correct"),
            "semantic_preserved_rate": _rate(rows, "semantic_preserved"),
            "structural_oracle_accuracy": _rate(rows, "structural_oracle_correct"),
        }
    )
    return out


def _invalid_reason(*checks: bool) -> str:
    names = (
        "bijection_invalid",
        "candidate_inventory_changed",
        "operation_changed",
        "gold_remapping_failed",
        "semantic_changed",
        "structural_oracle_failed",
    )
    return ";".join(name for name, ok in zip(names, checks) if not ok)


def _target_id(record: BindingRecord) -> str | None:
    target = record.gold.executable_target
    if isinstance(target, dict):
        value = target.get("target_id")
        return str(value) if value is not None else None
    return str(target) if target is not None else None


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
