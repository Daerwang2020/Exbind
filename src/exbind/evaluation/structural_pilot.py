from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from exbind.evaluation.capabilities import annotate_record
from exbind.evaluation.pilot import _text_baseline, _text_type_baseline
from exbind.generators.hierarchical import generate_hierarchical_batch
from exbind.generators.relational import generate_relational_batch
from exbind.schema import BindingRecord


def run_structural_pilot(*, out_dir: Path, n: int = 50, seed: int = 20260823, dry_run: bool = False) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    relational = generate_relational_batch(n, seed=seed, distractors=4)
    hierarchical = generate_hierarchical_batch(n, seed=seed)
    records = relational + hierarchical
    diagnostics = [case_diagnostics(record) for record in records]
    random_rows = random_baselines(records, seed=seed)
    oracle_rows = [structural_oracle_row(record) for record in records]
    summary = structural_summary(diagnostics, random_rows, oracle_rows)
    frozen_rows = [
        {
            "method": "frozen_vlm",
            "model": "not_run",
            "status": "blocked_requires_explicit_inference_approval_after_validation",
            "identity": "",
            "relation": "",
            "hierarchy": "",
            "role": "",
            "overall": "",
        }
    ]
    manifest = {
        "n_relational": len(relational),
        "n_hierarchical": len(hierarchical),
        "seed": seed,
        "dry_run": dry_run,
        "inference_run": False,
    }
    if not dry_run:
        _write_jsonl(out_dir / "relational_50.jsonl", [record.to_dict() for record in relational])
        _write_jsonl(out_dir / "hierarchical_50.jsonl", [record.to_dict() for record in hierarchical])
        _write_csv(out_dir / "structural_case_diagnostics.csv", diagnostics)
        _write_csv(out_dir / "random_baseline.csv", random_rows)
        _write_csv(out_dir / "structural_oracle.csv", oracle_rows)
        _write_csv(out_dir / "frozen_model_results.csv", frozen_rows)
        _write_csv(out_dir / "structural_pilot_summary.csv", summary)
        (out_dir / "structural_pilot_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
    return {
        "diagnostics": diagnostics,
        "random_baseline": random_rows,
        "structural_oracle": oracle_rows,
        "frozen_model_results": frozen_rows,
        "summary": summary,
        "manifest": manifest,
    }


def case_diagnostics(record: BindingRecord) -> dict[str, Any]:
    annotation = annotate_record(record)
    gold_id = _target_id(record)
    text_pred = _text_baseline(record)
    text_type_pred = _text_type_baseline(record)
    return {
        "case_id": record.case_id,
        "family": record.source_family,
        "gold_id": gold_id,
        "candidate_count": len(record.candidate_set),
        "same_type_distractor_count": annotation["same_type_distractor_count"],
        "structurally_plausible_distractor_count": annotation["structurally_plausible_distractor_count"],
        "random_expected_accuracy": 1 / len(record.candidate_set) if record.candidate_set else 0.0,
        "relation_depth": 1 if record.binding_structure.relation else 0,
        "hierarchy_depth": len(record.binding_structure.hierarchy_path or []),
        "number_of_relations_required": 1 if annotation["capabilities"]["relation"] else 0,
        "lexical_shortcut_available": annotation["lexical_shortcut_available"],
        "type_shortcut_available": annotation["type_shortcut_available"],
        "text_prediction": text_pred,
        "text_baseline_correct": text_pred == gold_id,
        "text_type_prediction": text_type_pred,
        "text_type_baseline_correct": text_type_pred == gold_id,
        "gold_unique": annotation["gold_unique"],
    }


def random_baselines(records: list[BindingRecord], *, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    groups: dict[str, list[BindingRecord]] = defaultdict(list)
    for record in records:
        groups[record.source_family].append(record)
    for family, items in sorted(groups.items()):
        expected = sum(1 / len(record.candidate_set) for record in items if record.candidate_set) / len(items)
        empirical_correct = 0
        for record in items:
            if not record.candidate_set:
                continue
            pred = rng.choice(record.candidate_set).candidate_id
            empirical_correct += int(pred == _target_id(record))
        rows.append(
            {
                "family": family,
                "n": len(items),
                "random_expected_accuracy": expected,
                "seeded_random_accuracy": empirical_correct / len(items) if items else None,
                "seed": seed,
            }
        )
    return rows


def structural_oracle_row(record: BindingRecord) -> dict[str, Any]:
    pred = structural_oracle(record)
    gold = _target_id(record)
    return {
        "case_id": record.case_id,
        "family": record.source_family,
        "oracle_prediction": pred,
        "gold_id": gold,
        "oracle_correct": pred == gold,
        "invalid_reason": "" if pred == gold else "oracle_failed_or_case_underspecified",
    }


def structural_oracle(record: BindingRecord) -> str | None:
    if record.source_family == "relational_synth":
        objects = record.observation.state_metadata.get("objects", [])
        relation = (record.binding_structure.relation or {}).get("type")
        anchor_id = (record.binding_structure.relation or {}).get("anchor")
        attrs = record.binding_structure.attributes or {}
        candidates = [
            obj
            for obj in objects
            if obj.get("id") != anchor_id
            and obj.get("shape") == attrs.get("shape")
            and obj.get("color") == attrs.get("color")
            and _satisfies_relation(obj, objects, anchor_id, relation)
        ]
        return candidates[0]["id"] if len(candidates) == 1 else None
    if record.source_family == "hierarchical_synth":
        attrs = record.binding_structure.attributes or {}
        candidates = [
            candidate
            for candidate in record.candidate_set
            if candidate.attributes.get("group") == attrs.get("group")
            and candidate.attributes.get("module") == attrs.get("module")
            and candidate.attributes.get("part") == attrs.get("part")
        ]
        return candidates[0].candidate_id if len(candidates) == 1 else None
    return None


def structural_summary(
    diagnostics: list[dict[str, Any]],
    random_rows: list[dict[str, Any]],
    oracle_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    oracle_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    random_by_family = {row["family"]: row for row in random_rows}
    for row in diagnostics:
        by_family[row["family"]].append(row)
    for row in oracle_rows:
        oracle_by_family[row["family"]].append(row)
    out = []
    for family, rows in sorted(by_family.items()):
        oracle = oracle_by_family[family]
        random_row = random_by_family[family]
        n = len(rows)
        out.append(
            {
                "family": family,
                "n": n,
                "gold_unique_rate": _rate(rows, "gold_unique"),
                "lexical_shortcut_rate": _rate(rows, "lexical_shortcut_available"),
                "type_shortcut_rate": _rate(rows, "type_shortcut_available"),
                "text_baseline_accuracy": _rate(rows, "text_baseline_correct"),
                "text_type_baseline_accuracy": _rate(rows, "text_type_baseline_correct"),
                "mean_candidate_count": sum(int(row["candidate_count"]) for row in rows) / n if n else None,
                "mean_same_type_distractors": sum(int(row["same_type_distractor_count"]) for row in rows) / n if n else None,
                "mean_structurally_plausible_distractors": sum(int(row["structurally_plausible_distractor_count"]) for row in rows) / n if n else None,
                "mean_relation_depth": sum(int(row["relation_depth"]) for row in rows) / n if n else None,
                "mean_hierarchy_depth": sum(int(row["hierarchy_depth"]) for row in rows) / n if n else None,
                "random_expected_accuracy": random_row["random_expected_accuracy"],
                "seeded_random_accuracy": random_row["seeded_random_accuracy"],
                "structural_oracle_accuracy": _rate(oracle, "oracle_correct"),
                "manual_review_count": sum(int(not row["gold_unique"] or row["lexical_shortcut_available"]) for row in rows)
                + sum(int(not row["oracle_correct"]) for row in oracle),
            }
        )
    return out


def _satisfies_relation(obj: dict[str, Any], objects: list[dict[str, Any]], anchor_id: str, relation: str) -> bool:
    anchor = next((item for item in objects if item.get("id") == anchor_id), None)
    if not anchor:
        return False
    if relation == "left of":
        return obj["x"] < anchor["x"]
    if relation == "right of":
        return obj["x"] > anchor["x"]
    if relation == "above":
        return obj["y"] < anchor["y"]
    if relation == "below":
        return obj["y"] > anchor["y"]
    if relation == "nearest":
        others = [item for item in objects if item.get("id") != anchor_id]
        distances = sorted((math.hypot(item["x"] - anchor["x"], item["y"] - anchor["y"]), item["id"]) for item in others)
        return bool(distances and distances[0][1] == obj["id"])
    return False


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
