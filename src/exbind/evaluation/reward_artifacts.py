from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from exbind.generators.hierarchical import generate_hierarchical_batch
from exbind.generators.relational import generate_relational_batch
from exbind.reward import exact_binding_reward
from exbind.reward.adapters import assert_no_reward_leakage, policy_observation
from exbind.reward.structured_reward import compute_structured_reward


def run_reward_artifacts(*, out_dir: Path, dry_run: bool = False) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = generate_relational_batch(2, seed=11, distractors=4) + generate_hierarchical_batch(2, seed=11)
    rows = []
    examples = []
    for record in records:
        gold_prediction = {
            **dict(record.gold.executable_target),
            "target_type": record.binding_structure.target_type,
        }
        if record.binding_structure.relation is not None:
            gold_prediction["relation"] = record.binding_structure.relation
        if record.binding_structure.hierarchy_path is not None:
            gold_prediction["hierarchy_path"] = record.binding_structure.hierarchy_path
        partial_prediction = {"target_type": record.binding_structure.target_type}
        if record.binding_structure.relation is not None:
            partial_prediction["relation"] = record.binding_structure.relation
        if record.binding_structure.hierarchy_path is not None:
            partial_prediction["hierarchy_path"] = record.binding_structure.hierarchy_path
        for name, prediction in (("gold", gold_prediction), ("partial_structured", partial_prediction)):
            reward = compute_structured_reward(prediction, record)
            rows.append(
                {
                    "case_id": record.case_id,
                    "family": record.source_family,
                    "prediction_name": name,
                    "exact_reward": exact_binding_reward(prediction, record),
                    **reward,
                }
            )
            examples.append({"case_id": record.case_id, "prediction_name": name, "prediction": prediction, "reward": reward})
        payload = policy_observation(record)
        assert_no_reward_leakage(payload, record)
    if not dry_run:
        _write_csv(out_dir / "reward_component_tests.csv", rows)
        _write_jsonl(out_dir / "reward_examples.jsonl", examples)
    return {"rows": rows, "examples": examples}


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
