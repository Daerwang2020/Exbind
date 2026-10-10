#!/usr/bin/env python3
"""Score surface-reference predictions against ExBind binding records."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exbind.evaluation.cross_format import map_prediction_to_canonical
from exbind.metrics.scoring import score_vector
from exbind.schema import BindingRecord


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize(payload: dict[str, Any], record: BindingRecord) -> dict[str, Any]:
    prediction = dict(payload.get("prediction", payload))
    if "executable_target" not in prediction:
        gold = record.gold.executable_target
        if isinstance(gold, dict):
            prediction["executable_target"] = {
                key: prediction[key] for key in gold if key in prediction
            }
        elif "target_id" in prediction:
            prediction["executable_target"] = prediction["target_id"]
    prediction["canonical_target"] = map_prediction_to_canonical(prediction, record)
    return prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = {row.case_id: row for row in (BindingRecord.from_dict(item) for item in read_jsonl(args.records))}
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    failure_counts: Counter[str] = Counter()
    for item in read_jsonl(args.predictions):
        case_id = str(item["case_id"])
        record = records[case_id]
        prediction = normalize(item, record)
        vector = score_vector(prediction, record)
        exact = bool(vector.get("executable_exact"))
        failure = "correct" if exact else ("invalid" if vector.get("candidate_valid") == 0 else "binding_mismatch")
        failure_counts[failure] += 1
        rows.append({"case_id": case_id, "score_vector": vector, "failure": failure})
    (args.out / "scores.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "n": len(rows),
        "exact": sum(row["failure"] == "correct" for row in rows),
        "valid": sum(row["score_vector"].get("candidate_valid") == 1 for row in rows),
        "failure_counts": dict(sorted(failure_counts.items())),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
