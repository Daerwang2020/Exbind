#!/usr/bin/env python3
"""Generate the small, separately versioned ExBind-v2 hard pilot."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from exbind.evaluation.cross_format import generator_gate_row, structural_oracle
from exbind.generators.hard_pilot import make_hard_hierarchy_latent, make_hard_table_latent
from exbind.representations import compile_table, compile_tree
from exbind.representations.base import rendered_to_record


LEVELS = (("easy", 2), ("medium", 3), ("hard", 4))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def compile_hard_record(latent, *, seed: int):
    if latent.metadata["hard_family"] == "hierarchy":
        rendered = compile_tree(latent)
        rendered.instruction = (
            "Select the output at hierarchy path "
            + " > ".join(latent.target_spec["hierarchy_path"])
            + "."
        )
    else:
        rendered = compile_table(latent)
    record = rendered_to_record(rendered, latent, generator_version="exbind_v2.hard_pilot.v1")
    record.split = "exbind_v2_hard_pilot_v1"
    record.metadata.update(latent.metadata)
    record.metadata["hard_level"] = latent.difficulty_tags["difficulty"]
    order_seed = seed + sum(ord(char) for char in record.case_id)
    random.Random(order_seed).shuffle(record.candidate_set)
    record.metadata["candidate_order_seed"] = order_seed
    record.metadata["surface_variant"] = "hard_pilot_v1"
    return record


def gate_report(records) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [generator_gate_row(record) for record in records]
    for row, record in zip(rows, records):
        row["hard_family"] = record.metadata["hard_family"]
        row["hard_level"] = record.metadata["hard_level"]
        row["difficulty_axis"] = "hierarchy_depth" if record.metadata["hard_family"] == "hierarchy" else "table_ambiguity"
        row["difficulty_value"] = record.metadata.get("hierarchy_depth") or record.metadata.get("table_ambiguity")
        row["structural_oracle_correct"] = structural_oracle(record) == record.gold.executable_target
        row["accepted"] = bool(row["accepted"] and row["structural_oracle_correct"])
    by_level: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_level[(row["hard_family"], row["hard_level"])].append(row)
    summary = {
        "version": "exbind_v2_hard_pilot_v1",
        "records": len(records),
        "accepted": sum(bool(row["accepted"]) for row in rows),
        "rejected": sum(not bool(row["accepted"]) for row in rows),
        "structural_oracle_accuracy": sum(bool(row["structural_oracle_correct"]) for row in rows) / len(rows),
        "levels": {
            f"{family}:{level}": {
                "n": len(items),
                "accepted": sum(bool(row["accepted"]) for row in items),
                "rejected": sum(not bool(row["accepted"]) for row in items),
                "structural_oracle_accuracy": sum(bool(row["structural_oracle_correct"]) for row in items) / len(items),
                "target_id_leakage_rate": sum(bool(row["target_id_leakage"]) for row in items) / len(items),
                "candidate_order_shortcut_rate": sum(bool(row["candidate_order_shortcut"]) for row in items) / len(items),
                "lexical_baseline_accuracy": sum(bool(row["lexical_baseline_correct"]) for row in items) / len(items),
                "type_baseline_accuracy": sum(bool(row["type_baseline_correct"]) for row in items) / len(items),
                "gold_candidate_ranks": sorted({row["gold_candidate_rank"] for row in items}),
            }
            for (family, level), items in sorted(by_level.items())
        },
        "failure_counts": dict(Counter(
            "structural_oracle" if not row["structural_oracle_correct"] else "gate_rejected"
            for row in rows
            if not row["accepted"]
        )),
    }
    return rows, summary


def write_gate_doc(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# ExBind-v2 Hard Pilot Gate Report",
        "",
        "Version: `exbind_v2_hard_pilot_v1`.",
        "",
        f"Generated records: {summary['records']}; accepted: {summary['accepted']}; rejected: {summary['rejected']}; structural oracle: {summary['structural_oracle_accuracy']:.3f}.",
        "",
        "## Gate contract",
        "",
        "- unique gold and deterministic latent-to-surface mapping",
        "- non-empty candidate set",
        "- structural oracle exact match",
        "- no target-ID leakage",
        "- lexical-only and type-only baselines do not solve the case",
        "- difficulty metadata matches the requested family and level",
        "",
        "| Family | Level | N | Accepted | Rejected | Oracle | ID leak | Order shortcut | Lexical | Type | Gold ranks |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for key, item in summary["levels"].items():
        family, level = key.split(":", 1)
        lines.append(f"| {family} | {level} | {item['n']} | {item['accepted']} | {item['rejected']} | {item['structural_oracle_accuracy']:.3f} | {item['target_id_leakage_rate']:.3f} | {item['candidate_order_shortcut_rate']:.3f} | {item['lexical_baseline_accuracy']:.3f} | {item['type_baseline_accuracy']:.3f} | {item['gold_candidate_ranks']} |")
    lines.extend(["", "No model inference is represented by this gate report. Rejected cases must not be included in model evaluation."])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts/exbind_v2/hard_pilot_v1"))
    parser.add_argument("--per-level", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()

    records = []
    latents = []
    index = 0
    for level, depth in LEVELS:
        for _ in range(args.per_level):
            latent = make_hard_hierarchy_latent(index, depth=depth, level=level)
            latents.append(latent)
            records.append(compile_hard_record(latent, seed=args.seed))
            index += 1
        for _ in range(args.per_level):
            latent = make_hard_table_latent(index, level=level)
            latents.append(latent)
            records.append(compile_hard_record(latent, seed=args.seed))
            index += 1
    gates, summary = gate_report(records)
    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "latent_instances.jsonl", [latent.to_dict() for latent in latents])
    write_jsonl(args.out / "binding_records.jsonl", [record.to_dict() for record in records if record.metadata.get("hard_level") in {"easy", "medium", "hard"}])
    write_jsonl(args.out / "cross_format_groups.jsonl", [])
    write_csv(args.out / "generator_gates.csv", gates)
    (args.out / "gate_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out / "hard_pilot_manifest.json").write_text(
        json.dumps(
            {
                "version": "exbind_v2_hard_pilot_v1",
                "records": len(records),
                "families": ["hierarchy", "table"],
                "levels": ["easy", "medium", "hard"],
                "per_family_per_level": args.per_level,
                "seed": args.seed,
                "generator_version": "exbind_v2.hard_pilot.v1",
                "records_path": str(args.out / "binding_records.jsonl"),
                "groups_path": str(args.out / "cross_format_groups.jsonl"),
                "gate_summary": str(args.out / "gate_summary.json"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_gate_doc(Path("docs/EXBIND_V2_HARD_PILOT_GATE_REPORT.md"), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["rejected"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
