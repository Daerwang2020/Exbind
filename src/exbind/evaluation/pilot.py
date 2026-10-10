from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from exbind.generators.hierarchical import generate_hierarchical_batch
from exbind.generators.relational import generate_relational_batch
from exbind.schema import BindingRecord


def run_pilot(*, out_dir: Path, n: int = 50, seed: int = 20260823, dry_run: bool = False) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    relational = generate_relational_batch(n, seed=seed, distractors=4)
    hierarchical = generate_hierarchical_batch(n, seed=seed)
    rows = [_baseline_row(record) for record in relational + hierarchical]
    summary = _summary(rows)
    manual = [row for row in rows if row["manual_review"]]
    manifest = {
        "n_relational": len(relational),
        "n_hierarchical": len(hierarchical),
        "seed": seed,
        "dry_run": dry_run,
        "generators": {
            "relational": "exbind.generators.relational.generate_relational_batch",
            "hierarchical": "exbind.generators.hierarchical.generate_hierarchical_batch",
        },
    }
    if not dry_run:
        _write_jsonl(out_dir / "relational_50.jsonl", [record.to_dict() for record in relational])
        _write_jsonl(out_dir / "hierarchical_50.jsonl", [record.to_dict() for record in hierarchical])
        _write_csv(out_dir / "pilot_baseline_rows.csv", rows)
        _write_csv(out_dir / "pilot_summary.csv", summary)
        _write_csv(out_dir / "pilot_manual_review.csv", manual)
        (out_dir / "pilot_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"rows": rows, "summary": summary, "manual": manual, "manifest": manifest}


def _baseline_row(record: BindingRecord) -> dict[str, Any]:
    gold_id = record.gold.executable_target.get("target_id") if isinstance(record.gold.executable_target, dict) else None
    text_pred = _text_baseline(record)
    text_type_pred = _text_type_baseline(record)
    unique = _unique_target(record, gold_id)
    gold_valid = gold_id in {candidate.candidate_id for candidate in record.candidate_set}
    relation_required = int(text_type_pred != gold_id)
    manual_review = not gold_valid or not unique
    return {
        "case_id": record.case_id,
        "family": record.source_family,
        "gold_id": gold_id,
        "gold_valid": gold_valid,
        "unique_target": unique,
        "candidate_count": len(record.candidate_set),
        "text_prediction": text_pred,
        "text_baseline_correct": text_pred == gold_id,
        "text_type_prediction": text_type_pred,
        "text_type_baseline_correct": text_type_pred == gold_id,
        "lexical_shortcut": text_pred == gold_id,
        "relation_required": relation_required,
        "manual_review": manual_review,
    }


def _text_baseline(record: BindingRecord) -> str | None:
    query = _tokens(record.instruction or "")
    scored = []
    for candidate in record.candidate_set:
        text = " ".join([candidate.label or "", candidate.candidate_id])
        tokens = _tokens(text)
        score = len(query & tokens) / len(query | tokens) if query | tokens else 0.0
        scored.append((score, candidate.candidate_id))
    scored.sort(reverse=True)
    return scored[0][1] if scored else None


def _text_type_baseline(record: BindingRecord) -> str | None:
    target_type = record.binding_structure.target_type
    query = _tokens(record.instruction or "")
    scored = []
    for candidate in record.candidate_set:
        text = " ".join([candidate.label or "", candidate.candidate_id, candidate.candidate_type or ""])
        tokens = _tokens(text)
        type_bonus = 1.0 if candidate.candidate_type == target_type else 0.0
        score = type_bonus + (len(query & tokens) / len(query | tokens) if query | tokens else 0.0)
        scored.append((score, candidate.candidate_id))
    scored.sort(reverse=True)
    return scored[0][1] if scored else None


def _unique_target(record: BindingRecord, gold_id: str | None) -> bool:
    if not gold_id:
        return False
    if record.source_family == "relational_synth":
        objects = record.observation.state_metadata.get("objects", [])
        relation = (record.binding_structure.relation or {}).get("type")
        anchor = (record.binding_structure.relation or {}).get("anchor")
        target_attrs = record.binding_structure.attributes
        matches = [
            obj["id"]
            for obj in objects
            if obj.get("id") != anchor
            and obj.get("shape") == target_attrs.get("shape")
            and obj.get("color") == target_attrs.get("color")
            and _satisfies_relation(obj, objects, anchor, relation)
        ]
        return matches == [gold_id]
    wanted = record.binding_structure.attributes
    matches = [
        candidate.candidate_id
        for candidate in record.candidate_set
        if all(candidate.attributes.get(key) == value for key, value in wanted.items())
    ]
    return matches == [gold_id]


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


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["family"]].append(row)
    out = []
    for family, items in sorted(groups.items()):
        n = len(items)
        out.append(
            {
                "family": family,
                "n": n,
                "gold_valid_rate": _rate(items, "gold_valid"),
                "unique_target_rate": _rate(items, "unique_target"),
                "mean_candidate_count": sum(int(row["candidate_count"]) for row in items) / n if n else None,
                "lexical_shortcut_rate": _rate(items, "lexical_shortcut"),
                "text_baseline_accuracy": _rate(items, "text_baseline_correct"),
                "text_type_baseline_accuracy": _rate(items, "text_type_baseline_correct"),
                "relation_required_rate": _rate(items, "relation_required"),
                "manual_review_count": sum(int(row["manual_review"]) for row in items),
            }
        )
    return out


def _rate(rows: list[dict[str, Any]], field: str) -> float | None:
    return sum(int(bool(row[field])) for row in rows) / len(rows) if rows else None


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


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
