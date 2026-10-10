from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from exbind.evaluation.legacy_adapter import identifier_regime, normalize_stage_a_row


DATASET_ROOTS = {
    "EVG-Bench-Perturbed": Path("vlm_diag/dataset/evg_bench_perturbed"),
    "SEB-perturbed": Path("vlm_diag/dataset/svgeditbench_v2_perturbed"),
}


def run_counterfactual_analysis(
    *,
    stage_a_path: Path,
    candidate_order_path: Path,
    out_dir: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    symbolic_rows, unmatched = symbolic_id_rename_rows(stage_a_path)
    order_rows = candidate_order_rows(candidate_order_path)
    all_rows = symbolic_rows + order_rows
    summary = summarize_counterfactual_rows(all_rows)
    symbolic_summary = summarize_counterfactual_rows(symbolic_rows)
    order_summary = summarize_counterfactual_rows(order_rows)
    manifest = {
        "stage_a_path": str(stage_a_path),
        "candidate_order_path": str(candidate_order_path),
        "dry_run": dry_run,
        "n_rows": len(all_rows),
        "n_symbolic_rows": len(symbolic_rows),
        "n_order_rows": len(order_rows),
        "n_unmatched": len(unmatched),
        "strict_symbolic_note": "Rows are unavailable unless the stored transformed gold equals T(base_gold).",
    }
    if not dry_run:
        _write_jsonl(out_dir / "counterfactual_consistency_rows.jsonl", all_rows)
        _write_csv(out_dir / "counterfactual_consistency_summary.csv", summary)
        _write_csv(out_dir / "symbolic_id_rename_summary.csv", symbolic_summary)
        _write_csv(out_dir / "candidate_order_summary.csv", order_summary)
        _write_csv(out_dir / "counterfactual_unmatched_rows.csv", unmatched)
        (out_dir / "counterfactual_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"rows": all_rows, "summary": summary, "unmatched": unmatched, "manifest": manifest}


def symbolic_id_rename_rows(stage_a_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = _read_jsonl(stage_a_path)
    groups: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        norm = normalize_stage_a_row(row)
        key = (str(norm.parent_case_id), str(norm.model), str(norm.suite))
        groups[key][norm.perturbation or "clean"] = row
    usable: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    for (parent, model, suite), by_pert in sorted(groups.items()):
        base = by_pert.get("clean")
        child = by_pert.get("symbolic_id_shuffle")
        if not base or not child:
            continue
        reason = None
        root = DATASET_ROOTS.get(str(child.get("dataset")))
        if root is None:
            reason = "no_dataset_root_for_symbolic_pair"
        else:
            mapping = reconstruct_svg_id_map(root / parent / "before.svg", root / str(child.get("sample_id")) / "before.svg")
            if not mapping:
                reason = "missing_or_empty_id_map"
            elif not _is_bijective(mapping):
                reason = "non_bijective_id_map"
            elif base.get("operation") != child.get("operation"):
                reason = "operation_changed"
            else:
                mapped_gold = transform_binding(base.get("gold"), mapping)
                if mapped_gold != child.get("gold"):
                    reason = "stored_transformed_gold_not_T_base_gold"
        if reason:
            unmatched.append(
                {
                    "case_id": child.get("sample_id"),
                    "parent_case_id": parent,
                    "model": model,
                    "suite": suite,
                    "counterfactual_type": "symbolic_id_rename",
                    "reason": reason,
                }
            )
            continue
        for method, base_field, child_field, exact_field in [
            ("Raw", "raw_prediction", "raw_prediction", "exact_raw"),
            ("ViCoB", "final_prediction", "final_prediction", "exact_final"),
        ]:
            mapping = reconstruct_svg_id_map(root / parent / "before.svg", root / str(child.get("sample_id")) / "before.svg")
            expected = transform_binding(base.get(base_field), mapping)
            observed = child.get(child_field)
            usable.append(
                _counterfactual_record(
                    method=method,
                    model=model,
                    suite=suite,
                    operation=str(base.get("operation") or base.get("task")),
                    identifier_regime=identifier_regime(list(mapping.keys())),
                    counterfactual_type="symbolic_id_rename",
                    case_id=str(base.get("sample_id")),
                    transformed_case_id=str(child.get("sample_id")),
                    base_correct=bool(base.get(exact_field)),
                    transformed_correct=bool(child.get(exact_field)),
                    consistent=canonical_binding(expected) == canonical_binding(observed),
                    metadata={"expected": expected, "observed": observed},
                )
            )
    return usable, unmatched


def candidate_order_rows(candidate_order_path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(candidate_order_path)
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("sample_id")),
            str(row.get("model_name") or row.get("model")),
            str(row.get("dataset")),
            str(row.get("method")),
        )
        grouped[key].append(row)
    out: list[dict[str, Any]] = []
    for (case_id, model, suite, method), items in sorted(grouped.items()):
        if len(items) < 2:
            continue
        items = sorted(items, key=lambda row: int(row.get("candidate_order_seed") or 0))
        base = items[0]
        for child in items[1:]:
            out.append(
                _counterfactual_record(
                    method=method,
                    model=model,
                    suite=suite,
                    operation=str(base.get("operation")),
                    identifier_regime=identifier_regime(list((base.get("scores") or {}).keys())),
                    counterfactual_type="candidate_order",
                    case_id=case_id,
                    transformed_case_id=f"{case_id}:seed_{child.get('candidate_order_seed')}",
                    base_correct=bool(base.get("exact_final", base.get("is_correct"))),
                    transformed_correct=bool(child.get("exact_final", child.get("is_correct"))),
                    consistent=canonical_binding(base.get("final_prediction", base.get("prediction"))) == canonical_binding(child.get("final_prediction", child.get("prediction"))),
                    metadata={"base_seed": base.get("candidate_order_seed"), "child_seed": child.get("candidate_order_seed")},
                )
            )
        # Raw prediction is present in the same cached order rows.  Use one method
        # group per case/model/suite to avoid double-counting the identical raw rows.
        if method == sorted({str(row.get("method")) for row in rows if str(row.get("sample_id")) == case_id and str(row.get("model_name") or row.get("model")) == model and str(row.get("dataset")) == suite})[0]:
            for child in items[1:]:
                out.append(
                    _counterfactual_record(
                        method="Raw",
                        model=model,
                        suite=suite,
                        operation=str(base.get("operation") or base.get("task")),
                        identifier_regime=identifier_regime(list((base.get("scores") or {}).keys())),
                        counterfactual_type="candidate_order",
                        case_id=case_id,
                        transformed_case_id=f"{case_id}:raw_seed_{child.get('candidate_order_seed')}",
                        base_correct=bool(base.get("exact_raw")),
                        transformed_correct=bool(child.get("exact_raw")),
                        consistent=canonical_binding(base.get("raw_prediction")) == canonical_binding(child.get("raw_prediction")),
                        metadata={"base_seed": base.get("candidate_order_seed"), "child_seed": child.get("candidate_order_seed")},
                    )
                )
    return out


def reconstruct_svg_id_map(base_svg: Path, child_svg: Path) -> dict[str, str]:
    if not base_svg.exists() or not child_svg.exists():
        return {}
    base = _svg_id_signatures(base_svg)
    child = _svg_id_signatures(child_svg)
    child_by_sig: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for cid, sig in child.items():
        child_by_sig[sig].append(cid)
    mapping: dict[str, str] = {}
    for bid, sig in base.items():
        matches = child_by_sig.get(sig, [])
        if len(matches) == 1:
            mapping[bid] = matches[0]
    return mapping


def transform_binding(binding: Any, mapping: dict[str, str]) -> Any:
    if isinstance(binding, str):
        return mapping.get(binding, binding)
    if isinstance(binding, list):
        return [transform_binding(value, mapping) for value in binding]
    if isinstance(binding, dict):
        return {key: transform_binding(value, mapping) for key, value in binding.items()}
    return binding


def canonical_binding(binding: Any) -> Any:
    if isinstance(binding, dict):
        return tuple(sorted((key, canonical_binding(value)) for key, value in binding.items()))
    if isinstance(binding, list):
        return tuple(canonical_binding(value) for value in binding)
    if isinstance(binding, set):
        return tuple(sorted(binding))
    return binding


def summarize_counterfactual_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(str(row.get(field) or "") for field in ("method", "model", "suite", "identifier_regime", "operation", "counterfactual_type"))
        groups[key].append(row)
    out: list[dict[str, Any]] = []
    for key, items in sorted(groups.items()):
        n = len(items)
        consistent = sum(int(item["consistent"]) for item in items)
        base_correct = sum(int(item["base_correct"]) for item in items)
        transformed_correct = sum(int(item["transformed_correct"]) for item in items)
        consistent_correct = sum(int(item["consistent"] and item["base_correct"]) for item in items)
        consistent_wrong = sum(int(item["consistent"] and not item["base_correct"]) for item in items)
        inconsistent_correct_to_wrong = sum(int((not item["consistent"]) and item["base_correct"] and not item["transformed_correct"]) for item in items)
        inconsistent_wrong_to_correct = sum(int((not item["consistent"]) and (not item["base_correct"]) and item["transformed_correct"]) for item in items)
        out.append(
            {
                "method": key[0],
                "model": key[1],
                "suite": key[2],
                "identifier_regime": key[3],
                "operation": key[4],
                "counterfactual_type": key[5],
                "n_pairs": n,
                "base_accuracy": base_correct / n if n else None,
                "transformed_accuracy": transformed_correct / n if n else None,
                "consistency": consistent / n if n else None,
                "consistent_correct": consistent_correct,
                "consistent_wrong": consistent_wrong,
                "inconsistent_correct_to_wrong": inconsistent_correct_to_wrong,
                "inconsistent_wrong_to_correct": inconsistent_wrong_to_correct,
            }
        )
    return out


def _counterfactual_record(**kwargs: Any) -> dict[str, Any]:
    return kwargs


def _svg_id_signatures(path: Path) -> dict[str, tuple[Any, ...]]:
    text = path.read_text(encoding="utf-8")
    out: dict[str, tuple[Any, ...]] = {}
    for match in re.finditer(r"<(rect|text)\b([^>]*)>", text):
        tag = match.group(1)
        attrs = dict(re.findall(r"([\w:-]+)=\"([^\"]*)\"", match.group(2)))
        elem_id = attrs.pop("id", None)
        if elem_id and (elem_id.startswith("rect_node_") or elem_id.startswith("text_node_")):
            out[elem_id] = (tag, tuple(sorted(attrs.items())))
    return out


def _is_bijective(mapping: dict[str, str]) -> bool:
    return len(set(mapping.values())) == len(mapping)


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
