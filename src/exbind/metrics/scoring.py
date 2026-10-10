from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Optional

from exbind.schema import BindingRecord


Score = Optional[int]
RewardVector = dict[str, Score]


DEFAULT_STAGE_ORDER = ("semantic", "attribute", "anchor", "relation", "hierarchy", "type", "candidate", "executable")


def _eq(pred: Any, gold: Any) -> Score:
    if gold is None:
        return None
    return 1 if pred == gold else 0


def _path_eq(pred: Any, gold: Any) -> Score:
    if gold is None:
        return None
    if pred is None:
        return 0
    return 1 if list(pred) == list(gold) else 0


def score_semantic(prediction: dict[str, Any], record: BindingRecord) -> Score:
    return _eq(prediction.get("semantic_target"), record.gold.semantic_target)


def score_identity(prediction: dict[str, Any], record: BindingRecord) -> Score:
    pred = prediction.get("canonical_target") or prediction.get("semantic_target")
    if pred is not None and record.gold.semantic_target is not None:
        return _eq(pred, record.gold.semantic_target)
    return score_candidate(prediction, record)


def score_attribute(prediction: dict[str, Any], record: BindingRecord) -> Score:
    gold = record.binding_structure.attributes or None
    return _eq(prediction.get("attributes"), gold)


def score_anchor(prediction: dict[str, Any], record: BindingRecord) -> Score:
    return _eq(prediction.get("anchor"), record.binding_structure.anchor)


def score_relation(prediction: dict[str, Any], record: BindingRecord) -> Score:
    return _eq(prediction.get("relation"), record.binding_structure.relation)


def score_hierarchy(prediction: dict[str, Any], record: BindingRecord) -> Score:
    return _path_eq(prediction.get("hierarchy_path"), record.binding_structure.hierarchy_path)


def score_role(prediction: dict[str, Any], record: BindingRecord) -> Score:
    gold = record.gold.executable_target
    if not isinstance(gold, dict) or not {"source_id", "target_id"} <= set(gold):
        return None
    return 1 if prediction.get("source_id") == gold.get("source_id") and prediction.get("target_id") == gold.get("target_id") else 0


def score_type(prediction: dict[str, Any], record: BindingRecord) -> Score:
    return _eq(prediction.get("target_type"), record.binding_structure.target_type)


def score_candidate(prediction: dict[str, Any], record: BindingRecord) -> Score:
    gold = _final_candidate_id(record.gold.executable_target)
    pred = prediction.get("candidate_id") or prediction.get("target_id") or prediction.get("dom_id")
    return _eq(pred, gold)


def score_candidate_validity(prediction: dict[str, Any], record: BindingRecord) -> Score:
    ids = {
        value
        for key in ("candidate_id", "target_id", "source_id", "group_id", "wrong_edge", "dom_id", "cell_id")
        if isinstance((value := prediction.get(key)), str)
    }
    if not ids:
        executable = prediction.get("executable_target")
        if isinstance(executable, dict):
            ids = {value for value in executable.values() if isinstance(value, str)}
        elif isinstance(executable, str):
            ids = {executable}
    if not ids:
        return 0
    candidate_ids = {candidate.candidate_id for candidate in record.candidate_set}
    return 1 if ids <= candidate_ids else 0


def score_executable(prediction: dict[str, Any], record: BindingRecord) -> Score:
    pred = prediction.get("executable_target")
    if pred is None:
        pred = {
            key: prediction.get(key)
            for key in ("target_id", "source_id", "wrong_edge", "group_id", "dom_id", "xpath", "css_selector")
            if key in prediction
        }
    return _eq(pred, record.gold.executable_target)


def score_executable_exact(prediction: dict[str, Any], record: BindingRecord) -> Score:
    return score_executable(prediction, record)


def score_vector(prediction: dict[str, Any], record: BindingRecord) -> RewardVector:
    return {
        "identity": score_identity(prediction, record),
        "semantic": score_semantic(prediction, record),
        "attribute": score_attribute(prediction, record),
        "anchor": score_anchor(prediction, record),
        "relation": score_relation(prediction, record),
        "hierarchy": score_hierarchy(prediction, record),
        "role": score_role(prediction, record),
        "type": score_type(prediction, record),
        "candidate": score_candidate(prediction, record),
        "candidate_valid": score_candidate_validity(prediction, record),
        "executable": score_executable(prediction, record),
        "executable_exact": score_executable_exact(prediction, record),
    }


def exact_binding_accuracy(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    total = 0
    correct = 0
    for row in rows:
        if "exact_final" in row:
            total += 1
            correct += int(bool(row.get("exact_final")))
        elif "exact" in row:
            total += 1
            correct += int(bool(row.get("exact")))
    return {"correct": correct, "total": total, "accuracy": correct / total if total else None}


def aggregate_dimension_accuracy(vectors: Iterable[RewardVector]) -> dict[str, dict[str, Any]]:
    totals: dict[str, int] = defaultdict(int)
    correct: dict[str, int] = defaultdict(int)
    for vector in vectors:
        for dim, value in vector.items():
            if value is None:
                continue
            totals[dim] += 1
            correct[dim] += int(value)
    return {
        dim: {
            "correct": correct[dim],
            "total": totals[dim],
            "accuracy": correct[dim] / totals[dim] if totals[dim] else None,
        }
        for dim in sorted(totals)
    }


def binding_survival_curve(
    vectors: Iterable[RewardVector],
    stage_order: Iterable[str] = DEFAULT_STAGE_ORDER,
) -> dict[str, dict[str, Any]]:
    order = list(stage_order)
    totals = Counter()
    survived = Counter()
    for vector in vectors:
        ok_so_far = True
        for stage in order:
            value = vector.get(stage)
            if value is None:
                continue
            totals[stage] += 1
            ok_so_far = ok_so_far and bool(value)
            survived[stage] += int(ok_so_far)
    return {
        stage: {
            "survived": survived[stage],
            "total": totals[stage],
            "survival": survived[stage] / totals[stage] if totals[stage] else None,
        }
        for stage in order
        if totals[stage]
    }


def failure_stage(vector: RewardVector, stage_order: Iterable[str] = DEFAULT_STAGE_ORDER) -> str | None:
    for stage in stage_order:
        value = vector.get(stage)
        if value == 0:
            return stage
    return None


def failure_stage_distribution(
    vectors: Iterable[RewardVector],
    stage_order: Iterable[str] = DEFAULT_STAGE_ORDER,
) -> dict[str, int]:
    counts = Counter()
    for vector in vectors:
        counts[failure_stage(vector, stage_order) or "none"] += 1
    return dict(counts)


def counterfactual_consistency(
    parent_prediction: dict[str, Any],
    child_prediction: dict[str, Any],
    parent_record: BindingRecord,
    child_record: BindingRecord,
) -> dict[str, bool | None]:
    semantic_expected = (
        child_record.transformation.invariant_semantic_target
        if child_record.transformation is not None
        else None
    )
    parent_sem = parent_prediction.get("semantic_target")
    child_sem = child_prediction.get("semantic_target")
    parent_exec = parent_prediction.get("executable_target")
    child_exec = child_prediction.get("executable_target")
    return {
        "semantic_consistency": (parent_sem == child_sem) if semantic_expected else None,
        "symbolic_consistency": parent_exec == child_exec if parent_exec is not None and child_exec is not None else None,
        "parent_exact_executable": parent_exec == parent_record.gold.executable_target if parent_exec is not None else None,
        "child_exact_executable": child_exec == child_record.gold.executable_target if child_exec is not None else None,
    }


def _final_candidate_id(executable_target: Any) -> str | None:
    if isinstance(executable_target, str):
        return executable_target
    if isinstance(executable_target, dict):
        for key in ("target_id", "dom_id", "wrong_edge", "group_id", "cell_id"):
            value = executable_target.get(key)
            if isinstance(value, str):
                return value
    return None
