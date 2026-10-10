from __future__ import annotations

from typing import Any

from exbind.schema import BindingRecord


def valid_reward(prediction: dict[str, Any], record: BindingRecord) -> float:
    ids = _prediction_ids(prediction)
    if not ids:
        return 0.0
    candidate_ids = {candidate.candidate_id for candidate in record.candidate_set}
    return 1.0 if ids <= candidate_ids else 0.0


def type_reward(prediction: dict[str, Any], record: BindingRecord) -> float | None:
    gold_type = record.binding_structure.target_type or record.binding_structure.executable_type
    if gold_type is None:
        return None
    pred_type = prediction.get("target_type") or prediction.get("executable_type") or prediction.get("candidate_type")
    if pred_type is not None:
        return 1.0 if pred_type == gold_type else 0.0
    ids = _prediction_ids(prediction)
    if not ids:
        return 0.0
    by_id = {candidate.candidate_id: candidate for candidate in record.candidate_set}
    predicted_types = {by_id[item].candidate_type for item in ids if item in by_id}
    gold_ids = _gold_ids(record.gold.executable_target)
    gold_types = {by_id[item].candidate_type for item in gold_ids if item in by_id}
    return 1.0 if predicted_types and predicted_types <= gold_types else 0.0


def identity_reward(prediction: dict[str, Any], record: BindingRecord) -> float:
    return 1.0 if _canonical_prediction(prediction) == record.gold.executable_target else 0.0


def relation_reward(prediction: dict[str, Any], record: BindingRecord) -> float | None:
    gold_relation = record.binding_structure.relation
    if gold_relation is None:
        return None
    return 1.0 if prediction.get("relation") == gold_relation else 0.0


def hierarchy_reward(prediction: dict[str, Any], record: BindingRecord) -> float | None:
    gold_path = record.binding_structure.hierarchy_path
    if gold_path is None:
        return None
    pred_path = prediction.get("hierarchy_path")
    return 1.0 if pred_path is not None and list(pred_path) == list(gold_path) else 0.0


def role_reward(prediction: dict[str, Any], record: BindingRecord) -> float | None:
    gold = record.gold.executable_target
    if not isinstance(gold, dict) or not {"source_id", "target_id"} <= set(gold):
        return None
    source_ok = prediction.get("source_id") == gold.get("source_id")
    target_ok = prediction.get("target_id") == gold.get("target_id")
    return (float(source_ok) + float(target_ok)) / 2.0


def exact_binding_reward(prediction: dict[str, Any], record: BindingRecord) -> float:
    return identity_reward(prediction, record)


def _prediction_ids(prediction: dict[str, Any]) -> set[str]:
    ids = set()
    for key in ("target_id", "source_id", "wrong_edge", "group_id", "dom_id"):
        value = prediction.get(key)
        if isinstance(value, str):
            ids.add(value)
    executable = prediction.get("executable_target")
    if isinstance(executable, dict):
        ids.update(value for value in executable.values() if isinstance(value, str))
    elif isinstance(executable, str):
        ids.add(executable)
    return ids


def _gold_ids(target: Any) -> set[str]:
    if isinstance(target, str):
        return {target}
    if isinstance(target, dict):
        return {value for value in target.values() if isinstance(value, str)}
    return set()


def _canonical_prediction(prediction: dict[str, Any]) -> Any:
    if "executable_target" in prediction:
        return prediction["executable_target"]
    return {
        key: prediction[key]
        for key in ("target_id", "source_id", "wrong_edge", "group_id", "dom_id")
        if key in prediction
    }
