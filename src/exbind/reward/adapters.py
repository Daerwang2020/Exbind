from __future__ import annotations

from typing import Any

from exbind.schema import BindingRecord


REWARD_ONLY_KEYS = {
    "gold",
    "gold_executable",
    "executable_target",
    "semantic_target",
    "structured_reward",
    "reward",
    "target_id",
}


def policy_observation(record: BindingRecord, *, include_candidates: bool = True) -> dict[str, Any]:
    data: dict[str, Any] = {
        "case_id": record.case_id,
        "source_family": record.source_family,
        "instruction": record.instruction,
        "operation": record.operation,
        "observation": {
            "image": record.observation.image,
            "source": record.observation.source,
            "source_type": record.observation.source_type,
        },
    }
    if include_candidates:
        data["candidates"] = [
            {
                "candidate_id": candidate.candidate_id,
                "candidate_type": candidate.candidate_type,
                "label": candidate.label,
                "attributes": candidate.attributes,
            }
            for candidate in record.candidate_set
        ]
    return data


def assert_no_reward_leakage(payload: dict[str, Any], record: BindingRecord) -> None:
    _assert_no_forbidden_keys(payload)
    instruction = str(payload.get("instruction") or "")
    for gold_id in _gold_ids(record.gold.executable_target):
        if gold_id and gold_id in instruction:
            raise AssertionError(f"gold executable id leaked into instruction: {gold_id}")


def _assert_no_forbidden_keys(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in REWARD_ONLY_KEYS:
                raise AssertionError(f"reward-only key leaked into policy payload: {key}")
            _assert_no_forbidden_keys(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_forbidden_keys(child)


def _gold_ids(gold: Any) -> set[str]:
    if isinstance(gold, dict):
        return {value for value in gold.values() if isinstance(value, str)}
    if isinstance(gold, str):
        return {gold}
    return set()
