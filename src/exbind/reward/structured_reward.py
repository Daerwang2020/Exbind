from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from exbind.reward.components import (
    hierarchy_reward,
    identity_reward,
    relation_reward,
    role_reward,
    type_reward,
    valid_reward,
)
from exbind.schema import BindingRecord


@dataclass(frozen=True)
class StructuredRewardConfig:
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "valid": 1.0,
            "type": 1.0,
            "identity": 1.0,
            "relation": 1.0,
            "hierarchy": 1.0,
            "role": 1.0,
        }
    )


def compute_structured_reward(
    prediction: dict[str, Any],
    record: BindingRecord,
    *,
    config: StructuredRewardConfig | None = None,
) -> dict[str, float | None]:
    config = config or StructuredRewardConfig()
    components = {
        "valid_reward": valid_reward(prediction, record),
        "type_reward": type_reward(prediction, record),
        "identity_reward": identity_reward(prediction, record),
        "relation_reward": relation_reward(prediction, record),
        "hierarchy_reward": hierarchy_reward(prediction, record),
        "role_reward": role_reward(prediction, record),
    }
    total = 0.0
    weight_sum = 0.0
    for name, value in components.items():
        if value is None:
            continue
        key = name.removesuffix("_reward")
        weight = float(config.weights.get(key, 0.0))
        total += weight * float(value)
        weight_sum += abs(weight)
    components["total_reward"] = total / weight_sum if weight_sum else 0.0
    return components
