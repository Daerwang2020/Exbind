from __future__ import annotations

import copy
import random
from typing import Any

from exbind.schema import BindingRecord, Transformation


def candidate_order_permutation(record: BindingRecord, *, seed: int = 0) -> BindingRecord:
    rng = random.Random(seed)
    child = copy.deepcopy(record)
    rng.shuffle(child.candidate_set)
    child.case_id = f"{record.case_id}_candidate_order_permutation"
    child.transformation = Transformation(
        type="candidate_order_permutation",
        parent_case_id=record.case_id,
        invariant_semantic_target=True,
        semantic_target_before=record.gold.semantic_target,
        semantic_target_after=child.gold.semantic_target,
        executable_target_before=record.gold.executable_target,
        executable_target_after=child.gold.executable_target,
    )
    return child


def symbolic_id_rename(record: BindingRecord, *, prefix: str = "renamed") -> BindingRecord:
    child = copy.deepcopy(record)
    id_map = {candidate.candidate_id: f"{prefix}_{idx}" for idx, candidate in enumerate(child.candidate_set)}
    for candidate in child.candidate_set:
        old = candidate.candidate_id
        candidate.candidate_id = id_map[old]
        candidate.executable_id = id_map[old]
        candidate.raw = _rename_refs(candidate.raw, id_map)
    child.case_id = f"{record.case_id}_symbolic_id_rename"
    child.observation.source = _rename_source(child.observation.source, id_map)
    child.observation.state_metadata = _rename_refs(child.observation.state_metadata, id_map)
    child.gold.executable_target = _rename_target(child.gold.executable_target, id_map)
    child.binding_structure.anchor = _rename_refs(child.binding_structure.anchor, id_map)
    child.binding_structure.relation = _rename_refs(child.binding_structure.relation, id_map)
    child.binding_structure.attributes = _rename_refs(child.binding_structure.attributes, id_map)
    child.binding_structure.symbolic_reference = child.gold.executable_target
    child.transformation = Transformation(
        type="symbolic_id_rename",
        parent_case_id=record.case_id,
        invariant_semantic_target=True,
        semantic_target_before=record.gold.semantic_target,
        semantic_target_after=child.gold.semantic_target,
        executable_target_before=record.gold.executable_target,
        executable_target_after=child.gold.executable_target,
        metadata={"id_map": id_map},
    )
    return child


def _rename_target(target, id_map: dict[str, str]):
    if isinstance(target, str):
        return id_map.get(target, target)
    if isinstance(target, dict):
        return {key: id_map.get(value, value) if isinstance(value, str) else value for key, value in target.items()}
    return target


def _rename_refs(value: Any, id_map: dict[str, str]) -> Any:
    if isinstance(value, str):
        return id_map.get(value, value)
    if isinstance(value, list):
        return [_rename_refs(item, id_map) for item in value]
    if isinstance(value, tuple):
        return tuple(_rename_refs(item, id_map) for item in value)
    if isinstance(value, dict):
        return {key: _rename_refs(item, id_map) for key, item in value.items()}
    return value


def _rename_source(source: str | None, id_map: dict[str, str]) -> str | None:
    if source is None:
        return None
    out = source
    for old, new in sorted(id_map.items(), key=lambda item: len(item[0]), reverse=True):
        out = out.replace(old, new)
    return out
