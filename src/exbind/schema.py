from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


JsonDict = dict[str, Any]


@dataclass
class Observation:
    image: str | None = None
    source: str | None = None
    source_type: str | None = None
    state_metadata: JsonDict = field(default_factory=dict)


@dataclass
class Candidate:
    candidate_id: str
    candidate_type: str | None = None
    label: str | None = None
    executable_id: str | None = None
    semantic_id: str | None = None
    bbox: list[float] | None = None
    attributes: JsonDict = field(default_factory=dict)
    raw: JsonDict = field(default_factory=dict)


@dataclass
class GoldTarget:
    semantic_target: Any = None
    executable_target: Any = None
    intermediate: JsonDict = field(default_factory=dict)


@dataclass
class BindingStructure:
    target_type: str | None = None
    attributes: JsonDict = field(default_factory=dict)
    anchor: Any = None
    relation: Any = None
    hierarchy_path: list[str] | None = None
    part: str | None = None
    executable_type: str | None = None
    symbolic_reference: Any = None


@dataclass
class Transformation:
    type: str | None = None
    parent_case_id: str | None = None
    invariant_semantic_target: bool | None = None
    semantic_target_before: Any = None
    semantic_target_after: Any = None
    executable_target_before: Any = None
    executable_target_after: Any = None
    metadata: JsonDict = field(default_factory=dict)


@dataclass
class BindingRecord:
    case_id: str
    source_family: str
    split: str | None = None
    generator_version: str | None = None
    observation: Observation = field(default_factory=Observation)
    instruction: str | None = None
    operation: str | None = None
    candidate_set: list[Candidate] = field(default_factory=list)
    gold: GoldTarget = field(default_factory=GoldTarget)
    binding_structure: BindingStructure = field(default_factory=BindingStructure)
    transformation: Transformation | None = None
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        data = asdict(self)
        if self.transformation is None:
            data["transformation"] = None
        return data

    @classmethod
    def from_dict(cls, data: JsonDict) -> "BindingRecord":
        observation = Observation(**dict(data.get("observation") or {}))
        candidates = [Candidate(**dict(row)) for row in data.get("candidate_set") or []]
        gold = GoldTarget(**dict(data.get("gold") or {}))
        structure = BindingStructure(**dict(data.get("binding_structure") or {}))
        transformation_data = data.get("transformation")
        transformation = (
            Transformation(**dict(transformation_data))
            if isinstance(transformation_data, dict)
            else None
        )
        return cls(
            case_id=str(data["case_id"]),
            source_family=str(data["source_family"]),
            split=data.get("split"),
            generator_version=data.get("generator_version"),
            observation=observation,
            instruction=data.get("instruction"),
            operation=data.get("operation"),
            candidate_set=candidates,
            gold=gold,
            binding_structure=structure,
            transformation=transformation,
            metadata=dict(data.get("metadata") or {}),
        )


def structured_target(record: BindingRecord) -> JsonDict:
    """Return the programmatically verifiable intermediate target object."""
    structure = record.binding_structure
    return {
        "target_type": structure.target_type,
        "attributes": structure.attributes or None,
        "anchor": structure.anchor,
        "relation": structure.relation,
        "hierarchy_path": structure.hierarchy_path,
        "part": structure.part,
        "candidate_type": structure.executable_type,
        "executable_target": record.gold.executable_target,
        "symbolic_reference": structure.symbolic_reference,
    }
