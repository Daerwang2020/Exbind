from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from exbind.schema import Candidate, Observation


JsonDict = dict[str, Any]


@dataclass
class LatentEntity:
    entity_id: str
    semantic_type: str | None = None
    attributes: JsonDict = field(default_factory=dict)
    role: str | None = None
    parent: str | None = None
    children: list[str] = field(default_factory=list)
    relations: list[JsonDict] = field(default_factory=list)
    canonical_target_key: str | None = None

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass
class LatentBindingInstance:
    latent_id: str
    entities: list[LatentEntity] = field(default_factory=list)
    relations: list[JsonDict] = field(default_factory=list)
    hierarchy: list[JsonDict] = field(default_factory=list)
    attributes: JsonDict = field(default_factory=dict)
    roles: JsonDict = field(default_factory=dict)
    target_spec: JsonDict = field(default_factory=dict)
    operation: str | None = None
    capability_tags: list[str] = field(default_factory=list)
    difficulty_tags: JsonDict = field(default_factory=dict)
    metadata: JsonDict = field(default_factory=dict)

    def entity(self, entity_id: str) -> LatentEntity | None:
        return next((entity for entity in self.entities if entity.entity_id == entity_id), None)

    @property
    def canonical_target(self) -> str | None:
        target = self.target_spec.get("canonical_target")
        return str(target) if target is not None else None

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass
class RenderedBindingInstance:
    latent_id: str
    format: str
    observation: Observation
    source_structure: JsonDict
    candidate_set: list[Candidate]
    instruction: str
    gold_executable_target: Any
    canonical_semantic_target: str
    mapping_latent_to_surface: dict[str, str]
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)
