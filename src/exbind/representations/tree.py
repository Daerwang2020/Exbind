from __future__ import annotations

from exbind.latent import LatentBindingInstance, RenderedBindingInstance
from exbind.schema import Candidate, Observation


def compile_tree(latent: LatentBindingInstance) -> RenderedBindingInstance:
    leaves = [entity for entity in latent.entities if not entity.children]
    mapping = {entity.entity_id: f"tree_node_{idx:02d}" for idx, entity in enumerate(latent.entities)}
    lines = []
    by_parent: dict[str | None, list] = {}
    for entity in latent.entities:
        by_parent.setdefault(entity.parent, []).append(entity)
    _append_tree(lines, by_parent, None, mapping, depth=0)
    candidates = [
        Candidate(
            candidate_id=mapping[entity.entity_id],
            candidate_type=entity.semantic_type or entity.role or "node",
            label=f"candidate {idx}",
            executable_id=mapping[entity.entity_id],
            semantic_id=entity.entity_id,
            attributes={
                **entity.attributes,
                "role": entity.role,
                "parent": entity.parent,
                "hierarchy_path": _hierarchy_path(entity, by_parent),
            },
        )
        for idx, entity in enumerate(leaves)
    ]
    canonical = latent.canonical_target or ""
    spec = latent.target_spec
    instruction = (
        f"Select the {spec.get('part', 'item')} of the {spec.get('object', 'object')} "
        f"in {spec.get('group', 'the group')}."
    )
    return RenderedBindingInstance(
        latent_id=latent.latent_id,
        format="tree",
        observation=Observation(source="\n".join(lines), source_type="tree_text", state_metadata={"tree": lines}),
        source_structure={"tree_lines": lines},
        candidate_set=candidates,
        instruction=instruction,
        gold_executable_target={"target_id": mapping[canonical]},
        canonical_semantic_target=canonical,
        mapping_latent_to_surface=mapping,
        metadata={"representation_family": "hierarchical_tree"},
    )


def _append_tree(lines: list[str], by_parent: dict[str | None, list], parent: str | None, mapping: dict[str, str], depth: int) -> None:
    for entity in sorted(by_parent.get(parent, []), key=lambda item: item.entity_id):
        label = entity.attributes.get("name") or entity.role or entity.semantic_type or entity.entity_id
        lines.append(f"{'  ' * depth}- {label} [{mapping[entity.entity_id]}]")
        _append_tree(lines, by_parent, entity.entity_id, mapping, depth + 1)


def _hierarchy_path(entity, by_parent: dict[str | None, list]) -> list[str]:
    by_id = {item.entity_id: item for members in by_parent.values() for item in members}
    path = []
    current = entity
    while current is not None:
        path.append(str(current.attributes.get("name") or current.role or current.semantic_type or current.entity_id))
        current = by_id.get(current.parent)
    return list(reversed(path))
