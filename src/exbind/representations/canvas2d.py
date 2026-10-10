from __future__ import annotations

import json

from exbind.latent import LatentBindingInstance, RenderedBindingInstance
from exbind.schema import Candidate, Observation


def compile_canvas2d(latent: LatentBindingInstance) -> RenderedBindingInstance:
    mapping = {entity.entity_id: f"canvas_obj_{idx:02d}" for idx, entity in enumerate(latent.entities)}
    commands = []
    candidates = []
    for idx, entity in enumerate(latent.entities):
        x = int(entity.attributes.get("x", 32 + 40 * idx))
        y = int(entity.attributes.get("y", 40 + 30 * idx))
        shape = str(entity.semantic_type or entity.attributes.get("shape") or "object")
        color = str(entity.attributes.get("color") or "gray")
        surface_id = mapping[entity.entity_id]
        commands.append({"draw": shape, "id": surface_id, "x": x, "y": y, "color": color})
        candidates.append(
            Candidate(
                candidate_id=surface_id,
                candidate_type=shape,
                label=f"candidate {idx}",
                executable_id=surface_id,
                semantic_id=entity.entity_id,
                bbox=[x - 10, y - 10, 20, 20],
                attributes={"shape": shape, "color": color, "x": x, "y": y},
            )
        )
    canonical = latent.canonical_target or ""
    relation = latent.target_spec.get("relation") or {}
    relation_text = str(relation.get("type") or relation).replace("_", " ")
    anchor = latent.entity(str(latent.target_spec.get("anchor")))
    anchor_text = _entity_phrase(anchor) if anchor else "the anchor"
    target_type = str(latent.target_spec.get("target_type") or "object")
    target_color = str((latent.target_spec.get("target_attributes") or {}).get("color") or "")
    instruction = f"Select the {target_color} {target_type} {relation_text} the {anchor_text}.".replace("  ", " ")
    return RenderedBindingInstance(
        latent_id=latent.latent_id,
        format="canvas2d",
        observation=Observation(
            source=json.dumps({"width": 320, "height": 240, "commands": commands}, sort_keys=True),
            source_type="canvas2d_json",
            state_metadata={"commands": commands},
        ),
        source_structure={"commands": commands},
        candidate_set=candidates,
        instruction=instruction,
        gold_executable_target={"target_id": mapping[canonical]},
        canonical_semantic_target=canonical,
        mapping_latent_to_surface=mapping,
        metadata={"representation_family": "relational_2d"},
    )


def _entity_phrase(entity) -> str:
    color = str(entity.attributes.get("color") or "")
    shape = str(entity.semantic_type or entity.attributes.get("shape") or "object")
    return f"{color} {shape}".strip()
