from __future__ import annotations

from exbind.latent import LatentBindingInstance, RenderedBindingInstance
from exbind.schema import Candidate, Observation


def compile_svg(latent: LatentBindingInstance) -> RenderedBindingInstance:
    mapping = {entity.entity_id: f"svg_obj_{idx:02d}" for idx, entity in enumerate(latent.entities)}
    rows = ['<svg xmlns="http://www.w3.org/2000/svg" width="320" height="240">']
    candidates = []
    for idx, entity in enumerate(latent.entities):
        x = int(entity.attributes.get("x", 32 + 40 * idx))
        y = int(entity.attributes.get("y", 40 + 30 * idx))
        shape = str(entity.semantic_type or entity.attributes.get("shape") or "object")
        color = str(entity.attributes.get("color") or "gray")
        surface_id = mapping[entity.entity_id]
        if shape == "circle":
            rows.append(f'<circle id="{surface_id}" cx="{x}" cy="{y}" r="12" fill="{color}"/>')
        elif shape == "square":
            rows.append(f'<rect id="{surface_id}" x="{x - 12}" y="{y - 12}" width="24" height="24" fill="{color}"/>')
        else:
            rows.append(f'<polygon id="{surface_id}" points="{x},{y-14} {x-14},{y+14} {x+14},{y+14}" fill="{color}"/>')
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
    rows.append("</svg>")
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
        format="svg",
        observation=Observation(source="\n".join(rows), source_type="svg", state_metadata={"svg_ids": mapping}),
        source_structure={"svg": rows},
        candidate_set=candidates,
        instruction=instruction,
        gold_executable_target={"target_id": mapping[canonical]},
        canonical_semantic_target=canonical,
        mapping_latent_to_surface=mapping,
        metadata={"representation_family": "relational_svg"},
    )


def _entity_phrase(entity) -> str:
    color = str(entity.attributes.get("color") or "")
    shape = str(entity.semantic_type or entity.attributes.get("shape") or "object")
    return f"{color} {shape}".strip()
