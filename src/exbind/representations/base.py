from __future__ import annotations

from typing import Any

from exbind.latent import LatentBindingInstance, RenderedBindingInstance
from exbind.schema import BindingRecord, BindingStructure, GoldTarget


def rendered_to_record(
    rendered: RenderedBindingInstance,
    latent: LatentBindingInstance,
    *,
    generator_version: str = "cross_format.v1",
) -> BindingRecord:
    target = latent.target_spec
    return BindingRecord(
        case_id=f"{latent.latent_id}__{rendered.format}",
        source_family=rendered.format,
        split="generated_cross_format",
        generator_version=generator_version,
        observation=rendered.observation,
        instruction=rendered.instruction,
        operation=latent.operation or "select",
        candidate_set=rendered.candidate_set,
        gold=GoldTarget(
            semantic_target=rendered.canonical_semantic_target,
            executable_target=rendered.gold_executable_target,
            intermediate={
                "canonical_target": rendered.canonical_semantic_target,
                "target_spec": target,
                "surface_target": _surface_target(rendered.gold_executable_target),
            },
        ),
        binding_structure=BindingStructure(
            target_type=target.get("target_type") or target.get("semantic_type"),
            attributes=dict(target.get("target_attributes") or target.get("attributes") or {}),
            anchor=target.get("anchor"),
            relation=target.get("relation"),
            hierarchy_path=list(target.get("hierarchy_path") or []) or None,
            part=target.get("part"),
            executable_type=target.get("executable_type") or "surface_id",
            symbolic_reference=rendered.gold_executable_target,
        ),
        metadata={
            "latent_id": latent.latent_id,
            "format": rendered.format,
            "mapping_latent_to_surface": rendered.mapping_latent_to_surface,
            "surface_to_latent": {value: key for key, value in rendered.mapping_latent_to_surface.items()},
            "difficulty": latent.difficulty_tags,
            "capability_tags": list(latent.capability_tags),
            **rendered.metadata,
        },
    )


def _surface_target(target: Any) -> str | None:
    if isinstance(target, str):
        return target
    if isinstance(target, dict):
        for key in ("target_id", "cell_id", "source_id", "dom_id"):
            value = target.get(key)
            if isinstance(value, str):
                return value
    return None
