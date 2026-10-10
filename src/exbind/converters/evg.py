from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from exbind.schema import (
    BindingRecord,
    BindingStructure,
    Candidate,
    GoldTarget,
    Observation,
    Transformation,
)


PERTURBATION_SUFFIXES = (
    "_anchor_mismatch",
    "_similar_id",
    "_symbolic_id_shuffle",
    "_visual_label_shuffle",
    "_counterfactual_a",
    "_counterfactual_b",
)


def base_case_id(case_id: str) -> str:
    for suffix in PERTURBATION_SUFFIXES:
        if case_id.endswith(suffix):
            return case_id[: -len(suffix)]
    return case_id


def _read_meta(sample_dir: Path) -> dict[str, Any]:
    with (sample_dir / "meta.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _object_by_id(meta: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(obj.get("object_id")): obj
        for obj in meta.get("objects") or []
        if isinstance(obj, dict) and obj.get("object_id")
    }


def _symbolic_by_id(meta: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(elem.get("element_id")): elem
        for elem in meta.get("symbolic_elements") or []
        if isinstance(elem, dict) and elem.get("element_id")
    }


def _candidate_from_id(
    candidate_id: str,
    objects: dict[str, dict[str, Any]],
    symbolic: dict[str, dict[str, Any]],
    meta: dict[str, Any],
) -> Candidate:
    elem = symbolic.get(candidate_id, {})
    linked = str(elem.get("linked_object") or "")
    obj = objects.get(linked, {})
    attrs = meta.get("candidate_attributes") or {}
    attr = attrs.get(candidate_id, {}) if isinstance(attrs, dict) else {}
    visuals = meta.get("candidate_visuals") or {}
    visual = visuals.get(candidate_id, {}) if isinstance(visuals, dict) else {}
    bbox = obj.get("bbox") or visual.get("bbox")
    return Candidate(
        candidate_id=candidate_id,
        candidate_type=str(elem.get("element_type") or obj.get("type") or ""),
        label=str(obj.get("text") or attr.get("descriptor") or ""),
        executable_id=candidate_id,
        semantic_id=linked or candidate_id,
        bbox=bbox if isinstance(bbox, list) else None,
        attributes=dict(attr) if isinstance(attr, dict) else {},
        raw={"symbolic": elem, "object": obj},
    )


def _candidate_set(meta: dict[str, Any]) -> list[Candidate]:
    objects = _object_by_id(meta)
    symbolic = _symbolic_by_id(meta)
    task = str(meta.get("task_type") or "")
    if task == "edge_rejection":
        return [
            Candidate(
                candidate_id=str(edge.get("id")),
                candidate_type="edge",
                executable_id=str(edge.get("id")),
                semantic_id=f"{edge.get('source')}->{edge.get('target')}",
                attributes={
                    "source": edge.get("source"),
                    "target": edge.get("target"),
                },
                raw=dict(edge),
            )
            for edge in meta.get("edges") or []
            if isinstance(edge, dict) and edge.get("id")
        ]
    if task == "group_grounding":
        return [
            Candidate(
                candidate_id=str(group.get("id")),
                candidate_type="group",
                label=str(group.get("label") or ""),
                executable_id=str(group.get("id")),
                semantic_id=str(group.get("id")),
                bbox=group.get("bbox") if isinstance(group.get("bbox"), list) else None,
                attributes={"children": list(group.get("children") or [])},
                raw=dict(group),
            )
            for group in meta.get("groups") or []
            if isinstance(group, dict) and group.get("id")
        ]
    return [
        _candidate_from_id(str(candidate_id), objects, symbolic, meta)
        for candidate_id in meta.get("candidate_ids") or []
        if isinstance(candidate_id, str)
    ]


def _semantic_target(meta: dict[str, Any]) -> Any:
    task = str(meta.get("task_type") or "")
    objects = _object_by_id(meta)
    symbolic = _symbolic_by_id(meta)
    stage_a = meta.get("stage_a_gold") if isinstance(meta.get("stage_a_gold"), dict) else {}
    if task == "endpoint_grounding":
        return {
            "source": _linked_object(stage_a.get("source_id"), symbolic),
            "target": _linked_object(stage_a.get("target_id"), symbolic),
        }
    if task == "edge_rejection":
        edge_id = stage_a.get("wrong_edge")
        for edge in meta.get("edges") or []:
            if isinstance(edge, dict) and edge.get("id") == edge_id:
                return {"source": edge.get("source"), "target": edge.get("target")}
        return edge_id
    if task == "group_grounding":
        group_id = stage_a.get("group_id")
        for group in meta.get("groups") or []:
            if isinstance(group, dict) and group.get("id") == group_id:
                return {"group_id": group_id, "children": list(group.get("children") or [])}
        return group_id
    target_id = stage_a.get("target_id") or (meta.get("binding_target") or {}).get("primary_id")
    linked = _linked_object(target_id, symbolic)
    obj = objects.get(str(linked), {})
    return {"object_id": linked, "text": obj.get("text"), "type": obj.get("type")}


def _linked_object(element_id: Any, symbolic: dict[str, dict[str, Any]]) -> str | None:
    if not isinstance(element_id, str):
        return None
    return str(symbolic.get(element_id, {}).get("linked_object") or element_id)


def _executable_target(meta: dict[str, Any]) -> Any:
    if isinstance(meta.get("stage_a_gold"), dict):
        return meta["stage_a_gold"]
    if isinstance(meta.get("binding_target"), dict):
        return meta["binding_target"].get("primary_id")
    return None


def _hierarchy_path(meta: dict[str, Any], semantic_target: Any) -> list[str] | None:
    target_objects: set[str] = set()
    if isinstance(semantic_target, dict):
        for key in ("object_id", "source", "target"):
            value = semantic_target.get(key)
            if isinstance(value, str):
                target_objects.add(value)
    for group in meta.get("groups") or []:
        if not isinstance(group, dict):
            continue
        children = {str(child) for child in group.get("children") or []}
        if target_objects & children:
            return [str(group.get("id"))]
    return None


def _relation(meta: dict[str, Any]) -> Any:
    if str(meta.get("task_type") or "") == "endpoint_grounding":
        stage_a = meta.get("stage_a_gold") if isinstance(meta.get("stage_a_gold"), dict) else {}
        return {
            "type": "ordered_endpoint_pair",
            "source": stage_a.get("source_id"),
            "target": stage_a.get("target_id"),
        }
    if str(meta.get("task_type") or "") == "edge_rejection":
        return {"type": "existing_edge"}
    return None


def _anchor(meta: dict[str, Any]) -> Any:
    if str(meta.get("task_type") or "") == "endpoint_grounding":
        stage_a = meta.get("stage_a_gold") if isinstance(meta.get("stage_a_gold"), dict) else {}
        return stage_a.get("source_id")
    return None


def _transformation(meta: dict[str, Any], semantic_target: Any, executable_target: Any) -> Transformation | None:
    perturbation = meta.get("perturbation")
    if not perturbation:
        return None
    case_id = str(meta.get("sample_id") or "")
    return Transformation(
        type=str(perturbation),
        parent_case_id=base_case_id(case_id),
        invariant_semantic_target=True,
        semantic_target_after=semantic_target,
        executable_target_after=executable_target,
        metadata={
            key: meta.get(key)
            for key in ("label_mapping", "id_mapping", "perturbation")
            if key in meta
        },
    )


def convert_evg_case(
    sample_dir: str | Path,
    *,
    source_family: str | None = None,
    split: str | None = None,
) -> BindingRecord:
    sample_path = Path(sample_dir)
    meta = _read_meta(sample_path)
    case_id = str(meta.get("sample_id") or sample_path.name)
    semantic = _semantic_target(meta)
    executable = _executable_target(meta)
    family = source_family or _source_family(sample_path, meta)
    source_path = sample_path / "before.svg"
    image_path = sample_path / "before.png"
    structure = BindingStructure(
        target_type=str(meta.get("task_type") or ""),
        attributes={
            "operation": meta.get("operation"),
            "stage_b": meta.get("stage_b_gold"),
            "difficulty": meta.get("difficulty"),
        },
        anchor=_anchor(meta),
        relation=_relation(meta),
        hierarchy_path=_hierarchy_path(meta, semantic),
        executable_type=_executable_type(meta),
        symbolic_reference=executable,
    )
    return BindingRecord(
        case_id=case_id,
        source_family=family,
        split=split or ("perturbed" if meta.get("perturbation") else "clean"),
        generator_version=str(meta.get("template") or ""),
        observation=Observation(
            image=str(image_path) if image_path.exists() else None,
            source=str(source_path) if source_path.exists() else None,
            source_type="svg",
            state_metadata={"template": meta.get("template")},
        ),
        instruction=str(meta.get("instruction") or ""),
        operation=str(meta.get("operation") or ""),
        candidate_set=_candidate_set(meta),
        gold=GoldTarget(
            semantic_target=semantic,
            executable_target=executable,
            intermediate={},
        ),
        binding_structure=structure,
        transformation=_transformation(meta, semantic, executable),
        metadata={
            "legacy_meta_path": str(sample_path / "meta.json"),
            "legacy_task_type": meta.get("task_type"),
            "candidate_count": len(meta.get("candidate_ids") or []),
            "ambiguity": meta.get("ambiguity"),
            "distractors": meta.get("distractors"),
            "gold_patch": meta.get("gold_patch"),
            "external_source": meta.get("_external_source"),
        },
    )


def convert_evg_dataset(
    root: str | Path,
    *,
    limit: int | None = None,
    source_family: str | None = None,
    split: str | None = None,
) -> list[BindingRecord]:
    root_path = Path(root)
    records: list[BindingRecord] = []
    for sample_dir in sorted(root_path.iterdir()):
        if not sample_dir.is_dir() or not (sample_dir / "meta.json").exists():
            continue
        records.append(convert_evg_case(sample_dir, source_family=source_family, split=split))
        if limit is not None and len(records) >= limit:
            break
    return records


def _source_family(sample_path: Path, meta: dict[str, Any]) -> str:
    name = sample_path.parent.name
    if "svgeditbench" in name or str(meta.get("template")) == "external_svgeditbench_v2":
        return "SEB"
    if "external" in name:
        return "Novel-SVG"
    return "EVG-Bench-Perturbed" if meta.get("perturbation") else "EVG-Bench"


def _executable_type(meta: dict[str, Any]) -> str:
    task = str(meta.get("task_type") or "")
    if task == "endpoint_grounding":
        return "ordered_endpoint_pair"
    if task == "edge_rejection":
        return "edge_id"
    if task == "group_grounding":
        return "group_id"
    return "element_id"
