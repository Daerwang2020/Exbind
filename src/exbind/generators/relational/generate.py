from __future__ import annotations

import random
from typing import Any

from exbind.schema import BindingRecord, BindingStructure, Candidate, GoldTarget, Observation


SHAPES = ("circle", "square", "triangle")
COLORS = ("blue", "red", "green", "yellow")
RELATIONS = ("left of", "right of", "above", "below", "nearest")


def generate_relational_batch(n: int, *, seed: int = 0, distractors: int = 2) -> list[BindingRecord]:
    rng = random.Random(seed)
    return [generate_relational_case(i, rng=rng, distractors=distractors) for i in range(n)]


def generate_relational_case(index: int, *, rng: random.Random | None = None, distractors: int = 2) -> BindingRecord:
    rng = rng or random.Random(index)
    anchor = {
        "id": f"rel_{index:04d}_obj_0",
        "shape": "circle",
        "color": "blue",
        "x": 160,
        "y": 120,
    }
    relation = rng.choice(RELATIONS)
    target = _target_for_relation(index, relation, anchor)
    objects = [anchor, target]
    objects.append(
        {
            "id": f"rel_{index:04d}_obj_2",
            "shape": target["shape"],
            "color": target["color"],
            "x": anchor["x"] + (90 if target["x"] <= anchor["x"] else -90),
            "y": anchor["y"] + (70 if target["y"] <= anchor["y"] else -70),
        }
    )
    for j in range(max(0, distractors - 1)):
        shape = rng.choice([shape for shape in SHAPES if shape != target["shape"]])
        color = rng.choice([color for color in COLORS if color != target["color"]])
        objects.append(
            {
                "id": f"rel_{index:04d}_obj_{j + 3}",
                "shape": shape,
                "color": color,
                "x": 36 + 48 * j,
                "y": 36 + 31 * j,
            }
        )
    candidates = [
        Candidate(
            candidate_id=obj["id"],
            candidate_type=obj["shape"],
            label=f"candidate {idx}",
            executable_id=obj["id"],
            semantic_id=obj["id"],
            bbox=[obj["x"] - 10, obj["y"] - 10, 20, 20],
            attributes={"shape": obj["shape"], "color": obj["color"]},
        )
        for idx, obj in enumerate(objects)
    ]
    instruction = f"Select the {target['shape']} {relation} the blue circle."
    return BindingRecord(
        case_id=f"rel_{index:04d}",
        source_family="relational_synth",
        split="generated",
        generator_version="relational.v1",
        observation=Observation(source=_svg(objects), source_type="svg", state_metadata={"objects": objects}),
        instruction=instruction,
        operation="select",
        candidate_set=candidates,
        gold=GoldTarget(
            semantic_target={"object_id": target["id"], "shape": target["shape"], "color": target["color"]},
            executable_target={"target_id": target["id"]},
        ),
        binding_structure=BindingStructure(
            target_type=target["shape"],
            attributes={"shape": target["shape"], "color": target["color"]},
            anchor=anchor["id"],
            relation={"type": relation, "anchor": anchor["id"]},
            executable_type="object_id",
            symbolic_reference=target["id"],
        ),
        metadata={"distractors": distractors},
    )


def _target_for_relation(index: int, relation: str, anchor: dict[str, Any]) -> dict[str, Any]:
    offsets = {
        "left of": (-70, 0),
        "right of": (70, 0),
        "above": (0, -60),
        "below": (0, 60),
        "nearest": (30, 0),
    }
    dx, dy = offsets[relation]
    return {
        "id": f"rel_{index:04d}_obj_1",
        "shape": "triangle",
        "color": "red",
        "x": anchor["x"] + dx,
        "y": anchor["y"] + dy,
    }


def _svg(objects: list[dict[str, Any]]) -> str:
    rows = ['<svg xmlns="http://www.w3.org/2000/svg" width="320" height="240">']
    for obj in objects:
        if obj["shape"] == "circle":
            rows.append(f'<circle id="{obj["id"]}" cx="{obj["x"]}" cy="{obj["y"]}" r="12" fill="{obj["color"]}"/>')
        elif obj["shape"] == "square":
            rows.append(f'<rect id="{obj["id"]}" x="{obj["x"] - 12}" y="{obj["y"] - 12}" width="24" height="24" fill="{obj["color"]}"/>')
        else:
            x, y = obj["x"], obj["y"]
            rows.append(f'<polygon id="{obj["id"]}" points="{x},{y-14} {x-14},{y+14} {x+14},{y+14}" fill="{obj["color"]}"/>')
    rows.append("</svg>")
    return "\n".join(rows)
