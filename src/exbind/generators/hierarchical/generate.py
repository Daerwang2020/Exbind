from __future__ import annotations

import random
from typing import Any

from exbind.schema import BindingRecord, BindingStructure, Candidate, GoldTarget, Observation


def generate_hierarchical_batch(n: int, *, seed: int = 0) -> list[BindingRecord]:
    rng = random.Random(seed)
    return [generate_hierarchical_case(i, rng=rng) for i in range(n)]


def generate_hierarchical_case(index: int, *, rng: random.Random | None = None) -> BindingRecord:
    rng = rng or random.Random(index)
    module = "left module"
    part = "output port"
    target_id = f"hier_{index:04d}_node_0"
    candidates = [
        Candidate(
            candidate_id=target_id,
            candidate_type="port",
            label="candidate port 0",
            executable_id=target_id,
            semantic_id=target_id,
            bbox=[92, 90, 12, 12],
            attributes={"group": "Group A", "module": "left", "part": "output", "direction": "right"},
        ),
        Candidate(
            candidate_id=f"hier_{index:04d}_node_1",
            candidate_type="port",
            label="candidate port 1",
            executable_id=f"hier_{index:04d}_node_1",
            semantic_id=f"hier_{index:04d}_node_1",
            bbox=[42, 90, 12, 12],
            attributes={"group": "Group A", "module": "left", "part": "input", "direction": "left"},
        ),
        Candidate(
            candidate_id=f"hier_{index:04d}_node_2",
            candidate_type="port",
            label="candidate port 2",
            executable_id=f"hier_{index:04d}_node_2",
            semantic_id=f"hier_{index:04d}_node_2",
            bbox=[92, 170, 12, 12],
            attributes={"group": "Group B", "module": "left", "part": "output", "direction": "right"},
        ),
    ]
    return BindingRecord(
        case_id=f"hier_{index:04d}",
        source_family="hierarchical_synth",
        split="generated",
        generator_version="hierarchical.v1",
        observation=Observation(source=_svg(index), source_type="svg", state_metadata={"groups": ["Group A", "Group B"]}),
        instruction=f"Select the {part} of the {module} in Group A.",
        operation="select_port",
        candidate_set=candidates,
        gold=GoldTarget(
            semantic_target={"group": "Group A", "module": "left", "part": "output port"},
            executable_target={"target_id": target_id},
        ),
        binding_structure=BindingStructure(
            target_type="port",
            attributes={"group": "Group A", "module": "left", "part": "output"},
            hierarchy_path=["Group A", "left module", "output port"],
            part="output port",
            executable_type="port_id",
            symbolic_reference=target_id,
        ),
    )


def _svg(index: int) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="260" height="220">
  <g id="hier_{index:04d}_group_a" data-label="Group A">
    <rect id="hier_{index:04d}_box_0" x="48" y="72" width="50" height="48" fill="#dbeafe" stroke="#1d4ed8"/>
    <circle id="hier_{index:04d}_node_1" cx="48" cy="96" r="6" fill="#ffffff" stroke="#1d4ed8"/>
    <circle id="hier_{index:04d}_node_0" cx="98" cy="96" r="6" fill="#1d4ed8"/>
  </g>
  <g id="hier_{index:04d}_group_b" data-label="Group B">
    <rect id="hier_{index:04d}_box_1" x="48" y="152" width="50" height="48" fill="#fef3c7" stroke="#92400e"/>
    <circle id="hier_{index:04d}_node_2" cx="98" cy="176" r="6" fill="#92400e"/>
  </g>
</svg>"""
