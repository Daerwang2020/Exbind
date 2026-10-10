from __future__ import annotations

from exbind.latent import LatentBindingInstance, LatentEntity


def make_hard_hierarchy_latent(index: int, *, depth: int, level: str) -> LatentBindingInstance:
    prefix = f"hard_hier_{level}_{index:04d}"
    group_a = LatentEntity(f"{prefix}_group_a", "group", {"name": "Group A"})
    group_b = LatentEntity(f"{prefix}_group_b", "group", {"name": "Group B"})
    entities = [group_a, group_b]
    target_parent = group_a
    distractor_parent = group_b

    for branch, parent in (("target", target_parent), ("distractor", distractor_parent)):
        for layer in range(depth):
            node = LatentEntity(
                f"{prefix}_{branch}_module_{layer}",
                "module",
                {"name": "Module", "module": "Module", "layer": layer + 1},
                parent=parent.entity_id,
            )
            parent.children.append(node.entity_id)
            entities.append(node)
            parent = node
        output = LatentEntity(
            f"{prefix}_{branch}_output",
            "port",
            {"group": "Group A" if branch == "target" else "Group B", "module": "Module", "part": "output"},
            role="output",
            parent=parent.entity_id,
        )
        input_port = LatentEntity(
            f"{prefix}_{branch}_input",
            "port",
            {"group": "Group A" if branch == "target" else "Group B", "module": "Module", "part": "input"},
            role="input",
            parent=parent.entity_id,
        )
        parent.children.extend([output.entity_id, input_port.entity_id])
        entities.extend([output, input_port])

    path = ["Group A", *(["Module"] * depth), "output"]
    target_entity = next(entity for entity in entities if entity.entity_id == f"{prefix}_target_output")
    return LatentBindingInstance(
        latent_id=prefix,
        entities=entities,
        hierarchy=[
            {"parent": entity.parent, "child": entity.entity_id}
            for entity in entities
            if entity.parent is not None
        ],
        target_spec={
            "group": "Group A",
            "object": "Module",
            "part": "output",
            "role": "output",
            "hierarchy_path": path,
            "canonical_target": target_entity.entity_id,
            "target_type": "port",
            "target_attributes": {"group": "Group A", "module": "Module", "part": "output"},
            "executable_type": "port_id",
        },
        operation="select_port",
        capability_tags=["identity", "hierarchy", "role", "type", "executable"],
        difficulty_tags={
            "difficulty": level,
            "hierarchy_depth": depth,
            "candidate_count": 4,
            "confusable_entities": 3,
        },
        metadata={"hard_family": "hierarchy", "hard_level": level, "hierarchy_depth": depth},
    )


def make_hard_table_latent(index: int, *, level: str) -> LatentBindingInstance:
    prefix = f"hard_table_{level}_{index:04d}"
    target_owner = "Team B"
    target_attribute = "status" if index % 2 == 0 else "priority"
    if level == "easy":
        rows = [
            ("Alpha", "Team A", "blocked", "low"),
            ("Beacon", "Team B", "review", "high"),
            ("Cobalt", "Team C", "ready", "medium"),
        ]
        conditions = {"owner": target_owner}
    elif level == "medium":
        rows = [
            ("Alpha", "Team A", "blocked", "low"),
            ("Beacon", "Team B", "review", "high"),
            ("Cobalt", "Team B", "ready", "medium"),
            ("Delta", "Team C", "review", "low"),
        ]
        conditions = {"owner": target_owner, "status": "review"}
    else:
        rows = [
            ("Alpha", "Team A", "blocked", "low"),
            ("Beacon", "Team B", "review", "high"),
            ("Cobalt", "Team B", "ready", "medium"),
            ("Delta", "Team B", "review", "low"),
            ("Ember", "Team B", "blocked", "high"),
            ("Fjord", "Team C", "ready", "medium"),
        ]
        conditions = {"owner": target_owner, "status": "review", "priority": "high"}
    entities = [
        LatentEntity(
            f"{prefix}_row_{row_index}",
            "row",
            {"project_label": project, "owner": owner, "status": status, "priority": priority},
        )
        for row_index, (project, owner, status, priority) in enumerate(rows)
    ]
    target_row = next(entity for entity in entities if all(entity.attributes[key] == value for key, value in conditions.items()))
    canonical = f"{target_row.entity_id}.{target_attribute}"
    return LatentBindingInstance(
        latent_id=prefix,
        entities=entities,
        attributes={"columns": ["owner", "status", "priority"]},
        target_spec={
            "canonical_target": canonical,
            "entity": target_row.entity_id,
            "attribute": target_attribute,
            "owner": target_owner,
            "row_conditions": conditions,
            "target_type": "cell",
            "target_attributes": {"owner": target_owner, "column": target_attribute},
            "executable_type": "cell_id",
        },
        operation="select_cell",
        capability_tags=["identity", "attribute", "executable"],
        difficulty_tags={
            "difficulty": level,
            "table_ambiguity": level,
            "candidate_count": len(entities) * 3,
            "confusable_entities": sum(entity.attributes["owner"] == target_owner for entity in entities),
        },
        metadata={"hard_family": "table", "hard_level": level, "table_ambiguity": level},
    )
