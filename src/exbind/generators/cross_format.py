from __future__ import annotations

import random

from exbind.latent import LatentBindingInstance, LatentEntity


RELATIONS = ("left_of", "right_of", "above", "below", "nearest")


def make_relational_latent(index: int, *, rng: random.Random | None = None, distractors: int = 4) -> LatentBindingInstance:
    rng = rng or random.Random(index)
    relation = RELATIONS[index % len(RELATIONS)]
    anchor = LatentEntity(
        entity_id=f"rel_lat_{index:04d}_anchor",
        semantic_type="circle",
        attributes={"color": "blue", "x": 160, "y": 120},
    )
    dx, dy = {
        "left_of": (-70, 0),
        "right_of": (70, 0),
        "above": (0, -60),
        "below": (0, 60),
        "nearest": (30, 0),
    }[relation]
    target = LatentEntity(
        entity_id=f"rel_lat_{index:04d}_target",
        semantic_type="triangle",
        attributes={"color": "red", "x": 160 + dx, "y": 120 + dy},
    )
    same_attrs_distractor = LatentEntity(
        entity_id=f"rel_lat_{index:04d}_same_attr_distractor",
        semantic_type="triangle",
        attributes={"color": "red", "x": 160 - dx if dx else 240, "y": 120 - dy if dy else 190},
    )
    entities = [anchor, target, same_attrs_distractor]
    for j in range(max(0, distractors - 1)):
        entities.append(
            LatentEntity(
                entity_id=f"rel_lat_{index:04d}_distractor_{j}",
                semantic_type=rng.choice(["circle", "square"]),
                attributes={"color": rng.choice(["green", "yellow"]), "x": 36 + 45 * j, "y": 36 + 31 * j},
            )
        )
    return LatentBindingInstance(
        latent_id=f"rel_lat_{index:04d}",
        entities=entities,
        relations=[{"type": relation, "source": target.entity_id, "anchor": anchor.entity_id}],
        target_spec={
            "target_type": "triangle",
            "target_attributes": {"color": "red"},
            "anchor": anchor.entity_id,
            "relation": {"type": relation, "anchor": anchor.entity_id},
            "canonical_target": target.entity_id,
            "executable_type": "object_id",
        },
        operation="select",
        capability_tags=["identity", "attribute", "anchor", "relation", "type", "executable"],
        difficulty_tags={
            "difficulty": "medium",
            "candidate_count": len(entities),
            "distractor_count": len(entities) - 2,
            "relation_depth": 1,
            "confusable_entities": 1,
        },
    )


def make_hierarchy_latent(index: int, *, varied: bool = False) -> LatentBindingInstance:
    if not varied:
        return _make_hierarchy_latent_v1(index)

    labels = (("Group A", "Group B"), ("Group Cedar", "Group Harbor"), ("Group North", "Group South"))
    group_a_name, group_b_name = labels[index % len(labels)]
    module_name = ("left module", "right module")[index % 2]
    module_position = module_name.split()[0]
    target_group_name = (group_a_name, group_b_name)[(index // 2) % 2]
    target_part = ("input", "output")[(index // 4) % 2]
    other_part = "output" if target_part == "input" else "input"
    prefix = f"hier_v2_lat_{index:04d}"
    group_a = LatentEntity(f"{prefix}_group_a", "group", {"name": group_a_name})
    group_b = LatentEntity(f"{prefix}_group_b", "group", {"name": group_b_name})
    module_a = LatentEntity(
        f"{prefix}_module_a",
        "module",
        {"name": module_name, "group": group_a_name, "module": module_position},
        parent=group_a.entity_id,
    )
    module_b = LatentEntity(
        f"{prefix}_module_b",
        "module",
        {"name": module_name, "group": group_b_name, "module": module_position},
        parent=group_b.entity_id,
    )
    target_group = group_a if target_group_name == group_a_name else group_b
    target_module = module_a if target_group is group_a else module_b
    other_module = module_b if target_module is module_a else module_a
    target = LatentEntity(
        f"{prefix}_target_{target_part}",
        "port",
        {"group": target_group_name, "module": module_position, "part": target_part},
        role=target_part,
        parent=target_module.entity_id,
    )
    sibling = LatentEntity(
        f"{prefix}_target_{other_part}",
        "port",
        {"group": target_group_name, "module": module_position, "part": other_part},
        role=other_part,
        parent=target_module.entity_id,
    )
    group_distractor = LatentEntity(
        f"{prefix}_other_group_{target_part}",
        "port",
        {"group": other_module.attributes["group"], "module": module_position, "part": target_part},
        role=target_part,
        parent=other_module.entity_id,
    )
    target_group.children = [target_module.entity_id]
    target_module.children = [target.entity_id, sibling.entity_id]
    other_module.children = [group_distractor.entity_id]
    other_group = group_b if target_group is group_a else group_a
    other_group.children = [other_module.entity_id]
    return LatentBindingInstance(
        latent_id=prefix,
        entities=[group_a, group_b, module_a, module_b, target, sibling, group_distractor],
        hierarchy=[
            {"parent": group_a.entity_id, "child": module_a.entity_id},
            {"parent": group_b.entity_id, "child": module_b.entity_id},
            {"parent": target_module.entity_id, "child": target.entity_id},
            {"parent": target_module.entity_id, "child": sibling.entity_id},
            {"parent": other_module.entity_id, "child": group_distractor.entity_id},
        ],
        target_spec={
            "group": target_group_name,
            "object": module_name,
            "part": target_part,
            "role": target_part,
            "hierarchy_path": [target_group_name, module_name, target_part],
            "canonical_target": target.entity_id,
            "target_type": "port",
            "target_attributes": {"group": target_group_name, "module": module_position, "part": target_part},
            "executable_type": "port_id",
        },
        operation="select_port",
        capability_tags=["identity", "hierarchy", "role", "type", "executable"],
        difficulty_tags={"difficulty": "medium", "candidate_count": 3, "hierarchy_depth": 3, "confusable_entities": 2},
        metadata={"surface_variant": "interface_gate_v1"},
    )


def _make_hierarchy_latent_v1(index: int) -> LatentBindingInstance:
    group_a = LatentEntity(f"hier_lat_{index:04d}_group_a", "group", {"name": "Group A"})
    group_b = LatentEntity(f"hier_lat_{index:04d}_group_b", "group", {"name": "Group B"})
    module_a_left = LatentEntity(
        f"hier_lat_{index:04d}_module_a_left",
        "module",
        {"name": "left module", "group": "Group A", "module": "left"},
        parent=group_a.entity_id,
    )
    in_port = LatentEntity(
        f"hier_lat_{index:04d}_a_left_input",
        "port",
        {"group": "Group A", "module": "left", "part": "input"},
        role="input",
        parent=module_a_left.entity_id,
    )
    out_port = LatentEntity(
        f"hier_lat_{index:04d}_a_left_output",
        "port",
        {"group": "Group A", "module": "left", "part": "output"},
        role="output",
        parent=module_a_left.entity_id,
    )
    b_out = LatentEntity(
        f"hier_lat_{index:04d}_b_left_output",
        "port",
        {"group": "Group B", "module": "left", "part": "output"},
        role="output",
        parent=group_b.entity_id,
    )
    group_a.children = [module_a_left.entity_id]
    module_a_left.children = [in_port.entity_id, out_port.entity_id]
    group_b.children = [b_out.entity_id]
    return LatentBindingInstance(
        latent_id=f"hier_lat_{index:04d}",
        entities=[group_a, group_b, module_a_left, in_port, out_port, b_out],
        hierarchy=[
            {"parent": group_a.entity_id, "child": module_a_left.entity_id},
            {"parent": module_a_left.entity_id, "child": in_port.entity_id},
            {"parent": module_a_left.entity_id, "child": out_port.entity_id},
            {"parent": group_b.entity_id, "child": b_out.entity_id},
        ],
        target_spec={
            "group": "Group A",
            "object": "left module",
            "part": "output",
            "role": "output",
            "hierarchy_path": ["Group A", "left module", "output"],
            "canonical_target": out_port.entity_id,
            "target_type": "port",
            "target_attributes": {"group": "Group A", "module": "left", "part": "output"},
            "executable_type": "port_id",
        },
        operation="select_port",
        capability_tags=["identity", "hierarchy", "role", "type", "executable"],
        difficulty_tags={
            "difficulty": "medium",
            "candidate_count": 3,
            "hierarchy_depth": 3,
            "confusable_entities": 2,
        },
    )


def make_graph_latent(index: int, *, varied: bool = False) -> LatentBindingInstance:
    if not varied:
        return _make_graph_latent_v1(index)

    label_sets = (
        ("Encoder A", "Decoder B", "Decoder C"),
        ("Parser A", "Renderer B", "Renderer C"),
        ("Router A", "Store B", "Store C"),
        ("Planner A", "Executor B", "Executor C"),
    )
    source_label, target_label, distractor_label = label_sets[index % len(label_sets)]
    prefix = f"graph_v2_lat_{index:04d}"
    source = LatentEntity(f"{prefix}_source", "node", {"label": source_label})
    target = LatentEntity(f"{prefix}_target", "node", {"label": target_label})
    distractor = LatentEntity(f"{prefix}_distractor", "node", {"label": distractor_label, "below": target.entity_id})
    source_out = LatentEntity(f"{prefix}_source_out", "port", {"node": source.entity_id}, role="output", parent=source.entity_id)
    target_in = LatentEntity(f"{prefix}_target_in", "port", {"node": target.entity_id}, role="input", parent=target.entity_id)
    distractor_in = LatentEntity(f"{prefix}_distractor_in", "port", {"node": distractor.entity_id}, role="input", parent=distractor.entity_id)
    source.children = [source_out.entity_id]
    target.children = [target_in.entity_id]
    distractor.children = [distractor_in.entity_id]
    canonical = f"{source_out.entity_id}->{target_in.entity_id}"
    entities = [source, target, distractor, source_out, target_in, distractor_in]
    rotate = index % len(entities)
    entities = entities[rotate:] + entities[:rotate]
    return LatentBindingInstance(
        latent_id=prefix,
        entities=entities,
        relations=[{"type": "directed_edge", "source": source.entity_id, "target": target.entity_id}],
        roles={"source": source_out.entity_id, "target": target_in.entity_id},
        target_spec={
            "canonical_target": canonical,
            "source_endpoint": source_out.entity_id,
            "target_endpoint": target_in.entity_id,
            "target_type": "ordered_endpoint_pair",
            "relation": {"type": "sends_to", "source": source.entity_id, "target": target.entity_id},
            "role": "source_target_pair",
            "executable_type": "ordered_endpoint_pair",
            "instruction": f"Identify the ordered endpoint pair required to connect {source_label} to {target_label}.",
        },
        operation="connect",
        capability_tags=["identity", "relation", "role", "type", "executable"],
        difficulty_tags={"difficulty": "medium", "candidate_count": 3, "ordered_role_ambiguity": 1, "relation_depth": 1},
        metadata={"surface_variant": "interface_gate_v1", "endpoint_explicit": True},
    )


def _make_graph_latent_v1(index: int) -> LatentBindingInstance:
    encoder = LatentEntity(f"graph_lat_{index:04d}_encoder", "node", {"label": "Encoder A"})
    decoder = LatentEntity(f"graph_lat_{index:04d}_decoder", "node", {"label": "Decoder B"})
    aux = LatentEntity(f"graph_lat_{index:04d}_aux", "node", {"label": "Decoder C", "below": decoder.entity_id})
    enc_out = LatentEntity(f"graph_lat_{index:04d}_encoder_out", "port", {"node": encoder.entity_id}, role="output", parent=encoder.entity_id)
    dec_in = LatentEntity(f"graph_lat_{index:04d}_decoder_in", "port", {"node": decoder.entity_id}, role="input", parent=decoder.entity_id)
    aux_in = LatentEntity(f"graph_lat_{index:04d}_aux_in", "port", {"node": aux.entity_id}, role="input", parent=aux.entity_id)
    encoder.children = [enc_out.entity_id]
    decoder.children = [dec_in.entity_id]
    aux.children = [aux_in.entity_id]
    canonical = f"{enc_out.entity_id}->{dec_in.entity_id}"
    return LatentBindingInstance(
        latent_id=f"graph_lat_{index:04d}",
        entities=[encoder, decoder, aux, enc_out, dec_in, aux_in],
        relations=[{"type": "directed_edge", "source": encoder.entity_id, "target": decoder.entity_id}],
        roles={"source": enc_out.entity_id, "target": dec_in.entity_id},
        target_spec={
            "canonical_target": canonical,
            "source_endpoint": enc_out.entity_id,
            "target_endpoint": dec_in.entity_id,
            "target_type": "ordered_endpoint_pair",
            "relation": {"type": "sends_to", "source": encoder.entity_id, "target": decoder.entity_id},
            "role": "source_target_pair",
            "executable_type": "ordered_endpoint_pair",
            "instruction": "Identify the ordered endpoint pair required to connect Encoder A to Decoder B.",
        },
        operation="connect",
        capability_tags=["identity", "relation", "role", "type", "executable"],
        difficulty_tags={"difficulty": "medium", "candidate_count": 3, "ordered_role_ambiguity": 1, "relation_depth": 1},
    )


def make_table_latent(index: int, *, varied: bool = False) -> LatentBindingInstance:
    if not varied:
        return _make_table_latent_v1(index)

    teams = (("Team Aster", "Team Birch", "Team Cedar"), ("Team Harbor", "Team Indigo", "Team Juniper"), ("Team Kestrel", "Team Lumen", "Team Meridian"))[index % 3]
    projects = (("Atlas", "Beacon", "Cobalt"), ("Delta", "Ember", "Fjord"), ("Grove", "Helix", "Ion"))[index % 3]
    statuses = ("blocked", "ready", "review")
    priorities = ("high", "medium", "low")
    target_row_index = (index // 3) % 3
    target_attribute = ("status", "priority")[(index // 9) % 2]
    prefix = f"table_v2_lat_{index:04d}"
    rows = [
        LatentEntity(
            f"{prefix}_project_{row_index}",
            "row",
            {
                "project_label": projects[row_index],
                "owner": teams[row_index],
                "status": statuses[(row_index + index) % len(statuses)],
                "priority": priorities[(row_index + index) % len(priorities)],
            },
        )
        for row_index in range(3)
    ]
    target_row = rows[target_row_index].entity_id
    canonical = f"{target_row}.{target_attribute}"
    return LatentBindingInstance(
        latent_id=prefix,
        entities=rows,
        attributes={"columns": ["owner", "status", "priority"]},
        target_spec={
            "canonical_target": canonical,
            "entity": target_row,
            "attribute": target_attribute,
            "owner": teams[target_row_index],
            "target_type": "cell",
            "target_attributes": {"owner": teams[target_row_index], "column": target_attribute},
            "executable_type": "cell_id",
        },
        operation="select_cell",
        capability_tags=["identity", "attribute", "type", "executable"],
        difficulty_tags={"difficulty": "easy", "candidate_count": 9, "confusable_entities": 1},
        metadata={"surface_variant": "interface_gate_v1"},
    )


def _make_table_latent_v1(index: int) -> LatentBindingInstance:
    rows = [
        LatentEntity(
            f"table_lat_{index:04d}_project_a",
            "row",
            {"project_label": "Project 1", "owner": "Team A", "status": "blocked", "priority": "high"},
        ),
        LatentEntity(
            f"table_lat_{index:04d}_project_b",
            "row",
            {"project_label": "Project 2", "owner": "Team B", "status": "ready", "priority": "medium"},
        ),
        LatentEntity(
            f"table_lat_{index:04d}_project_c",
            "row",
            {"project_label": "Project 3", "owner": "Team C", "status": "ready", "priority": "low"},
        ),
    ]
    target_row = rows[1].entity_id
    canonical = f"{target_row}.status"
    return LatentBindingInstance(
        latent_id=f"table_lat_{index:04d}",
        entities=rows,
        attributes={"columns": ["owner", "status", "priority"]},
        target_spec={
            "canonical_target": canonical,
            "entity": target_row,
            "attribute": "status",
            "owner": "Team B",
            "target_type": "cell",
            "target_attributes": {"owner": "Team B", "column": "status"},
            "executable_type": "cell_id",
        },
        operation="select_cell",
        capability_tags=["identity", "attribute", "type", "executable"],
        difficulty_tags={"difficulty": "easy", "candidate_count": 9, "confusable_entities": 1},
    )
