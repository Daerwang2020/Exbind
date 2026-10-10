from __future__ import annotations

from exbind.latent import LatentBindingInstance, RenderedBindingInstance
from exbind.schema import Candidate, Observation


def compile_graph(latent: LatentBindingInstance) -> RenderedBindingInstance:
    mapping = {entity.entity_id: f"graph_item_{idx:02d}" for idx, entity in enumerate(latent.entities)}
    nodes = [entity for entity in latent.entities if entity.semantic_type == "node"]
    ports = [entity for entity in latent.entities if entity.semantic_type == "port"]
    lines = ["digraph {"]
    for node in nodes:
        label = node.attributes.get("label") or node.entity_id
        lines.append(f'  {mapping[node.entity_id]} [label="{label}"];')
    for relation in latent.relations:
        if relation.get("type") == "directed_edge":
            lines.append(f'  {mapping[str(relation["source"])]} -> {mapping[str(relation["target"])]};')
    if latent.metadata.get("endpoint_explicit"):
        for port in ports:
            parent = latent.entity(str(port.parent))
            parent_label = parent.attributes.get("label") if parent else port.parent
            role = port.role or "port"
            lines.append(f'  {mapping[port.entity_id]} [shape=box, label="{role} endpoint of {parent_label}"];')
            if parent:
                lines.append(f'  {mapping[parent.entity_id]} -> {mapping[port.entity_id]} [style=dotted, label="has {role} endpoint"];')
    lines.append("}")
    candidates = [
        Candidate(
            candidate_id=mapping[port.entity_id],
            candidate_type=port.role or "port",
            label=f"candidate {idx}",
            executable_id=mapping[port.entity_id],
            semantic_id=port.entity_id,
            attributes={**port.attributes, "role": port.role, "parent": port.parent},
        )
        for idx, port in enumerate(ports)
    ]
    spec = latent.target_spec
    canonical = latent.canonical_target or ""
    source = str(spec.get("source_endpoint"))
    target = str(spec.get("target_endpoint"))
    instruction = str(spec.get("instruction") or "Identify the ordered endpoint pair required by the graph relation.")
    metadata = {"representation_family": "directed_graph"}
    if latent.metadata.get("endpoint_explicit"):
        metadata["endpoint_explicit"] = True
    return RenderedBindingInstance(
        latent_id=latent.latent_id,
        format="graph",
        observation=Observation(source="\n".join(lines), source_type="dot_graph", state_metadata={"relations": latent.relations}),
        source_structure={"dot": lines, "relations": latent.relations},
        candidate_set=candidates,
        instruction=instruction,
        gold_executable_target={"source_id": mapping[source], "target_id": mapping[target]},
        canonical_semantic_target=canonical,
        mapping_latent_to_surface=mapping,
        metadata=metadata,
    )
