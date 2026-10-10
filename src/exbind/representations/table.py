from __future__ import annotations

from exbind.latent import LatentBindingInstance, RenderedBindingInstance
from exbind.schema import Candidate, Observation


def compile_table(latent: LatentBindingInstance) -> RenderedBindingInstance:
    rows = [entity for entity in latent.entities if entity.semantic_type == "row"]
    columns = list(latent.attributes.get("columns") or ["owner", "status", "priority"])
    mapping = {entity.entity_id: f"table_row_{idx:02d}" for idx, entity in enumerate(rows)}
    cell_mapping = {
        f"{entity.entity_id}.{column}": f"cell_r{row_idx:02d}_c{col_idx:02d}"
        for row_idx, entity in enumerate(rows)
        for col_idx, column in enumerate(columns)
    }
    html = ["<table>", "<thead><tr><th>project</th>"]
    html.extend(f"<th>{column}</th>" for column in columns)
    html.append("</tr></thead><tbody>")
    for entity in rows:
        html.append(f'<tr id="{mapping[entity.entity_id]}"><th>{entity.attributes.get("project_label", "project")}</th>')
        for column in columns:
            key = f"{entity.entity_id}.{column}"
            html.append(f'<td id="{cell_mapping[key]}">{entity.attributes.get(column, "")}</td>')
        html.append("</tr>")
    html.append("</tbody></table>")
    candidates = [
        Candidate(
            candidate_id=cell_id,
            candidate_type="cell",
            label=f"candidate {idx}",
            executable_id=cell_id,
            semantic_id=semantic,
            attributes={
                "row": semantic.split(".")[0],
                "column": semantic.split(".")[1],
                "value": _cell_value(rows, semantic),
            },
        )
        for idx, (semantic, cell_id) in enumerate(sorted(cell_mapping.items()))
    ]
    canonical = latent.canonical_target or ""
    spec = latent.target_spec
    conditions = dict(spec.get("row_conditions") or {})
    if spec.get("owner") is not None:
        conditions.setdefault("owner", spec.get("owner"))
    condition_text = " and ".join(f"{column} {value}" for column, value in conditions.items())
    if not condition_text:
        condition_text = "the specified project"
    instruction = f"Select the {spec.get('attribute', 'attribute')} cell for the project where {condition_text}."
    return RenderedBindingInstance(
        latent_id=latent.latent_id,
        format="table",
        observation=Observation(source="\n".join(html), source_type="html_table", state_metadata={"columns": columns}),
        source_structure={"html": html, "columns": columns},
        candidate_set=candidates,
        instruction=instruction,
        gold_executable_target={"cell_id": cell_mapping[canonical]},
        canonical_semantic_target=canonical,
        mapping_latent_to_surface={**mapping, **cell_mapping},
        metadata={"representation_family": "structured_table"},
    )


def _cell_value(rows, semantic: str):
    row_id, column = semantic.split(".")
    entity = next(item for item in rows if item.entity_id == row_id)
    return entity.attributes.get(column)
