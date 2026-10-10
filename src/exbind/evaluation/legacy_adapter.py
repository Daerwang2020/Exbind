from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from exbind.converters.evg import base_case_id


@dataclass(frozen=True)
class NormalizedLegacyRecord:
    case_id: str
    parent_case_id: str | None = None
    source_case_id: str | None = None
    split: str | None = None
    suite: str | None = None
    source_family: str | None = None
    model: str | None = None
    method: str | None = None
    operation: str | None = None
    role: str | None = None
    raw_binding: Any = None
    pred_binding: Any = None
    vicob_binding: Any = None
    gold_binding: Any = None
    candidate_ids: list[str] = field(default_factory=list)
    identifier_regime: str | None = None
    perturbation: str | None = None
    is_perturbed: bool | None = None
    raw_valid: bool | None = None
    pred_valid: bool | None = None
    raw_correct: bool | None = None
    pred_correct: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_stage_a_row(row: dict[str, Any]) -> NormalizedLegacyRecord:
    case_id = str(row.get("sample_id") or row.get("instance_id") or "")
    parent = row.get("base_sample_id") or base_case_id(case_id)
    perturbation = row.get("perturbation")
    if perturbation == "clean":
        perturbation = None
    candidate_ids = normalize_candidate_ids(
        row.get("candidate_ids")
        or row.get("candidate_id")
        or row.get("scores")
        or row.get("logit_scores")
        or row.get("candidate_scores")
    )
    method = str(row.get("method") or row.get("base_method") or "") or None
    pred = row.get("final_prediction", row.get("prediction", row.get("prediction_parsed")))
    return NormalizedLegacyRecord(
        case_id=case_id,
        parent_case_id=str(parent) if parent else None,
        source_case_id=str(row.get("instance_id") or case_id),
        split=_none_if_empty(row.get("split")),
        suite=_none_if_empty(row.get("dataset") or row.get("split")),
        source_family=_source_family(row),
        model=_none_if_empty(row.get("model_name") or row.get("model")),
        method=method,
        operation=_none_if_empty(row.get("operation")),
        role=_none_if_empty(row.get("task")),
        raw_binding=row.get("raw_prediction"),
        pred_binding=pred,
        vicob_binding=pred if method and "vicob" in method.lower() else None,
        gold_binding=row.get("gold"),
        candidate_ids=candidate_ids,
        identifier_regime=identifier_regime(candidate_ids),
        perturbation=perturbation,
        is_perturbed=bool(perturbation),
        raw_valid=_optional_bool(row.get("valid_raw")),
        pred_valid=_optional_bool(row.get("valid_final", row.get("is_valid"))),
        raw_correct=_optional_bool(row.get("exact_raw")),
        pred_correct=_optional_bool(row.get("exact_final", row.get("is_correct", row.get("exact")))),
        metadata={
            key: row.get(key)
            for key in (
                "error_type_raw",
                "error_type_final",
                "score_raw",
                "score_final",
                "changed_by_method",
                "candidate_order_seed",
                "base_method",
                "domain",
                "difficulty",
                "density_bucket",
            )
            if key in row
        },
    )


def normalize_candidate_ids(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        values = raw.keys()
    elif isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, Iterable):
        values = raw
    else:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if isinstance(value, dict):
            cid = value.get("candidate_id") or value.get("element_id") or value.get("id") or value.get("dom_id")
        else:
            cid = value
        if cid is None:
            continue
        text = str(cid)
        if text not in seen:
            out.append(text)
            seen.add(text)
    return out


def identifier_regime(candidate_ids: list[str]) -> str | None:
    if not candidate_ids:
        return None
    generic = 0
    for cid in candidate_ids:
        tail = cid.rsplit("_", 1)[-1]
        if tail.isdigit() or cid.startswith(("path_", "shape_", "dom_")):
            generic += 1
    return "generic_id" if generic == len(candidate_ids) else "semantic_id"


def find_duplicate_keys(records: Iterable[NormalizedLegacyRecord], fields: tuple[str, ...]) -> dict[tuple[Any, ...], int]:
    counts: dict[tuple[Any, ...], int] = {}
    for record in records:
        data = record.to_dict()
        key = tuple(data.get(field) for field in fields)
        counts[key] = counts.get(key, 0) + 1
    return {key: count for key, count in counts.items() if count > 1}


def _source_family(row: dict[str, Any]) -> str | None:
    suite = str(row.get("dataset") or row.get("split") or "")
    if suite.startswith("EVG"):
        return "EVG"
    if suite.startswith("SEB"):
        return "SEB"
    if "Novel" in suite:
        return "Novel-SVG"
    if "MiniWoB" in suite or "DOM" in suite:
        return "DOM"
    return suite or None


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _none_if_empty(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None
