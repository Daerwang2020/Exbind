from __future__ import annotations

from typing import Any

from exbind.schema import BindingRecord, BindingStructure, Candidate, GoldTarget, Observation


def convert_dom_row(row: dict[str, Any]) -> BindingRecord:
    candidates = [
        Candidate(
            candidate_id=str(candidate.get("dom_id")),
            candidate_type=str(candidate.get("role") or candidate.get("tag") or "dom_node"),
            label=_candidate_label(candidate),
            executable_id=str(candidate.get("dom_id")),
            semantic_id=str(candidate.get("xpath") or candidate.get("css_selector") or candidate.get("dom_id")),
            bbox=candidate.get("bbox") if isinstance(candidate.get("bbox"), list) else None,
            attributes={
                "tag": candidate.get("tag"),
                "role": candidate.get("role"),
                "type": candidate.get("type"),
                "xpath": candidate.get("xpath"),
                "css_selector": candidate.get("css_selector"),
                "visible_text": candidate.get("visible_text"),
                "aria_label": candidate.get("aria_label"),
                "placeholder": candidate.get("placeholder"),
            },
            raw=dict(candidate),
        )
        for candidate in row.get("candidates") or []
        if isinstance(candidate, dict) and candidate.get("dom_id")
    ]
    gold = row.get("gold") if isinstance(row.get("gold"), dict) else {}
    gold_candidate = next(
        (candidate for candidate in candidates if candidate.candidate_id == gold.get("dom_id")),
        None,
    )
    return BindingRecord(
        case_id=str(row.get("sample_id") or ""),
        source_family=str(row.get("dataset") or "DOM"),
        split=str((row.get("metadata") or {}).get("split") or ""),
        generator_version=str((row.get("metadata") or {}).get("generator") or ""),
        observation=Observation(
            image=row.get("screenshot_path"),
            source=row.get("html_path"),
            source_type="html",
            state_metadata=dict(row.get("metadata") or {}),
        ),
        instruction=str(row.get("instruction") or ""),
        operation=str(row.get("operation") or ""),
        candidate_set=candidates,
        gold=GoldTarget(
            semantic_target={
                "visible_text": gold_candidate.label if gold_candidate else None,
                "xpath": gold.get("xpath"),
            },
            executable_target={
                "dom_id": gold.get("dom_id"),
                "xpath": gold.get("xpath"),
                "css_selector": gold.get("css_selector"),
            },
        ),
        binding_structure=BindingStructure(
            target_type=str(row.get("task") or "dom_reference_selection"),
            attributes={"operation": row.get("operation")},
            hierarchy_path=_xpath_path(gold.get("xpath")),
            executable_type="dom_id",
            symbolic_reference=gold.get("dom_id"),
        ),
        metadata={"raw_prediction": row.get("raw_prediction")},
    )


def _candidate_label(candidate: dict[str, Any]) -> str:
    return " ".join(
        str(candidate.get(key) or "").strip()
        for key in ("visible_text", "aria_label", "placeholder", "name")
        if str(candidate.get(key) or "").strip()
    )


def _xpath_path(xpath: Any) -> list[str] | None:
    if not isinstance(xpath, str) or not xpath:
        return None
    return [part for part in xpath.strip("/").split("/") if part]
