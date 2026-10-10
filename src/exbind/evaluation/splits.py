from __future__ import annotations

from collections import defaultdict

from exbind.schema import BindingRecord


def split_family_id(record: BindingRecord) -> str:
    return "::".join(
        [
            record.source_family,
            record.generator_version or "unknown_generator",
            str(record.metadata.get("template_id") or record.metadata.get("template") or "unknown_template"),
        ]
    )


def validate_split_family_isolation(records: list[BindingRecord], split_by_case_id: dict[str, str]) -> dict[str, object]:
    splits_by_family: dict[str, set[str]] = defaultdict(set)
    for record in records:
        split = split_by_case_id.get(record.case_id)
        if split is not None:
            splits_by_family[split_family_id(record)].add(split)
    violating = {family: sorted(splits) for family, splits in splits_by_family.items() if len(splits) > 1}
    return {
        "valid": not violating,
        "violating_families": violating,
        "n_families": len(splits_by_family),
    }
