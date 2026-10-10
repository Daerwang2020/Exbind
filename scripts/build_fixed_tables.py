#!/usr/bin/env python3
"""Construct an independent, paired table-binding evaluation before model scoring.

Each source has one image and eight eligible cell marks. Direct, composed, and
metadata-conditioned questions keep the image, candidates, order, and target
fixed. A counterfactual metadata question changes only the active record and
therefore changes the gold cell. The content is independently generated and
does not reuse the visual-primary table rows, values, or CSS.
"""
from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
import random

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fixed_tables"
SEED = 20261003
N_SOURCES = 96
REGIONS = ("North", "South", "East", "West")
STAGES = ("Open", "Review", "Pending", "Closed")
PROJECT_WORDS = ("Aster", "Beryl", "Cedar", "Drift", "Ember", "Fjord", "Grove", "Harbor",
                 "Ivory", "Juniper", "Kite", "Lumen", "Mosaic", "Nova", "Orchid", "Pine",
                 "Quartz", "Ridge", "Solace", "Talon", "Umber", "Vale", "Willow", "Zephyr")
OWNERS = ("Ava Chen", "Noah Park", "Mina Rao", "Leo Brooks", "Sara Kim", "Owen Li",
          "Nina Patel", "Eli Moore", "Iris Hall", "Theo Reed", "Maya Fox", "Ravi Shah")
SKINS = ("indigo", "teal", "slate", "plum")
SOURCE_ASSIGNMENTS = [(region, stage, col, SKINS[(ri + si + ci + rep) % 4])
                      for ri, region in enumerate(REGIONS)
                      for si, stage in enumerate(STAGES)
                      for ci, col in enumerate(("owner", "amount"))
                      for rep in range(3)]
random.Random(SEED + 11).shuffle(SOURCE_ASSIGNMENTS)
SHIFTS = (1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5)
MARK_ASSIGNMENTS = [(t, ((t - 1 + shift) % 8) + 1) for shift in SHIFTS for t in range(1, 9)]
random.Random(SEED + 17).shuffle(MARK_ASSIGNMENTS)


CSS = """
*{box-sizing:border-box}html,body{margin:0;padding:0}body{font:14px/1.35 -apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;color:#223047;background:#eef2f8}
.app{min-height:900px;display:grid;grid-template-columns:208px 1fr}.rail{background:#14263f;color:#bed0e5;padding:26px 15px;display:flex;flex-direction:column}.brand{color:white;font-size:18px;font-weight:800;padding:0 13px 27px}.brand i{font-style:normal;background:#547cf1;border-radius:7px;padding:5px;margin-right:8px}.nav{padding:12px;border-radius:8px;margin:3px 0;font-size:13px}.nav.on{background:#304a73;color:white;font-weight:700}.railfoot{margin-top:auto;padding:16px 12px;border-top:1px solid #36506a;font-size:12px}.main{min-width:0}.top{height:65px;background:white;border-bottom:1px solid #dfe5ee;display:flex;align-items:center;justify-content:space-between;padding:0 29px;color:#6a7b91}.crumb b{color:#243751}.search{border:1px solid #e2e8f0;border-radius:7px;padding:7px 15px;background:#f8fafc}.content{max-width:1200px;margin:26px auto;padding:0 22px}.eyebrow{font-size:11px;letter-spacing:1.3px;color:#637891;text-transform:uppercase;font-weight:800}h1{font-size:25px;margin:6px 0 5px}.sub{font-size:13px;color:#6f7f94;margin:0 0 20px}.bar{display:flex;gap:9px;margin-bottom:16px}.chip{background:white;border:1px solid #dce4ee;border-radius:7px;padding:8px 12px;font-size:12px;color:#566a82}.spacer{flex:1}.primary{background:#416bdd;border-radius:7px;padding:8px 13px;color:white;font-weight:700}.card{background:white;border:1px solid #dfe6ef;border-radius:11px;box-shadow:0 4px 14px #1e33510b;overflow:hidden}.cardhead{padding:19px 20px;border-bottom:1px solid #e4eaf2;font-weight:750}.cardhead small{font-weight:500;color:#8a98a9;margin-left:6px}.cardhead em{font-style:normal;float:right;font-size:12px;color:#78899f}table{border-collapse:collapse;width:100%;table-layout:fixed}th{background:#f6f8fc;color:#63758d;font-size:11px;text-transform:uppercase;letter-spacing:.5px;text-align:left;padding:13px 15px;border-bottom:1px solid #e2e9f1}td{height:59px;border-bottom:1px solid #edf1f6;padding:9px 15px;font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}tr:nth-child(even) td{background:#fcfdff}.project{font-weight:740;color:#283d5c}.muted{font-size:11px;color:#95a3b5}.stage{display:inline-block;padding:5px 9px;border-radius:16px;background:#edf2ff;color:#4e69b8;font-size:11px;font-weight:700}.amount{font-variant-numeric:tabular-nums;font-weight:720}.candidate{position:relative;padding-left:43px !important}.candidate::before{content:attr(data-mark);position:absolute;left:9px;top:50%;transform:translateY(-50%);display:grid;place-items:center;width:24px;height:22px;border:1px solid #bd8a21;border-radius:5px;background:#ffd35c;color:#3a2b06;font-size:12px;font-weight:850;line-height:1;z-index:2}.candidate::after{content:'';position:absolute;inset:5px 4px;border:1px solid #d3a53d;border-radius:5px;pointer-events:none}.foot{padding:14px 20px;font-size:11px;color:#8c9bad}.indigo .rail{background:#14263f}.teal .rail{background:#163c43}.teal .primary{background:#208c92}.slate .rail{background:#243343}.slate .primary{background:#557186}.plum .rail{background:#342746}.plum .primary{background:#8660a7}
"""


def choose_rows(source_i: int, rng: random.Random, target_region: str, target_stage: str) -> tuple[list[dict], int, int]:
    ri, si = REGIONS.index(target_region), STAGES.index(target_stage)
    other_region = REGIONS[(ri + 1 + rng.randrange(3)) % 4]
    other_stage = STAGES[(si + 1 + rng.randrange(3)) % 4]
    pairs = [(target_region, target_stage), (target_region, other_stage),
             (other_region, target_stage), (other_region, other_stage)]
    remaining = [(a, b) for a in REGIONS for b in STAGES if (a, b) not in pairs]
    rng.shuffle(remaining)
    pairs += remaining[:4]
    rows = []
    words = rng.sample(PROJECT_WORDS, 8)
    people = rng.sample(OWNERS, 8)
    ref_codes = rng.sample(range(100, 1000), 8)
    for j, (region, stage) in enumerate(pairs):
        ref = f"R-{source_i+1:03d}-{ref_codes[j]}"
        rows.append({"ref": ref, "project": f"{words[j]} {rng.randrange(100, 999)}",
                     "region": region, "stage": stage, "owner": people[j],
                     "amount": f"${rng.randrange(14, 480):,},{rng.randrange(0, 1000):03d}"})
    target_ref, alternate_ref = rows[0]["ref"], rows[1]["ref"]
    rng.shuffle(rows)
    return rows, next(i for i, r in enumerate(rows) if r["ref"] == target_ref), next(i for i, r in enumerate(rows) if r["ref"] == alternate_ref)


def make_source(source_i: int) -> dict:
    rng = random.Random(SEED + source_i * 104729)
    target_region_assigned, target_stage_assigned, target_col, skin = SOURCE_ASSIGNMENTS[source_i]
    rows, target, alternate = choose_rows(source_i, rng, target_region_assigned, target_stage_assigned)
    other_col = "amount" if target_col == "owner" else "owner"
    target_region, target_stage = rows[target]["region"], rows[target]["stage"]
    assert sum(r["region"] == target_region and r["stage"] == target_stage for r in rows) == 1
    assert sum(r["region"] == target_region for r in rows) >= 2
    assert sum(r["stage"] == target_stage for r in rows) >= 2
    same_stage = next(i for i, r in enumerate(rows) if i != target and r["stage"] == target_stage)
    other_row = next(i for i in range(8) if i not in (target, alternate, same_stage))
    extra_row = next(i for i in range(8) if i not in (target, alternate, same_stage, other_row))
    marked = [(target, target_col), (alternate, target_col), (same_stage, target_col),
              (other_row, target_col), (target, other_col), (alternate, other_col),
              (extra_row, other_col), (same_stage, other_col)]
    rng.shuffle(marked)
    target_mark_fixed, alternate_mark_fixed = MARK_ASSIGNMENTS[source_i]
    marks = [m for m in range(1, 9) if m not in (target_mark_fixed, alternate_mark_fixed)]
    rng.shuffle(marks)
    marks_by_key = {f"{rows[target]['ref']}::{target_col}": target_mark_fixed,
                    f"{rows[alternate]['ref']}::{target_col}": alternate_mark_fixed}
    for ri, col in marked:
        key = f"{rows[ri]['ref']}::{col}"
        if key not in marks_by_key:
            marks_by_key[key] = marks.pop()
    assert not marks and len(marks_by_key) == 8
    mark_by_cell = marks_by_key
    mark_map = {str(mark_by_cell[f"{rows[ri]['ref']}::{col}"]):
                f"cell_{rows[ri]['ref'].replace('-', '_')}_{col}" for ri, col in marked}
    target_mark = mark_by_cell[f"{rows[target]['ref']}::{target_col}"]
    alternate_mark = mark_by_cell[f"{rows[alternate]['ref']}::{target_col}"]
    columns = ["project", "ref"] + rng.sample(["region", "stage", "owner", "amount"], 4)
    return {"source_id": f"table_v3_{source_i+1:03d}", "skin": skin,
            "split": "development" if source_i < 24 else "heldout",
            "rows": rows, "columns": columns, "target_index": target, "alternate_index": alternate,
            "target_column": target_col, "mark_by_cell": mark_by_cell, "mark_map": mark_map,
            "target_mark": target_mark, "alternate_mark": alternate_mark,
            "target_region": target_region, "target_stage": target_stage}


def render_html(src: dict) -> str:
    colnames = {"project": "Project", "ref": "Ref", "region": "Region", "stage": "Stage", "owner": "Owner", "amount": "Amount"}
    header = "".join(f"<th>{colnames[c]}</th>" for c in src["columns"])
    body = []
    for row in src["rows"]:
        cells = []
        for col in src["columns"]:
            val = escape(row[col])
            if col == "project": val = f'<span class="project">{val}</span>'
            if col == "stage": val = f'<span class="stage">{val}</span>'
            if col == "amount": val = f'<span class="amount">{val}</span>'
            key = f"{row['ref']}::{col}"
            if key in src["mark_by_cell"]:
                mark = src["mark_by_cell"][key]
                sid = src["mark_map"][str(mark)]
                cells.append(f'<td id="{sid}" class="candidate" data-mark="{mark}">{val}</td>')
            else:
                cells.append(f"<td>{val}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (f'<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head>'
            f'<body class="{src["skin"]}"><div class="app"><aside class="rail"><div class="brand"><i>▥</i>LedgerPilot</div>'
            '<div class="nav">◫ &nbsp; Overview</div><div class="nav on">▦ &nbsp; Portfolio</div><div class="nav">◷ &nbsp; Activity</div><div class="nav">⚙ &nbsp; Settings</div>'
            '<div class="railfoot">Operations workspace<br><span class="muted">Quarterly review</span></div></aside><main class="main">'
            '<div class="top"><span class="crumb">Workspace / <b>Portfolio records</b></span><span class="search">⌕ &nbsp; Search records</span></div>'
            '<section class="content"><div class="eyebrow">Operations / Portfolio</div><h1>Portfolio records</h1>'
            '<p class="sub">Review active records, ownership, and financial details.</p><div class="bar"><span class="chip">All records ⌄</span>'
            '<span class="chip">Current quarter ⌄</span><span class="spacer"></span><span class="primary">+ New record</span></div>'
            f'<div class="card"><div class="cardhead">Records <small>· 8 shown</small><em>Columns · Export</em></div>'
            f'<table><thead><tr>{header}</tr></thead><tbody>{"".join(body)}</tbody></table>'
            '<div class="foot">Showing 8 records &nbsp; · &nbsp; Page 1 of 1</div></div></section></main></div></body></html>')


def case_rows(src: dict, image_path: str, html_path: str) -> list[dict]:
    target = src["rows"][src["target_index"]]
    alternate = src["rows"][src["alternate_index"]]
    col = src["target_column"].title()
    refs = [target["ref"], alternate["ref"]] + [r["ref"] for r in src["rows"] if r["ref"] not in (target["ref"], alternate["ref"])][:2]
    observations = []
    for active_ref in (target["ref"], alternate["ref"]):
        entries = [{"ref": ref, "active": ref == active_ref} for ref in refs]
        observations.append({"status_list": entries})
    specs = [
        ("direct", f'Select the numbered cell in the "{target["project"]}" row and "{col}" column.', None, src["target_mark"]),
        ("compose", f'Select the numbered "{col}" cell for the record whose Region is "{src["target_region"]}" and Stage is "{src["target_stage"]}".', None, src["target_mark"]),
        ("context", f'Select the numbered "{col}" cell for the record marked active in the supplied status list.', observations[0], src["target_mark"]),
        ("context_flip", f'Select the numbered "{col}" cell for the record marked active in the supplied status list.', observations[1], src["alternate_mark"]),
    ]
    out = []
    for condition, instruction, obs, gold in specs:
        out.append({"version": "exbind.table_controlled.v3", "case_id": f'{src["source_id"]}__{condition}',
                    "parent_latent_id": src["source_id"], "family": "data_table", "condition": condition,
                    "skin": src["skin"], "split": src["split"],
                    "image_path": image_path, "html_path": html_path,
                    "instruction": instruction, "structured_observation": obs,
                    "candidate_marks": list(range(1, 9)), "mark_map": src["mark_map"],
                    "gold_mark": gold, "gold_surface_id": src["mark_map"][str(gold)],
                    "target_column": src["target_column"]})
    return out


def text_controls(src: dict, cases: list[dict]) -> list[dict]:
    column_names = {"project": "Project", "ref": "Ref", "region": "Region",
                    "stage": "Stage", "owner": "Owner", "amount": "Amount"}
    by_mark = {int(mark): sid for mark, sid in src["mark_map"].items()}
    flat = []
    for mark in sorted(by_mark):
        sid = by_mark[mark]
        ref, col = sid.removeprefix("cell_").rsplit("_", 1)
        row = next(r for r in src["rows"] if r["ref"].replace("-", "_") == ref)
        flat.append(f"{mark}: {row[col]}")
    ordered = ["Columns from left to right: " + ", ".join(column_names[c] for c in src["columns"])]
    for row in src["rows"]:
        values = []
        for col in src["columns"]:
            key = f"{row['ref']}::{col}"
            mark = src["mark_by_cell"].get(key)
            values.append(f"{column_names[col]}={row[col]}" + (f" [mark {mark}]" if mark is not None else ""))
        ordered.append("; ".join(values))
    return [{"case_id": c["case_id"], "parent_latent_id": src["source_id"],
             "split": src["split"], "flat_labels": "\n".join(flat),
             "reading_order": "\n".join(ordered)} for c in cases]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    out = args.out
    for sub in ("images", "html"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    sources = [make_source(i) for i in range(N_SOURCES)]
    rows, controls = [], []
    for src in sources:
        img = f"images/{src['source_id']}.png"
        html = f"html/{src['source_id']}.html"
        (out / html).write_text(render_html(src), encoding="utf-8")
        source_cases = case_rows(src, img, html)
        rows.extend(source_cases)
        controls.extend(text_controls(src, source_cases))
    assert len(rows) == 384 and len({r["case_id"] for r in rows}) == 384
    assert all(r["mark_map"][str(r["gold_mark"])] == r["gold_surface_id"] for r in rows)
    assert all(rows[i]["image_path"] == rows[i+1]["image_path"] == rows[i+2]["image_path"] == rows[i+3]["image_path"]
               for i in range(0, len(rows), 4))
    assert all(rows[i]["gold_mark"] == rows[i+1]["gold_mark"] == rows[i+2]["gold_mark"]
               and rows[i+3]["gold_mark"] != rows[i]["gold_mark"] for i in range(0, len(rows), 4))
    if args.render:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
            for src in sources:
                page.goto((out / "html" / (src["source_id"] + ".html")).as_uri())
                page.screenshot(path=str(out / "images" / (src["source_id"] + ".png")), full_page=True)
                if page.locator("[data-mark]").count() != 8:
                    raise ValueError(f"{src['source_id']}: mark count")
                boxes = []
                for mark in range(1, 9):
                    el = page.locator(f'[data-mark="{mark}"]')
                    box = el.bounding_box()
                    if not box or box["width"] < 75 or box["height"] < 35:
                        raise ValueError(f"{src['source_id']}: malformed cell {mark}")
                    boxes.append(box)
                for i, a in enumerate(boxes):
                    for b in boxes[i+1:]:
                        if min(a["x"]+a["width"],b["x"]+b["width"]) > max(a["x"],b["x"]) and min(a["y"]+a["height"],b["y"]+b["height"]) > max(a["y"],b["y"]):
                            raise ValueError(f"{src['source_id']}: overlapping candidates")
            browser.close()
    (out / "private_sources.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in sources), encoding="utf-8")
    (out / "cases.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in rows), encoding="utf-8")
    (out / "text_controls.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in controls), encoding="utf-8")
    manifest = {"version": "exbind.table_controlled.v3", "seed": SEED, "n_source": N_SOURCES,
                "n_questions": len(rows),
                "n_images": sum((out / "images" / f"table_v3_{i+1:03d}.png").is_file() for i in range(N_SOURCES)),
                "development_sources": 24, "heldout_sources": 72,
                "conditions": ["direct", "compose", "context", "context_flip"],
                "generation_script": "scripts/build_exbind_table_controlled_v3.py",
                "status": "constructed; not human-audited or model-scored",
                "design": "same image, marks, mark order, and target for direct/compose/context; flipped metadata changes only observation and gold"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), **manifest}))


if __name__ == "__main__":
    main()
