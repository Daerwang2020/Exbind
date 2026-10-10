# ExBind benchmark release

Anonymous review release for the ExBind visual-to-symbolic binding benchmark. The package includes two complementary generated benchmark resources and offline evaluators.

## Contents

| Collection | Unit | Count | Split or role |
|---|---:|---:|---|
| Broad structural suite | Case | 250 | Six symbolic/rendered representations |
| Targeted structural suite | Case | 240 | Table and hierarchy diagnostics |
| Cross-format suite | Latent group | 50 | Two linked surfaces per group |
| Fixed-table suite | Source | 96 | 24 development, 72 held out |
| Fixed-table suite | Request | 384 | Direct, Compose, Context, Flip |

Counts use each collection's own unit and are not summed. The table suite provides rendered images, HTML, requests, candidate mappings, separated model inputs and reference files. Scripts score predictions without model calls.

## Important release boundary

The paper inventories 3,451 SVG/DOM binding records from inherited resources. Those raw assets are **not included here** because their upstream redistribution terms have not been established in the source project. The release therefore does not claim to provide those 3,451 records. EVG, SEB, SVGEditBench, MiniWoB/DOM transfer source assets, model weights, cached model predictions, paper drafts, and author-specific experiment files are excluded. See `THIRD_PARTY_NOTICES.md`.

## Quick start

Python 3.9 or later is required. For the fixed-table generator, install Playwright and its Chromium browser. To score already generated fixed-table predictions, Python standard library is sufficient.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Score fixed-table predictions, one JSON object per request with `case_id` and either an integer `candidate` or a `raw` response:

```bash
python evaluation/score_fixed_tables.py \
  --split heldout \
  --predictions predictions.jsonl \
  --out scored/heldout
```

Reproduce the core structural collections:

```bash
python scripts/generate_broad.py --out outputs/broad --seed 20260823
python scripts/generate_targeted.py --out outputs/targeted --seed 20260824
```

Regenerate the fixed tables with Playwright and Chromium:

```bash
python scripts/build_fixed_tables.py
```

See `REPRODUCE.md` for schemas, expected outputs, and split handling. Gold/reference files are intended for benchmark evaluation and should not be supplied to models during inference.

## Licensing

Code under `src/` and the listed generation/evaluation scripts is licensed under MIT. Newly generated data under `data/exbind_v2/` and `data/fixed_tables/` is licensed under CC BY 4.0. These licenses do not cover excluded third-party materials or model checkpoints. See the scoped license files and `THIRD_PARTY_NOTICES.md`.

## Citation

Please cite the ExBind paper associated with this release. The anonymous repository intentionally omits author-identifying citation metadata during review.
