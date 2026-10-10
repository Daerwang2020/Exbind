# Reproduction and scoring

## Structure

- `data/exbind_v2/broad/`: 250 generated structural cases and reference mappings.
- `data/exbind_v2/targeted/`: 240 targeted cases and mappings.
- `data/exbind_v2/cross_format/`: 50 paired latent groups and consistency labels.
- `data/fixed_tables/`: 96 generated interfaces and 384 paired requests.
- `evaluation/`: fixed-table prediction parser, reference checks, scorer, and frozen prompts.
- `scripts/`: deterministic collection generators and structural scorer.

## Fixed-table splits

Use `inference_inputs/development_*` for prompt or configuration selection and `inference_inputs/heldout_*` for final evaluation. Gold state and mapping files are provided for evaluation. Keep them out of model inputs. `score_fixed_tables.py` requires all requested case IDs and rejects duplicate or unknown IDs. It reports per-request exactness/validity and paired Context/Flip behavior; it does not compute confidence intervals.

## Structural suites

Install package dependencies with `pip install -e .`. Generate broad and targeted cases with the fixed seeds shown in the root README. Score predictions with `scripts/score_predictions.py`; its `--help` lists exact arguments and accepted JSONL fields. Structural and table scores use separate collection-specific evaluators and denominators.

## Model inference

Model weights and API clients are not included. Obtain checkpoints separately and follow their terms. Do not provide reference annotations, gold mappings, or held-out gold files to the model.
