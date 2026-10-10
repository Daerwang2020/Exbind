"""Score frozen ExBind binding and state outputs. No model calls."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from reference_checks import oracle_state, valid_state, resolver, errors, parse_api_mark


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def index(rows, key):
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f'Duplicate {key}')
    return result


def score(data, prediction_rows, state_rows=None, split='heldout'):
    sources = index([r for r in read(data / 'private_sources.jsonl') if r['split'] == split], 'source_id')
    cases = [r for r in read(data / 'cases.jsonl') if r['split'] == split]
    predictions = index(prediction_rows, 'case_id')
    if set(predictions) != {c['case_id'] for c in cases}:
        raise ValueError('Predictions must contain exactly the case IDs in the requested split')
    states = index(state_rows, 'source_id') if state_rows is not None else None
    if states is not None and set(states) != set(sources):
        raise ValueError('States must contain exactly one entry for every source in the requested split')
    rows = []
    for c in cases:
        pred = predictions[c['case_id']]
        mark = pred.get('candidate')
        if 'raw' in pred:
            mark, _ = parse_api_mark(pred['raw'])
        valid = type(mark) is int and mark in c['candidate_marks']
        row = dict(case_id=c['case_id'], source_id=c['parent_latent_id'], requirement=c['condition'],
                   candidate=mark, valid=valid, exact=bool(valid and mark == c['gold_mark']))
        if states is not None:
            row['sufficient'] = resolver(states[row['source_id']].get('state'), c) == c['gold_mark']
        rows.append(row)
    requirements = ('direct', 'compose', 'context', 'context_flip')
    by_source = {}
    for row in rows:
        by_source.setdefault(row['source_id'], {})[row['requirement']] = row
    if any(set(group) != set(requirements) for group in by_source.values()):
        raise ValueError('Every source must retain all four paired requirements')
    def metrics(items):
        return dict(n=len(items), correct=sum(r['exact'] for r in items),
                    valid=sum(r['valid'] for r in items),
                    exact_accuracy=sum(r['exact'] for r in items) / len(items),
                    validity=sum(r['valid'] for r in items) / len(items))
    result = dict(split=split, sources=len(sources), overall=metrics(rows),
                  by_requirement={q: metrics([r for r in rows if r['requirement'] == q]) for q in requirements})
    result['paired'] = dict(
        context_flip_both_correct=sum(g['context']['exact'] and g['context_flip']['exact'] for g in by_source.values()),
        context_flip_answer_changed=sum(g['context']['candidate'] != g['context_flip']['candidate'] for g in by_source.values()),
        requirement_minus_direct_pp={q: 100 * sum(int(g[q]['exact']) - int(g['direct']['exact']) for g in by_source.values()) / len(sources) for q in requirements[1:]})
    if states is not None:
        field_errors = {}
        for sid, state_row in states.items():
            state = state_row.get('state')
            field_errors[sid] = errors(state, oracle_state(sources[sid])) if valid_state(state) else None
        result['state'] = dict(valid_records=sum(v is not None for v in field_errors.values()),
                               exact_records=sum(v == [] for v in field_errors.values()),
                               sufficient_requests=sum(r['sufficient'] for r in rows), field_errors=field_errors,
                               joint_counts={f'n{s}{a}': sum(r['sufficient'] == bool(s) and r['exact'] == bool(a) for r in rows) for s in (0, 1) for a in (0, 1)})
    return result, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path(__file__).resolve().parents[1] / 'fixed_tables')
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--states', type=Path)
    p.add_argument('--split', choices=['development', 'heldout'], default='heldout')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    summary, rows = score(args.data, read(args.predictions), read(args.states) if args.states else None, args.split)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (args.out / 'scored_cases.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
