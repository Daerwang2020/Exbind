"""Frozen ExBind reference checks. No model calls."""

from __future__ import annotations

import re
import copy
import json

FIELDS = ("project", "ref", "stage", "owner", "amount", "region")

def oracle_state(source: dict) -> dict:
    # Explicit allowlist: target_index, target_mark and alternate_* never enter the state.
    rows = [{"row_index": index, **{field: row[field] for field in
             ("project", "ref", "stage", "owner", "amount", "region")}}
            for index, row in enumerate(source["rows"])]
    by_ref = {row["ref"]: index for index, row in enumerate(source["rows"])}
    candidates = []
    for cell, mark in source["mark_by_cell"].items():
        ref, column = cell.split("::")
        candidates.append({"candidate": mark, "row_index": by_ref[ref],
                           "column": column,
                           "visible_value": source["rows"][by_ref[ref]][column]})
    return {"rows": rows, "candidates": sorted(candidates, key=lambda x: x["candidate"])}

def valid_state(state: dict | None) -> bool:
    if not isinstance(state, dict) or set(state) != {"rows", "candidates"}:
        return False
    rows, candidates = state["rows"], state["candidates"]
    if not isinstance(rows, list) or not isinstance(candidates, list) or len(rows) != 8 or len(candidates) != 8:
        return False
    if not all(isinstance(x, dict) and type(x.get("row_index")) is int for x in rows):
        return False
    if {x["row_index"] for x in rows} != set(range(8)):
        return False
    if not all(isinstance(x, dict) and type(x.get("candidate")) is int for x in candidates):
        return False
    if {x["candidate"] for x in candidates} != set(range(1, 9)):
        return False
    return all(isinstance(x, dict) and all(f in x for f in FIELDS) for x in rows) and all(
        isinstance(x, dict) and x.get("row_index") in range(8) and
        x.get("column") in ("owner", "amount") for x in candidates)

def resolver(state: dict | None, case: dict) -> int | None:
    if not valid_state(state):
        return None
    instruction = case["instruction"]
    column = re.search(r'"(Owner|Amount)"', instruction)
    if not column:
        return None
    column = column.group(1).lower()
    requirement = case["condition"]
    if requirement == "direct":
        match = re.search(r'in the "([^"]+)" row', instruction)
        if not match:
            return None
        field, value = "project", match.group(1)
        matching = lambda row: row.get(field) == value
        uncertain = lambda row: row.get(field) is None
    elif requirement == "compose":
        match = re.search(r'Region is "([^"]+)" and Stage is "([^"]+)"', instruction)
        if not match:
            return None
        region, stage = match.groups()
        matching = lambda row: row.get("region") == region and row.get("stage") == stage
        uncertain = lambda row: (row.get("region") in (region, None) and row.get("stage") in (stage, None)
                                 and (row.get("region") is None or row.get("stage") is None))
    else:
        entries = case.get("structured_observation", {}).get("status_list", [])
        active = [x["ref"] for x in entries if x.get("active") is True]
        if len(active) != 1:
            return None
        field, value = "ref", active[0]
        matching = lambda row: row.get(field) == value
        uncertain = lambda row: row.get(field) is None
    row_by_index = {x["row_index"]: x for x in state["rows"]}
    eligible = []
    for candidate in state["candidates"]:
        if candidate["column"] != column:
            continue
        row = row_by_index[candidate["row_index"]]
        if matching(row):
            eligible.append(candidate["candidate"])
        elif uncertain(row):
            return None
    return eligible[0] if len(eligible) == 1 else None


def parse_mark(reply: str, allow_single_mark: bool = False) -> tuple[int | None, str]:
    value = reply.strip()
    if not value:
        return None, "empty"
    if allow_single_mark and re.fullmatch(r"[0-9]+", value):
        return int(value), "single_number"
    try:
        obj = json.loads(value)
    except json.JSONDecodeError:
        match = re.search(r"\{[^{}]*\}", value, flags=re.S)
        if not match:
            return None, "invalid_json"
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None, "invalid_json"
    if not isinstance(obj, dict) or set(obj) != {"candidate"}:
        return None, "wrong_schema"
    mark = obj["candidate"]
    if isinstance(mark, bool) or not isinstance(mark, int):
        return None, "wrong_type"
    return mark, "ok"

def parse_state(reply: str) -> dict | None:
    answer = re.search(r"<answer>(.*?)</answer>", reply, re.S)
    if answer:
        reply = answer.group(1)
    else:
        reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.S)
    try:
        state = json.loads(reply.strip())
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", reply, re.S)
        if not match:
            return None
        try:
            state = json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return state if isinstance(state, dict) else None

def parse_api_mark(reply: str) -> tuple[int | None, str]:
    answer = re.search(r"<answer>(.*?)</answer>", reply, re.S)
    if answer:
        reply = answer.group(1)
    else:
        reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.S)
    reply = reply.strip().replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "")
    return parse_mark(reply, allow_single_mark=True)

def errors(state,truth):
 result=[]
 for collection,idkey,fields in [('rows','row_index',FIELDS),('candidates','candidate',('row_index','column','visible_value'))]:
  true={x[idkey]:x for x in truth[collection]}
  for x in state[collection]:
   for field in fields:
    if x.get(field)!=true[x[idkey]].get(field):result.append(f'{collection}:{x[idkey]}:{field}')
 return result

def constraints(source,case,mark):
 truth=oracle_state(source);candidate=next((x for x in truth['candidates'] if x['candidate']==mark),None)
 if candidate is None:return dict(candidate_valid=False,row_satisfied=False,column_satisfied=False)
 row=truth['rows'][candidate['row_index']];q=case['instruction'];column=re.search(r'"(Owner|Amount)"',q).group(1).lower()
 if case['condition']=='direct':row_ok=row['project']==re.search(r'in the "([^"]+)" row',q).group(1)
 elif case['condition']=='compose':
  region,stage=re.search(r'Region is "([^"]+)" and Stage is "([^"]+)"',q).groups();row_ok=row['region']==region and row['stage']==stage
 else:
  active=[x['ref'] for x in case['structured_observation']['status_list'] if x['active']];assert len(active)==1;row_ok=row['ref']==active[0]
 return dict(candidate_valid=True,row_satisfied=row_ok,column_satisfied=candidate['column']==column)

def apply_patch(state,raw):
 obj=parse_state(raw)
 if not isinstance(obj,dict) or set(obj)!={'state_patch','candidate'}:raise ValueError('Invalid repair response schema')
 if type(obj['candidate']) is not int or obj['candidate'] not in range(1,9):raise ValueError('Invalid candidate')
 if not isinstance(obj['state_patch'],list) or len(obj['state_patch'])>16:raise ValueError('Invalid patch length')
 updated=copy.deepcopy(state);seen=set()
 for op in obj['state_patch']:
  if not isinstance(op,dict) or set(op)!={'collection','index','field','value'}:raise ValueError('Invalid patch operation')
  coll,index,field,value=[op[k] for k in ('collection','index','field','value')]
  if coll not in ('rows','candidates') or type(index) is not int:raise ValueError('Invalid patch address')
  idkey='row_index' if coll=='rows' else 'candidate';allowed=FIELDS if coll=='rows' else ('row_index','column','visible_value')
  if field not in allowed or (coll,index,field) in seen:raise ValueError('Invalid or duplicate patch field')
  if field=='row_index':
   if type(value) is not int or value not in range(8):raise ValueError('Invalid row index value')
  elif field=='column':
   if value not in ('owner','amount'):raise ValueError('Invalid column value')
  elif value is not None and (not isinstance(value,str) or len(value)>256):raise ValueError('Invalid field value')
  target=next((x for x in updated[coll] if x[idkey]==index),None)
  if target is None:raise ValueError('Unknown patch address')
  target[field]=value;seen.add((coll,index,field))
 if not valid_state(updated):raise ValueError('Patch violates interface schema')
 return updated,obj['candidate'],obj['state_patch']

def guarded_state(before,proposal,truth,case):
 accepted=[];rejected=[]
 for op in proposal:
  coll=op['collection'];idkey='row_index' if coll=='rows' else 'candidate';target=next(x for x in truth[coll] if x[idkey]==op['index'])
  (accepted if target[op['field']]==op['value'] else rejected).append(op)
 state,_,_=apply_patch(before,json.dumps({'state_patch':accepted,'candidate':1}))
 rollback=resolver(before,case)==case['gold_mark'] and resolver(state,case)!=case['gold_mark']
 if rollback:rejected+=accepted;accepted=[];state=copy.deepcopy(before)
 assert set(errors(state,truth))<=set(errors(before,truth))
 assert resolver(before,case)!=case['gold_mark'] or resolver(state,case)==case['gold_mark']
 return state,accepted,rejected,rollback
