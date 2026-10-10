"""Original frozen prompt templates. No API clients."""
import json
import random

EXTRACT_PROMPT = (
    "Transcribe only the visible table in this screenshot into JSON. Do not answer a question. "
    "There are eight data rows and eight yellow numbered candidate marks. "
    "Return exactly one JSON object with keys rows and candidates. "
    "rows must have eight entries in top-to-bottom order, each with row_index (0-7), "
    "project, ref, stage, owner, amount, region as visible strings. "
    "candidates must have eight entries ordered by candidate number 1-8, each with "
    "candidate (integer), row_index (integer), column (owner or amount), "
    "visible_value (visible cell string). Use null for unreadable values. "
    "Do not invent IDs, hidden metadata, a target answer, or an external status list."
)

def prompt(case: dict) -> str:
    marks = case["candidate_marks"]
    # Paired conditions with the same marked pool must see the same list order.
    choices = marks[:]
    pool_key = ",".join(str(x) for x in sorted(marks))
    random.Random(case["parent_latent_id"] + ":" + case["skin"] + ":" + pool_key + ":prompt.v4").shuffle(choices)
    context = case.get("structured_observation")
    context_block = ("\nStructured observation from a separate metadata source:\n" +
                     json.dumps(context, ensure_ascii=False, sort_keys=True) + "\n") if context else "\n"
    return ("A screenshot contains numbered yellow marks on interface elements. "
            "Only these mark numbers are eligible executable candidates: "
            + ", ".join(str(x) for x in choices) + ".\n"
            + case["instruction"] + context_block +
            "Choose exactly one eligible mark. Return one JSON object with a single "
            'integer field named "candidate" and no explanation.')

def answer_prompt(case: dict, representation: str) -> str:
    context = case.get("structured_observation")
    context_block = ("\nStructured observation from a separate metadata source:\n" +
                     json.dumps(context, ensure_ascii=False, sort_keys=True)) if context else ""
    return ("Only the following numbered marks are eligible executable candidates: "
            + ", ".join(str(x) for x in case["candidate_marks"]) + ".\n"
            + "Representation of the interface:\n" + representation + "\n"
            + case["instruction"] + context_block + "\n"
            + 'Choose exactly one eligible mark. Return one JSON object with a single integer field named "candidate" and no explanation.')
