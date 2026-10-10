"""Deterministic candidate-constrained decoding for structured references."""

from __future__ import annotations

from typing import Any, Iterable


def candidate_json_strings(candidate_ids: Iterable[str], field: str = "cell_id") -> list[str]:
    """Return the finite JSON output language for one reference field."""
    return [f'{{"{field}": "{candidate_id}"}}' for candidate_id in candidate_ids]


class CandidateJsonPrefixConstraint:
    """Allow only token prefixes of one of a finite set of JSON answers.

    The constraint changes the output language only. It does not score, reorder,
    or select candidates, so semantic binding remains the model's task.
    """

    def __init__(
        self,
        tokenizer: Any,
        output_strings: Iterable[str],
        prompt_length: int,
        eos_token_id: int,
    ) -> None:
        self.prompt_length = int(prompt_length)
        self.eos_token_id = int(eos_token_id)
        self.sequences = [
            tuple(tokenizer.encode(text, add_special_tokens=False))
            for text in output_strings
        ]
        self.sequences = [sequence for sequence in self.sequences if sequence]
        if not self.sequences:
            raise ValueError("candidate output language is empty")

    def __call__(self, batch_id: int, input_ids: Any) -> list[int]:
        del batch_id
        generated = tuple(int(value) for value in input_ids[self.prompt_length :].tolist())
        matching = [sequence for sequence in self.sequences if sequence[: len(generated)] == generated]
        if not matching:
            return [self.eos_token_id]

        allowed: set[int] = set()
        for sequence in matching:
            if len(generated) == len(sequence):
                allowed.add(self.eos_token_id)
            else:
                allowed.add(sequence[len(generated)])
        return sorted(allowed)

