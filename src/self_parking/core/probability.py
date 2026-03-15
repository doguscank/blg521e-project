"""Probability helpers."""

from __future__ import annotations

import random


def weighted_random[T](
    items: list[T],
    weights: list[float],
    rng: random.Random,
) -> tuple[T, int]:
    """Pick a random item based on non-negative weights."""
    if len(items) != len(weights):
        raise ValueError("Items and weights must be of the same size")
    if not items:
        raise ValueError("Items must not be empty")

    cumulative_weights: list[float] = []
    running = 0.0
    for weight in weights:
        running += weight
        cumulative_weights.append(running)

    max_cumulative_weight = cumulative_weights[-1]
    if max_cumulative_weight <= 0:
        # Fallback to uniform if all weights are zero/non-positive.
        idx = rng.randrange(len(items))
        return items[idx], idx

    random_number = max_cumulative_weight * rng.random()
    for idx, threshold in enumerate(cumulative_weights):
        if threshold >= random_number:
            return items[idx], idx

    last_idx = len(items) - 1
    return items[last_idx], last_idx
