"""Math helper functions."""

from __future__ import annotations

import math

from self_parking.core.types import Command


def sigmoid(x: float) -> float:
    # Keep the canonical form: 1 / (1 + exp(-x)).
    # Python may overflow for very negative x when evaluating exp(-x),
    # in that case sigmoid saturates to 0.
    try:
        z = math.exp(-x)
    except OverflowError:
        return 0.0
    return 1 / (1 + z)


def sigmoid_to_categories(sigmoid_value: float, around_zero_margin: float = 0.49999) -> Command:
    if sigmoid_value < (0.5 - around_zero_margin):
        return -1
    if sigmoid_value > (0.5 + around_zero_margin):
        return 1
    return 0


def linear_polynomial(coefficients: list[float], variables: list[float]) -> float:
    if len(coefficients) != len(variables) + 1:
        raise ValueError(
            "Incompatible number polynomial coefficients and variables: "
            f"{len(coefficients)} and {len(variables)}"
        )

    result = 0.0
    for coefficient_index, coefficient in enumerate(coefficients):
        if coefficient_index < len(variables):
            result += coefficient * variables[coefficient_index]
        else:
            result += coefficient
    return result


def euclidean_distance_2d(from_point: tuple[float, float], to_point: tuple[float, float]) -> float:
    from_x, from_z = from_point
    to_x, to_z = to_point
    return math.sqrt((from_x - to_x) ** 2 + (from_z - to_z) ** 2)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
