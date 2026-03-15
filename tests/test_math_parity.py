from __future__ import annotations

import random

from self_parking.core.floats import bits_to_float10, bits_to_float16
from self_parking.core.math_utils import (
    euclidean_distance_2d,
    linear_polynomial,
    sigmoid,
    sigmoid_to_categories,
)
from self_parking.core.probability import weighted_random


def _bits(binary: str) -> list[int]:
    return [1 if ch == "1" else 0 for ch in binary]


def test_bits_to_float10_parity_cases() -> None:
    cases = [
        (-504, "1111111111"),
        (-496, "1111111110"),
        (-0.0146484375, "1000011100"),
        (0.0078125, "0000000000"),
        (0.008056640625, "0000000001"),
        (0.0625, "0001100000"),
        (244, "0111011101"),
        (256, "0111100000"),
        (384, "0111110000"),
        (488, "0111111101"),
        (496, "0111111110"),
        (504, "0111111111"),
    ]
    for expected, binary in cases:
        assert bits_to_float10(_bits(binary)) == expected


def test_bits_to_float16_selected_cases() -> None:
    cases = [
        (-65504, "1111101111111111"),
        (-1, "1011110000000000"),
        (0, "0000000000000000"),
        (1.5, "0011111000000000"),
        (65504, "0111101111111111"),
    ]
    for expected, binary in cases:
        assert round(bits_to_float16(_bits(binary)), 6) == round(expected, 6)


def test_sigmoid_parity_values() -> None:
    def floor7(value: float) -> float:
        return int(value * 10_000_000) / 10_000_000

    assert floor7(sigmoid(-1)) == 0.2689414
    assert floor7(sigmoid(0)) == 0.5
    assert floor7(sigmoid(1)) == 0.7310585


def test_sigmoid_to_categories_parity() -> None:
    assert sigmoid_to_categories(sigmoid(-100)) == -1
    assert sigmoid_to_categories(sigmoid(0)) == 0
    assert sigmoid_to_categories(sigmoid(100)) == 1


def test_linear_polynomial_parity() -> None:
    assert linear_polynomial([1, 2, 3], [4, 5]) == 17
    assert linear_polynomial([1, 2, 3], [0, 0]) == 3


def test_geometry_distance_xz_parity() -> None:
    assert round(euclidean_distance_2d((0, 0), (1, 1)), 2) == 1.41
    assert round(euclidean_distance_2d((5, 8), (10, 12)), 1) == 6.4


def test_weighted_random_distribution_shape() -> None:
    rng = random.Random(123)
    counts = [0, 0, 0]
    for _ in range(1000):
        _, idx = weighted_random([0, 1, 2], [10, 30, 60], rng)
        counts[idx] += 1

    assert 70 <= counts[0] <= 130
    assert 260 <= counts[1] <= 340
    assert 540 <= counts[2] <= 660
