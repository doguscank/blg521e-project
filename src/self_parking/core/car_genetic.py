"""Car-specific genome decoding and control formulas."""

from __future__ import annotations

from dataclasses import dataclass

from self_parking.core.floats import bits_to_float10, precision_configs
from self_parking.core.genetic import Genome
from self_parking.core.math_utils import (
    euclidean_distance_2d,
    linear_polynomial,
    sigmoid,
    sigmoid_to_categories,
)
from self_parking.core.types import Command, WheelPoints

CAR_SENSORS_NUM = 8
BIAS_UNITS = 1
GENES_PER_NUMBER = precision_configs.custom.total_bits_count

ENGINE_FORMULA_GENES_NUM = (CAR_SENSORS_NUM + BIAS_UNITS) * GENES_PER_NUMBER
WHEELS_FORMULA_GENES_NUM = (CAR_SENSORS_NUM + BIAS_UNITS) * GENES_PER_NUMBER
GENOME_LENGTH = ENGINE_FORMULA_GENES_NUM + WHEELS_FORMULA_GENES_NUM


@dataclass(frozen=True)
class DecodedGenome:
    engine_formula_coefficients: list[float]
    wheels_formula_coefficients: list[float]


def car_loss(wheels_position: WheelPoints, parking_lot_corners: WheelPoints) -> float:
    fl_distance = euclidean_distance_2d(wheels_position.fl, parking_lot_corners.fl)
    fr_distance = euclidean_distance_2d(wheels_position.fr, parking_lot_corners.fr)
    br_distance = euclidean_distance_2d(wheels_position.br, parking_lot_corners.br)
    bl_distance = euclidean_distance_2d(wheels_position.bl, parking_lot_corners.bl)
    return (fl_distance + fr_distance + br_distance + bl_distance) / 4


def car_loss_to_fitness(loss: float, alpha: float = 1.0) -> float:
    return 1 / (alpha * loss + 1)


def genome_to_numbers(genome: Genome, genes_per_number: int) -> list[float]:
    if len(genome) % genes_per_number != 0:
        raise ValueError("Wrong number of genes in the numbers genome")

    numbers: list[float] = []
    for i in range(0, len(genome), genes_per_number):
        numbers.append(bits_to_float10(genome[i : i + genes_per_number]))
    return numbers


def decode_genome(genome: Genome) -> DecodedGenome:
    engine_genes = genome[:ENGINE_FORMULA_GENES_NUM]
    wheels_genes = genome[ENGINE_FORMULA_GENES_NUM : ENGINE_FORMULA_GENES_NUM + WHEELS_FORMULA_GENES_NUM]

    return DecodedGenome(
        engine_formula_coefficients=genome_to_numbers(engine_genes, GENES_PER_NUMBER),
        wheels_formula_coefficients=genome_to_numbers(wheels_genes, GENES_PER_NUMBER),
    )


def engine_formula(genome: Genome, sensors: list[float]) -> Command:
    coefficients = decode_genome(genome).engine_formula_coefficients
    raw_result = linear_polynomial(coefficients, sensors)
    normalized_result = sigmoid(raw_result)
    return sigmoid_to_categories(normalized_result)


def wheels_formula(genome: Genome, sensors: list[float]) -> Command:
    coefficients = decode_genome(genome).wheels_formula_coefficients
    raw_result = linear_polynomial(coefficients, sensors)
    normalized_result = sigmoid(raw_result)
    return sigmoid_to_categories(normalized_result)
