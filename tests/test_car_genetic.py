from __future__ import annotations

from self_parking.core.car_genetic import decode_genome, engine_formula
from self_parking.core.genetic import genome_string_to_genome


def test_rear_sensor_forward_behavior_parity() -> None:
    rear_sensor_index = 4
    sensors = [0.0] * 8
    sensors[rear_sensor_index] = 2.4

    genome = genome_string_to_genome(
        "1 0 1 1 1 1 0 1 0 0 1 1 0 1 1 1 1 0 0 0 1 1 1 1 1 1 0 0 0 0 1 0 1 1 1 0 0 1 1 1 0 1 1 1 1 1 0 1 0 0 1 1 0 1 0 0 1 1 0 0 0 1 1 1 0 1 0 1 0 0 1 0 1 1 1 1 0 0 0 1 1 1 0 0 0 0 0 0 1 1 0 0 1 1 1 0 1 1 1 1 1 0 0 0 1 0 0 1 1 0 1 1 0 1 0 1 0 1 0 0 1 0 0 1 0 1 1 0 1 1 1 1 0 0 0 0 1 1 0 1 0 1 0 0 0 0 0 0 0 0 1 0 0 1 1 1 1 0 1 1 0 1 0 1 0 0 1 0 0 0 0 0 0 1 0 0 0 0 1 1"
    )

    decoded = decode_genome(genome)
    assert decoded.engine_formula_coefficients[rear_sensor_index] == 416
    assert engine_formula(genome, sensors) == 1
