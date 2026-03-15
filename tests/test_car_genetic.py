from __future__ import annotations

from self_parking.core.car_genetic import (
    CAR_SENSORS_NUM,
    GENES_PER_NUMBER,
    GENOME_LENGTH,
    decode_genome,
    engine_formula,
)


def test_rear_sensor_forward_behavior_parity() -> None:
    rear_sensor_index = 4
    sensors = [0.0] * CAR_SENSORS_NUM
    sensors[rear_sensor_index] = 2.4

    # Build a deterministic genome where only rear ray coefficient is large positive.
    # 416 in custom 10-bit format is 0111110100.
    zero_number = [0] * GENES_PER_NUMBER
    four_hundred_sixteen = [0, 1, 1, 1, 1, 1, 0, 1, 0, 0]

    coefficients_num = CAR_SENSORS_NUM + 1
    engine_numbers = [list(zero_number) for _ in range(coefficients_num)]
    engine_numbers[rear_sensor_index] = four_hundred_sixteen
    wheels_numbers = [list(zero_number) for _ in range(coefficients_num)]

    genome = [bit for number in engine_numbers + wheels_numbers for bit in number]
    assert len(genome) == GENOME_LENGTH

    decoded = decode_genome(genome)
    assert decoded.engine_formula_coefficients[rear_sensor_index] == 416
    assert engine_formula(genome, sensors) == 1
