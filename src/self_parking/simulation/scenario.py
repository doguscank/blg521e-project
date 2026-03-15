"""Scenario builders for default parking world."""

from __future__ import annotations

import random
from dataclasses import dataclass

from self_parking.core.types import Point2D, WheelPoints
from self_parking.evolution.config import ScenarioConfig
from self_parking.simulation.constants import (
    CHASSIS_LENGTH,
    CHASSIS_WIDTH,
    PARKING_INNER_LENGTH,
    PARKING_INNER_WIDTH,
    PARKING_SPOT_POSITION,
)
from self_parking.simulation.geometry import Polygon, rectangle_polygon


@dataclass(frozen=True)
class WorldScenario:
    start: Point2D
    start_yaw: float
    parking_spot: WheelPoints
    static_obstacles: list[Polygon]


def parking_spot_points() -> WheelPoints:
    x, z = PARKING_SPOT_POSITION

    outer_w = CHASSIS_WIDTH + 0.3
    outer_l = CHASSIS_LENGTH + 0.3

    inner_x = x + (outer_w - PARKING_INNER_WIDTH) / 2
    inner_z = z + (outer_l - PARKING_INNER_LENGTH) / 2

    return WheelPoints(
        fl=(inner_x + PARKING_INNER_WIDTH, inner_z + PARKING_INNER_LENGTH),
        fr=(inner_x, inner_z + PARKING_INNER_LENGTH),
        br=(inner_x, inner_z),
        bl=(inner_x + PARKING_INNER_WIDTH, inner_z),
    )


def build_static_obstacles(
    rows: int,
    cols: int,
    skip_cells: tuple[tuple[int, int], ...],
) -> list[Polygon]:
    obstacles: list[Polygon] = []

    margined_length = 1.4 * CHASSIS_LENGTH
    margined_width = 3.5 * CHASSIS_WIDTH

    for row in range(rows):
        for col in range(cols):
            if (row, col) in skip_cells:
                continue
            x = -0.5 * margined_width + row * margined_width
            z = -2 * margined_length + col * margined_length
            obstacles.append(
                rectangle_polygon(center_x=x, center_z=z, width=CHASSIS_WIDTH, length=CHASSIS_LENGTH)
            )

    return obstacles


def start_point(position: str, with_random_start: bool, rng: random.Random) -> Point2D:
    if position == "rear":
        z = -7 - 2 * rng.random() if with_random_start else -7
    elif position == "middle":
        z = 0
    else:
        z = 7 + 2 * rng.random() if with_random_start else 7
    return (0.0, float(z))


def build_scenario(config: ScenarioConfig, rng: random.Random) -> WorldScenario:
    return WorldScenario(
        start=start_point(config.start_position, config.with_random_start, rng),
        start_yaw=0.0,
        parking_spot=parking_spot_points(),
        static_obstacles=build_static_obstacles(
            rows=config.static_rows,
            cols=config.static_cols,
            skip_cells=config.static_skip_cells,
        ),
    )
