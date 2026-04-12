"""Car state and controller abstractions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from self_parking.core.car_genetic import engine_formula, wheels_formula
from self_parking.core.types import Command, Genome
from self_parking.evolution.config import SimulationConfig
from self_parking.simulation.geometry import car_body_polygon, intersects_any

GLOBAL_ACCELERATION = 1.0
GLOBAL_ACCELERATION_ANGLE = 0.0  # measured in radians, against +z axis, clockwise


class CarController(Protocol):
    """Controller contract: maps sensor values to (engine, wheel) commands."""

    def __call__(self, sensor_values: list[float | None]) -> tuple[Command, Command]:
        ...


@dataclass(frozen=True)
class GenomeCarController:
    """Genome-driven controller using ported formulas."""

    genome: Genome
    sensor_distance_fallback: float = 10.0  # This is important

    def __call__(self, sensor_values: list[float | None]) -> tuple[Command, Command]:
        cleaned = [
            self.sensor_distance_fallback if value is None else float(value)
            for value in sensor_values
        ]
        return (engine_formula(self.genome, cleaned), wheels_formula(self.genome, cleaned))


@dataclass(frozen=True)
class CarState:
    x: float
    z: float
    yaw: float
    speed: float
    steering: float
    collided: bool = False


@dataclass(frozen=True)
class CarStepResult:
    state: CarState
    collision_happened: bool


def step_car(
    state: CarState,
    engine_cmd: Command,
    wheel_cmd: Command,
    config: SimulationConfig,
    obstacles: list[list[tuple[float, float]]],
) -> CarStepResult:
    """Advance one fixed-dt step with kinematic bicycle model and polygon collision."""
    steering = wheel_cmd * config.max_steer
    acceleration = engine_cmd * config.max_acceleration - config.drag * state.speed

    acceleration = GLOBAL_ACCELERATION * math.cos(GLOBAL_ACCELERATION_ANGLE - state.yaw) + acceleration

    speed = max(-config.max_speed, min(config.max_speed, state.speed + acceleration * config.dt))

    yaw_rate = 0.0
    if abs(config.wheelbase) > 1e-8:
        yaw_rate = speed / config.wheelbase * math.tan(steering)

    candidate_yaw = state.yaw + yaw_rate * config.dt
    # when yaw is zero, we move in the +z direction
    candidate_x = state.x + speed * math.sin(candidate_yaw) * config.dt
    candidate_z = state.z + speed * math.cos(candidate_yaw) * config.dt

    candidate_poly = car_body_polygon(candidate_x, candidate_z, candidate_yaw)
    collision = intersects_any(candidate_poly, obstacles)

    if collision:
        return CarStepResult(
            state=CarState(
                x=state.x,
                z=state.z,
                yaw=state.yaw,
                speed=0.0,
                steering=steering,
                collided=True,
            ),
            collision_happened=True,
        )

    return CarStepResult(
        state=CarState(
            x=candidate_x,
            z=candidate_z,
            yaw=candidate_yaw,
            speed=speed,
            steering=steering,
            collided=False,
        ),
        collision_happened=False,
    )
