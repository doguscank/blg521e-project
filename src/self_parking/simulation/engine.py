"""Episode simulation engine."""

from __future__ import annotations

import random
from dataclasses import dataclass

from self_parking.core.car_genetic import car_loss
from self_parking.core.types import Genome, WheelPoints
from self_parking.evolution.config import ScenarioConfig, SimulationConfig
from self_parking.simulation.car import CarController, CarState, GenomeCarController, step_car
from self_parking.simulation.geometry import raycast_polygons, wheel_points
from self_parking.simulation.scenario import WorldScenario, build_scenario


@dataclass(frozen=True)
class EpisodeStep:
    index: int
    state: CarState
    sensors: list[float | None]
    engine_cmd: int
    wheel_cmd: int


@dataclass(frozen=True)
class EpisodeResult:
    loss: float
    steps: list[EpisodeStep]
    final_state: CarState
    final_wheel_points: WheelPoints
    collisions: int


def sensor_distances(
    state: CarState,
    scenario: WorldScenario,
    config: SimulationConfig,
) -> list[float | None]:
    distances: list[float | None] = []
    angle_step = 2 * 3.141592653589793 / config.sensor_count

    for sensor_idx in range(config.sensor_count):
        angle = state.yaw + angle_step * sensor_idx
        distance = raycast_polygons(
            origin=(state.x, state.z),
            angle=angle,
            obstacles=scenario.static_obstacles,
            max_distance=config.sensor_max_distance,
        )
        distances.append(distance)

    return distances


def simulate_episode_with_controller(
    controller: CarController,
    simulation_config: SimulationConfig,
    scenario_config: ScenarioConfig,
    seed: int,
) -> EpisodeResult:
    rng = random.Random(seed)
    scenario = build_scenario(scenario_config, rng)

    state = CarState(
        x=scenario.start[0],
        z=scenario.start[1],
        yaw=scenario.start_yaw,
        speed=0.0,
        steering=0.0,
        collided=False,
    )

    steps_num = int(simulation_config.episode_seconds / simulation_config.dt)
    collisions = 0
    history: list[EpisodeStep] = []

    for step_idx in range(steps_num):
        sensors = sensor_distances(state, scenario, simulation_config)
        engine_cmd, wheel_cmd = controller(sensors)

        step_result = step_car(
            state=state,
            engine_cmd=engine_cmd,
            wheel_cmd=wheel_cmd,
            config=simulation_config,
            obstacles=scenario.static_obstacles,
        )

        state = step_result.state
        collisions += 1 if step_result.collision_happened else 0

        history.append(
            EpisodeStep(
                index=step_idx,
                state=state,
                sensors=sensors,
                engine_cmd=engine_cmd,
                wheel_cmd=wheel_cmd,
            )
        )

    final_wheels = wheel_points(state.x, state.z, state.yaw)
    loss = car_loss(final_wheels, scenario.parking_spot)

    return EpisodeResult(
        loss=loss,
        steps=history,
        final_state=state,
        final_wheel_points=final_wheels,
        collisions=collisions,
    )


def simulate_episode(
    genome: Genome,
    simulation_config: SimulationConfig,
    scenario_config: ScenarioConfig,
    seed: int,
) -> EpisodeResult:
    controller = GenomeCarController(
        genome=genome,
        sensor_distance_fallback=simulation_config.sensor_distance_fallback,
    )
    return simulate_episode_with_controller(
        controller=controller,
        simulation_config=simulation_config,
        scenario_config=scenario_config,
        seed=seed,
    )
