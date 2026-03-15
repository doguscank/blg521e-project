from __future__ import annotations

from self_parking.core.car_genetic import car_loss
from self_parking.evolution.config import ScenarioConfig, SimulationConfig
from self_parking.simulation.car import CarState, step_car
from self_parking.simulation.constants import CHASSIS_LENGTH, CHASSIS_WIDTH
from self_parking.simulation.engine import simulate_episode_with_controller
from self_parking.simulation.geometry import raycast_polygons, rectangle_polygon
from self_parking.simulation.scenario import parking_spot_points


class ForwardController:
    def __call__(self, sensor_values: list[float | None]) -> tuple[int, int]:
        return 1, 0


def test_sensor_raycast_distance_against_polygon() -> None:
    obstacle = rectangle_polygon(center_x=0.0, center_z=3.0, width=1.0, length=1.0)
    distance = raycast_polygons(origin=(0.0, 0.0), angle=0.0, obstacles=[obstacle], max_distance=10.0)
    assert distance is not None
    assert round(distance, 2) == 2.5


def test_collision_blocks_movement() -> None:
    cfg = SimulationConfig(dt=0.2, drag=0.0)
    obstacle = rectangle_polygon(center_x=0.0, center_z=5.0, width=1.5, length=4.0)
    state = CarState(x=0.0, z=0.0, yaw=0.0, speed=8.0, steering=0.0)

    step = step_car(state=state, engine_cmd=0, wheel_cmd=0, config=cfg, obstacles=[obstacle])

    assert step.collision_happened is True
    assert step.state.x == state.x
    assert step.state.z == state.z
    assert step.state.speed == 0.0


def test_loss_zero_for_identical_points() -> None:
    points = parking_spot_points()

    assert car_loss(points, points) == 0.0


def test_parking_slot_size_matches_car_size() -> None:
    points = parking_spot_points()
    parking_width = abs(points.fl[0] - points.fr[0])
    parking_length = abs(points.fl[1] - points.bl[1])

    assert parking_width == CHASSIS_WIDTH
    assert parking_length == CHASSIS_LENGTH


def test_episode_simulation_runs() -> None:
    episode = simulate_episode_with_controller(
        controller=ForwardController(),
        simulation_config=SimulationConfig(episode_seconds=1.0, dt=0.1),
        scenario_config=ScenarioConfig(start_position="front"),
        seed=7,
    )
    assert episode.loss >= 0
    assert len(episode.steps) == 10
