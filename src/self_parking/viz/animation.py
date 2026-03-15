"""Episode animation helpers."""

from __future__ import annotations

import random
from pathlib import Path

from self_parking.core.car_genetic import RAY_SENSORS_NUM
from self_parking.evolution.config import ScenarioConfig
from self_parking.simulation.engine import EpisodeResult
from self_parking.simulation.geometry import car_body_polygon
from self_parking.simulation.scenario import build_scenario


def _require_matplotlib():
    try:
        import matplotlib.animation as animation  # noqa: PLC0415
        import matplotlib.pyplot as plt  # noqa: PLC0415
        from matplotlib.patches import Polygon as PatchPolygon  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for visualization. Install with: pip install '.[viz]'"
        ) from exc
    return plt, animation, PatchPolygon


def animate_episode(
    episode_result: EpisodeResult,
    scenario_config: ScenarioConfig,
    save_path: str | Path | None = None,
    show_sensors: bool = True,
    interval_ms: int = 40,
):
    """Animate a simulated episode with polygons and optional sensor rays."""
    plt, animation, PatchPolygon = _require_matplotlib()

    scenario = build_scenario(scenario_config, random.Random(0))

    fig, ax = plt.subplots(figsize=(7, 7))

    all_x = []
    all_z = []

    for obstacle in scenario.static_obstacles:
        obs_patch = PatchPolygon(obstacle, closed=True, fill=False, edgecolor="dimgray", linewidth=1.0)
        ax.add_patch(obs_patch)
        all_x.extend(point[0] for point in obstacle)
        all_z.extend(point[1] for point in obstacle)

    parking = scenario.parking_spot
    parking_poly = [parking.fl, parking.fr, parking.br, parking.bl]
    parking_patch = PatchPolygon(
        parking_poly,
        closed=True,
        fill=False,
        edgecolor="tab:green",
        linewidth=2,
        linestyle="--",
    )
    ax.add_patch(parking_patch)

    car_poly0 = car_body_polygon(
        episode_result.steps[0].state.x,
        episode_result.steps[0].state.z,
        episode_result.steps[0].state.yaw,
    )
    car_patch = PatchPolygon(car_poly0, closed=True, fill=False, edgecolor="tab:blue", linewidth=2)
    ax.add_patch(car_patch)

    sensor_lines = []
    ray_count = min(RAY_SENSORS_NUM, len(episode_result.steps[0].sensors))
    if show_sensors:
        for _ in range(ray_count):
            line, = ax.plot([], [], color="tab:red", linewidth=0.8, alpha=0.6)
            sensor_lines.append(line)

    all_x.extend(point[0] for point in parking_poly)
    all_z.extend(point[1] for point in parking_poly)
    for step in episode_result.steps:
        all_x.append(step.state.x)
        all_z.append(step.state.z)

    margin = 2.5
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_z) - margin, max(all_z) + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Automatic Parking Playback")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.grid(True, alpha=0.2)

    trajectory_line, = ax.plot([], [], color="tab:blue", linewidth=1.0, alpha=0.5)

    def _frame(frame_index: int):
        step = episode_result.steps[frame_index]
        state = step.state
        poly = car_body_polygon(state.x, state.z, state.yaw)
        car_patch.set_xy(poly)

        trajectory_x = [s.state.x for s in episode_result.steps[: frame_index + 1]]
        trajectory_z = [s.state.z for s in episode_result.steps[: frame_index + 1]]
        trajectory_line.set_data(trajectory_x, trajectory_z)

        if show_sensors:
            import math  # noqa: PLC0415

            angle_step = 2 * math.pi / ray_count
            for sensor_idx, sensor_distance in enumerate(step.sensors[:ray_count]):
                angle = state.yaw + angle_step * sensor_idx
                max_d = sensor_distance if sensor_distance is not None else 4.0
                end_x = state.x + max_d * math.sin(angle)
                end_z = state.z + max_d * math.cos(angle)
                sensor_lines[sensor_idx].set_data([state.x, end_x], [state.z, end_z])

        artists = [car_patch, trajectory_line]
        artists.extend(sensor_lines)
        return artists

    ani = animation.FuncAnimation(
        fig,
        _frame,
        frames=len(episode_result.steps),
        interval=interval_ms,
        blit=False,
        repeat=False,
    )

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".gif":
            writer = animation.PillowWriter(fps=max(1, int(1000 / interval_ms)))
            ani.save(path, writer=writer)
        else:
            ani.save(path)

    return fig, ani
