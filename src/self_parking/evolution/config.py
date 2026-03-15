"""Configuration dataclasses for evolution and simulation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

StartPosition = Literal["front", "middle", "rear", "random"]
ExecutionMode = Literal["single", "parallel"]


@dataclass(frozen=True)
class SimulationConfig:
    dt: float = 0.1
    episode_seconds: float = 17.0
    max_speed: float = 8.0
    max_acceleration: float = 2.0
    drag: float = 0.35
    max_steer: float = 0.6
    wheelbase: float = 2.45
    sensor_count: int = 8
    sensor_max_distance: float = 4.0
    sensor_distance_fallback: float = 0.0
    collision_penalty: float = 1000.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ScenarioConfig:
    start_position: StartPosition = "front"
    with_random_start: bool = False
    static_rows: int = 2
    static_cols: int = 5
    static_skip_cells: tuple[tuple[int, int], ...] = ((0, 2),)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["static_skip_cells"] = [list(item) for item in self.static_skip_cells]
        return data


@dataclass(frozen=True)
class RetryPolicyConfig:
    enabled: bool = True
    batch_index_check: int = 1
    retries: int = 1
    min_loss_increase_percentage: float = 110.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EvolutionConfig:
    generation_size: int = 100
    generations: int = 50
    batch_size: int = 2
    mutation_probability: float = 0.04
    long_living_champions_percentage: float = 6.0
    fitness_alpha: float = 0.5
    execution_mode: ExecutionMode = "single"
    workers: int | None = None
    seed: int = 42
    initial_generation: list[list[int]] | None = field(default=None)

    def to_dict(self) -> dict:
        return asdict(self)
