"""Shared core type aliases and small value objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Gene = Literal[0, 1]
Genome = list[Gene]
Generation = list[Genome]

Probability = float
Percentage = float
Command = Literal[-1, 0, 1]

Point2D = tuple[float, float]


@dataclass(frozen=True)
class WheelPoints:
    """Wheel corner points (front-left/front-right/back-right/back-left)."""

    fl: Point2D
    fr: Point2D
    br: Point2D
    bl: Point2D


@dataclass(frozen=True)
class RunMetadata:
    """Run metadata attached to checkpoints."""

    run_id: str
    seed: int
    created_at: str
