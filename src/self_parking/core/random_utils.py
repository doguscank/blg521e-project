"""Random and reproducibility utilities."""

from __future__ import annotations

import datetime as dt
import random
import uuid

from self_parking.core.types import RunMetadata


def seed_rng(seed: int) -> random.Random:
    """Create a dedicated RNG for deterministic simulation/training."""
    rng = random.Random(seed)
    return rng


def build_run_metadata(seed: int) -> RunMetadata:
    """Create run metadata for persistence and reproducibility."""
    return RunMetadata(
        run_id=str(uuid.uuid4()),
        seed=seed,
        created_at=dt.datetime.now(tz=dt.UTC).isoformat(),
    )
