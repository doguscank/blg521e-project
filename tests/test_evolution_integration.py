from __future__ import annotations

from self_parking.evolution.config import (
    EvolutionConfig,
    RetryPolicyConfig,
    ScenarioConfig,
    SimulationConfig,
)
from self_parking.evolution.trainer import EvolutionTrainer


def _trainer(mode: str) -> EvolutionTrainer:
    return EvolutionTrainer(
        evolution_config=EvolutionConfig(
            generation_size=12,
            generations=4,
            batch_size=3,
            mutation_probability=0.05,
            long_living_champions_percentage=6,
            execution_mode=mode,
            workers=2 if mode == "parallel" else None,
            seed=123,
        ),
        simulation_config=SimulationConfig(episode_seconds=1.2, dt=0.1),
        scenario_config=ScenarioConfig(start_position="front"),
        retry_policy=RetryPolicyConfig(enabled=True, retries=1),
    )


def test_small_run_generation_invariants() -> None:
    result = _trainer("single").run()
    assert len(result.generation) == 12
    assert len(result.loss_history) == 4
    assert len(result.avg_loss_history) == 4
    assert all(loss >= 0 for loss in result.loss_history)


def test_deterministic_reproducibility_single_mode() -> None:
    first = _trainer("single").run()
    second = _trainer("single").run()

    assert first.loss_history == second.loss_history
    assert first.avg_loss_history == second.avg_loss_history
    assert first.best_loss == second.best_loss
    assert first.best_genome == second.best_genome


def test_parallel_consistency_against_single_mode() -> None:
    single = _trainer("single").run()
    parallel = _trainer("parallel").run()

    assert single.loss_history == parallel.loss_history
    assert single.avg_loss_history == parallel.avg_loss_history
    assert single.best_loss == parallel.best_loss
