from __future__ import annotations

from pathlib import Path

from self_parking.evolution.config import (
    EvolutionConfig,
    RetryPolicyConfig,
    ScenarioConfig,
    SimulationConfig,
)
from self_parking.evolution.trainer import EvolutionTrainer
from self_parking.io.checkpoint import CHECKPOINT_VERSION, CheckpointStore


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    evolution_config = EvolutionConfig(generation_size=8, generations=2, seed=11)
    simulation_config = SimulationConfig(episode_seconds=0.8, dt=0.1)
    scenario_config = ScenarioConfig(start_position="front")
    retry_policy = RetryPolicyConfig(enabled=False)

    result = EvolutionTrainer(
        evolution_config=evolution_config,
        simulation_config=simulation_config,
        scenario_config=scenario_config,
        retry_policy=retry_policy,
    ).run()

    checkpoint = CheckpointStore.from_training_result(
        training_result=result,
        evolution_config=evolution_config,
        simulation_config=simulation_config,
        scenario_config=scenario_config,
        retry_policy=retry_policy,
    )

    path = tmp_path / "ckpt.json"
    CheckpointStore.save(path, checkpoint)
    loaded = CheckpointStore.load(path)

    assert loaded.version == CHECKPOINT_VERSION
    assert loaded.generation_index == result.generation_index
    assert loaded.best_genome == result.best_genome


def test_checkpoint_inspect(tmp_path: Path) -> None:
    evolution_config = EvolutionConfig(generation_size=6, generations=1, seed=5)
    simulation_config = SimulationConfig(episode_seconds=0.5)
    scenario_config = ScenarioConfig()
    retry_policy = RetryPolicyConfig(enabled=False)

    result = EvolutionTrainer(
        evolution_config=evolution_config,
        simulation_config=simulation_config,
        scenario_config=scenario_config,
        retry_policy=retry_policy,
    ).run()

    checkpoint = CheckpointStore.from_training_result(
        training_result=result,
        evolution_config=evolution_config,
        simulation_config=simulation_config,
        scenario_config=scenario_config,
        retry_policy=retry_policy,
    )

    path = tmp_path / "inspect.json"
    CheckpointStore.save(path, checkpoint)

    info = CheckpointStore.inspect(path)
    assert info["generation_size"] == 6
    assert info["genome_length"] > 0
