"""Checkpoint persistence for Python-native schema."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from self_parking.core.car_genetic import GENOME_LENGTH
from self_parking.core.types import Generation, Genome, RunMetadata
from self_parking.evolution.config import (
    EvolutionConfig,
    RetryPolicyConfig,
    ScenarioConfig,
    SimulationConfig,
)
from self_parking.evolution.trainer import TrainingResult

CHECKPOINT_VERSION = "checkpoint-v1"


@dataclass(frozen=True)
class CheckpointV1:
    version: str
    metadata: dict[str, Any]
    evolution_config: dict[str, Any]
    simulation_config: dict[str, Any]
    scenario_config: dict[str, Any]
    retry_policy: dict[str, Any]
    generation_index: int
    generation: Generation
    loss_history: list[float]
    avg_loss_history: list[float]
    best_genome: Genome
    best_loss: float


class CheckpointStore:
    """Read/write API for checkpoints."""

    @staticmethod
    def from_training_result(
        training_result: TrainingResult,
        evolution_config: EvolutionConfig,
        simulation_config: SimulationConfig,
        scenario_config: ScenarioConfig,
        retry_policy: RetryPolicyConfig,
    ) -> CheckpointV1:
        return CheckpointV1(
            version=CHECKPOINT_VERSION,
            metadata=asdict(training_result.metadata),
            evolution_config=evolution_config.to_dict(),
            simulation_config=simulation_config.to_dict(),
            scenario_config=scenario_config.to_dict(),
            retry_policy=retry_policy.to_dict(),
            generation_index=training_result.generation_index,
            generation=training_result.generation,
            loss_history=training_result.loss_history,
            avg_loss_history=training_result.avg_loss_history,
            best_genome=training_result.best_genome,
            best_loss=training_result.best_loss,
        )

    @staticmethod
    def validate(checkpoint: CheckpointV1) -> None:
        if checkpoint.version != CHECKPOINT_VERSION:
            raise ValueError(
                f"Unsupported checkpoint version: {checkpoint.version} (expected {CHECKPOINT_VERSION})"
            )
        if checkpoint.generation_index < 0:
            raise ValueError("Generation index must be non-negative")
        if not checkpoint.generation:
            raise ValueError("Generation must not be empty")
        if len(checkpoint.loss_history) != len(checkpoint.avg_loss_history):
            raise ValueError("Loss history and avg loss history lengths must match")

        for genome in checkpoint.generation:
            if len(genome) != GENOME_LENGTH:
                raise ValueError(
                    f"Genome length mismatch ({len(genome)}), expected {GENOME_LENGTH}"
                )
        if len(checkpoint.best_genome) != GENOME_LENGTH:
            raise ValueError(
                f"Best genome length mismatch ({len(checkpoint.best_genome)}), expected {GENOME_LENGTH}"
            )

        metadata = checkpoint.metadata
        required_fields = {"run_id", "seed", "created_at"}
        if not required_fields.issubset(metadata.keys()):
            missing = sorted(required_fields - set(metadata.keys()))
            raise ValueError(f"Missing metadata fields: {', '.join(missing)}")

    @staticmethod
    def save(path: str | Path, checkpoint: CheckpointV1) -> Path:
        CheckpointStore.validate(checkpoint)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        serializable = asdict(checkpoint)
        with target.open("w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

        return target

    @staticmethod
    def load(path: str | Path) -> CheckpointV1:
        source = Path(path)
        with source.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        checkpoint = CheckpointV1(
            version=payload["version"],
            metadata=payload["metadata"],
            evolution_config=payload["evolution_config"],
            simulation_config=payload["simulation_config"],
            scenario_config=payload["scenario_config"],
            retry_policy=payload["retry_policy"],
            generation_index=int(payload["generation_index"]),
            generation=[
                [1 if int(gene) == 1 else 0 for gene in genome]
                for genome in payload["generation"]
            ],
            loss_history=[float(v) for v in payload["loss_history"]],
            avg_loss_history=[float(v) for v in payload["avg_loss_history"]],
            best_genome=[1 if int(gene) == 1 else 0 for gene in payload["best_genome"]],
            best_loss=float(payload["best_loss"]),
        )
        CheckpointStore.validate(checkpoint)
        return checkpoint

    @staticmethod
    def inspect(path: str | Path) -> dict[str, Any]:
        ckpt = CheckpointStore.load(path)
        return {
            "version": ckpt.version,
            "run_id": ckpt.metadata["run_id"],
            "seed": ckpt.metadata["seed"],
            "created_at": ckpt.metadata["created_at"],
            "generation_index": ckpt.generation_index,
            "generation_size": len(ckpt.generation),
            "genome_length": len(ckpt.generation[0]),
            "best_loss": ckpt.best_loss,
        }

    @staticmethod
    def export_best_genomes_by_position(
        path: str | Path,
        genomes_by_position: dict[str, list[Genome]],
        metadata: RunMetadata | None = None,
    ) -> Path:
        payload = {
            "version": "best-genomes-v1",
            "metadata": asdict(metadata) if metadata else None,
            "genomes_by_position": genomes_by_position,
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return target


__all__ = ["CheckpointStore", "CheckpointV1", "CHECKPOINT_VERSION"]
