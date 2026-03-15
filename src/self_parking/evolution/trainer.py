"""Evolution trainer orchestrating generation lifecycle."""

from __future__ import annotations

import math
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import TypeVar

from self_parking.core.car_genetic import GENOME_LENGTH, car_loss_to_fitness
from self_parking.core.genetic import create_generation, select
from self_parking.core.random_utils import build_run_metadata, seed_rng
from self_parking.core.types import Generation, Genome, RunMetadata
from self_parking.evolution.config import (
    EvolutionConfig,
    RetryPolicyConfig,
    ScenarioConfig,
    SimulationConfig,
)
from self_parking.simulation.engine import EpisodeResult, simulate_episode

T = TypeVar("T")


@dataclass(frozen=True)
class GenerationSummary:
    generation_index: int
    min_loss: float
    avg_p50_loss: float
    best_genome: Genome


@dataclass(frozen=True)
class TrainingResult:
    metadata: RunMetadata
    generation_index: int
    generation: Generation
    loss_history: list[float]
    avg_loss_history: list[float]
    summaries: list[GenerationSummary]
    best_genome: Genome
    best_loss: float


@dataclass(frozen=True)
class _EvalTask:
    genome: Genome
    genome_index: int
    generation_index: int
    simulation_config: SimulationConfig
    scenario_config: ScenarioConfig
    seed: int


def _evaluate_task(task: _EvalTask) -> tuple[int, float]:
    episode = simulate_episode(
        genome=task.genome,
        simulation_config=task.simulation_config,
        scenario_config=task.scenario_config,
        seed=task.seed,
    )
    return task.genome_index, episode.loss


def _avg_p50(losses: list[float]) -> float:
    if not losses:
        return float("inf")
    sorted_losses = sorted(losses)
    p50_slice = sorted_losses[: max(1, int(len(sorted_losses) * 0.5 + 0.9999))]
    return sum(p50_slice) / len(p50_slice)


def _seed_for(
    base_seed: int,
    generation_index: int,
    genome_index: int,
    attempt: int,
) -> int:
    return base_seed + generation_index * 100_000 + genome_index * 101 + attempt * 10_000_000


class EvolutionTrainer:
    """Main evolution runner."""

    def __init__(
        self,
        evolution_config: EvolutionConfig,
        simulation_config: SimulationConfig,
        scenario_config: ScenarioConfig,
        retry_policy: RetryPolicyConfig,
        show_progress: bool = False,
    ) -> None:
        self.evolution_config = evolution_config
        self.simulation_config = simulation_config
        self.scenario_config = scenario_config
        self.retry_policy = retry_policy
        self.show_progress = show_progress

    def _with_progress(
        self,
        iterable: Iterable[T],
        *,
        desc: str,
        total: int | None = None,
        leave: bool = True,
        position: int = 0,
    ) -> Iterable[T]:
        if not self.show_progress:
            return iterable
        try:
            from tqdm.auto import tqdm  # noqa: PLC0415
        except ImportError:
            return iterable
        return tqdm(iterable, desc=desc, total=total, leave=leave, position=position)

    def _initial_generation(self, rng_seed: int) -> Generation:
        rng = seed_rng(rng_seed)
        if self.evolution_config.initial_generation is not None:
            generation: Generation = []
            for genome in self.evolution_config.initial_generation:
                if len(genome) != GENOME_LENGTH:
                    raise ValueError(
                        f"Initial genome has length {len(genome)}; expected {GENOME_LENGTH}"
                    )
                generation.append([1 if int(gene) == 1 else 0 for gene in genome])

            if len(generation) != self.evolution_config.generation_size:
                raise ValueError(
                    "Initial generation size mismatch: "
                    f"{len(generation)} vs {self.evolution_config.generation_size}"
                )
            return generation

        return create_generation(
            generation_size=self.evolution_config.generation_size,
            genome_length=GENOME_LENGTH,
            rng=rng,
        )

    def _evaluate_generation(
        self,
        generation: Generation,
        generation_index: int,
        attempt: int,
    ) -> list[float]:
        losses = [float("inf")] * len(generation)

        batch_size = max(1, self.evolution_config.batch_size)
        workers = self.evolution_config.workers

        batch_starts = range(0, len(generation), batch_size)
        batch_starts_iter = self._with_progress(
            batch_starts,
            desc=f"Gen {generation_index:03d} batches",
            total=math.ceil(len(generation) / batch_size),
            leave=False,
            position=1,
        )
        for batch_start in batch_starts_iter:
            batch_end = min(len(generation), batch_start + batch_size)
            tasks = [
                _EvalTask(
                    genome=generation[genome_index],
                    genome_index=genome_index,
                    generation_index=generation_index,
                    simulation_config=self.simulation_config,
                    scenario_config=self.scenario_config,
                    seed=_seed_for(
                        base_seed=self.evolution_config.seed,
                        generation_index=generation_index,
                        genome_index=genome_index,
                        attempt=attempt,
                    ),
                )
                for genome_index in range(batch_start, batch_end)
            ]

            if self.evolution_config.execution_mode == "parallel":
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    for genome_index, loss in executor.map(_evaluate_task, tasks):
                        losses[genome_index] = loss
            else:
                for task in tasks:
                    genome_index, loss = _evaluate_task(task)
                    losses[genome_index] = loss

        return losses

    def _retry_needed(self, generation_index: int, loss_history: list[float], current_min_loss: float) -> bool:
        if not self.retry_policy.enabled:
            return False
        if generation_index <= self.retry_policy.batch_index_check:
            return False
        if not loss_history:
            return False

        previous_min_loss = loss_history[-1]
        threshold = previous_min_loss * self.retry_policy.min_loss_increase_percentage / 100
        return current_min_loss > threshold

    def _fitness_function(self, generation: Generation, losses: list[float]):
        key_to_loss = {
            "".join(str(gene) for gene in genome): loss
            for genome, loss in zip(generation, losses, strict=True)
        }

        def fitness(genome: Genome) -> float:
            key = "".join(str(gene) for gene in genome)
            if key not in key_to_loss:
                raise ValueError("Fitness value for specified genome is undefined")
            return car_loss_to_fitness(key_to_loss[key], alpha=self.evolution_config.fitness_alpha)

        return fitness

    def run(self) -> TrainingResult:
        metadata = build_run_metadata(self.evolution_config.seed)
        generation = self._initial_generation(self.evolution_config.seed)

        loss_history: list[float] = []
        avg_loss_history: list[float] = []
        summaries: list[GenerationSummary] = []

        best_genome: Genome = list(generation[0])
        best_loss = float("inf")

        rng = seed_rng(self.evolution_config.seed + 1)

        generation_indexes = range(self.evolution_config.generations)
        generation_indexes_iter = self._with_progress(
            generation_indexes,
            desc="Generations",
            total=self.evolution_config.generations,
            leave=True,
            position=0,
        )

        for generation_index in generation_indexes_iter:
            attempt = 0
            losses = self._evaluate_generation(generation, generation_index, attempt)
            min_loss = min(losses)

            while attempt < self.retry_policy.retries and self._retry_needed(
                generation_index, loss_history, min_loss
            ):
                attempt += 1
                losses = self._evaluate_generation(generation, generation_index, attempt)
                min_loss = min(losses)

            avg_p50_loss = _avg_p50(losses)

            best_index = min(range(len(losses)), key=losses.__getitem__)
            generation_best = list(generation[best_index])

            loss_history.append(min_loss)
            avg_loss_history.append(avg_p50_loss)
            summaries.append(
                GenerationSummary(
                    generation_index=generation_index,
                    min_loss=min_loss,
                    avg_p50_loss=avg_p50_loss,
                    best_genome=generation_best,
                )
            )

            if min_loss < best_loss:
                best_loss = min_loss
                best_genome = generation_best

            if generation_index < self.evolution_config.generations - 1:
                generation = select(
                    generation=generation,
                    fitness=self._fitness_function(generation, losses),
                    mutation_probability=self.evolution_config.mutation_probability,
                    long_living_champions_percentage=self.evolution_config.long_living_champions_percentage,
                    rng=rng,
                )

        return TrainingResult(
            metadata=metadata,
            generation_index=self.evolution_config.generations - 1,
            generation=generation,
            loss_history=loss_history,
            avg_loss_history=avg_loss_history,
            summaries=summaries,
            best_genome=best_genome,
            best_loss=best_loss,
        )


__all__ = [
    "EpisodeResult",
    "EvolutionTrainer",
    "GenerationSummary",
    "TrainingResult",
]
