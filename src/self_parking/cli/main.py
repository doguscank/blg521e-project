"""Command-line interface for training and playback."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from self_parking.evolution.config import (
    EvolutionConfig,
    RetryPolicyConfig,
    ScenarioConfig,
    SimulationConfig,
)
from self_parking.evolution.trainer import EvolutionTrainer
from self_parking.io.checkpoint import CheckpointStore
from self_parking.simulation.engine import simulate_episode


def _scenario_from_dict(data: dict) -> ScenarioConfig:
    skip_cells = tuple(tuple(int(v) for v in pair) for pair in data.get("static_skip_cells", [(0, 2)]))
    return ScenarioConfig(
        start_position=data.get("start_position", "front"),
        with_random_start=bool(data.get("with_random_start", False)),
        static_rows=int(data.get("static_rows", 2)),
        static_cols=int(data.get("static_cols", 5)),
        static_skip_cells=skip_cells,
    )


def _simulation_from_dict(data: dict) -> SimulationConfig:
    return SimulationConfig(**data)


def _evolution_from_dict(data: dict) -> EvolutionConfig:
    return EvolutionConfig(**data)


def _retry_from_dict(data: dict) -> RetryPolicyConfig:
    return RetryPolicyConfig(**data)


def _cmd_train(args: argparse.Namespace) -> int:
    evolution_config = EvolutionConfig(
        generation_size=args.generation_size,
        generations=args.generations,
        batch_size=args.batch_size,
        mutation_probability=args.mutation,
        long_living_champions_percentage=args.champions,
        execution_mode=args.mode,
        workers=args.workers,
        seed=args.seed,
    )

    simulation_config = SimulationConfig(
        dt=args.dt,
        episode_seconds=args.episode_seconds,
    )

    scenario_config = ScenarioConfig(
        start_position=args.start_position,
        with_random_start=args.random_start,
    )

    retry_policy = RetryPolicyConfig(
        enabled=not args.disable_retry,
        retries=args.retries,
        batch_index_check=args.batch_index_check,
        min_loss_increase_percentage=args.min_loss_increase_percentage,
    )

    trainer = EvolutionTrainer(
        evolution_config=evolution_config,
        simulation_config=simulation_config,
        scenario_config=scenario_config,
        retry_policy=retry_policy,
        show_progress=not args.no_progress,
    )

    result = trainer.run()

    for summary in result.summaries:
        print(
            f"gen={summary.generation_index:03d} "
            f"min_loss={summary.min_loss:.4f} "
            f"p50_avg={summary.avg_p50_loss:.4f}"
        )

    checkpoint = CheckpointStore.from_training_result(
        training_result=result,
        evolution_config=evolution_config,
        simulation_config=simulation_config,
        scenario_config=scenario_config,
        retry_policy=retry_policy,
    )
    output_path = CheckpointStore.save(args.output, checkpoint)
    print(f"Checkpoint saved: {output_path}")

    if args.plot is not None:
        from self_parking.viz.plots import plot_training_history  # noqa: PLC0415

        fig, _ = plot_training_history(result.loss_history, result.avg_loss_history)
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=140)
        print(f"Training plot saved: {args.plot}")

    return 0


def _cmd_playback(args: argparse.Namespace) -> int:
    checkpoint = CheckpointStore.load(args.checkpoint)

    simulation_config = _simulation_from_dict(checkpoint.simulation_config)
    scenario_config = _scenario_from_dict(checkpoint.scenario_config)

    genome = checkpoint.best_genome
    if args.genome_index is not None:
        genome = checkpoint.generation[args.genome_index]

    episode = simulate_episode(
        genome=genome,
        simulation_config=simulation_config,
        scenario_config=scenario_config,
        seed=checkpoint.metadata["seed"],
    )

    print(f"Playback loss: {episode.loss:.6f}")
    print(f"Collisions: {episode.collisions}")

    if args.output is not None:
        from self_parking.viz.animation import animate_episode  # noqa: PLC0415

        animate_episode(
            episode_result=episode,
            scenario_config=scenario_config,
            save_path=args.output,
            show_sensors=not args.hide_sensors,
        )
        print(f"Playback artifact saved: {args.output}")

    return 0


def _cmd_checkpoint_inspect(args: argparse.Namespace) -> int:
    info = CheckpointStore.inspect(args.checkpoint)
    print(json.dumps(info, indent=2))
    return 0


def _cmd_checkpoint_export_best(args: argparse.Namespace) -> int:
    checkpoint = CheckpointStore.load(args.checkpoint)
    payload = {
        args.position: [checkpoint.best_genome],
    }
    path = CheckpointStore.export_best_genomes_by_position(args.output, payload)
    print(f"Best genome export saved: {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spce", description="Self-parking car evolution CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Run evolution training")
    train.add_argument("--generations", type=int, default=20)
    train.add_argument("--generation-size", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--mutation", type=float, default=0.04)
    train.add_argument("--champions", type=float, default=6.0)
    train.add_argument("--mode", choices=["single", "parallel"], default="single")
    train.add_argument("--workers", type=int, default=None)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--start-position", choices=["front", "middle", "rear", "random"], default="front")
    train.add_argument("--random-start", action="store_true")
    train.add_argument("--dt", type=float, default=0.1)
    train.add_argument("--episode-seconds", type=float, default=17.0)
    train.add_argument("--disable-retry", action="store_true")
    train.add_argument("--retries", type=int, default=1)
    train.add_argument("--batch-index-check", type=int, default=1)
    train.add_argument("--min-loss-increase-percentage", type=float, default=110.0)
    train.add_argument("--output", type=Path, default=Path("artifacts/checkpoints/latest.json"))
    train.add_argument("--plot", type=Path, default=None)
    train.add_argument("--no-progress", action="store_true")
    train.set_defaults(func=_cmd_train)

    playback = subparsers.add_parser("playback", help="Replay checkpoint best genome")
    playback.add_argument("--checkpoint", type=Path, required=True)
    playback.add_argument("--genome-index", type=int, default=None)
    playback.add_argument("--output", type=Path, default=None)
    playback.add_argument("--hide-sensors", action="store_true")
    playback.set_defaults(func=_cmd_playback)

    ckpt = subparsers.add_parser("checkpoint-inspect", help="Inspect checkpoint metadata")
    ckpt.add_argument("--checkpoint", type=Path, required=True)
    ckpt.set_defaults(func=_cmd_checkpoint_inspect)

    export_best = subparsers.add_parser(
        "checkpoint-export-best",
        help="Export best genome for automatic demo by start position",
    )
    export_best.add_argument("--checkpoint", type=Path, required=True)
    export_best.add_argument("--position", choices=["front", "middle", "rear"], default="front")
    export_best.add_argument("--output", type=Path, default=Path("artifacts/best_genomes.json"))
    export_best.set_defaults(func=_cmd_checkpoint_export_best)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
