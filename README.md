# Self-Parking Car Evolution (Python)

Python reimplementation of `self-parking-car-evolution-js` with the v1 scope:
- `Evolution` mode (genetic training)
- `Automatic parking` mode (best-genome playback)
- Polygon-based 2D simulation (no high-level visuals/3D models)

## Features

- Ported GA core and car-genome formulas from the TypeScript project
- Deterministic kinematic car dynamics with polygon collision + 8 ray sensors
- Versioned Python-native checkpoint format (`checkpoint-v1`)
- Selectable execution mode:
  - `single` (deterministic default)
  - `parallel` (process-based evaluation)
- CLI workflows for training, playback, and checkpoint inspection
- Notebook workflows for interactive training analytics and playback animation

## Project Structure

- `src/self_parking/core`: GA primitives, numeric helpers, genome decoding
- `src/self_parking/simulation`: 2D world geometry, sensors, kinematic engine
- `src/self_parking/evolution`: configuration and trainer loop
- `src/self_parking/io`: checkpoint save/load/validate
- `src/self_parking/viz`: matplotlib plots and animation
- `src/self_parking/cli`: command-line entrypoint
- `notebooks/`: training and playback notebooks
- `tests/`: parity, simulation, integration, and smoke tests

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,viz,notebook]'
```

## CLI Usage

Train and save checkpoint:

```bash
spce train \
  --generations 20 \
  --generation-size 100 \
  --output artifacts/checkpoints/run.json \
  --plot artifacts/plots/loss.png
```

`tqdm` progress bars are enabled by default in `spce train`. Use `--no-progress` to disable them.

Replay the best genome from a checkpoint:

```bash
spce playback \
  --checkpoint artifacts/checkpoints/run.json \
  --output artifacts/playback/run.gif
```

Inspect checkpoint metadata:

```bash
spce checkpoint-inspect --checkpoint artifacts/checkpoints/run.json
```

Export best genome for a start-position demo set:

```bash
spce checkpoint-export-best \
  --checkpoint artifacts/checkpoints/run.json \
  --position front \
  --output artifacts/best_genomes.json
```

## Notebooks

- `notebooks/evolution_training.ipynb`
- `notebooks/automatic_playback.ipynb`

## Testing

```bash
pytest
```

## Notes

- Manual-driving mode is intentionally deferred in this v1.
- Checkpoint compatibility with original JS JSON files is intentionally not included.
- See migration details in [`docs/MIGRATION.md`](docs/MIGRATION.md).
