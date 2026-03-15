# Migration Notes (JS -> Python)

## What Was Ported

- Genetic algorithm core:
  - generation creation
  - weighted selection
  - uniform crossover + mutation
  - champion carry-over
- Numeric helpers:
  - 10-bit and 16-bit float decoding
  - sigmoid and `{-1,0,1}` category mapping
  - linear polynomial
  - XZ-plane euclidean distance
- Car genome logic:
  - genome decoding into engine/wheel coefficients
  - controller formulas with 8 ray sensors + 5 state sensors (`pos_x`, `pos_y`, `vel_x`, `vel_y`, `heading`)
  - parking loss and fitness transform
- Runtime behavior:
  - generation history (min loss + P50 average)
  - retry policy hooks for unstable generations
  - checkpoint persistence and restoration

## Intentional v1 Differences

- Physics engine replaced by deterministic kinematic bicycle model.
- Rendering replaced by notebook/CLI polygon visualization (matplotlib), no 3D scene.
- Manual-driving tab is not implemented.
- Checkpoint format is Python-native (`checkpoint-v1`) and not JS-compatible.

## Equivalent Scenario Defaults

- Dynamic car start positions: `front=+7`, `middle=0`, `rear=-7` (`z` axis)
- Static parked cars grid: 2x5 with skipped cell `(0,2)`
- Car dimensions and sensor count follow original constants
- Parking spot inner rectangle generated from original dimensions and offsets

## Execution Modes

- `single` mode is deterministic and default.
- `parallel` mode evaluates genomes with process pools while keeping deterministic seed mapping.
