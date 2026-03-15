# Implementation Plan (Executed)

This repository implements the approved conversion plan from `self-parking-car-evolution-js` to Python with these locked decisions:

- Scope: Evolution + Automatic playback
- Physics/rendering: deterministic 2D kinematic + polygons
- Checkpoints: Python-native only (`checkpoint-v1`)
- Execution: single-process deterministic default + optional parallel mode

See details in:
- `README.md`
- `docs/MIGRATION.md`
- `docs/CONTRIBUTING.md`
