# AeroPipeRL

Modular multi-agent UAV pipe-network navigation training project.

## Features

- Standalone headless training entry: `py -3.10 main.py --resume none`
- Environment separated into its own module for reuse in other RL experiments
- High-level path planning and low-level obstacle avoidance split into different Python files
- MAPPO-style trainer with centralized critic and modular checkpoint loading

## Project Layout

```text
AeroPipeRL/
|- main.py
|- requirements.txt
|- aeropipe_rl/
|  |- config.py
|  |- environment.py
|  |- runner.py
|  |- algorithms/
|  |  |- path_planning.py
|  |  |- obstacle_avoidance.py
|  |- models/
|  |  |- critic.py
|  |  |- policy.py
|  |- training/
|     |- buffer.py
|     |- trainer.py
```

## Quick Start

```bash
py -3.10 -m pip install -r requirements.txt
py -3.10 main.py --resume none
```

## Resume Options

- `--resume none`
- `--resume latest`
- `--resume best`

Checkpoints are written into `checkpoints/`.

## Notes

- This version is focused on training reproducibility and codebase structure.
- Rendering was intentionally removed from the default entry so the project is easier to run and open-source.
