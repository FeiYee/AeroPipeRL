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
|- scripts/
|  |- watch_model.py # GUI可视化工具，查看训练好的模型运行效果
|- aeropipe_rl/
|  |- config.py
|  |- environment.py
|  |- runner.py
|  |- algorithms/
|  |  |- path_planning.py
|  |  |- obstacle_avoidance.py
|  |  |- hierarchy.py
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

## 可视化查看模型效果

训练完成后，可以使用GUI工具查看模型在3D管道网络中的运行效果：

```bash
# 查看最优模型
py -3.10 scripts/watch_model.py --ckpt best

# 查看最新模型
py -3.10 scripts/watch_model.py --ckpt latest
```

### 可视化快捷键：
- `R`：重新生成随机管道网络
- `SPACE`：重置运行状态
- `Q / ESC`：退出
- 鼠标拖拽：旋转视角
- 滚轮：缩放

## Notes

- This version is focused on training reproducibility and codebase structure.
- Rendering was intentionally removed from the default entry so the project is easier to run and open-source.
