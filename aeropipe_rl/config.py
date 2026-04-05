from __future__ import annotations

from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LATEST_CKPT_PATH = CHECKPOINT_DIR / "pipe_marl_latest.pt"
BEST_CKPT_PATH = CHECKPOINT_DIR / "pipe_marl_best.pt"

N_AGENTS = 6                  # FIX: 20 agents in a 26-node graph is severely overcrowded
                               # ~0.77 agents/node causes constant collisions & reward explosion
                               # 6 agents gives room to learn, scale up later
MAX_N_NODES = 26
MAX_N_EDGES = MAX_N_NODES * MAX_N_NODES
MIN_PATH_LEN = 4              # FIX: was 5, eased slightly for sparser graph connectivity

PIPE_R = 2.0                  # FIX: was 1.6 — slightly wider pipe so agent can actually fit + move
HUB_R = 2.2                   # FIX: was 1.8 — proportional to pipe
WP_LOOKAHEAD_R = HUB_R * 0.6  # slightly more lookahead
AGENT_R = 0.1
AGENT_COL_R = 0.15            # FIX: was 0.12, small increase to match agent radius better
MAX_SPEED = 0.55              # FIX: was 0.42, allow slightly faster traversal
MAX_ACC = 0.20                # FIX: was 0.15, match speed increase
DT = 1.0

EGO_DIM = 17
LOCAL_TOPK = 10
NODE_DIM = 7
NBR_RADIUS = 10.0
NBR_DIM = 7
MAX_NBR = N_AGENTS - 1
ACT_DIM = 3
GLOBAL_DIM = EGO_DIM * N_AGENTS

HIGH_LEVEL_UPDATE_FREQ = 5
TIME_WINDOW_SIZE = 20
R_ON_TIME = 0.2
R_CAPACITY = 0.3
R_CONFLICT = -0.5
FLOW_ROLLOUT_STEPS = MAX_N_NODES
TERMINATION_THRESHOLD = 0.55  # FIX: was 0.60, slightly easier to trigger option switch
TERMINATION_TAU = 2.5
TERMINATION_V_THRESHOLD = 1.0
TERMINATION_V_RESET = 0.0

HIDDEN = 128
N_HEADS = 4
N_LAYERS = 2
DROPOUT = 0.05                # FIX: was 0.0, small dropout prevents rapid overfitting

ENABLE_COMM = True
COMM_ALPHA_INIT = 0.05
COMM_ALPHA_MAX = 0.40

# ── PPO / Training ──────────────────────────────────────────────────────────
GAMMA = 0.99
GAE_LAMBDA = 0.95
LOW_LEVEL_GAMMA = 0.995       # FIX: was 0.99, longer horizon needed for pipe navigation
LOW_LEVEL_GAE_LAMBDA = 0.97  # FIX: was 0.95
OPTION_GAMMA = 0.98           # FIX: was 0.97
OPTION_GAE_LAMBDA = 0.95      # FIX: was 0.93
PLANNER_GAMMA = 0.97          # FIX: was 0.95
PLANNER_GAE_LAMBDA = 0.93     # FIX: was 0.90
LR_ACTOR = 2e-4               # FIX: was 3e-4, slightly slower for stability
LR_CRITIC = 6e-4              # FIX: was 8e-4
T_HORIZON = 256               # FIX: was 512, shorter horizon → more frequent updates
K_EPOCHS = 3                  # FIX: was 4, less reuse prevents destructive updates
MINI_BATCH = 16               # FIX: was 24, smaller batch fits shorter rollout
CLIP_EPS = 0.15               # FIX: was 0.20, tighter clip for early stability
ENT_COEF = 0.025              # FIX: was 0.015, more exploration early
VAL_COEF = 0.50
GRAD_NORM = 0.40              # FIX: was 0.50, tighter gradient clipping
MAX_EP_STEPS = 500
STEP_BUDGET = 0               # 0 = auto-compute per episode

# ── Reward shaping ───────────────────────────────────────────────────────────
# FIX: This was the reward explosion source.
# R_SHAPING * (progress/seg_len) could blow up when seg_len ≈ 0.
# Solution: cap shaping, reduce magnitude, rely more on sparse goal rewards.
R_GOAL_BASE = 50.0            # FIX: was 30, clearer goal signal
R_GOAL_TEAM_INC = 3.0         # FIX: was 5.0, reduced team bonus
R_WALL = -3.0                 # FIX: was -8.0. Large wall penalty causes panic/thrashing.
                               # Smaller penalty + inward guidance is more informative.
R_AGENT_COL = -2.0            # FIX: was -4.0, same reasoning
R_AGENT_COL_PERSIST = -0.2   # FIX: was -0.5
COL_PERSIST_STEPS = 2         # FIX: was 3
R_WP = 2.0                    # FIX: was 5.0, waypoint bonus was too dominant
R_STEP_BASE = -0.02           # FIX: was -0.05, lighter step penalty
R_STEP_OT_START = -0.06       # FIX: was -0.12
R_STEP_OT_INC = -0.005        # FIX: was -0.01
R_STEP_OT_MIN = -0.15         # FIX: was -0.30
R_SHAPING = 0.5               # FIX: was 2.0! This was the explosion source.
                               # Also added a cap in environment._update_progress
R_SPEED = 0.08                # FIX: was 0.15
DIR_REWARD_COS_TH = 0.1       # FIX: was 0.0, require slight alignment before rewarding speed
WALL_SAFE_MARGIN = 0.60
R_WALL_NEAR = 0.0             # disabled

AUTO_BUDGET_SLACK = 2.0       # FIX: was 1.6, give more budget
AUTO_BUDGET_BUF = 30.0        # FIX: was 20.0

# ── INNOVATION: Curriculum learning flags ────────────────────────────────────
# Agents start with N_AGENTS_CURRICULUM and scale up to N_AGENTS
CURRICULUM_ENABLED = True
N_AGENTS_START = 2            # start with just 2 agents
N_AGENTS_TARGET = 6           # ramp to full 6
CURRICULUM_RAMP_EP = 2000     # episodes to reach full agent count

# ── INNOVATION: Adaptive entropy ─────────────────────────────────────────────
ENT_COEF_MIN = 0.003
ENT_COEF_DECAY = 0.9997      # FIX: was 0.9995, slightly slower decay

GLOBAL_SEED = 42
LOG_EVERY_EP = 10
SAVE_EVERY_EP = 100
MAX_TRAIN_EPISODES = 200_000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GUI / watch mode defaults
W = 1280
H = 800
WATCH_SPF = 1
WATCH_PAUSE = 80
AGENT_COLORS = [
    (0.10, 0.90, 1.00, 1.0),
    (1.00, 0.55, 0.05, 1.0),
    (0.20, 1.00, 0.35, 1.0),
    (0.90, 0.20, 1.00, 1.0),
    (1.00, 0.20, 0.20, 1.0),
    (0.20, 0.40, 1.00, 1.0),
    (1.00, 0.90, 0.20, 1.0),
    (0.00, 0.80, 0.80, 1.0),
    (0.80, 0.40, 0.00, 1.0),
    (1.00, 0.00, 0.70, 1.0),
    (0.50, 1.00, 0.00, 1.0),
    (0.00, 0.50, 1.00, 1.0),
    (0.60, 0.00, 1.00, 1.0),
    (1.00, 0.30, 0.60, 1.0),
    (0.40, 0.80, 0.20, 1.0),
    (0.30, 0.30, 0.30, 1.0),
    (0.70, 0.70, 0.70, 1.0),
    (0.00, 0.60, 0.40, 1.0),
    (0.80, 0.20, 0.00, 1.0),
    (0.20, 0.00, 0.60, 1.0),
]


def ensure_runtime_dirs() -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
