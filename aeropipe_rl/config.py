from __future__ import annotations

from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LATEST_CKPT_PATH = CHECKPOINT_DIR / "pipe_marl_latest.pt"
BEST_CKPT_PATH = CHECKPOINT_DIR / "pipe_marl_best.pt"

N_AGENTS = 20
MAX_N_NODES = 26
MAX_N_EDGES = MAX_N_NODES * MAX_N_NODES
MIN_PATH_LEN = 5

PIPE_R = 1.6
HUB_R = 1.8
WP_LOOKAHEAD_R = HUB_R * 0.5
AGENT_R = 0.1
AGENT_COL_R = 0.12
MAX_SPEED = 0.42
MAX_ACC = 0.15
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
TERMINATION_THRESHOLD = 0.60
TERMINATION_TAU = 2.5
TERMINATION_V_THRESHOLD = 1.0
TERMINATION_V_RESET = 0.0

HIDDEN = 128
N_HEADS = 4
N_LAYERS = 2
DROPOUT = 0.0

ENABLE_COMM = True
COMM_ALPHA_INIT = 0.05
COMM_ALPHA_MAX = 0.40

GAMMA = 0.99
GAE_LAMBDA = 0.95
LOW_LEVEL_GAMMA = 0.99
LOW_LEVEL_GAE_LAMBDA = 0.95
OPTION_GAMMA = 0.97
OPTION_GAE_LAMBDA = 0.93
PLANNER_GAMMA = 0.95
PLANNER_GAE_LAMBDA = 0.90
LR_ACTOR = 3e-4
LR_CRITIC = 8e-4
T_HORIZON = 512
K_EPOCHS = 4
MINI_BATCH = 24
CLIP_EPS = 0.20
ENT_COEF = 0.015
VAL_COEF = 0.50
GRAD_NORM = 0.50
MAX_EP_STEPS = 600
STEP_BUDGET = 300

R_GOAL_BASE = 30.0
R_GOAL_TEAM_INC = 5.0
R_WALL = -8.0
R_AGENT_COL = -4.0
R_AGENT_COL_PERSIST = -0.5
COL_PERSIST_STEPS = 3
R_WP = 5.0
R_STEP_BASE = -0.05
R_STEP_OT_START = -0.12
R_STEP_OT_INC = -0.01
R_STEP_OT_MIN = -0.30
R_SHAPING = 2.0
R_SPEED = 0.15
DIR_REWARD_COS_TH = 0.0
WALL_SAFE_MARGIN = 0.60
R_WALL_NEAR = 2.0

AUTO_BUDGET_SLACK = 1.6
AUTO_BUDGET_BUF = 20.0

GLOBAL_SEED = 42
LOG_EVERY_EP = 10
SAVE_EVERY_EP = 100
MAX_TRAIN_EPISODES = 200000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 可视化配置
W = 1280
H = 800
WATCH_SPF = 1         # 每渲染帧运行的环境步数（观看模式）
WATCH_PAUSE = 80      # 观看模式下两局之间的暂停帧数

# Agent颜色调色板 (RGBA 0-1)
AGENT_COLORS = [
    (0.10, 0.90, 1.00, 1.0),   # cyan
    (1.00, 0.55, 0.05, 1.0),   # orange
    (0.20, 1.00, 0.35, 1.0),   # green
    (0.90, 0.20, 1.00, 1.0),   # purple
    (1.00, 0.20, 0.20, 1.0),   # red
    (0.20, 0.40, 1.00, 1.0),   # blue
    (1.00, 0.90, 0.20, 1.0),   # yellow
    (0.00, 0.80, 0.80, 1.0),   # teal
    (0.80, 0.40, 0.00, 1.0),   # brown
    (1.00, 0.00, 0.70, 1.0),   # magenta
    (0.50, 1.00, 0.00, 1.0),   # lime
    (0.00, 0.50, 1.00, 1.0),   # sky blue
    (0.60, 0.00, 1.00, 1.0),   # violet
    (1.00, 0.30, 0.60, 1.0),   # pink
    (0.40, 0.80, 0.20, 1.0),   # olive green
    (0.30, 0.30, 0.30, 1.0),   # dark gray
    (0.70, 0.70, 0.70, 1.0),   # light gray
    (0.00, 0.60, 0.40, 1.0),   # sea green
    (0.80, 0.20, 0.00, 1.0),   # dark orange-red
    (0.20, 0.00, 0.60, 1.0),   # indigo
]



def ensure_runtime_dirs() -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
