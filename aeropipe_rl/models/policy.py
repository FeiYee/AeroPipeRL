from __future__ import annotations

import torch.nn as nn

from aeropipe_rl.algorithms.obstacle_avoidance import BetaActor
from aeropipe_rl.algorithms.path_planning import AdversaryActor, PlannerActor
from aeropipe_rl.config import DEVICE
from aeropipe_rl.models.critic import CentralizedCritic


class MARLPolicy(nn.Module):
    """Composite policy used for saving/loading and training orchestration."""

    def __init__(self) -> None:
        super().__init__()
        self.executor = BetaActor().to(DEVICE)
        self.planner = PlannerActor().to(DEVICE)
        self.adversary = AdversaryActor().to(DEVICE)
        self.critic = CentralizedCritic().to(DEVICE)
