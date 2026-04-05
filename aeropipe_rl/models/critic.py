from __future__ import annotations

import torch
import torch.nn as nn

from aeropipe_rl.config import EGO_DIM, N_AGENTS


class CentralizedCritic(nn.Module):
    """Centralized multi-timescale value network over all agent ego states."""

    def __init__(self) -> None:
        super().__init__()
        self.agent_encoder = nn.Sequential(
            nn.Linear(EGO_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        self.shared = nn.Sequential(
            nn.Linear(32 * N_AGENTS, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.step_head = nn.Linear(128, 1)
        self.option_head = nn.Linear(128, 1)
        self.planner_head = nn.Linear(128, 1)

    def encode(self, global_obs: torch.Tensor) -> torch.Tensor:
        batch_size = global_obs.shape[0]
        x = global_obs.view(batch_size, N_AGENTS, EGO_DIM)
        x = self.agent_encoder(x)
        x = x.reshape(batch_size, N_AGENTS * 32)
        return self.shared(x)

    def forward(self, global_obs: torch.Tensor, head: str = "step"):
        x = self.encode(global_obs)
        outputs = {
            "step": self.step_head(x).squeeze(-1),
            "option": self.option_head(x).squeeze(-1),
            "planner": self.planner_head(x).squeeze(-1),
        }
        if head == "all":
            return outputs
        return outputs[head]
