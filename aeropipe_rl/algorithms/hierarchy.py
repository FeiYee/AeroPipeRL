from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from aeropipe_rl.config import (
    FLOW_ROLLOUT_STEPS,
    HIDDEN,
    TERMINATION_TAU,
    TERMINATION_V_RESET,
    TERMINATION_V_THRESHOLD,
)


class SubgoalFiLM(nn.Module):
    """Feature-wise conditioning blocks keyed by layer name."""

    def __init__(self, channels: dict[str, int], cond_dim: int = HIDDEN) -> None:
        super().__init__()
        self.adapters = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(cond_dim, cond_dim),
                    nn.GELU(),
                    nn.Linear(cond_dim, channels[name] * 2),
                )
                for name in channels
            }
        )

    def apply(self, name: str, x: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        if cond is None or name not in self.adapters:
            return x

        gamma_beta = self.adapters[name](cond)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = 1.0 + 0.1 * torch.tanh(gamma)
        beta = 0.1 * torch.tanh(beta)

        while gamma.dim() < x.dim():
            gamma = gamma.unsqueeze(-2)
            beta = beta.unsqueeze(-2)
        return gamma * x + beta


class TerminationLIF(nn.Module):
    """Event-driven option termination head driven by LIF membrane dynamics."""

    def __init__(
        self,
        ctx_dim: int,
        metric_dim: int,
        tau: float = TERMINATION_TAU,
        v_threshold: float = TERMINATION_V_THRESHOLD,
        v_reset: float = TERMINATION_V_RESET,
        sharpness: float = 6.0,
    ) -> None:
        super().__init__()
        self.tau = tau
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.sharpness = sharpness
        hidden_dim = max(ctx_dim, metric_dim * 4)
        self.current_encoder = nn.Sequential(
            nn.Linear(ctx_dim + metric_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        ctx: torch.Tensor,
        metrics: torch.Tensor,
        membrane: torch.Tensor | None = None,
        reset_mask: torch.Tensor | None = None,
        agent_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = metrics[..., 0:1]
        progress = metrics[..., 1:2]
        align = metrics[..., 2:3]
        in_hub = metrics[..., 3:4]
        wall_clear = metrics[..., 4:5]

        proximity = torch.relu(torch.exp(-8.0 * dist) - 0.2)
        heuristic_current = (
            0.85 * proximity
            + 0.45 * torch.relu(progress)
            + 0.15 * torch.relu(align)
            + 0.10 * in_hub
            + 0.15 * torch.relu(wall_clear)
        )
        current = self.current_encoder(torch.cat([ctx, metrics], dim=-1)) + heuristic_current
        if membrane is None:
            membrane = torch.full_like(current, self.v_reset)
        else:
            membrane = membrane.to(device=current.device, dtype=current.dtype)

        if reset_mask is not None:
            while reset_mask.dim() < membrane.dim():
                reset_mask = reset_mask.unsqueeze(-1)
            membrane = torch.where(reset_mask, torch.full_like(membrane, self.v_reset), membrane)

        membrane = membrane * (1.0 - 1.0 / self.tau) + current
        terminate_prob = torch.sigmoid((membrane - self.v_threshold) * self.sharpness)
        spike = (membrane >= self.v_threshold).float()
        membrane = torch.where(spike > 0, torch.full_like(membrane, self.v_reset), membrane)

        if agent_mask is not None:
            while agent_mask.dim() < terminate_prob.dim():
                agent_mask = agent_mask.unsqueeze(-1)
            terminate_prob = terminate_prob.masked_fill(agent_mask, 0.0)
            spike = spike.masked_fill(agent_mask, 0.0)
            membrane = membrane.masked_fill(agent_mask, self.v_reset)

        return terminate_prob.squeeze(-1), spike.squeeze(-1), membrane.detach()


class DifferentiablePathFlowLayer(nn.Module):
    """Turns per-edge flow preferences into differentiable path distributions."""

    def __init__(self, rollout_steps: int = FLOW_ROLLOUT_STEPS) -> None:
        super().__init__()
        self.rollout_steps = rollout_steps

    def forward(
        self,
        edge_flow_weights: torch.Tensor,
        adj: torch.Tensor,
        node_mask: torch.Tensor,
        cur_node_ids: torch.Tensor,
        goal_node_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, agent_count = edge_flow_weights.shape[:2]
        node_count = adj.shape[-1]
        edge_flow = edge_flow_weights.view(batch_size, agent_count, node_count, node_count)

        eye = torch.eye(node_count, device=edge_flow.device, dtype=torch.bool)
        valid_edges = (adj.unsqueeze(1) > 0) & ~eye.view(1, 1, node_count, node_count)
        valid_edges = valid_edges & node_mask.unsqueeze(1).unsqueeze(-1) & node_mask.unsqueeze(1).unsqueeze(-2)

        masked_scores = edge_flow.masked_fill(~valid_edges, -1e9)
        transition = torch.softmax(masked_scores, dim=-1)

        fallback = eye.to(dtype=transition.dtype).view(1, 1, node_count, node_count).expand_as(transition)
        has_outgoing = valid_edges.any(dim=-1, keepdim=True)
        transition = torch.where(has_outgoing, transition, fallback)

        goal_onehot = F.one_hot(goal_node_ids, num_classes=node_count).to(dtype=transition.dtype)
        goal_rows = goal_onehot.unsqueeze(-1)
        transition = transition * (1.0 - goal_rows) + fallback * goal_rows
        transition = transition * node_mask.unsqueeze(1).unsqueeze(-1).to(dtype=transition.dtype)

        state = F.one_hot(cur_node_ids, num_classes=node_count).to(dtype=transition.dtype)
        visit = state.clone()
        edge_path = torch.zeros_like(transition)

        for _ in range(self.rollout_steps):
            step_edge = state.unsqueeze(-1) * transition
            edge_path = edge_path + step_edge
            state = torch.matmul(state.unsqueeze(-2), transition).squeeze(-2)
            visit = visit + state

        visit = visit * node_mask.unsqueeze(1).to(dtype=visit.dtype)
        visit = visit / visit.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        edge_path = edge_path * valid_edges.to(dtype=edge_path.dtype)
        edge_path = edge_path / edge_path.sum(dim=(-1, -2), keepdim=True).clamp(min=1e-6)

        cur_onehot = F.one_hot(cur_node_ids, num_classes=node_count).to(dtype=visit.dtype)
        subgoal_probs = visit * (1.0 - cur_onehot)
        need_goal_fallback = subgoal_probs.sum(dim=-1, keepdim=True) <= 1e-6
        goal_fallback = goal_onehot
        subgoal_probs = torch.where(need_goal_fallback, goal_fallback, subgoal_probs)
        subgoal_probs = subgoal_probs / subgoal_probs.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        return {
            "transition_probs": transition,
            "edge_path_probs": edge_path.view(batch_size, agent_count, -1),
            "node_visit_probs": visit,
            "subgoal_node_probs": subgoal_probs,
        }
