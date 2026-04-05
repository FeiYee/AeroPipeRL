from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from aeropipe_rl.algorithms.hierarchy import SubgoalFiLM, TerminationLIF
from aeropipe_rl.config import (
    ACT_DIM,
    COMM_ALPHA_INIT,
    COMM_ALPHA_MAX,
    DROPOUT,
    EGO_DIM,
    ENABLE_COMM,
    HIDDEN,
    LOCAL_TOPK,
    MAX_ACC,
    MAX_NBR,
    NBR_DIM,
    N_HEADS,
    N_LAYERS,
    NODE_DIM,
)


class LIFNeuron(nn.Module):
    def __init__(self, tau: float = 2.0, v_threshold: float = 1.0, v_reset: float = 0.0, surrogate: str = "sigmoid", alpha: float = 1.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.tau = tau
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)
        if surrogate == "sigmoid":
            self.surrogate = self._sigmoid_surrogate
        elif surrogate == "relu":
            self.surrogate = self._relu_surrogate
        else:
            raise ValueError(f"Unsupported surrogate: {surrogate}")

    def _sigmoid_surrogate(self, x: torch.Tensor) -> torch.Tensor:
        sig = torch.sigmoid(self.alpha * x)
        return sig * (1 - sig) * self.alpha

    def _relu_surrogate(self, x: torch.Tensor) -> torch.Tensor:
        return (x > 0).float()

    def forward(self, x: torch.Tensor, v: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if v is None:
            v = torch.full_like(x, self.v_reset, device=x.device)
        v = v * (1 - 1 / self.tau) + x
        spike = (v >= self.v_threshold).float()
        v = torch.where(spike > 0, torch.full_like(v, self.v_reset, device=v.device), v)
        spike = self.dropout(spike)
        spike = spike + (self.surrogate(v - self.v_threshold) - self.surrogate(v - self.v_threshold)).detach()
        return spike, v


class SpikeMultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, tau: float = 2.0, num_time_steps: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_time_steps = num_time_steps
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_lif = LIFNeuron(tau=tau, dropout=dropout)
        self.k_lif = LIFNeuron(tau=tau, dropout=dropout)
        self.v_lif = LIFNeuron(tau=tau, dropout=dropout)
        self.attn_lif = LIFNeuron(tau=tau, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None, need_weights: bool = False):
        batch_size, n_tokens, embed_dim = q.shape
        q_p = self.q_proj(q).view(batch_size, n_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_p = self.k_proj(k).view(batch_size, n_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v_p = self.v_proj(v).view(batch_size, n_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q_p = (q_p - q_p.min()) / (q_p.max() - q_p.min() + 1e-6)
        k_p = (k_p - k_p.min()) / (k_p.max() - k_p.min() + 1e-6)
        v_p = (v_p - v_p.min()) / (v_p.max() - v_p.min() + 1e-6)

        q_v = k_v = v_v = attn_v = None
        out_spikes = torch.zeros(batch_size, self.num_heads, n_tokens, self.head_dim, device=q.device)
        for _ in range(self.num_time_steps):
            q_spike, q_v = self.q_lif(q_p, q_v)
            k_spike, k_v = self.k_lif(k_p, k_v)
            v_spike, v_v = self.v_lif(v_p, v_v)

            attn = torch.matmul(q_spike, k_spike.transpose(-2, -1)) * self.scale
            if key_padding_mask is not None:
                mask = key_padding_mask.unsqueeze(1).unsqueeze(1)
                attn = attn.masked_fill(mask, float("-inf"))
            attn_spike, attn_v = self.attn_lif(attn, attn_v)
            attn_spike = self.dropout(attn_spike)
            out_spikes += torch.matmul(attn_spike, v_spike)

        out = out_spikes / self.num_time_steps
        out = out.permute(0, 2, 1, 3).reshape(batch_size, n_tokens, embed_dim)
        out = self.out_proj(out)
        return out, None


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.proj = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.attn_src = nn.Parameter(torch.empty(heads, out_dim))
        self.attn_dst = nn.Parameter(torch.empty(heads, out_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        batch_size, node_count, _ = x.shape
        h = self.proj(x).view(batch_size, node_count, self.heads, self.out_dim)
        src = (h * self.attn_src.unsqueeze(0).unsqueeze(0)).sum(-1)
        dst = (h * self.attn_dst.unsqueeze(0).unsqueeze(0)).sum(-1)
        e = self.leaky_relu(src.unsqueeze(2) + dst.unsqueeze(1))
        mask = adj.unsqueeze(-1) > 0
        e = e.masked_fill(~mask, float("-inf"))
        a = torch.softmax(e, dim=2)
        a = self.dropout(a)
        h2 = h.permute(0, 2, 1, 3)
        a2 = a.permute(0, 3, 1, 2)
        out = torch.matmul(a2, h2)
        out = out.permute(0, 2, 1, 3).reshape(batch_size, node_count, self.heads * self.out_dim)
        return out, a


class LocalGraphEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        gat_head_dim = HIDDEN // N_HEADS
        self.gat1 = GraphAttentionLayer(NODE_DIM, gat_head_dim, heads=N_HEADS, dropout=DROPOUT)
        self.gat2 = GraphAttentionLayer(HIDDEN, gat_head_dim, heads=N_HEADS, dropout=DROPOUT)
        self.norm1 = nn.LayerNorm(HIDDEN)
        self.norm2 = nn.LayerNorm(HIDDEN)
        self.ff = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, HIDDEN))
        self._last_attn: Optional[torch.Tensor] = None

    def forward(self, node_feat: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x, _ = self.gat1(node_feat, adj)
        x = self.norm1(x)
        x, attn = self.gat2(x, adj)
        self._last_attn = attn.detach()
        x = self.norm2(x + self.ff(x))
        return x


class AgentCommBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.0, tau: float = 2.0, num_time_steps: int = 4) -> None:
        super().__init__()
        self.attn = SpikeMultiheadAttention(dim, heads, dropout=dropout, tau=tau, num_time_steps=num_time_steps)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.output_activation = nn.Identity()

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        q = x
        if key_padding_mask is not None:
            q = q.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        attn_out, _ = self.attn(q, q, q, key_padding_mask=key_padding_mask, need_weights=False)
        if key_padding_mask is not None:
            attn_out = attn_out.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        if key_padding_mask is not None:
            ff_out = ff_out.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        x = self.norm2(x + ff_out)
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return self.output_activation(x)


class BetaActor(nn.Module):
    """Low-level obstacle avoidance / motion executor."""

    def __init__(self) -> None:
        super().__init__()
        self.graph_encoder = LocalGraphEncoder()
        self.ego_mlp = nn.Sequential(nn.Linear(EGO_DIM, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN))
        self.nbr_mlp = nn.Sequential(nn.Linear(NBR_DIM, HIDDEN // 2), nn.ReLU(), nn.Linear(HIDDEN // 2, HIDDEN // 2))
        self.subgoal_pos_encoder = nn.Sequential(
            nn.Linear(10, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN),
        )
        self.goal_condition_proj = nn.Sequential(
            nn.Linear(HIDDEN * 2, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN),
        )
        film_channels = {
            "node_input": NODE_DIM,
            "graph_output": HIDDEN,
            "ego_input": EGO_DIM,
            "ego_output": HIDDEN,
            "local_input": HIDDEN + HIDDEN + HIDDEN // 2,
            "local_output": HIDDEN,
        }
        for layer_idx in range(N_LAYERS):
            film_channels[f"comm_pre_{layer_idx}"] = HIDDEN
            film_channels[f"comm_post_{layer_idx}"] = HIDDEN
        self.subgoal_film = SubgoalFiLM(film_channels)
        self.local_fuse = nn.Sequential(
            nn.Linear(HIDDEN + HIDDEN + HIDDEN // 2, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
        )
        self.comm_blocks = nn.ModuleList(
            [AgentCommBlock(HIDDEN, heads=N_HEADS, dropout=DROPOUT) for _ in range(N_LAYERS)]
        )
        self.temporal_gru = nn.GRUCell(HIDDEN, HIDDEN // 2)
        self._h = None
        self._termination_v = None
        self._prev_subgoal_dist = None
        self.post_fuse = nn.Sequential(nn.Linear(HIDDEN + HIDDEN // 2, HIDDEN // 2), nn.Tanh())
        self.termination_lif = TerminationLIF(ctx_dim=HIDDEN // 2, metric_dim=5)
        self.head = nn.Sequential(nn.Linear(HIDDEN // 2, ACT_DIM * 2), nn.Softplus())
        self.eps = 1e-6
        self.comm_alpha = COMM_ALPHA_INIT

    def reset_temporal(self) -> None:
        self._h = None
        self._termination_v = None
        self._prev_subgoal_dist = None

    def _ensure_step_batch(self, ego, node_feat, adj, nbrs, nbr_mask, subgoal_pos, subgoal_token, agent_mask, option_start):
        added_batch = False
        if ego.dim() == 2:
            added_batch = True
            ego = ego.unsqueeze(0)
            node_feat = node_feat.unsqueeze(0)
            adj = adj.unsqueeze(0)
            nbrs = nbrs.unsqueeze(0)
            nbr_mask = nbr_mask.unsqueeze(0)
            subgoal_pos = subgoal_pos.unsqueeze(0) if subgoal_pos is not None else None
            subgoal_token = subgoal_token.unsqueeze(0) if subgoal_token is not None else None
            if agent_mask is not None and agent_mask.dim() == 1:
                agent_mask = agent_mask.unsqueeze(0)
            if option_start is not None and option_start.dim() == 1:
                option_start = option_start.unsqueeze(0)
        return ego, node_feat, adj, nbrs, nbr_mask, subgoal_pos, subgoal_token, agent_mask, option_start, added_batch

    def _expand_ep_start(self, ep_start, batch_size: int, agent_count: int, device: torch.device) -> torch.Tensor:
        if isinstance(ep_start, bool):
            ep_tensor = torch.full((batch_size, agent_count), ep_start, dtype=torch.bool, device=device)
        elif isinstance(ep_start, torch.Tensor):
            ep_tensor = ep_start.to(device=device, dtype=torch.bool)
            if ep_tensor.dim() == 0:
                ep_tensor = ep_tensor.view(1, 1).expand(batch_size, agent_count)
            elif ep_tensor.dim() == 1:
                ep_tensor = ep_tensor.unsqueeze(-1).expand(-1, agent_count)
        else:
            ep_tensor = torch.zeros((batch_size, agent_count), dtype=torch.bool, device=device)
        return ep_tensor

    def _build_goal_condition(self, ego, subgoal_pos, subgoal_token):
        ego_pos = ego[..., :3]
        ego_vel = ego[..., 3:6]
        if subgoal_pos is None:
            subgoal_pos = torch.zeros(*ego.shape[:-1], 3, device=ego.device, dtype=ego.dtype)
        if subgoal_token is None:
            subgoal_token = torch.zeros(*ego.shape[:-1], HIDDEN, device=ego.device, dtype=ego.dtype)

        delta = subgoal_pos - ego_pos
        subgoal_dist = torch.linalg.norm(delta, dim=-1)
        subgoal_dir = delta / subgoal_dist.unsqueeze(-1).clamp(min=1e-6)
        speed_norm = torch.linalg.norm(ego_vel, dim=-1).clamp(min=1e-6)
        align = (ego_vel * subgoal_dir).sum(dim=-1) / speed_norm
        geometry = ego[..., 14:16]

        goal_pos_features = torch.cat(
            [
                subgoal_pos,
                subgoal_dir,
                subgoal_dist.unsqueeze(-1),
                align.unsqueeze(-1),
                geometry,
            ],
            dim=-1,
        )
        pos_token = self.subgoal_pos_encoder(goal_pos_features)
        goal_condition = self.goal_condition_proj(torch.cat([subgoal_token, pos_token], dim=-1))
        static_metrics = torch.cat([align.unsqueeze(-1), geometry], dim=-1)
        return goal_condition, subgoal_dist, static_metrics

    def _encode(
        self,
        ego,
        node_feat,
        adj,
        nbrs,
        nbr_mask,
        subgoal_pos=None,
        subgoal_token=None,
        agent_mask=None,
        option_start=None,
        ep_start=False,
        training_mode=False,
    ):
        ego, node_feat, adj, nbrs, nbr_mask, subgoal_pos, subgoal_token, agent_mask, option_start, added_batch = self._ensure_step_batch(
            ego,
            node_feat,
            adj,
            nbrs,
            nbr_mask,
            subgoal_pos,
            subgoal_token,
            agent_mask,
            option_start,
        )

        batch_size, agent_count = ego.shape[:2]
        if agent_mask is None:
            agent_mask = torch.zeros((batch_size, agent_count), dtype=torch.bool, device=ego.device)
        if option_start is None:
            option_start = torch.zeros((batch_size, agent_count), dtype=torch.bool, device=ego.device)

        goal_condition, subgoal_dist, static_metrics = self._build_goal_condition(ego, subgoal_pos, subgoal_token)
        ego_flat = ego.reshape(batch_size * agent_count, EGO_DIM)
        node_flat = node_feat.reshape(batch_size * agent_count, LOCAL_TOPK, NODE_DIM)
        adj_flat = adj.reshape(batch_size * agent_count, LOCAL_TOPK, LOCAL_TOPK)
        nbr_flat = nbrs.reshape(batch_size * agent_count, MAX_NBR, NBR_DIM)
        nbr_mask_flat = nbr_mask.reshape(batch_size * agent_count, MAX_NBR)
        goal_flat = goal_condition.reshape(batch_size * agent_count, HIDDEN)

        node_flat = self.subgoal_film.apply("node_input", node_flat, goal_flat)
        graph_nodes = self.graph_encoder(node_flat, adj_flat)
        graph_nodes = self.subgoal_film.apply("graph_output", graph_nodes, goal_flat)
        cur_weights = node_flat[..., 5:6]
        goal_weights = node_flat[..., 6:7]
        cur_weights = cur_weights + 0.5 * goal_weights
        cur_den = cur_weights.sum(dim=1, keepdim=True) + 1e-6
        graph_ctx = (graph_nodes * cur_weights).sum(dim=1) / cur_den.squeeze(1)

        nbr_h = self.nbr_mlp(nbr_flat)
        nbr_h = nbr_h.masked_fill(nbr_mask_flat.unsqueeze(-1), float("-inf"))
        nbr_ctx = nbr_h.max(dim=1).values
        nbr_ctx = torch.where(torch.isinf(nbr_ctx), torch.zeros_like(nbr_ctx), nbr_ctx)

        ego_flat = self.subgoal_film.apply("ego_input", ego_flat, goal_flat)
        ego_ctx = self.ego_mlp(ego_flat)
        ego_ctx = self.subgoal_film.apply("ego_output", ego_ctx, goal_flat)
        local_input = torch.cat([ego_ctx, graph_ctx, nbr_ctx], dim=-1)
        local_input = self.subgoal_film.apply("local_input", local_input, goal_flat)
        local_ctx = self.local_fuse(local_input).view(batch_size, agent_count, HIDDEN)
        local_ctx = self.subgoal_film.apply("local_output", local_ctx, goal_condition)

        if ENABLE_COMM and len(self.comm_blocks) > 0:
            comm_ctx = local_ctx
            for layer_idx, block in enumerate(self.comm_blocks):
                comm_ctx = self.subgoal_film.apply(f"comm_pre_{layer_idx}", comm_ctx, goal_condition)
                comm_ctx = block(comm_ctx, key_padding_mask=agent_mask)
                comm_ctx = self.subgoal_film.apply(f"comm_post_{layer_idx}", comm_ctx, goal_condition)
            alpha = float(np.clip(getattr(self, "comm_alpha", COMM_ALPHA_INIT), 0.0, COMM_ALPHA_MAX))
            ctx_full = local_ctx + alpha * (comm_ctx - local_ctx)
        else:
            ctx_full = local_ctx

        ep_start_mask = self._expand_ep_start(ep_start, batch_size, agent_count, ego.device)
        state_reset = ep_start_mask | option_start

        if (
            training_mode
            or self._h is None
            or self._termination_v is None
            or self._prev_subgoal_dist is None
            or self._h.shape[0] != agent_count
        ):
            h_state = torch.zeros(agent_count, HIDDEN // 2, device=ego.device, dtype=ctx_full.dtype)
            term_v = torch.zeros(agent_count, 1, device=ego.device, dtype=ctx_full.dtype)
            prev_dist = subgoal_dist[0].detach()
        else:
            h_state = self._h.to(device=ego.device, dtype=ctx_full.dtype)
            term_v = self._termination_v.to(device=ego.device, dtype=ctx_full.dtype)
            prev_dist = self._prev_subgoal_dist.to(device=ego.device, dtype=subgoal_dist.dtype)

        h_steps = []
        for t in range(batch_size):
            reset_t = state_reset[t]
            h_state = torch.where(reset_t.unsqueeze(-1), torch.zeros_like(h_state), h_state)
            h_prev = h_state
            h_state = self.temporal_gru(ctx_full[t], h_state)
            h_state = torch.where(agent_mask[t].unsqueeze(-1), h_prev, h_state)
            if not training_mode:
                active_t = (~agent_mask[t]).unsqueeze(-1).expand_as(h_state)
                h_state = torch.where(active_t, h_state + 0.01 * torch.randn_like(h_state), h_state)
            h_steps.append(h_state)

        h_view = torch.stack(h_steps, dim=0)
        ctx = self.post_fuse(torch.cat([ctx_full, h_view], dim=-1))
        ctx = ctx.masked_fill(agent_mask.unsqueeze(-1), 0.0)

        term_probs = []
        for t in range(batch_size):
            reset_t = state_reset[t]
            prev_dist = torch.where(reset_t, subgoal_dist[t].detach(), prev_dist)
            progress = (prev_dist - subgoal_dist[t]).unsqueeze(-1)
            term_metrics = torch.cat([subgoal_dist[t].unsqueeze(-1), progress, static_metrics[t]], dim=-1)
            term_prob_t, _, term_v = self.termination_lif(
                ctx[t],
                term_metrics,
                membrane=term_v,
                reset_mask=reset_t,
                agent_mask=agent_mask[t],
            )
            prev_dist = torch.where(agent_mask[t], prev_dist, subgoal_dist[t].detach())
            term_probs.append(term_prob_t)
        terminate_prob = torch.stack(term_probs, dim=0)

        if not training_mode:
            self._h = h_state.detach()
            self._termination_v = term_v.detach()
            self._prev_subgoal_dist = prev_dist.detach()

        if added_batch:
            ctx = ctx.squeeze(0)
            terminate_prob = terminate_prob.squeeze(0)
        return ctx, terminate_prob

    def forward(
        self,
        ego,
        node_feat,
        adj,
        nbrs,
        nbr_mask,
        subgoal_pos=None,
        subgoal_token=None,
        agent_mask=None,
        option_start=None,
        ep_start=False,
        training_mode=False,
    ):
        ctx, terminate_prob = self._encode(
            ego,
            node_feat,
            adj,
            nbrs,
            nbr_mask,
            subgoal_pos=subgoal_pos,
            subgoal_token=subgoal_token,
            agent_mask=agent_mask,
            option_start=option_start,
            ep_start=ep_start,
            training_mode=training_mode,
        )
        params = self.head(ctx).reshape(*ctx.shape[:-1], ACT_DIM, 2) + self.eps
        return params[..., 0], params[..., 1], terminate_prob

    def sample_policy(
        self,
        ego,
        node_feat,
        adj,
        nbrs,
        nbr_mask,
        subgoal_pos=None,
        subgoal_token=None,
        agent_mask=None,
        option_start=None,
        deterministic=False,
        ep_start=False,
        training_mode=False,
    ):
        alpha, beta, terminate_prob = self.forward(
            ego,
            node_feat,
            adj,
            nbrs,
            nbr_mask,
            subgoal_pos=subgoal_pos,
            subgoal_token=subgoal_token,
            agent_mask=agent_mask,
            option_start=option_start,
            ep_start=ep_start,
            training_mode=training_mode,
        )
        dist = torch.distributions.Beta(alpha, beta)
        if deterministic:
            sample = alpha / (alpha + beta)
            log_prob = torch.zeros_like(alpha[..., 0])
        else:
            sample = dist.rsample()
            log_prob = dist.log_prob(sample).sum(-1)

        terminate_prob = terminate_prob.clamp(self.eps, 1.0 - self.eps)
        term_dist = torch.distributions.Bernoulli(probs=terminate_prob)
        if deterministic:
            terminate_action = terminate_prob > 0.5
            terminate_log_prob = torch.zeros_like(terminate_prob)
        else:
            terminate_action = term_dist.sample().bool()
            terminate_log_prob = term_dist.log_prob(terminate_action.float())

        action = (sample * 2 - 1) * MAX_ACC
        if agent_mask is not None:
            action = action.masked_fill(agent_mask.unsqueeze(-1), 0.0)
            log_prob = log_prob.masked_fill(agent_mask, 0.0)
            terminate_log_prob = terminate_log_prob.masked_fill(agent_mask, 0.0)
            terminate_prob = terminate_prob.masked_fill(agent_mask, 0.0)
            terminate_action = terminate_action.masked_fill(agent_mask, False)
        return action, log_prob, terminate_action, terminate_log_prob, terminate_prob

    def get_action(self, *args, **kwargs):
        action, log_prob, _, _, _ = self.sample_policy(*args, **kwargs)
        return action, log_prob

    def evaluate_policy(
        self,
        ego,
        node_feat,
        adj,
        nbrs,
        nbr_mask,
        action,
        terminate_action,
        subgoal_pos=None,
        subgoal_token=None,
        agent_mask=None,
        option_start=None,
        ep_start=False,
        training_mode=False,
    ):
        alpha, beta, terminate_prob = self.forward(
            ego,
            node_feat,
            adj,
            nbrs,
            nbr_mask,
            subgoal_pos=subgoal_pos,
            subgoal_token=subgoal_token,
            agent_mask=agent_mask,
            option_start=option_start,
            ep_start=ep_start,
            training_mode=training_mode,
        )
        dist = torch.distributions.Beta(alpha, beta)
        sample = (action / MAX_ACC + 1) / 2
        sample = sample.clamp(self.eps, 1 - self.eps)
        action_log_prob = dist.log_prob(sample).sum(-1)
        action_entropy = dist.entropy().sum(-1)

        terminate_prob = terminate_prob.clamp(self.eps, 1.0 - self.eps)
        term_dist = torch.distributions.Bernoulli(probs=terminate_prob)
        term_action = terminate_action.float()
        term_log_prob = term_dist.log_prob(term_action)
        term_entropy = term_dist.entropy()
        if agent_mask is not None:
            action_log_prob = action_log_prob.masked_fill(agent_mask, 0.0)
            action_entropy = action_entropy.masked_fill(agent_mask, 0.0)
            term_log_prob = term_log_prob.masked_fill(agent_mask, 0.0)
            term_entropy = term_entropy.masked_fill(agent_mask, 0.0)
        return action_log_prob, action_entropy, term_log_prob, term_entropy

    def evaluate_action(self, ego, node_feat, adj, nbrs, nbr_mask, action, agent_mask=None, ep_start=False, training_mode=False):
        action_log_prob, action_entropy, _, _ = self.evaluate_policy(
            ego,
            node_feat,
            adj,
            nbrs,
            nbr_mask,
            action,
            terminate_action=torch.zeros(action.shape[:-1], device=action.device),
            agent_mask=agent_mask,
            ep_start=ep_start,
            training_mode=training_mode,
        )
        return action_log_prob, action_entropy

    def graph_attn_weights(self) -> Optional[np.ndarray]:
        attn = self.graph_encoder._last_attn
        if attn is None:
            return None
        return attn[0].mean(dim=-1).cpu().numpy()
