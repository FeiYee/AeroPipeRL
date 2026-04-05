from __future__ import annotations

import torch
import torch.nn as nn

from aeropipe_rl.algorithms.obstacle_avoidance import GraphAttentionLayer
from aeropipe_rl.config import DROPOUT, EGO_DIM, HIDDEN, MAX_N_NODES, MAX_SPEED, N_HEADS, NODE_DIM, TIME_WINDOW_SIZE


class PlannerActor(nn.Module):
    """High-level path planner that proposes the next graph node per UAV."""

    def __init__(self) -> None:
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(NODE_DIM, HIDDEN // 2),
            nn.ReLU(),
            nn.Linear(HIDDEN // 2, HIDDEN // 2),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(4, HIDDEN // 4),
            nn.ReLU(),
            nn.Linear(HIDDEN // 4, HIDDEN // 4),
        )
        self.agent_state_encoder = nn.Sequential(
            nn.Linear(EGO_DIM + MAX_N_NODES + 1, HIDDEN // 2),
            nn.ReLU(),
            nn.Linear(HIDDEN // 2, HIDDEN // 2),
        )
        self.graph_attn = GraphAttentionLayer(
            in_dim=HIDDEN // 2 + HIDDEN // 4,
            out_dim=HIDDEN // 4,
            heads=N_HEADS,
            dropout=DROPOUT,
        )
        self.head = nn.Sequential(
            nn.Linear(HIDDEN + HIDDEN // 2, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, MAX_N_NODES + TIME_WINDOW_SIZE + 2),
        )

    def forward(self, node_feat, edge_feats, agent_ego_feats, adj, cur_node_ids, existing_route_onehot=None, plan_mask=None):
        batch_size = node_feat.shape[0]
        agent_count = agent_ego_feats.shape[1]

        node_h = self.node_encoder(node_feat)
        edge_h = self.edge_encoder(edge_feats)
        edge_h_agg = edge_h.mean(dim=2)
        graph_input = torch.cat([node_h, edge_h_agg], dim=-1)
        graph_h, _ = self.graph_attn(graph_input, adj)

        if existing_route_onehot is None:
            existing_route_onehot = torch.zeros(batch_size, agent_count, MAX_N_NODES, device=node_feat.device)
        if plan_mask is None:
            plan_mask = torch.zeros(batch_size, agent_count, 1, device=node_feat.device)
        else:
            plan_mask = plan_mask.unsqueeze(-1)

        agent_state_input = torch.cat([agent_ego_feats, existing_route_onehot, plan_mask], dim=-1)
        agent_h = self.agent_state_encoder(agent_state_input)

        global_context = torch.cat(
            [graph_h.mean(dim=1).unsqueeze(1).expand(-1, agent_count, -1), agent_h],
            dim=-1,
        )
        output = self.head(global_context)

        next_node_logits = output[..., :MAX_N_NODES]
        time_slot_logits = output[..., MAX_N_NODES : MAX_N_NODES + TIME_WINDOW_SIZE]
        speed_ref = torch.tanh(output[..., MAX_N_NODES + TIME_WINDOW_SIZE]) * MAX_SPEED
        wait_prob = torch.sigmoid(output[..., MAX_N_NODES + TIME_WINDOW_SIZE + 1])
        return next_node_logits, time_slot_logits, speed_ref, wait_prob


class AdversaryActor(nn.Module):
    """Adversarial congestion generator for robust planner training."""

    def __init__(self) -> None:
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(NODE_DIM + 4 + EGO_DIM, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
        )
        self.head = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN // 2),
            nn.ReLU(),
            nn.Linear(HIDDEN // 2, MAX_N_NODES),
        )

    def forward(self, node_feat, edge_feats, agent_ego_feats):
        _, node_count = node_feat.shape[:2]
        agent_count = agent_ego_feats.shape[1]

        edge_feat_agg = edge_feats.mean(dim=2).unsqueeze(1).expand(-1, agent_count, -1, -1)
        node_feat_exp = node_feat.unsqueeze(1).expand(-1, agent_count, -1, -1)
        agent_feat_exp = agent_ego_feats.unsqueeze(2).expand(-1, -1, node_count, -1)

        state_input = torch.cat([node_feat_exp, edge_feat_agg, agent_feat_exp], dim=-1)
        state_h = self.state_encoder(state_input)
        node_scores = self.head(state_h.mean(dim=1))

        capacity_compress = torch.sigmoid(node_scores.mean(dim=-1, keepdim=True)).unsqueeze(1)
        capacity_compress = capacity_compress.expand(-1, node_count, -1, -1)
        return capacity_compress
