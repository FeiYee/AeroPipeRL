from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from aeropipe_rl.algorithms.hierarchy import DifferentiablePathFlowLayer
from aeropipe_rl.algorithms.obstacle_avoidance import GraphAttentionLayer
from aeropipe_rl.config import (
    DROPOUT,
    EGO_DIM,
    HIDDEN,
    MAX_N_EDGES,
    MAX_N_NODES,
    MAX_SPEED,
    N_HEADS,
    NODE_DIM,
    TIME_WINDOW_SIZE,
    ENABLE_GLOBAL_COLLISION_PENALTY,
    FLOW_PENALTY_COEF,
)


class PlannerActor(nn.Module):
    """Flow-aware high-level planner that produces differentiable route distributions."""

    def __init__(self) -> None:
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(NODE_DIM, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(4, HIDDEN // 2),
            nn.ReLU(),
            nn.Linear(HIDDEN // 2, HIDDEN // 2),
        )
        self.agent_state_encoder = nn.Sequential(
            nn.Linear(EGO_DIM + MAX_N_NODES * 3 + 1, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
        )
        self.graph_attn = GraphAttentionLayer(
            in_dim=HIDDEN + HIDDEN // 2,
            out_dim=HIDDEN // N_HEADS,
            heads=N_HEADS,
            dropout=DROPOUT,
        )
        self.edge_flow_head = nn.Sequential(
            nn.Linear(HIDDEN * 3 + HIDDEN // 2, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, 1),
        )
        self.subgoal_proj = nn.Sequential(
            nn.Linear(HIDDEN * 2, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN),
        )
        self.aux_head = nn.Sequential(
            nn.Linear(HIDDEN * 2, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, TIME_WINDOW_SIZE + 2),
        )
        self.flow_layer = DifferentiablePathFlowLayer()

    def forward(
        self,
        node_feat: torch.Tensor,
        edge_feats: torch.Tensor,
        agent_ego_feats: torch.Tensor,
        adj: torch.Tensor,
        cur_node_ids: torch.Tensor,
        goal_node_ids: torch.Tensor,
        existing_route_onehot: torch.Tensor | None = None,
        plan_mask: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size, node_count = node_feat.shape[:2]
        agent_count = agent_ego_feats.shape[1]

        if node_mask is None:
            node_mask = adj.diagonal(dim1=-2, dim2=-1) > 0
        node_mask_f = node_mask.to(dtype=node_feat.dtype)

        node_h = self.node_encoder(node_feat)
        edge_h = self.edge_encoder(edge_feats)
        edge_h_agg = edge_h.mean(dim=2)
        graph_input = torch.cat([node_h, edge_h_agg], dim=-1)
        graph_h, _ = self.graph_attn(graph_input, adj)
        graph_h = graph_h * node_mask_f.unsqueeze(-1)

        if existing_route_onehot is None:
            existing_route_onehot = torch.zeros(batch_size, agent_count, MAX_N_NODES, device=node_feat.device)
        if plan_mask is None:
            plan_mask = torch.zeros(batch_size, agent_count, 1, device=node_feat.device)
        else:
            plan_mask = plan_mask.unsqueeze(-1).to(dtype=node_feat.dtype)

        cur_onehot = F.one_hot(cur_node_ids, num_classes=node_count).to(dtype=node_feat.dtype)
        goal_onehot = F.one_hot(goal_node_ids, num_classes=node_count).to(dtype=node_feat.dtype)
        agent_state_input = torch.cat(
            [agent_ego_feats, cur_onehot, goal_onehot, existing_route_onehot, plan_mask],
            dim=-1,
        )
        agent_h = self.agent_state_encoder(agent_state_input)

        src_h = graph_h.unsqueeze(1).unsqueeze(3).expand(-1, agent_count, -1, node_count, -1)
        dst_h = graph_h.unsqueeze(1).unsqueeze(2).expand(-1, agent_count, node_count, -1, -1)
        edge_h_agent = edge_h.unsqueeze(1).expand(-1, agent_count, -1, -1, -1)
        agent_h_exp = agent_h.unsqueeze(2).unsqueeze(3).expand(-1, -1, node_count, node_count, -1)
        edge_input = torch.cat([src_h, dst_h, edge_h_agent, agent_h_exp], dim=-1)
        edge_flow_scores = self.edge_flow_head(edge_input).squeeze(-1)

        valid_edges = (adj.unsqueeze(1) > 0)

        # ========== 新增：全局流量约束，避免多智能体路径冲突 ==========
        if ENABLE_GLOBAL_COLLISION_PENALTY and existing_route_onehot is not None:
            batch_size, agent_count = existing_route_onehot.shape[:2]
            # 计算每个节点的全局占用率（所有智能体已选路径的节点概率之和）
            node_occupancy = existing_route_onehot.sum(dim=1)  # [batch, MAX_N_NODES]
            # 扩展维度到和edge_flow_scores匹配：[batch, agent_count, MAX_N_NODES, MAX_N_NODES]
            # src_occupancy是每个边起点的占用率：[batch, agent_count, src_node, dst_node]
            src_occupancy = node_occupancy.unsqueeze(1).unsqueeze(-1).expand(-1, agent_count, -1, MAX_N_NODES)
            edge_occupancy = src_occupancy * adj.unsqueeze(1)  # 邻接矩阵是有向的，仅保留存在的边
            # 占用越多的边，分数越低，引导智能体选择空闲路径
            edge_flow_scores = edge_flow_scores - FLOW_PENALTY_COEF * edge_occupancy
        # ============================================================

        edge_flow_scores = edge_flow_scores.masked_fill(~valid_edges, -1e4)
        flow_out = self.flow_layer(
            edge_flow_scores.view(batch_size, agent_count, MAX_N_EDGES),
            adj,
            node_mask,
            cur_node_ids,
            goal_node_ids,
        )

        node_visit_probs = flow_out["node_visit_probs"]
        subgoal_node_probs = flow_out["subgoal_node_probs"]
        node_positions = node_feat[..., :3]
        subgoal_position = (subgoal_node_probs.unsqueeze(-1) * node_positions.unsqueeze(1)).sum(dim=2)

        path_context = (node_visit_probs.unsqueeze(-1) * graph_h.unsqueeze(1)).sum(dim=2)
        subgoal_token = self.subgoal_proj(torch.cat([agent_h, path_context], dim=-1))

        graph_den = node_mask_f.sum(dim=1, keepdim=True).clamp(min=1.0).unsqueeze(-1)
        global_graph = (graph_h * node_mask_f.unsqueeze(-1)).sum(dim=1, keepdim=True) / graph_den
        global_graph = global_graph.expand(-1, agent_count, -1)
        aux_out = self.aux_head(torch.cat([global_graph, subgoal_token], dim=-1))

        time_slot_logits = aux_out[..., :TIME_WINDOW_SIZE]
        speed_ref = torch.tanh(aux_out[..., TIME_WINDOW_SIZE]) * MAX_SPEED
        wait_prob = torch.sigmoid(aux_out[..., TIME_WINDOW_SIZE + 1])

        # ========== 新增：子目标约束参数 ==========
        # 子目标预计到达时间（步）：speed_ref越大，deadline越短
        subgoal_deadline = torch.clamp(speed_ref * 20.0, min=20.0, max=200.0)
        # 子目标优先级：0-1，越高越优先
        subgoal_priority = torch.sigmoid(time_slot_logits.mean(dim=-1))
        # 允许偏离最大距离（米）：3-5米
        subgoal_tolerance = 3.0 + 2.0 * torch.sigmoid(wait_prob)
        # ==========================================

        return {
            "edge_flow_weights": edge_flow_scores.view(batch_size, agent_count, MAX_N_EDGES),
            "edge_path_probs": flow_out["edge_path_probs"],
            "transition_probs": flow_out["transition_probs"],
            "node_visit_probs": node_visit_probs,
            "subgoal_node_probs": subgoal_node_probs,
            "subgoal_position": subgoal_position,
            "subgoal_token": subgoal_token,
            "subgoal_deadline": subgoal_deadline,  # 新增
            "subgoal_priority": subgoal_priority,  # 新增
            "subgoal_tolerance": subgoal_tolerance,  # 新增
            "time_slot_logits": time_slot_logits,
            "speed_ref": speed_ref,
            "wait_prob": wait_prob,
        }

    def evaluate_action(
        self,
        node_feat: torch.Tensor,
        edge_feats: torch.Tensor,
        agent_ego_feats: torch.Tensor,
        adj: torch.Tensor,
        cur_node_ids: torch.Tensor,
        goal_node_ids: torch.Tensor,
        subgoal_nodes: torch.Tensor,
        existing_route_onehot: torch.Tensor | None = None,
        plan_mask: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        outputs = self.forward(
            node_feat,
            edge_feats,
            agent_ego_feats,
            adj,
            cur_node_ids,
            goal_node_ids,
            existing_route_onehot=existing_route_onehot,
            plan_mask=plan_mask,
            node_mask=node_mask,
        )
        dist = torch.distributions.Categorical(outputs["subgoal_node_probs"])
        log_prob = dist.log_prob(subgoal_nodes)
        entropy = dist.entropy()
        return outputs, log_prob, entropy


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
