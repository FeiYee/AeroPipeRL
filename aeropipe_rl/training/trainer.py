from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from aeropipe_rl.config import (
    CLIP_EPS,
    COMM_ALPHA_INIT,
    COMM_ALPHA_MAX,
    DEVICE,
    ENABLE_COMM,
    ENT_COEF,
    ENT_COEF_MIN,
    ENT_COEF_DECAY,
    EGO_DIM,
    GRAD_NORM,
    HIDDEN,
    K_EPOCHS,
    LOW_LEVEL_GAE_LAMBDA,
    LOW_LEVEL_GAMMA,
    LR_ACTOR,
    LR_CRITIC,
    MINI_BATCH,
    N_AGENTS,
    OPTION_GAE_LAMBDA,
    OPTION_GAMMA,
    PLANNER_GAE_LAMBDA,
    PLANNER_GAMMA,
    TERMINATION_THRESHOLD,
    VAL_COEF,
    ENABLE_GLOBAL_COLLISION_PENALTY,
    CONFLICT_LOSS_COEF,
)
from aeropipe_rl.models.policy import MARLPolicy
from aeropipe_rl.training.buffer import RolloutBuffer


class MAPPOTrainer:
    def __init__(self) -> None:
        self.policy = MARLPolicy()
        self.opt_exec = optim.Adam(self.policy.executor.parameters(), lr=LR_ACTOR)
        self.opt_planner = optim.Adam(self.policy.planner.parameters(), lr=LR_ACTOR * 0.5)
        self.opt_adversary = optim.Adam(self.policy.adversary.parameters(), lr=LR_ACTOR * 0.3)
        self.opt_c = optim.Adam(self.policy.critic.parameters(), lr=LR_CRITIC)
        self.buf = RolloutBuffer()

        self.ep = 0
        self.total_t = 0
        self.update_cnt = 0
        self._cur_n_agents = N_AGENTS   # tracks curriculum agent count

        self.r_hist = deque(maxlen=500)
        self.loss_hist = deque(maxlen=500)
        self.sr_hist = deque(maxlen=50)
        self.r50 = deque(maxlen=50)
        self.stp50 = deque(maxlen=50)
        self.goal_step_avg50 = deque(maxlen=50)
        self.goal_step_med50 = deque(maxlen=50)
        self.goal_step_std50 = deque(maxlen=50)
        self.budget50 = deque(maxlen=50)
        self.wall_rate100 = deque(maxlen=100)
        self.agent_col_rate100 = deque(maxlen=100)
        self.timeout20_rate100 = deque(maxlen=100)
        self.timeout50_rate100 = deque(maxlen=100)
        self.timeout100_rate100 = deque(maxlen=100)
        self.collision_rate_ma100 = deque(maxlen=100)
        self.time_gauss_pen100 = deque(maxlen=100)
        self.best_score = -1e9
        self._ent_coef = ENT_COEF

        self.active_subgoal_pos = np.zeros((N_AGENTS, 3), dtype=np.float32)
        self.active_subgoal_token = np.zeros((N_AGENTS, HIDDEN), dtype=np.float32)
        self.active_subgoal_node = np.zeros(N_AGENTS, dtype=np.int64)
        self.has_active_option = np.zeros(N_AGENTS, dtype=bool)
        self.pending_replan = np.ones(N_AGENTS, dtype=bool)
        # ========== 新增：子目标约束参数存储 ==========
        self.subgoal_deadline = np.zeros(N_AGENTS, dtype=np.float32)
        self.subgoal_priority = np.zeros(N_AGENTS, dtype=np.float32)
        self.subgoal_tolerance = np.zeros(N_AGENTS, dtype=np.float32)
        self.subgoal_elapsed_steps = np.zeros(N_AGENTS, dtype=np.int32)
        # ============================================

    def set_n_agents(self, n: int) -> None:
        """Curriculum: update active agent count. Resets per-agent state."""
        self._cur_n_agents = n
        self.active_subgoal_pos = np.zeros((N_AGENTS, 3), dtype=np.float32)
        self.active_subgoal_token = np.zeros((N_AGENTS, HIDDEN), dtype=np.float32)
        self.active_subgoal_node = np.zeros(N_AGENTS, dtype=np.int64)
        self.has_active_option = np.zeros(N_AGENTS, dtype=bool)
        self.pending_replan = np.ones(N_AGENTS, dtype=bool)
        # ========== 新增：子目标约束参数重置 ==========
        self.subgoal_deadline = np.zeros(N_AGENTS, dtype=np.float32)
        self.subgoal_priority = np.zeros(N_AGENTS, dtype=np.float32)
        self.subgoal_tolerance = np.zeros(N_AGENTS, dtype=np.float32)
        self.subgoal_elapsed_steps = np.zeros(N_AGENTS, dtype=np.int32)
        # ============================================

    def reset_episode(self) -> None:
        self.policy.executor.reset_temporal()
        self.active_subgoal_pos.fill(0.0)
        self.active_subgoal_token.fill(0.0)
        self.active_subgoal_node.fill(0)
        self.has_active_option[:] = False
        self.pending_replan[:] = True
        # ========== 新增：子目标约束参数重置 ==========
        self.subgoal_deadline.fill(0.0)
        self.subgoal_priority.fill(0.0)
        self.subgoal_tolerance.fill(0.0)
        self.subgoal_elapsed_steps.fill(0)
        # ============================================

    def after_step(self, dones: np.ndarray) -> None:
        done_mask = np.asarray(dones, dtype=bool)
        n = len(done_mask)
        self.pending_replan[:n][done_mask] = False
        self.has_active_option[:n][done_mask] = False
        self.active_subgoal_pos[:n][done_mask] = 0.0
        self.active_subgoal_token[:n][done_mask] = 0.0
        self.active_subgoal_node[:n][done_mask] = 0
        # ========== 新增：完成的智能体重置子目标约束 ==========
        self.subgoal_deadline[:n][done_mask] = 0.0
        self.subgoal_priority[:n][done_mask] = 0.0
        self.subgoal_tolerance[:n][done_mask] = 0.0
        self.subgoal_elapsed_steps[:n][done_mask] = 0
        # ============================================

    def _critic_global_from_egos(self, egos, dones_mask=None, n_agents=None):
        """
        Build global obs tensor padded to N_AGENTS for critic.
        Curriculum: active agents fill first n_agents*EGO_DIM entries; rest = 0.
        """
        n = n_agents if n_agents is not None else len(egos)
        global_obs = np.zeros(EGO_DIM * N_AGENTS, dtype=np.float32)
        for i, ego in enumerate(egos[:n]):
            if dones_mask is not None and dones_mask[i]:
                continue
            global_obs[i * EGO_DIM:(i + 1) * EGO_DIM] = ego
        return global_obs

    def _normalize_advantages(self, adv: np.ndarray, mask: np.ndarray) -> np.ndarray:
        valid = mask > 0.5
        if not np.any(valid):
            return adv
        mean = float(np.mean(adv[valid]))
        std = float(np.std(adv[valid]) + 1e-6)
        out = adv.copy()
        out[valid] = (out[valid] - mean) / std
        return out

    def _compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        active_masks: np.ndarray,
        terminal_mask: np.ndarray,
        ep_start: np.ndarray,
        last_value: np.ndarray,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        rollout_len = rewards.shape[0]
        adv = np.zeros_like(rewards)
        ret = np.zeros_like(rewards)
        gae = np.zeros(rewards.shape[1], dtype=np.float32)
        v_next = last_value.copy()

        for t in reversed(range(rollout_len)):
            if t + 1 < rollout_len and bool(ep_start[t + 1]):
                gae[:] = 0.0
                v_next[:] = 0.0
            terminal = terminal_mask[t].astype(np.float32)
            delta = rewards[t] + gamma * v_next * (1.0 - terminal) - values[t]
            gae = delta + gamma * gae_lambda * (1.0 - terminal) * gae
            gae *= active_masks[t]
            adv[t] = gae
            ret[t] = gae + values[t]
            v_next = values[t]
        return adv, ret

    @torch.no_grad()
    def act(
        self,
        egos,
        node_feats,
        adjs,
        nbrs_list,
        nbr_masks,
        global_obs,
        env,
        agent_mask=None,
        deterministic=False,
        ep_start=False,
    ):
        n = len(egos)
        if agent_mask is None:
            agent_mask = np.zeros(n, dtype=bool)

        ego_tensor = torch.FloatTensor(np.stack(egos)).unsqueeze(0).to(DEVICE)
        node_tensor = torch.FloatTensor(np.stack(node_feats)).unsqueeze(0).to(DEVICE)
        adj_tensor = torch.FloatTensor(np.stack(adjs)).unsqueeze(0).to(DEVICE)
        nbr_tensor = torch.FloatTensor(np.stack(nbrs_list)).unsqueeze(0).to(DEVICE)
        nbr_mask_tensor = torch.BoolTensor(np.stack(nbr_masks)).unsqueeze(0).to(DEVICE)
        agent_mask_tensor = torch.BoolTensor(agent_mask).unsqueeze(0).to(DEVICE)
        clean_global = self._critic_global_from_egos(egos, dones_mask=agent_mask, n_agents=n)
        global_tensor = torch.FloatTensor(clean_global).unsqueeze(0).to(DEVICE)

        critic_heads = self.policy.critic(global_tensor, head="all")
        step_value = np.full(n, float(critic_heads["step"].item()), dtype=np.float32)
        option_value = np.full(n, float(critic_heads["option"].item()), dtype=np.float32)
        planner_value = np.full(n, float(critic_heads["planner"].item()), dtype=np.float32)
        step_value[agent_mask] = 0.0
        option_value[agent_mask] = 0.0
        planner_value[agent_mask] = 0.0

        option_start = self.pending_replan[:n] & (~agent_mask)
        planner_action = np.zeros(n, dtype=np.int64)
        planner_log_prob = np.zeros(n, dtype=np.float32)
        planner_reward = np.zeros(n, dtype=np.float32)
        planner_active_mask = option_start.copy()

        planner_state = env.get_planner_state()
        planner_node_tensor = torch.FloatTensor(planner_state["node_feat"]).unsqueeze(0).to(DEVICE)
        planner_edge_tensor = torch.FloatTensor(planner_state["edge_feat"]).unsqueeze(0).to(DEVICE)
        planner_adj_tensor = torch.FloatTensor(planner_state["adj"]).unsqueeze(0).to(DEVICE)
        planner_node_mask_tensor = torch.BoolTensor(planner_state["node_mask"]).unsqueeze(0).to(DEVICE)
        planner_cur_nodes = planner_state["cur_node_ids"][:n]
        planner_goal_nodes = planner_state["goal_node_ids"][:n]
        planner_route_onehot = planner_state["route_onehot"][:n]
        planner_cur_nodes_tensor = torch.LongTensor(planner_cur_nodes).unsqueeze(0).to(DEVICE)
        planner_goal_nodes_tensor = torch.LongTensor(planner_goal_nodes).unsqueeze(0).to(DEVICE)
        planner_route_tensor = torch.FloatTensor(planner_route_onehot).unsqueeze(0).to(DEVICE)
        plan_mask_tensor = torch.BoolTensor(option_start).unsqueeze(0).to(DEVICE)

        if np.any(planner_active_mask):
            planner_out = self.policy.planner(
                planner_node_tensor,
                planner_edge_tensor,
                ego_tensor,
                planner_adj_tensor,
                planner_cur_nodes_tensor,
                planner_goal_nodes_tensor,
                existing_route_onehot=planner_route_tensor,
                plan_mask=plan_mask_tensor,
                node_mask=planner_node_mask_tensor,
            )
            planner_dist = torch.distributions.Categorical(planner_out["subgoal_node_probs"])
            if deterministic:
                sampled_nodes = planner_out["subgoal_node_probs"].argmax(dim=-1)
            else:
                sampled_nodes = planner_dist.sample()
            sampled_lp = planner_dist.log_prob(sampled_nodes)

            sampled_nodes_np = sampled_nodes.squeeze(0).cpu().numpy()
            sampled_lp_np = sampled_lp.squeeze(0).cpu().numpy()
            subgoal_pos_np = planner_out["subgoal_position"].squeeze(0).cpu().numpy()
            subgoal_tok_np = planner_out["subgoal_token"].squeeze(0).cpu().numpy()
            transition_np = planner_out["transition_probs"].squeeze(0).cpu().numpy()
            route_gain = np.zeros(n, dtype=np.float32)

            subgoal_deadline_np = planner_out["subgoal_deadline"].squeeze(0).cpu().numpy()
            subgoal_priority_np = planner_out["subgoal_priority"].squeeze(0).cpu().numpy()
            subgoal_tolerance_np = planner_out["subgoal_tolerance"].squeeze(0).cpu().numpy()

            for agent_id in np.where(planner_active_mask)[0]:
                planner_action[agent_id] = int(sampled_nodes_np[agent_id])
                planner_log_prob[agent_id] = float(sampled_lp_np[agent_id])
                self.active_subgoal_node[agent_id] = planner_action[agent_id]
                self.active_subgoal_pos[agent_id] = subgoal_pos_np[agent_id]
                self.active_subgoal_token[agent_id] = subgoal_tok_np[agent_id]
                # ========== 新增：存储子目标约束参数 ==========
                self.subgoal_deadline[agent_id] = float(subgoal_deadline_np[agent_id])
                self.subgoal_priority[agent_id] = float(subgoal_priority_np[agent_id])
                self.subgoal_tolerance[agent_id] = float(subgoal_tolerance_np[agent_id])
                self.subgoal_elapsed_steps[agent_id] = 0  # 重置计时
                # ============================================
                route_gain[agent_id] = env.apply_flow_subgoal(
                    agent_id,
                    transition_np[agent_id],
                    planner_action[agent_id],
                    planner_state["node_ids"],
                )
                self.has_active_option[agent_id] = True
            flow_eff = env.flow_efficiency_score()
            planner_reward[planner_active_mask] = flow_eff + route_gain[planner_active_mask]

        subgoal_pos_tensor = torch.FloatTensor(self.active_subgoal_pos[:n]).unsqueeze(0).to(DEVICE)
        subgoal_tok_tensor = torch.FloatTensor(self.active_subgoal_token[:n]).unsqueeze(0).to(DEVICE)
        option_start_tensor = torch.BoolTensor(option_start).unsqueeze(0).to(DEVICE)

        actions, action_log_probs, terminate_action, terminate_log_prob, terminate_prob = self.policy.executor.sample_policy(
            ego_tensor,
            node_tensor,
            adj_tensor,
            nbr_tensor,
            nbr_mask_tensor,
            subgoal_pos=subgoal_pos_tensor,
            subgoal_token=subgoal_tok_tensor,
            agent_mask=agent_mask_tensor,
            option_start=option_start_tensor,
            deterministic=deterministic,
            ep_start=ep_start,
            training_mode=False,
        )

        termination_active_mask = self.has_active_option[:n] & (~option_start) & (~agent_mask)
        terminate_prob_np = terminate_prob.squeeze(0).cpu().numpy()
        terminate_action_np = terminate_prob_np >= TERMINATION_THRESHOLD
        terminate_log_prob_np = np.where(
            terminate_action_np,
            np.log(np.clip(terminate_prob_np, 1e-6, 1.0)),
            np.log(np.clip(1.0 - terminate_prob_np, 1e-6, 1.0)),
        ).astype(np.float32)
        terminate_action_np[~termination_active_mask] = False
        terminate_log_prob_np[~termination_active_mask] = 0.0
        terminate_prob_np[~termination_active_mask] = 0.0
        self.pending_replan[:n] = terminate_action_np[:n] & termination_active_mask

        act_np = actions.squeeze(0).cpu().numpy()
        action_log_prob_np = action_log_probs.squeeze(0).cpu().numpy()
        act_np[agent_mask] = 0.0
        action_log_prob_np[agent_mask] = 0.0

        return {
            "actions": act_np,
            "action_log_probs": action_log_prob_np,
            "step_values": step_value,
            "option_values": option_value,
            "planner_values": planner_value,
            "subgoal_positions": self.active_subgoal_pos[:n].copy(),
            "subgoal_tokens": self.active_subgoal_token[:n].copy(),
            "option_start": option_start.copy(),
            "termination_actions": terminate_action_np,
            "termination_log_probs": terminate_log_prob_np,
            "termination_probs": terminate_prob_np,
            "termination_active_masks": termination_active_mask.copy(),
            "planner_node_feats": planner_state["node_feat"].copy(),
            "planner_edge_feats": planner_state["edge_feat"].copy(),
            "planner_adjs": planner_state["adj"].copy(),
            "planner_node_masks": planner_state["node_mask"].copy(),
            "planner_cur_nodes": planner_cur_nodes.copy(),
            "planner_goal_nodes": planner_goal_nodes.copy(),
            "planner_route_onehot": planner_route_onehot.copy(),
            "planner_actions": planner_action,
            "planner_log_probs": planner_log_prob,
            "planner_rewards": planner_reward,
            "planner_active_masks": planner_active_mask,
        }

    def update(self, last_egos, last_dones, n_agents=None) -> None:
        n = n_agents if n_agents is not None else self._cur_n_agents
        rollout_len = len(self.buf)
        if rollout_len == 0:
            return

        with torch.no_grad():
            clean_last_g = self._critic_global_from_egos(last_egos, dones_mask=last_dones, n_agents=n)
            global_tensor = torch.FloatTensor(clean_last_g).unsqueeze(0).to(DEVICE)
            last_heads = self.policy.critic(global_tensor, head="all")
            last_step_v = np.full(N_AGENTS, float(last_heads["step"].item()), dtype=np.float32)
            last_option_v = np.full(N_AGENTS, float(last_heads["option"].item()), dtype=np.float32)
            last_planner_v = np.full(N_AGENTS, float(last_heads["planner"].item()), dtype=np.float32)
        alive_mask = (~np.asarray(last_dones, dtype=bool)).astype(np.float32)
        # Pad to N_AGENTS
        alive_full = np.zeros(N_AGENTS, dtype=np.float32)
        alive_full[:len(alive_mask)] = alive_mask
        last_step_v *= alive_full
        last_option_v *= alive_full
        last_planner_v *= alive_full

        rewards = np.array(self.buf.rewards, dtype=np.float32)        # (T, n)
        dones = np.array(self.buf.dones, dtype=np.float32)
        masks = np.array(self.buf.active_masks, dtype=np.float32)
        ep_start = np.array(self.buf.ep_start, dtype=bool)

        # Pad to N_AGENTS on agent axis for GAE (critic is N_AGENTS wide)
        T = rewards.shape[0]
        cur_n = rewards.shape[1]

        def pad_agent(arr, fill=0.0):
            """Pad 2D (T, cur_n) → (T, N_AGENTS)."""
            if arr.shape[1] == N_AGENTS:
                return arr
            out = np.full((T, N_AGENTS), fill, dtype=arr.dtype)
            out[:, :cur_n] = arr
            return out

        rewards_p = pad_agent(rewards)
        dones_p = pad_agent(dones)
        masks_p = pad_agent(masks)

        step_values = pad_agent(np.array(self.buf.step_values, dtype=np.float32)) * masks_p
        option_values = pad_agent(np.array(self.buf.option_values, dtype=np.float32)) * masks_p
        planner_values = pad_agent(np.array(self.buf.planner_values, dtype=np.float32)) * masks_p

        step_terminal = np.maximum(dones_p, 1.0 - masks_p)
        option_terminal = np.maximum(pad_agent(np.array(self.buf.option_end, dtype=np.float32)), 1.0 - masks_p)
        planner_rewards = pad_agent(np.array(self.buf.planner_rewards, dtype=np.float32)) * pad_agent(
            np.array(self.buf.planner_active_masks, dtype=np.float32)
        )

        step_adv, step_ret = self._compute_gae(rewards_p, step_values, masks_p, step_terminal, ep_start, last_step_v, LOW_LEVEL_GAMMA, LOW_LEVEL_GAE_LAMBDA)
        option_adv, option_ret = self._compute_gae(rewards_p, option_values, masks_p, option_terminal, ep_start, last_option_v, OPTION_GAMMA, OPTION_GAE_LAMBDA)
        planner_adv, planner_ret = self._compute_gae(planner_rewards, planner_values, masks_p, step_terminal, ep_start, last_planner_v, PLANNER_GAMMA, PLANNER_GAE_LAMBDA)

        term_mask_t = pad_agent(np.array(self.buf.termination_active_masks, dtype=np.float32))
        planner_mask_t = pad_agent(np.array(self.buf.planner_active_masks, dtype=np.float32))
        step_adv = self._normalize_advantages(step_adv, masks_p)
        option_adv = self._normalize_advantages(option_adv, term_mask_t)
        planner_adv = self._normalize_advantages(planner_adv, planner_mask_t)

        # For actor update, use un-padded (cur_n) slices
        step_adv_n = step_adv[:, :cur_n]
        option_adv_n = option_adv[:, :cur_n]
        step_ret_n = step_ret[:, :cur_n]
        option_ret_n = option_ret[:, :cur_n]
        planner_ret_n = planner_ret[:, :cur_n]
        masks_n = masks_p[:, :cur_n]

        ego_t = np.array(self.buf.egos, dtype=np.float32)              # (T, cur_n, EGO_DIM)
        node_t = np.array(self.buf.node_feats, dtype=np.float32)
        adj_t = np.array(self.buf.adjs, dtype=np.float32)
        nbr_t = np.array(self.buf.nbrs, dtype=np.float32)
        nbr_mask_t = np.array(self.buf.nbr_masks, dtype=bool)
        global_t = np.array(self.buf.globals, dtype=np.float32)
        act_t = np.array(self.buf.actions, dtype=np.float32)
        act_lp_t = np.array(self.buf.action_log_probs, dtype=np.float32)
        subgoal_pos_t = np.array(self.buf.subgoal_positions, dtype=np.float32)   # (T, N_AGENTS, 3)
        subgoal_token_t = np.array(self.buf.subgoal_tokens, dtype=np.float32)    # (T, N_AGENTS, HIDDEN)
        option_start_t = np.array(self.buf.option_start, dtype=bool)             # (T, N_AGENTS)
        term_action_t = np.array(self.buf.termination_actions, dtype=np.float32) # (T, N_AGENTS)
        term_lp_t = np.array(self.buf.termination_log_probs, dtype=np.float32)   # (T, N_AGENTS)
        term_mask_n = np.array(self.buf.termination_active_masks, dtype=np.float32)

        planner_node_t = np.array(self.buf.planner_node_feats, dtype=np.float32)
        planner_edge_t = np.array(self.buf.planner_edge_feats, dtype=np.float32)
        planner_adj_t = np.array(self.buf.planner_adjs, dtype=np.float32)
        planner_node_mask_t = np.array(self.buf.planner_node_masks, dtype=bool)
        planner_cur_t = np.array(self.buf.planner_cur_nodes, dtype=np.int64)
        planner_goal_t = np.array(self.buf.planner_goal_nodes, dtype=np.int64)
        planner_route_t = np.array(self.buf.planner_route_onehot, dtype=np.float32)
        planner_action_t = np.array(self.buf.planner_actions, dtype=np.int64)
        planner_lp_t = np.array(self.buf.planner_log_probs, dtype=np.float32)
        planner_mask_n = np.array(self.buf.planner_active_masks, dtype=np.float32)

        valid_t = np.any(masks_n > 0.5, axis=1)
        idxs = np.where(valid_t)[0]
        if len(idxs) == 0:
            self.buf.clear()
            return

        segments: list[tuple[int, int]] = []
        seg_start = None
        prev_t = -1
        for t in idxs:
            new_seg = (seg_start is None) or (t != prev_t + 1) or bool(ep_start[t])
            if new_seg:
                if seg_start is not None:
                    segments.append((seg_start, prev_t + 1))
                seg_start = int(t)
            prev_t = int(t)
        if seg_start is not None:
            segments.append((seg_start, prev_t + 1))

        chunk_len = max(1, int(MINI_BATCH))
        chunks: list[tuple[int, int]] = []
        for start, end in segments:
            cur = start
            while cur < end:
                nxt = min(cur + chunk_len, end)
                chunks.append((cur, nxt))
                cur = nxt

        actor_losses = []
        for _ in range(K_EPOCHS):
            np.random.shuffle(chunks)
            for start, end in chunks:
                b = np.arange(start, end, dtype=np.int64)

                bE = torch.FloatTensor(ego_t[b]).to(DEVICE)            # (B, cur_n, EGO_DIM)
                bNF = torch.FloatTensor(node_t[b]).to(DEVICE)
                bADJ = torch.FloatTensor(adj_t[b]).to(DEVICE)
                bNB = torch.FloatTensor(nbr_t[b]).to(DEVICE)
                bNM = torch.BoolTensor(nbr_mask_t[b]).to(DEVICE)
                bSGP = torch.FloatTensor(subgoal_pos_t[b]).to(DEVICE)  # (B, N_AGENTS, 3)
                bSGT = torch.FloatTensor(subgoal_token_t[b]).to(DEVICE)
                bA = torch.FloatTensor(act_t[b]).to(DEVICE)
                bOldLP = torch.FloatTensor(act_lp_t[b]).to(DEVICE)
                bStepADV = torch.FloatTensor(step_adv_n[b]).to(DEVICE)
                bOptionADV = torch.FloatTensor(option_adv_n[b]).to(DEVICE)
                bStepRET = torch.FloatTensor(step_ret_n[b]).to(DEVICE)
                bOptionRET = torch.FloatTensor(option_ret_n[b]).to(DEVICE)
                bPlannerRET = torch.FloatTensor(planner_ret_n[b]).to(DEVICE)
                bM = torch.FloatTensor(masks_n[b]).to(DEVICE)
                bAM = bM < 0.5
                bOS = torch.BoolTensor(option_start_t[b]).to(DEVICE)   # (B, N_AGENTS)
                bEP = torch.BoolTensor(ep_start[b]).to(DEVICE)

                bTA = torch.FloatTensor(term_action_t[b]).to(DEVICE)   # (B, N_AGENTS)
                bOldTermLP = torch.FloatTensor(term_lp_t[b]).to(DEVICE)
                bTM = torch.FloatTensor(term_mask_n[b]).to(DEVICE)     # (B, N_AGENTS)

                # Executor forward: pass cur_n agents, full N_AGENTS buffers for subgoal/term
                # Slice subgoal to cur_n agent dimension for executor
                bSGP_n = bSGP[:, :cur_n, :]
                bSGT_n = bSGT[:, :cur_n, :]
                bOS_n = bOS[:, :cur_n]
                bTA_n = bTA[:, :cur_n]
                bOldTermLP_n = bOldTermLP[:, :cur_n]
                bTM_n = bTM[:, :cur_n]
                bOptionADV_n = bOptionADV

                action_lp, action_ent, term_lp, term_ent = self.policy.executor.evaluate_policy(
                    bE, bNF, bADJ, bNB, bNM,
                    bA, bTA_n,
                    subgoal_pos=bSGP_n,
                    subgoal_token=bSGT_n,
                    agent_mask=bAM,
                    option_start=bOS_n,
                    ep_start=bEP,
                    training_mode=True,
                )

                ratio_action = (action_lp - bOldLP).exp() * bM
                s1 = ratio_action * bStepADV
                s2 = ratio_action.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * bStepADV
                action_loss = (-torch.min(s1, s2) - self._ent_coef * action_ent) * bM
                action_loss = action_loss.sum() / (bM.sum() + 1e-8)

                ratio_term = (term_lp - bOldTermLP_n).exp() * bTM_n
                t1 = ratio_term * bOptionADV_n
                t2 = ratio_term.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * bOptionADV_n
                term_loss = (-torch.min(t1, t2) - 0.5 * self._ent_coef * term_ent) * bTM_n
                term_loss = term_loss.sum() / (bTM_n.sum() + 1e-8)

                self.opt_exec.zero_grad()
                exec_loss = action_loss + term_loss
                exec_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.executor.parameters(), GRAD_NORM)
                self.opt_exec.step()

                bPNF = torch.FloatTensor(planner_node_t[b]).to(DEVICE)
                bPEF = torch.FloatTensor(planner_edge_t[b]).to(DEVICE)
                bPADJ = torch.FloatTensor(planner_adj_t[b]).to(DEVICE)
                bPNM = torch.BoolTensor(planner_node_mask_t[b]).to(DEVICE)
                bPCUR = torch.LongTensor(planner_cur_t[b]).to(DEVICE)
                bPGOAL = torch.LongTensor(planner_goal_t[b]).to(DEVICE)
                bPROUTE = torch.FloatTensor(planner_route_t[b]).to(DEVICE)
                bPA = torch.LongTensor(planner_action_t[b]).to(DEVICE)
                bOldPlanLP = torch.FloatTensor(planner_lp_t[b]).to(DEVICE)
                bPlanMaskBool = torch.BoolTensor(planner_mask_n[b] > 0.5).to(DEVICE)
                bPlanMask = torch.FloatTensor(planner_mask_n[b]).to(DEVICE)
                bPlanADV = torch.FloatTensor(planner_adv[:, :cur_n][b]).to(DEVICE)

                outputs, plan_lp, plan_ent = self.policy.planner.evaluate_action(
                    bPNF, bPEF, bE, bPADJ, bPCUR, bPGOAL, bPA,
                    existing_route_onehot=bPROUTE,
                    plan_mask=bPlanMaskBool,
                    node_mask=bPNM,
                )
                ratio_plan = (plan_lp - bOldPlanLP).exp() * bPlanMask
                p1 = ratio_plan * bPlanADV
                p2 = ratio_plan.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * bPlanADV
                planner_loss = (-torch.min(p1, p2) - 0.5 * self._ent_coef * plan_ent) * bPlanMask
                planner_loss = planner_loss.sum() / (bPlanMask.sum() + 1e-8)

                # ========== 新增：全局路径冲突正则项 ==========
                if ENABLE_GLOBAL_COLLISION_PENALTY and float(bPlanMask.sum().item()) > 0.0:
                    edge_path_probs = outputs["edge_path_probs"]  # [batch, agent_count, MAX_N_EDGES]
                    # 计算路径重叠率：sum(所有智能体边概率之和的平方)，重叠越多值越大
                    route_overlap = torch.sum(torch.square(torch.sum(edge_path_probs, dim=1)), dim=-1)
                    # 归一化除以智能体数量，保证损失量级和主损失匹配
                    route_overlap = route_overlap / edge_path_probs.shape[1]
                    conflict_loss = CONFLICT_LOSS_COEF * torch.mean(route_overlap)
                    planner_loss = planner_loss + conflict_loss
                # ==============================================

                if float(bPlanMask.sum().item()) > 0.0:
                    self.opt_planner.zero_grad()
                    planner_loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.planner.parameters(), GRAD_NORM)
                    self.opt_planner.step()

                bG = torch.FloatTensor(global_t[b]).to(DEVICE)
                critic_heads = self.policy.critic(bG, head="all")
                active_den = bM.sum(dim=-1).clamp(min=1.0)
                step_target = (bStepRET * bM).sum(dim=-1) / active_den
                option_target = (bOptionRET * bM).sum(dim=-1) / active_den
                planner_target = (bPlannerRET * bM).sum(dim=-1) / active_den
                critic_loss = VAL_COEF * (
                    nn.functional.mse_loss(critic_heads["step"], step_target)
                    + nn.functional.mse_loss(critic_heads["option"], option_target)
                    + nn.functional.mse_loss(critic_heads["planner"], planner_target)
                )

                self.opt_c.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.critic.parameters(), GRAD_NORM)
                self.opt_c.step()

                actor_losses.append(float((exec_loss + planner_loss).item()))

        self.loss_hist.append(np.mean(actor_losses) if actor_losses else 0.0)
        self.total_t += int(np.sum(masks_n))
        self.buf.clear()
        self.update_cnt += 1

        if ENABLE_COMM:
            progress = min(1.0, self.update_cnt / 3000.0)
            self.policy.executor.comm_alpha = COMM_ALPHA_INIT + (COMM_ALPHA_MAX - COMM_ALPHA_INIT) * progress
        else:
            self.policy.executor.comm_alpha = 0.0

        # INNOVATION: adaptive entropy with slower decay
        self._ent_coef = max(ENT_COEF_MIN, ENT_COEF * (ENT_COEF_DECAY ** self.update_cnt))

        if self.update_cnt % 100 == 0:
            for param_group in self.opt_exec.param_groups:
                param_group["lr"] *= 0.997
            for param_group in self.opt_planner.param_groups:
                param_group["lr"] *= 0.997
            for param_group in self.opt_c.param_groups:
                param_group["lr"] *= 0.997

    def end_ep(self, ep_r, results, n_steps, done_steps, step_budget) -> None:
        self.ep += 1
        self.r_hist.append(ep_r)
        self.r50.append(ep_r)
        success = float(np.mean([result == "goal" for result in results]))
        self.sr_hist.append(success)
        self.stp50.append(n_steps)
        self.budget50.append(step_budget)

        goal_steps = [int(done_steps[i]) for i, result in enumerate(results) if result == "goal" and done_steps[i] >= 0]
        if goal_steps:
            self.goal_step_avg50.append(float(np.mean(goal_steps)))
            self.goal_step_med50.append(float(np.median(goal_steps)))
            self.goal_step_std50.append(float(np.std(goal_steps)))
        else:
            self.goal_step_avg50.append(float("nan"))
            self.goal_step_med50.append(float("nan"))
            self.goal_step_std50.append(float("nan"))

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "executor": self.policy.executor.state_dict(),
                "planner": self.policy.planner.state_dict(),
                "adversary": self.policy.adversary.state_dict(),
                "critic": self.policy.critic.state_dict(),
                "ep": self.ep,
                "best_score": self.best_score,
            },
            path,
        )
        print(f"[AeroPipeRL] Saved -> {path}")

    def load(self, path: Path | str) -> None:
        path = Path(path)
        if not path.exists():
            print(f"[AeroPipeRL] Checkpoint not found: {path}")
            return

        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        executor_sd = ckpt.get("executor", ckpt.get("actor", {}))
        planner_sd = ckpt.get("planner", {})
        adversary_sd = ckpt.get("adversary", {})
        critic_sd = ckpt.get("critic", {})

        cur_executor = self.policy.executor.state_dict()
        cur_planner = self.policy.planner.state_dict()
        cur_adversary = self.policy.adversary.state_dict()
        cur_critic = self.policy.critic.state_dict()

        executor_sd = {k: v for k, v in executor_sd.items() if k in cur_executor and tuple(v.shape) == tuple(cur_executor[k].shape)}
        planner_sd = {k: v for k, v in planner_sd.items() if k in cur_planner and tuple(v.shape) == tuple(cur_planner[k].shape)}
        adversary_sd = {k: v for k, v in adversary_sd.items() if k in cur_adversary and tuple(v.shape) == tuple(cur_adversary[k].shape)}
        critic_sd = {k: v for k, v in critic_sd.items() if k in cur_critic and tuple(v.shape) == tuple(cur_critic[k].shape)}

        executor_res = self.policy.executor.load_state_dict(executor_sd, strict=False)
        planner_res = self.policy.planner.load_state_dict(planner_sd, strict=False)
        adversary_res = self.policy.adversary.load_state_dict(adversary_sd, strict=False)
        critic_res = self.policy.critic.load_state_dict(critic_sd, strict=False)

        self.ep = ckpt.get("ep", 0)
        self.best_score = ckpt.get("best_score", -1e9)

        missing = list(executor_res.missing_keys) + list(planner_res.missing_keys) + list(adversary_res.missing_keys) + list(critic_res.missing_keys)
        unexpected = list(executor_res.unexpected_keys) + list(planner_res.unexpected_keys) + list(adversary_res.unexpected_keys) + list(critic_res.unexpected_keys)
        if missing or unexpected:
            print(f"[AeroPipeRL] Partial checkpoint load -> {path} (ep={self.ep})")
        else:
            print(f"[AeroPipeRL] Loaded -> {path} (ep={self.ep})")
