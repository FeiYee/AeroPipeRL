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
    EGO_DIM,
    GAMMA,
    GAE_LAMBDA,
    GRAD_NORM,
    HIGH_LEVEL_UPDATE_FREQ,
    K_EPOCHS,
    LR_ACTOR,
    LR_CRITIC,
    MINI_BATCH,
    N_AGENTS,
    R_CAPACITY,
    R_CONFLICT,
    R_ON_TIME,
    VAL_COEF,
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
        self.high_level_step_counter = 0

        self.ep = 0
        self.total_t = 0
        self.update_cnt = 0
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

    def _critic_global_from_egos(self, egos, dones_mask=None):
        global_obs = np.concatenate(egos).astype(np.float32)
        if dones_mask is not None:
            global_obs = global_obs.copy()
            for i in range(N_AGENTS):
                if dones_mask[i]:
                    start = i * EGO_DIM
                    global_obs[start : start + EGO_DIM] = 0.0
        return global_obs

    @torch.no_grad()
    def act(self, egos, node_feats, adjs, nbrs_list, nbr_masks, global_obs, env, agent_mask=None, deterministic=False):
        if agent_mask is None:
            agent_mask = np.zeros(N_AGENTS, dtype=bool)

        ego_tensor = torch.FloatTensor(np.stack(egos)).unsqueeze(0).to(DEVICE)
        node_tensor = torch.FloatTensor(np.stack(node_feats)).unsqueeze(0).to(DEVICE)
        adj_tensor = torch.FloatTensor(np.stack(adjs)).unsqueeze(0).to(DEVICE)
        nbr_tensor = torch.FloatTensor(np.stack(nbrs_list)).unsqueeze(0).to(DEVICE)
        nbr_mask_tensor = torch.BoolTensor(np.stack(nbr_masks)).unsqueeze(0).to(DEVICE)
        agent_mask_tensor = torch.BoolTensor(agent_mask).unsqueeze(0).to(DEVICE)
        global_tensor = torch.FloatTensor(global_obs).unsqueeze(0).to(DEVICE)

        actions, log_probs = self.policy.executor.get_action(
            ego_tensor,
            node_tensor,
            adj_tensor,
            nbr_tensor,
            nbr_mask_tensor,
            agent_mask=agent_mask_tensor,
            deterministic=deterministic,
            ep_start=False,
            training_mode=False,
        )
        value = self.policy.critic(global_tensor)

        planner_next_node = np.zeros(N_AGENTS, dtype=np.int32)
        planner_log_prob = np.zeros(N_AGENTS, dtype=np.float32)
        planner_reward = np.zeros(N_AGENTS, dtype=np.float32)

        if self.high_level_step_counter % HIGH_LEVEL_UPDATE_FREQ == 0:
            global_node_feat, global_edge_feat, _, global_adj, _ = env._get_global_planner_state()
            cur_node_ids = [env._nearest_node_id(env.positions[i]) for i in range(N_AGENTS)]
            cur_node_tensor = torch.LongTensor(cur_node_ids).unsqueeze(0).to(DEVICE)
            next_node_logits, _, _, _ = self.policy.planner(
                torch.FloatTensor(global_node_feat).unsqueeze(0).to(DEVICE),
                torch.FloatTensor(global_edge_feat).unsqueeze(0).to(DEVICE),
                ego_tensor,
                torch.FloatTensor(global_adj).unsqueeze(0).to(DEVICE),
                cur_node_tensor,
            )
            next_node_probs = torch.softmax(next_node_logits, dim=-1)
            next_node_dist = torch.distributions.Categorical(next_node_probs)
            if deterministic:
                planner_next_node = next_node_probs.argmax(dim=-1).squeeze(0).cpu().numpy()
            else:
                planner_next_node = next_node_dist.sample().squeeze(0).cpu().numpy()
            planner_log_prob = next_node_dist.log_prob(torch.LongTensor(planner_next_node).to(DEVICE)).squeeze(0).cpu().numpy()

            for i in range(N_AGENTS):
                if agent_mask[i]:
                    continue
                cur_node = cur_node_ids[i]
                next_node = planner_next_node[i]
                if next_node in env.net.adj[cur_node] and next_node != env.goal_nodes[i]:
                    new_path = env.net.bfs(cur_node, env.goal_nodes[i])
                    if new_path:
                        env.route_plan[i] = [cur_node] + new_path[1:]
                        env.waypoints[i] = [env.net.nodes[nid].copy() for nid in env.route_plan[i]]
                        env.wp_idx[i] = 1
                        planner_reward[i] += R_ON_TIME + R_CAPACITY + max(R_CONFLICT, 0.0)

        self.high_level_step_counter += 1

        act_np = actions.squeeze(0).cpu().numpy()
        lp_np = log_probs.squeeze(0).cpu().numpy() if log_probs is not None else np.zeros(N_AGENTS, dtype=np.float32)
        val_np = np.full(N_AGENTS, float(value.item()), dtype=np.float32)
        act_np[agent_mask] = 0.0
        lp_np[agent_mask] = 0.0
        val_np[agent_mask] = 0.0
        return act_np, lp_np, val_np, planner_next_node, planner_log_prob, planner_reward

    def update(self, last_egos, last_dones) -> None:
        rollout_len = len(self.buf)
        if rollout_len == 0:
            return

        with torch.no_grad():
            clean_last_g = self._critic_global_from_egos(last_egos, dones_mask=last_dones)
            global_tensor = torch.FloatTensor(clean_last_g).unsqueeze(0).to(DEVICE)
            last_v = np.full(N_AGENTS, float(self.policy.critic(global_tensor).item()), dtype=np.float32)
        last_v = last_v * (~last_dones).astype(np.float32)

        rewards = np.array(self.buf.rewards, dtype=np.float32)
        dones = np.array(self.buf.dones, dtype=np.float32)
        masks = np.array(self.buf.active_masks, dtype=np.float32)
        values = np.array(self.buf.values, dtype=np.float32)
        ep_start = np.array(self.buf.ep_start, dtype=bool)
        dones = np.maximum(dones, 1.0 - masks)
        rewards = rewards * masks
        values = values * masks

        adv = np.zeros_like(rewards)
        ret = np.zeros_like(rewards)
        gae = np.zeros(N_AGENTS, dtype=np.float32)
        v_next = last_v
        for t in reversed(range(rollout_len)):
            if t + 1 < rollout_len and ep_start[t + 1]:
                gae[:] = 0.0
                v_next[:] = 0.0
            delta = rewards[t] + GAMMA * v_next * (1 - dones[t]) - values[t]
            gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * gae
            adv[t] = gae
            ret[t] = gae + values[t]
            v_next = values[t]

        ego_t = np.array(self.buf.egos, dtype=np.float32)
        node_t = np.array(self.buf.node_feats, dtype=np.float32)
        adj_t = np.array(self.buf.adjs, dtype=np.float32)
        nbr_t = np.array(self.buf.nbrs, dtype=np.float32)
        nbr_mask_t = np.array(self.buf.nbr_masks, dtype=bool)
        global_t = np.array(self.buf.globals, dtype=np.float32)
        act_t = np.array(self.buf.actions, dtype=np.float32)
        log_prob_t = np.array(self.buf.log_probs, dtype=np.float32)
        mask_t = np.array(self.buf.active_masks, dtype=np.float32)

        valid_t = np.any(mask_t > 0.5, axis=1)
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
                bE = torch.FloatTensor(ego_t[b]).to(DEVICE)
                bNF = torch.FloatTensor(node_t[b]).to(DEVICE)
                bADJ = torch.FloatTensor(adj_t[b]).to(DEVICE)
                bNB = torch.FloatTensor(nbr_t[b]).to(DEVICE)
                bNM = torch.BoolTensor(nbr_mask_t[b]).to(DEVICE)
                bG = torch.FloatTensor(global_t[b]).to(DEVICE)
                bA = torch.FloatTensor(act_t[b]).to(DEVICE)
                bLP = torch.FloatTensor(log_prob_t[b]).to(DEVICE)
                bADV = torch.FloatTensor(adv[b]).to(DEVICE)
                bRET = torch.FloatTensor(ret[b]).to(DEVICE)
                bM = torch.FloatTensor(mask_t[b]).to(DEVICE)
                bAM = bM < 0.5

                new_lp, entropy = self.policy.executor.evaluate_action(
                    bE,
                    bNF,
                    bADJ,
                    bNB,
                    bNM,
                    bA,
                    agent_mask=bAM,
                    ep_start=True,
                    training_mode=True,
                )
                new_v = self.policy.critic(bG)

                ratio = (new_lp - bLP).exp() * bM
                s1 = ratio * bADV
                s2 = ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * bADV
                actor_loss = (-torch.min(s1, s2) * bM - self._ent_coef * entropy * bM).sum() / (bM.sum() + 1e-8)

                ret_mean = (bRET * bM).sum(-1) / bM.sum(-1).clamp(min=1.0)
                critic_loss = VAL_COEF * nn.functional.mse_loss(new_v, ret_mean)

                self.opt_exec.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.executor.parameters(), GRAD_NORM)
                self.opt_exec.step()

                self.opt_c.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.critic.parameters(), GRAD_NORM)
                self.opt_c.step()

                actor_losses.append(actor_loss.item())

        self.loss_hist.append(np.mean(actor_losses))
        self.total_t += int(np.sum(mask_t))
        self.buf.clear()
        self.update_cnt += 1

        if ENABLE_COMM:
            progress = min(1.0, self.update_cnt / 3000.0)
            self.policy.executor.comm_alpha = COMM_ALPHA_INIT + (COMM_ALPHA_MAX - COMM_ALPHA_INIT) * progress
        else:
            self.policy.executor.comm_alpha = 0.0

        self._ent_coef = max(0.003, ENT_COEF * (0.9995 ** self.update_cnt))
        if self.update_cnt % 100 == 0:
            for param_group in self.opt_exec.param_groups:
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
