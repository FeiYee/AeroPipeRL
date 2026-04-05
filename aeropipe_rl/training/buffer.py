from __future__ import annotations

import numpy as np

from aeropipe_rl.config import N_AGENTS


class RolloutBuffer:
    """Stores multi-agent rollout fragments for PPO updates."""

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.egos = []
        self.node_feats = []
        self.adjs = []
        self.nbrs = []
        self.nbr_masks = []
        self.globals = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.active_masks = []
        self.values = []
        self.wall_hits = []
        self.ep_start = []
        self.planner_next_node = []
        self.planner_log_prob = []
        self.planner_rewards = []

    def push(self, egos, node_feats, adjs, nbrs, nbr_masks, global_obs, actions, log_probs, rewards, dones, active_masks, values, wall_hits, ep_start=False, planner_next_node=None, planner_log_prob=None, planner_rewards=None):
        self.egos.append(egos)
        self.node_feats.append(node_feats)
        self.adjs.append(adjs)
        self.nbrs.append(nbrs)
        self.nbr_masks.append(nbr_masks)
        self.globals.append(global_obs)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.rewards.append(rewards)
        self.dones.append(dones)
        self.active_masks.append(active_masks)
        self.values.append(values)
        self.wall_hits.append(wall_hits)
        self.ep_start.append(bool(ep_start))

        if planner_next_node is None:
            planner_next_node = np.zeros(N_AGENTS, dtype=np.int32)
        if planner_log_prob is None:
            planner_log_prob = np.zeros(N_AGENTS, dtype=np.float32)
        if planner_rewards is None:
            planner_rewards = np.zeros(N_AGENTS, dtype=np.float32)
        self.planner_next_node.append(planner_next_node)
        self.planner_log_prob.append(planner_log_prob)
        self.planner_rewards.append(planner_rewards)

    def __len__(self) -> int:
        return len(self.rewards)
