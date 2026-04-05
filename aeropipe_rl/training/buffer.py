from __future__ import annotations


class RolloutBuffer:
    """Stores hierarchical multi-agent rollout fragments for PPO updates."""

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
        self.action_log_probs = []
        self.rewards = []
        self.dones = []
        self.active_masks = []
        self.wall_hits = []
        self.ep_start = []

        self.step_values = []
        self.option_values = []
        self.planner_values = []

        self.subgoal_positions = []
        self.subgoal_tokens = []
        self.option_start = []
        self.option_end = []

        self.termination_actions = []
        self.termination_log_probs = []
        self.termination_probs = []
        self.termination_active_masks = []

        self.planner_node_feats = []
        self.planner_edge_feats = []
        self.planner_adjs = []
        self.planner_node_masks = []
        self.planner_cur_nodes = []
        self.planner_goal_nodes = []
        self.planner_route_onehot = []
        self.planner_actions = []
        self.planner_log_probs = []
        self.planner_rewards = []
        self.planner_active_masks = []

    def push(
        self,
        egos,
        node_feats,
        adjs,
        nbrs,
        nbr_masks,
        global_obs,
        actions,
        action_log_probs,
        rewards,
        dones,
        active_masks,
        step_values,
        option_values,
        planner_values,
        wall_hits,
        ep_start,
        subgoal_positions,
        subgoal_tokens,
        option_start,
        option_end,
        termination_actions,
        termination_log_probs,
        termination_probs,
        termination_active_masks,
        planner_node_feats,
        planner_edge_feats,
        planner_adjs,
        planner_node_masks,
        planner_cur_nodes,
        planner_goal_nodes,
        planner_route_onehot,
        planner_actions,
        planner_log_probs,
        planner_rewards,
        planner_active_masks,
    ) -> None:
        self.egos.append(egos)
        self.node_feats.append(node_feats)
        self.adjs.append(adjs)
        self.nbrs.append(nbrs)
        self.nbr_masks.append(nbr_masks)
        self.globals.append(global_obs)

        self.actions.append(actions)
        self.action_log_probs.append(action_log_probs)
        self.rewards.append(rewards)
        self.dones.append(dones)
        self.active_masks.append(active_masks)
        self.wall_hits.append(wall_hits)
        self.ep_start.append(ep_start)

        self.step_values.append(step_values)
        self.option_values.append(option_values)
        self.planner_values.append(planner_values)

        self.subgoal_positions.append(subgoal_positions)
        self.subgoal_tokens.append(subgoal_tokens)
        self.option_start.append(option_start)
        self.option_end.append(option_end)

        self.termination_actions.append(termination_actions)
        self.termination_log_probs.append(termination_log_probs)
        self.termination_probs.append(termination_probs)
        self.termination_active_masks.append(termination_active_masks)

        self.planner_node_feats.append(planner_node_feats)
        self.planner_edge_feats.append(planner_edge_feats)
        self.planner_adjs.append(planner_adjs)
        self.planner_node_masks.append(planner_node_masks)
        self.planner_cur_nodes.append(planner_cur_nodes)
        self.planner_goal_nodes.append(planner_goal_nodes)
        self.planner_route_onehot.append(planner_route_onehot)
        self.planner_actions.append(planner_actions)
        self.planner_log_probs.append(planner_log_probs)
        self.planner_rewards.append(planner_rewards)
        self.planner_active_masks.append(planner_active_masks)

    def __len__(self) -> int:
        return len(self.rewards)
