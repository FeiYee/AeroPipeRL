from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from aeropipe_rl.config import (
    ACT_DIM,
    BEST_CKPT_PATH,
    CURRICULUM_ENABLED,
    CURRICULUM_RAMP_EP,
    DEVICE,
    EGO_DIM,
    GLOBAL_DIM,
    GLOBAL_SEED,
    HIDDEN,
    LOG_EVERY_EP,
    LATEST_CKPT_PATH,
    MAX_TRAIN_EPISODES,
    N_AGENTS,
    N_AGENTS_START,
    N_AGENTS_TARGET,
    NBR_DIM,
    N_HEADS,
    N_LAYERS,
    SAVE_EVERY_EP,
    T_HORIZON,
    TIMEOUT_PENALTY_PER_STEP,
    DEVIATION_PENALTY_PER_STEP,
    OVERDUE_REPLAN_PENALTY,
    SUBGOAL_CHECK_INTERVAL,
    ensure_runtime_dirs,
)
from aeropipe_rl.environment import MAEnv, PipeNet
from aeropipe_rl.training.trainer import MAPPOTrainer


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_banner() -> None:
    print(f"[AeroPipeRL] seed={GLOBAL_SEED}")
    print(f"[AeroPipeRL] device={DEVICE} N_AGENTS={N_AGENTS} (curriculum: {N_AGENTS_START}->{N_AGENTS_TARGET})")
    print(f"[AeroPipeRL] EGO={EGO_DIM} NBR={NBR_DIM} ACT={ACT_DIM} HIDDEN={HIDDEN} heads={N_HEADS} layers={N_LAYERS}")
    print(f"[AeroPipeRL] GLOBAL_DIM={GLOBAL_DIM} SAVE_EVERY_EP={SAVE_EVERY_EP}")


def get_curriculum_n_agents(ep: int) -> int:
    """
    INNOVATION: Agent-count curriculum.
    Start with 2 agents, linearly ramp to N_AGENTS_TARGET over CURRICULUM_RAMP_EP episodes.
    This lets the policy learn basic navigation before facing full multi-agent coordination.
    """
    if not CURRICULUM_ENABLED:
        return N_AGENTS_TARGET
    frac = min(1.0, ep / max(CURRICULUM_RAMP_EP, 1))
    n = int(N_AGENTS_START + frac * (N_AGENTS_TARGET - N_AGENTS_START))
    return max(N_AGENTS_START, min(N_AGENTS_TARGET, n))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train AeroPipeRL with headless MAPPO.")
    parser.add_argument("--resume", choices=["none", "latest", "best"], default="latest")
    parser.add_argument("--max-episodes", type=int, default=MAX_TRAIN_EPISODES)
    args = parser.parse_args(argv)

    ensure_runtime_dirs()
    seed_all(GLOBAL_SEED)
    print_banner()

    net = PipeNet()
    # Start with N_AGENTS_START agents for curriculum
    init_n = get_curriculum_n_agents(0)
    env = MAEnv(net, n_agents=init_n)
    trainer = MAPPOTrainer()

    if args.resume == "latest":
        trainer.load(LATEST_CKPT_PATH)
    elif args.resume == "best":
        trainer.load(BEST_CKPT_PATH)
    else:
        print("[AeroPipeRL] Resume disabled, starting from scratch.")

    print(f"[AeroPipeRL] Network: {len(net.nodes)} nodes, {len(net.edges)} edges")
    print(f"[AeroPipeRL] Starting with {init_n} agents (curriculum)")

    def new_episode():
        # INNOVATION: update agent count per curriculum schedule
        target_n = get_curriculum_n_agents(trainer.ep)
        if env.n_agents != target_n:
            prev_n = env.n_agents
            env.set_n_agents(target_n)
            trainer.set_n_agents(target_n)
            print(f"[AeroPipeRL] Curriculum: agents {prev_n}->{target_n} at ep {trainer.ep}")
        obs = env.reset()
        trainer.reset_episode()
        return obs, 0.0, 0

    def on_episode_end(ep_reward, results, ep_steps):
        nonlocal tr_obs, tr_ep_r, tr_ep_steps
        n = env.n_agents
        trainer.end_ep(ep_reward, results, ep_steps, env.done_steps.copy(), env.step_budget)

        wall_rate_ep = float(np.mean(env.ep_wall_hit.astype(np.float32)))
        timeout_rate_ep = float(np.mean([result == "timeout" for result in results]))
        agent_col_rate_ep = float(np.mean(env.ep_agent_collided.astype(np.float32)))
        timeout20_ep = 1.0 if timeout_rate_ep >= 0.20 else 0.0
        timeout50_ep = 1.0 if timeout_rate_ep >= 0.50 else 0.0
        timeout100_ep = 1.0 if timeout_rate_ep >= 0.999 else 0.0

        trainer.wall_rate100.append(wall_rate_ep)
        trainer.agent_col_rate100.append(agent_col_rate_ep)
        trainer.timeout20_rate100.append(timeout20_ep)
        trainer.timeout50_rate100.append(timeout50_ep)
        trainer.timeout100_rate100.append(timeout100_ep)
        trainer.collision_rate_ma100.append(0.5 * wall_rate_ep + 0.5 * agent_col_rate_ep)
        trainer.time_gauss_pen100.append(float(env.last_time_gauss_penalty_applied_mean))

        ep_success = float(np.mean([result == "goal" for result in results]))
        score = ep_success * 1000.0 + ep_reward
        if score > trainer.best_score:
            trainer.best_score = score
            trainer.save(BEST_CKPT_PATH)

        if trainer.ep % SAVE_EVERY_EP == 0:
            trainer.save(LATEST_CKPT_PATH)

        if trainer.ep % LOG_EVERY_EP == 0:
            loss20 = float(np.mean(list(trainer.loss_hist)[-20:])) if trainer.loss_hist else 0.0
            goal_cnt = int(sum(result == "goal" for result in results))
            timeout_cnt = int(sum(result == "timeout" for result in results))
            goal_steps = [int(env.done_steps[i]) for i, result in enumerate(results) if result == "goal" and env.done_steps[i] >= 0]
            eta_avg = float(np.mean(goal_steps)) if goal_steps else float("nan")
            eta_med = float(np.median(goal_steps)) if goal_steps else float("nan")
            eta_std = float(np.std(goal_steps)) if goal_steps else float("nan")
            speed_avg = env.ep_speed_sum / max(env.ep_speed_count, 1)
            lr_a = float(trainer.opt_exec.param_groups[0]["lr"])
            lr_p = float(trainer.opt_planner.param_groups[0]["lr"])
            lr_c = float(trainer.opt_c.param_groups[0]["lr"])

            r100 = float(np.mean(list(trainer.r_hist)[-100:])) if trainer.r_hist else 0.0
            sr100 = float(np.mean(list(trainer.sr_hist)[-100:]) * 100.0) if trainer.sr_hist else 0.0
            step100 = float(np.mean(list(trainer.stp50)[-100:])) if trainer.stp50 else 0.0
            wall100 = float(np.mean(trainer.wall_rate100) * 100.0) if trainer.wall_rate100 else 0.0
            aco100 = float(np.mean(trainer.agent_col_rate100) * 100.0) if trainer.agent_col_rate100 else 0.0
            to20_100 = float(np.mean(trainer.timeout20_rate100) * 100.0) if trainer.timeout20_rate100 else 0.0
            to50_100 = float(np.mean(trainer.timeout50_rate100) * 100.0) if trainer.timeout50_rate100 else 0.0
            to100_100 = float(np.mean(trainer.timeout100_rate100) * 100.0) if trainer.timeout100_rate100 else 0.0
            col_ma100 = float(np.mean(trainer.collision_rate_ma100) * 100.0) if trainer.collision_rate_ma100 else 0.0
            time_pen_cur = float(env.last_time_gauss_penalty_applied_mean)
            time_pen100 = float(np.mean(trainer.time_gauss_pen100)) if trainer.time_gauss_pen100 else 0.0

            print(
                f"[EP {trainer.ep:6d}|n={n}] R={ep_reward:7.3f} | goal={goal_cnt:2d}/{n:2d} timeout={timeout_cnt:2d} "
                f"| steps={ep_steps:4d}/{env.step_budget:4d} | ETA(avg/med/std)={eta_avg:6.1f}/{eta_med:6.1f}/{eta_std:5.1f}"
            )
            print(
                f"           100ep: SR={sr100:6.2f}% R={r100:7.3f} steps={step100:6.1f} "
                f"wall={wall100:5.2f}% aCol={aco100:5.2f}% to20/50/100={to20_100:5.1f}/{to50_100:5.1f}/{to100_100:5.1f}%"
            )
            print(
                f"           train: spd={speed_avg:4.2f} loss20={loss20:7.4f} "
                f"lr(a/p/c)={lr_a:.2e}/{lr_p:.2e}/{lr_c:.2e} upd={trainer.update_cnt:5d} best={trainer.best_score:7.3f}"
            )

        tr_obs, tr_ep_r, tr_ep_steps = new_episode()

    tr_obs, tr_ep_r, tr_ep_steps = new_episode()

    while True:
        n = env.n_agents
        egos, node_feats, adjs, nbrs, nbr_masks, global_obs = tr_obs
        dones_before = env.dones.copy()
        ep_start = tr_ep_steps == 0

        step_out = trainer.act(
            egos,
            node_feats,
            adjs,
            nbrs,
            nbr_masks,
            global_obs,
            env,
            agent_mask=dones_before,
            ep_start=ep_start,
        )
        next_obs, rewards, dones, results = env.step(step_out["actions"])

        # ========== 新增：子目标超时/偏离惩罚 ==========
        n = env.n_agents
        trainer.subgoal_elapsed_steps[:n] += 1
        for i in range(n):
            if trainer.has_active_option[i] and not dones[i]:
                # 超时惩罚
                if trainer.subgoal_elapsed_steps[i] > trainer.subgoal_deadline[i]:
                    rewards[i] -= TIMEOUT_PENALTY_PER_STEP
                # 偏离惩罚：ego状态前3维是位置
                current_pos = egos[i][:3]
                subgoal_pos = trainer.active_subgoal_pos[i]
                dist = np.linalg.norm(current_pos - subgoal_pos)
                if dist > trainer.subgoal_tolerance[i]:
                    rewards[i] -= DEVIATION_PENALTY_PER_STEP
                    trainer.pending_replan[i] = True
                    rewards[i] -= OVERDUE_REPLAN_PENALTY  # 偏离触发重规划额外惩罚
        # ============================================

        # ========== 新增：高层主动检查子目标进度 ==========
        if tr_ep_steps % SUBGOAL_CHECK_INTERVAL == 0:
            for i in range(n):
                if trainer.has_active_option[i] and not dones[i]:
                    progress = trainer.subgoal_elapsed_steps[i] / max(trainer.subgoal_deadline[i], 1.0)
                    # 进度超过2倍deadline还没到，主动重规划
                    if progress > 2.0:
                        trainer.pending_replan[i] = True
                        rewards[i] -= OVERDUE_REPLAN_PENALTY
        # ============================================

        tr_ep_r += rewards.sum()
        tr_ep_steps += 1
        trainer.after_step(dones)

        option_end = np.logical_or(step_out["termination_actions"], dones)

        clean_global = trainer._critic_global_from_egos(egos, dones_mask=dones_before, n_agents=n)
        trainer.buf.push(
            np.stack(egos),
            np.stack(node_feats),
            np.stack(adjs),
            np.stack(nbrs),
            np.stack(nbr_masks),
            clean_global,
            step_out["actions"],
            step_out["action_log_probs"],
            rewards,
            dones,
            ~dones_before,
            step_out["step_values"],
            step_out["option_values"],
            step_out["planner_values"],
            env._step_wall_hit.copy(),
            ep_start,
            step_out["subgoal_positions"],
            step_out["subgoal_tokens"],
            step_out["option_start"],
            option_end,
            step_out["termination_actions"],
            step_out["termination_log_probs"],
            step_out["termination_probs"],
            step_out["termination_active_masks"],
            step_out["planner_node_feats"],
            step_out["planner_edge_feats"],
            step_out["planner_adjs"],
            step_out["planner_node_masks"],
            step_out["planner_cur_nodes"],
            step_out["planner_goal_nodes"],
            step_out["planner_route_onehot"],
            step_out["planner_actions"],
            step_out["planner_log_probs"],
            step_out["planner_rewards"],
            step_out["planner_active_masks"],
        )
        tr_obs = next_obs

        if np.all(dones):
            on_episode_end(tr_ep_r, results, tr_ep_steps)
            if trainer.ep >= args.max_episodes:
                print(f"[AeroPipeRL] Reached max_episodes={args.max_episodes}, stopping.")
                break

        if len(trainer.buf) >= T_HORIZON:
            next_egos, _, _, _, _, _ = tr_obs
            trainer.update(next_egos, env.dones, n_agents=n)


if __name__ == "__main__":
    main()
