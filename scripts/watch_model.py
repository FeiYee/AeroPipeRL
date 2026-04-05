#!/usr/bin/env python3
"""Watch a trained model on CPU with GUI rendering."""
import argparse
import numpy as np
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, QUIT, KEYDOWN, K_ESCAPE, K_q, K_r, K_SPACE
from OpenGL.GL import *
from OpenGL.GLU import *
import torch

from aeropipe_rl.config import DEVICE, BEST_CKPT_PATH, LATEST_CKPT_PATH, N_AGENTS, AGENT_COLORS, W, H, WATCH_SPF, WATCH_PAUSE, AGENT_R
from aeropipe_rl.environment import PipeNet, MAEnv
from aeropipe_rl.render import Camera, HUD, draw_pipe_network
from aeropipe_rl.training.trainer import MAPPOTrainer


# 渲染配置
RENDER_PLANNER_PATH = True  # 设为False关闭规划路径渲染
PLANNER_PATH_ALPHA = 0.3  # 路径透明度
PLANNER_PATH_WIDTH = 1.0  # 路径线宽（极细）


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", choices=["best", "latest"], default="best",
                        help="Which checkpoint to load")
    args = parser.parse_args()

    # Force CPU
    DEVICE = torch.device("cpu")
    DEBUG_GUI = True

    # Build environment and trainer
    net = PipeNet()
    env = MAEnv(net)
    trainer = MAPPOTrainer()

    # Load checkpoint
    ckpt_path = BEST_CKPT_PATH if args.ckpt == "best" else LATEST_CKPT_PATH
    trainer.load(ckpt_path)

    # Start GUI watch loop (deterministic)
    pygame.init()
    pygame.display.set_mode((W, H), DOUBLEBUF | OPENGL)
    pygame.display.set_caption(
        "Multi-Agent Pipe MARL | WATCH (CPU) | SPACE:mode  R:regen")

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    cam = Camera()
    hud = HUD()

    mode = 'watch'
    flash_red = 0
    last_results = [''] * N_AGENTS

    wa_obs = None
    wa_done_all = True
    wa_pause = 0
    wa_trails = [[] for _ in range(N_AGENTS)]
    wa_attn_cache = None
    wa_positions = np.zeros((N_AGENTS, 3))

    clock = pygame.time.Clock()

    while True:
        for ev in pygame.event.get():
            if ev.type == QUIT:
                pygame.quit(); return
            cam.event(ev)
            if ev.type == KEYDOWN:
                if ev.key in (K_ESCAPE, K_q):
                    pygame.quit(); return
                elif ev.key == K_r:
                    net.regenerate()
                    env = MAEnv(net)
                    wa_done_all = True; wa_pause = 0
                    last_results = [''] * N_AGENTS
                elif ev.key == K_SPACE:
                    mode = 'watch'

        if wa_pause > 0:
            wa_pause -= 1
            cam_tgt = np.mean(wa_positions, axis=0)
        elif wa_done_all:
            wa_obs = env.reset()
            trainer.policy.executor.reset_temporal()
            wa_trails = [[env.positions[i].copy()] for i in range(N_AGENTS)]
            wa_done_all = False
            wa_positions = env.positions.copy()
            cam_tgt = np.mean(wa_positions, axis=0)
            wa_attn_cache = None
        else:
            for _ in range(WATCH_SPF):
                if wa_done_all:
                    break
                egos, node_feats, adjs, nbrs, nbr_masks, g_obs = wa_obs
                acts, _, _, _, _, _ = trainer.act(
                    egos, node_feats, adjs, nbrs, nbr_masks, g_obs, env,
                    agent_mask=env.dones.copy(),
                    deterministic=True)
                wa_attn_cache = trainer.policy.executor.graph_attn_weights()

                wa_obs, _, dones, results = env.step(acts)
                wa_positions = env.positions.copy()

                for i in range(N_AGENTS):
                    if not dones[i]:
                        wa_trails[i].append(wa_positions[i].copy())

                if np.all(dones):
                    last_results = results[:]
                    if any(r == 'wall' for r in results):
                        flash_red = 12
                    wa_done_all = True
                    wa_pause = WATCH_PAUSE

            cam_tgt = np.mean(wa_positions, axis=0)

        if flash_red > 0:
            glClearColor(0.28, 0.0, 0.05, 1.0); flash_red -= 1
        else:
            glClearColor(0.03, 0.03, 0.09, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        gluPerspective(45, W / H, 0.1, 1000.0)
        glMatrixMode(GL_MODELVIEW); glLoadIdentity()
        cam.apply(cam_tgt)

        attn_lines = None
        if wa_attn_cache is not None and wa_obs is not None:
            attn_lines = []
            _, wa_node_feats, _, _, _, _ = wa_obs
            nf0 = wa_node_feats[0]
            cur_idx = int(np.argmax(nf0[:, 6]))
            row = wa_attn_cache[cur_idx]
            top_idx = np.argsort(-row)[:min(4, len(row))]
            p0 = wa_positions[0].copy()
            for k in top_idx:
                rel = nf0[k, :3] * env.net.extent
                pk = p0 + rel
                attn_lines.append((p0, pk, float(np.clip(row[k], 0.0, 1.0))))

        draw_pipe_network(
            net,
            starts=env.start_nodes[:],
            goals=env.goal_nodes[:],
            trails=wa_trails,
            waypoints_list=env.waypoints[:],
            attn_lines=attn_lines)

        # 渲染规划器生成的全局路径
        if RENDER_PLANNER_PATH:
            glLineWidth(PLANNER_PATH_WIDTH)
            for i in range(N_AGENTS):
                if env.dones[i]:
                    continue
                c = AGENT_COLORS[i % len(AGENT_COLORS)]
                route = env.route_plan[i]
                if len(route) < 2:
                    continue
                glBegin(GL_LINE_STRIP)
                glColor4f(c[0], c[1], c[2], PLANNER_PATH_ALPHA)
                for node_id in route:
                    pos = env.net.nodes[node_id]
                    glVertex3f(*pos)
                glEnd()
            glLineWidth(1.0)

        # Ultra-thin dashed guide lines: agent -> its goal (real-time)
        glLineWidth(1.0)
        for i in range(N_AGENTS):
            if env.dones[i]:
                continue
            c = AGENT_COLORS[i % len(AGENT_COLORS)]
            p0 = wa_positions[i]
            p1 = env.goals[i]
            segs = 28
            for k in range(0, segs, 2):
                t0 = k / segs
                t1 = (k + 1) / segs
                a = p0 * (1.0 - t0) + p1 * t0
                b = p0 * (1.0 - t1) + p1 * t1
                glBegin(GL_LINES)
                glColor4f(c[0], c[1], c[2], 0.28)
                glVertex3f(*a)
                glVertex3f(*b)
                glEnd()
        glLineWidth(1.0)

        for i in range(N_AGENTS):
            if env.dones[i]:
                continue
            c = AGENT_COLORS[i % len(AGENT_COLORS)]
            glPushMatrix()
            glTranslatef(*wa_positions[i])
            glColor3f(*c[:3])
            gluSphere(net.q, AGENT_R * 3.0, 16, 16)
            glPopMatrix()

        hud.render(trainer, 'watch', env, last_results)
        pygame.display.flip()
        clock.tick(60)


if __name__ == '__main__':
    main()
