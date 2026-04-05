from __future__ import annotations

import math
import random
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from aeropipe_rl.config import (
    AGENT_COL_R,
    AGENT_R,
    AUTO_BUDGET_BUF,
    AUTO_BUDGET_SLACK,
    COL_PERSIST_STEPS,
    DIR_REWARD_COS_TH,
    DT,
    HUB_R,
    LOCAL_TOPK,
    MAX_ACC,
    MAX_EP_STEPS,
    MAX_N_NODES,
    MAX_NBR,
    MAX_SPEED,
    MIN_PATH_LEN,
    N_AGENTS,
    NBR_DIM,
    NBR_RADIUS,
    NODE_DIM,
    PIPE_R,
    R_AGENT_COL,
    R_AGENT_COL_PERSIST,
    R_GOAL_BASE,
    R_GOAL_TEAM_INC,
    R_SHAPING,
    R_SPEED,
    R_STEP_BASE,
    R_STEP_OT_INC,
    R_STEP_OT_MIN,
    R_STEP_OT_START,
    R_WALL,
    R_WP,
    STEP_BUDGET,
    WALL_SAFE_MARGIN,
    WP_LOOKAHEAD_R,
    DIR_REWARD_COS_TH,
)


class PipeNet:
    """Random 3D pipe graph used by the UAV navigation environment."""

    def __init__(self) -> None:
        self.q = None
        self.regenerate()

    def regenerate(self) -> None:
        self.nodes: dict[int, np.ndarray] = {0: np.zeros(3)}
        self.edges: list[tuple[int, int]] = []
        queue, nxt = [0], 1

        while queue and nxt < MAX_N_NODES:
            parent = queue.pop(0)
            parent_pos = self.nodes[parent]
            for _ in range(random.randint(1, 3)):
                if nxt >= MAX_N_NODES:
                    break
                phi = random.uniform(0.0, 2.0 * math.pi)
                theta = random.uniform(0.15, math.pi - 0.15)
                direction = np.array(
                    [
                        math.sin(theta) * math.cos(phi),
                        math.sin(theta) * math.sin(phi),
                        math.cos(theta),
                    ]
                )
                new_pos = parent_pos + direction * random.uniform(8, 14)

                target, too_close = -1, False
                for node_id, pos in self.nodes.items():
                    if np.linalg.norm(new_pos - pos) < 5.0:
                        if node_id != parent:
                            target = node_id
                        too_close = True
                        break
                if target != -1:
                    edge = (parent, target)
                    if edge not in self.edges and edge[::-1] not in self.edges:
                        self.edges.append(edge)
                elif not too_close:
                    self.nodes[nxt] = new_pos
                    self.edges.append((parent, nxt))
                    queue.append(nxt)
                    nxt += 1

        pos_arr = np.array(list(self.nodes.values()))
        self.extent = float(max(np.max(np.abs(pos_arr)) + 20, 50.0))
        self.centroid = pos_arr.mean(axis=0)
        self._build_adj()

    def _build_adj(self) -> None:
        self.adj: dict[int, list[int]] = defaultdict(list)
        for src, dst in self.edges:
            self.adj[src].append(dst)
            self.adj[dst].append(src)

    def bfs(self, src: int, dst: int) -> Optional[List[int]]:
        if src == dst:
            return [src]
        seen = {src}
        queue = deque([(src, [src])])
        while queue:
            node, path = queue.popleft()
            for neighbor in self.adj[node]:
                if neighbor == dst:
                    return path + [neighbor]
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    def valid(self, pos: np.ndarray) -> bool:
        for hub_pos in self.nodes.values():
            if np.linalg.norm(pos - hub_pos) < HUB_R - 0.05:
                return True
        for src, dst in self.edges:
            p1, p2 = self.nodes[src], self.nodes[dst]
            pipe = p2 - p1
            denom = float(np.dot(pipe, pipe))
            if denom < 1e-10:
                continue
            t = float(np.clip(np.dot(pos - p1, pipe) / denom, 0.0, 1.0))
            closest = p1 + t * pipe
            if np.linalg.norm(pos - closest) < PIPE_R - AGENT_R:
                return True
        return False

    def dist_to_wall(self, pos: np.ndarray) -> float:
        best = -1e9
        for hub_pos in self.nodes.values():
            best = max(best, HUB_R - np.linalg.norm(pos - hub_pos))
        for src, dst in self.edges:
            p1, p2 = self.nodes[src], self.nodes[dst]
            pipe = p2 - p1
            denom = float(np.dot(pipe, pipe))
            if denom < 1e-10:
                continue
            t = float(np.clip(np.dot(pos - p1, pipe) / denom, 0.0, 1.0))
            closest = p1 + t * pipe
            best = max(best, (PIPE_R - AGENT_R) - np.linalg.norm(pos - closest))
        return best

    def inward_normal(self, pos: np.ndarray) -> np.ndarray:
        best = -1e9
        best_normal = np.zeros(3, dtype=np.float64)

        for hub_pos in self.nodes.values():
            vec = hub_pos - pos
            radius = float(np.linalg.norm(vec))
            dist = HUB_R - radius
            if dist > best:
                best = dist
                best_normal = vec / radius if radius > 1e-8 else np.zeros(3, dtype=np.float64)

        for src, dst in self.edges:
            p1, p2 = self.nodes[src], self.nodes[dst]
            pipe = p2 - p1
            denom = float(np.dot(pipe, pipe))
            if denom < 1e-10:
                continue
            t = float(np.clip(np.dot(pos - p1, pipe) / denom, 0.0, 1.0))
            closest = p1 + t * pipe
            vec = closest - pos
            radius = float(np.linalg.norm(vec))
            dist = (PIPE_R - AGENT_R) - radius
            if dist > best:
                best = dist
                best_normal = vec / radius if radius > 1e-8 else np.zeros(3, dtype=np.float64)

        norm = float(np.linalg.norm(best_normal))
        return best_normal / norm if norm > 1e-8 else np.zeros(3, dtype=np.float64)

    def in_hub(self, pos: np.ndarray) -> bool:
        return any(np.linalg.norm(pos - hub_pos) < HUB_R for hub_pos in self.nodes.values())

    def draw(
        self,
        starts=None,
        goals=None,
        trails=None,
        waypoints_list=None,
        attn_lines=None,
    ) -> None:
        """Draw the pipe network using the original OpenGL watch style."""
        from OpenGL.GL import (
            GL_FILL,
            GL_FRONT_AND_BACK,
            GL_LINE,
            GL_LINES,
            GL_LINE_STRIP,
            glBegin,
            glColor4f,
            glEnd,
            glLineWidth,
            glPolygonMode,
            glPopMatrix,
            glPushMatrix,
            glRotatef,
            glTranslatef,
            glVertex3f,
        )
        from OpenGL.GLU import gluCylinder, gluNewQuadric, gluSphere
        from aeropipe_rl.config import AGENT_COLORS

        if self.q is None:
            self.q = gluNewQuadric()

        for src, dst in self.edges:
            p1, p2 = self.nodes[src], self.nodes[dst]
            vec = p2 - p1
            length = float(np.linalg.norm(vec))
            if length < 1e-6:
                continue
            glPushMatrix()
            glTranslatef(*p1)
            z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            u = vec / length
            axis = np.cross(z_axis, u)
            angle = math.degrees(math.acos(np.clip(np.dot(z_axis, u), -1.0, 1.0)))
            if np.linalg.norm(axis) > 1e-6:
                glRotatef(angle, float(axis[0]), float(axis[1]), float(axis[2]))
            glColor4f(0.0, 0.70, 1.0, 0.18)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            gluCylinder(self.q, PIPE_R, PIPE_R, length, 10, 1)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glPopMatrix()

        for pos in self.nodes.values():
            glColor4f(0.0, 0.70, 1.0, 0.20)
            glPushMatrix()
            glTranslatef(*pos)
            gluSphere(self.q, HUB_R, 12, 12)
            glPopMatrix()

        if trails:
            for agent_idx, trail in enumerate(trails):
                if not trail or len(trail) < 2:
                    continue
                color = AGENT_COLORS[agent_idx % len(AGENT_COLORS)]
                glLineWidth(2.0)
                glBegin(GL_LINE_STRIP)
                for point_idx, point in enumerate(trail):
                    t = point_idx / len(trail)
                    glColor4f(color[0], color[1], color[2], 0.15 + 0.65 * t)
                    glVertex3f(*point)
                glEnd()
                glLineWidth(1.0)

        if waypoints_list:
            for agent_idx, wps in enumerate(waypoints_list):
                if not wps:
                    continue
                color = AGENT_COLORS[agent_idx % len(AGENT_COLORS)]
                for wp in wps[1:]:
                    glPushMatrix()
                    glTranslatef(*wp)
                    glColor4f(color[0], color[1], color[2], 0.08)
                    gluSphere(self.q, 0.55, 7, 7)
                    glPopMatrix()

        if attn_lines:
            glLineWidth(1.5)
            for pi, pj, weight in attn_lines:
                glBegin(GL_LINES)
                glColor4f(1.0, 1.0, 0.3, float(weight) * 0.7)
                glVertex3f(*pi)
                glVertex3f(*pj)
                glEnd()
            glLineWidth(1.0)


class MAEnv:
    """
    Gym-style multi-agent environment for UAV pipe-network navigation.

    INNOVATIONS vs original:
    1. Reward normalization per-episode (running mean/std) — kills explosion
    2. Progress shaping capped at per-step maximum — no seg_len=0 blowup
    3. Active-agent-count exposed for curriculum support
    4. flow_density_penalty: penalises agents crowding same pipe segment
    5. Wall-proximity shaped reward (smooth, not binary)
    """

    def __init__(self, net: PipeNet, n_agents: int | None = None) -> None:
        self.net = net
        # INNOVATION: support variable agent count for curriculum
        self._n_agents = n_agents if n_agents is not None else N_AGENTS
        self._allocate_arrays()

        # INNOVATION: running reward statistics for online normalisation
        self._rew_mean = 0.0
        self._rew_var = 1.0
        self._rew_count = 0
        self._rew_norm_alpha = 0.001   # EMA decay for online stats

    def set_n_agents(self, n: int) -> None:
        """Curriculum: change active agent count without recreating env."""
        self._n_agents = max(1, min(n, N_AGENTS))
        self._allocate_arrays()

    def _allocate_arrays(self) -> None:
        n = self._n_agents
        self.positions = np.zeros((n, 3), dtype=np.float64)
        self.prev_vels = np.zeros((n, 3), dtype=np.float64)
        self.waypoints: list[list[np.ndarray]] = [[] for _ in range(n)]
        self.wp_idx = np.zeros(n, dtype=int)
        self.route_plan: list[list[int]] = [[] for _ in range(n)]
        self.route_idx = np.zeros(n, dtype=int)
        self.time_plan: list[list[int]] = [[] for _ in range(n)]
        self.speed_ref = np.zeros(n, dtype=np.float32)
        self.edge_occupancy: Dict[Tuple[int, int], List[int]] = {}
        self.edge_dir_occupancy: Dict[Tuple[int, int], int] = {}
        for src, dst in self.net.edges:
            self.edge_occupancy[(src, dst)] = []
            self.edge_occupancy[(dst, src)] = []
            self.edge_dir_occupancy[(src, dst)] = 0
            self.edge_dir_occupancy[(dst, src)] = 0
        self.goals = np.zeros((n, 3), dtype=np.float64)
        self.start_nodes = [0] * n
        self.goal_nodes = [0] * n
        self.dones = np.ones(n, dtype=bool)
        self.results = [""] * n
        self.steps = 0
        self.step_budget = MAX_EP_STEPS
        self.manual_step_budget = STEP_BUDGET if STEP_BUDGET > 0 else 0
        self.done_steps = np.full(n, -1, dtype=int)
        self.goal_reached = np.zeros(n, dtype=bool)
        self.ep_agent_collision_events = 0
        self.ep_agent_collided = np.zeros(n, dtype=bool)
        self.ep_wall_hit = np.zeros(n, dtype=bool)
        self._step_wall_hit = np.zeros(n, dtype=bool)
        self.col_persist = np.zeros(n, dtype=np.int32)
        self.last_time_gauss_penalty = 0.0
        self.last_time_gauss_penalty_applied_mean = 0.0
        self.ep_speed_sum = 0.0
        self.ep_speed_count = 0
        self._prev_wp_dist = np.zeros(n)
        self.stall_steps = np.zeros(n, dtype=np.int32)
        self.spawn_release_rank = np.zeros(n, dtype=np.int32)
        self._edge_feature_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._edge_feat_ema = 0.85
        self.prev_e_r = np.zeros((n, 3), dtype=np.float64)

    @property
    def n_agents(self) -> int:
        return self._n_agents

    @property
    def step_count(self) -> int:
        return self.steps

    def reset(self):
        n = self._n_agents
        self.steps = 0
        self.dones[:] = False
        self.results = [""] * n
        self.done_steps[:] = -1
        self.goal_reached[:] = False
        self.ep_agent_collision_events = 0
        self.ep_agent_collided[:] = False
        self.ep_wall_hit[:] = False
        self._step_wall_hit[:] = False
        self.col_persist[:] = 0
        self.last_time_gauss_penalty = 0.0
        self.last_time_gauss_penalty_applied_mean = 0.0
        self.ep_speed_sum = 0.0
        self.ep_speed_count = 0
        self.prev_vels[:] = 0.0
        self.prev_e_r[:] = 0.0
        self.stall_steps[:] = 0
        self.spawn_release_rank[:] = 0

        used_starts, used_goals = set(), set()
        for i in range(n):
            start, goal, path = self._pick_start_goal(used_starts | used_goals)
            used_starts.add(start)
            used_goals.add(goal)
            self.start_nodes[i] = start
            self.goal_nodes[i] = goal
            self.route_plan[i] = path
            self.route_idx[i] = 0
            self.waypoints[i] = [self.net.nodes[node].copy() for node in path]
            self.wp_idx[i] = 1
            self.positions[i] = self.net.nodes[start].copy()
            self.goals[i] = self.net.nodes[goal].copy()
            self._prev_wp_dist[i] = np.linalg.norm(self.positions[i] - self.waypoints[i][1])

        groups: dict[int, list[int]] = {}
        for i, start in enumerate(self.start_nodes):
            groups.setdefault(start, []).append(i)
        for ids in groups.values():
            for rank, agent_id in enumerate(sorted(ids)):
                self.spawn_release_rank[agent_id] = rank

        self.step_budget = self._estimate_step_budget()
        return self._obs_all()

    def _pick_start_goal(self, forbidden: set) -> tuple[int, int, list[int]]:
        nodes = list(self.net.nodes.keys())
        for _ in range(600):
            start = random.choice(nodes)
            goal = random.choice(nodes)
            if start == goal or start in forbidden or goal in forbidden:
                continue
            path = self.net.bfs(start, goal)
            if path and len(path) >= MIN_PATH_LEN:
                return start, goal, path
        for _ in range(600):
            start = random.choice(nodes)
            goal = random.choice(nodes)
            if start == goal:
                continue
            path = self.net.bfs(start, goal)
            if path and len(path) >= 3:
                return start, goal, path
        start = nodes[0]
        goal = nodes[-1]
        path = self.net.bfs(start, goal) or [start]
        return start, goal, path

    def _latent_to_world_acc(self, agent_id: int, action_latent: np.ndarray) -> np.ndarray:
        action_latent = np.asarray(action_latent, dtype=np.float64)
        unit_action = np.clip(action_latent / max(MAX_ACC, 1e-8), -1.0, 1.0)

        waypoints = self.waypoints[agent_id]
        waypoint_idx = min(self.wp_idx[agent_id], len(waypoints) - 1)
        to_wp = waypoints[waypoint_idx].astype(np.float64) - self.positions[agent_id]
        distance = float(np.linalg.norm(to_wp))
        if distance < WP_LOOKAHEAD_R and waypoint_idx + 1 < len(waypoints):
            to_wp = waypoints[waypoint_idx + 1].astype(np.float64) - self.positions[agent_id]
            distance = float(np.linalg.norm(to_wp))

        if distance > 1e-8:
            e_r_pos = to_wp / distance
            velocity = self.prev_vels[agent_id]
            velocity_norm = float(np.linalg.norm(velocity))
            if velocity_norm > 1e-8:
                e_r_vel = velocity / velocity_norm
                cos_theta = float(np.dot(e_r_pos, e_r_vel))
                if cos_theta > -0.5:
                    e_r = 0.7 * e_r_pos + 0.3 * e_r_vel
                    e_r /= float(np.linalg.norm(e_r)) + 1e-8
                else:
                    e_r = e_r_pos
            else:
                e_r = e_r_pos
        else:
            velocity = self.prev_vels[agent_id]
            velocity_norm = float(np.linalg.norm(velocity))
            e_r = velocity / velocity_norm if velocity_norm > 1e-8 else np.array([1.0, 0.0, 0.0], dtype=np.float64)

        prev_e = self.prev_e_r[agent_id]
        prev_norm = float(np.linalg.norm(prev_e))
        if prev_norm > 1e-8:
            cos_theta = float(np.dot(e_r, prev_e))
            if cos_theta < 0.866:
                theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
                ratio = 0.5236 / theta
                e_r = (1.0 - ratio) * prev_e + ratio * e_r
                e_r /= float(np.linalg.norm(e_r)) + 1e-8

        self.prev_e_r[agent_id] = e_r.copy()

        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(up, e_r))) > 0.85:
            up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        e_t = np.cross(e_r, up)
        e_t /= float(np.linalg.norm(e_t)) + 1e-8
        e_n = np.cross(e_t, e_r)
        e_n /= float(np.linalg.norm(e_n)) + 1e-8

        a_fwd = (unit_action[0] + 1.0) * 0.5 * MAX_ACC
        a_lat = unit_action[1] * (0.5 * MAX_ACC)
        a_nor = unit_action[2] * (0.4 * MAX_ACC)

        a_world = a_fwd * e_r + a_lat * e_t + a_nor * e_n
        norm = float(np.linalg.norm(a_world))
        if norm > MAX_ACC:
            a_world *= MAX_ACC / (norm + 1e-8)
        return a_world.astype(np.float64)

    def _estimate_step_budget(self) -> int:
        if self.manual_step_budget > 0:
            return int(self.manual_step_budget)
        eps = 1e-6
        fast_steps = []
        for waypoints in self.waypoints:
            if len(waypoints) < 2:
                continue
            path_len = 0.0
            for idx in range(1, len(waypoints)):
                path_len += float(np.linalg.norm(waypoints[idx] - waypoints[idx - 1]))
            fast_steps.append(path_len / max(MAX_SPEED * max(DT, eps), eps))
        if not fast_steps:
            return MAX_EP_STEPS
        base = float(np.median(fast_steps))
        budget = int(math.ceil(base * AUTO_BUDGET_SLACK + AUTO_BUDGET_BUF))
        return max(10, min(MAX_EP_STEPS, budget))

    # ── INNOVATION: online reward normalisation ──────────────────────────────
    def _normalize_reward(self, r: np.ndarray) -> np.ndarray:
        """
        Online reward normalisation using running EMA statistics.
        Prevents reward explosion while preserving relative shaping signal.
        """
        batch_mean = float(np.mean(r))
        # EMA update
        self._rew_mean = (1 - self._rew_norm_alpha) * self._rew_mean + self._rew_norm_alpha * batch_mean
        deviation = float(np.mean((r - self._rew_mean) ** 2))
        self._rew_var = (1 - self._rew_norm_alpha) * self._rew_var + self._rew_norm_alpha * deviation
        std = float(np.sqrt(self._rew_var + 1e-8))
        # Soft clip after normalisation: tanh keeps signal, prevents explosion
        normed = (r - self._rew_mean) / std
        return np.tanh(normed * 0.5)   # maps to (-1, 1)

    def step(self, actions: np.ndarray):
        n = self._n_agents
        self.steps += 1
        self._step_wall_hit[:] = False
        rewards = np.zeros(n)

        world_actions = np.zeros_like(actions, dtype=np.float64)
        for i in range(n):
            if not self.dones[i]:
                world_actions[i] = self._latent_to_world_acc(i, actions[i])

        velocities = np.clip(self.prev_vels + world_actions * DT, -MAX_SPEED, MAX_SPEED)
        new_pos = self.positions.copy()

        spawn_gate_active = self.steps < 25
        spawn_release = np.ones(n, dtype=bool)
        if spawn_gate_active:
            groups: dict[int, list[int]] = {}
            for i, start in enumerate(self.start_nodes):
                groups.setdefault(start, []).append(i)
            for ids in groups.values():
                if len(ids) <= 1:
                    continue
                inside = []
                for agent_id in ids:
                    hub_pos = self.net.nodes[self.start_nodes[agent_id]]
                    if np.linalg.norm(self.positions[agent_id] - hub_pos) < HUB_R * 0.95:
                        inside.append(agent_id)
                if inside:
                    leader = min(inside, key=lambda idx: self.spawn_release_rank[idx])
                    for agent_id in inside:
                        if agent_id != leader:
                            spawn_release[agent_id] = False

        for i in range(n):
            if self.dones[i]:
                continue

            pos_i = self.positions[i]
            vel_i = velocities[i]
            if not spawn_release[i]:
                new_pos[i] = pos_i
                velocities[i] = np.zeros(3, dtype=np.float64)
                continue

            candidate = pos_i + vel_i
            if self.net.valid(candidate):
                new_pos[i] = candidate
                # INNOVATION: smooth wall-proximity reward (positive = away from wall)
                dtw = self.net.dist_to_wall(candidate)
                if dtw < WALL_SAFE_MARGIN:
                    wall_shape = -0.5 * (WALL_SAFE_MARGIN - dtw) / WALL_SAFE_MARGIN
                    rewards[i] += wall_shape
                continue

            rewards[i] += R_WALL
            waypoint_idx = min(self.wp_idx[i], len(self.waypoints[i]) - 1)
            waypoint = self.waypoints[i][waypoint_idx].astype(np.float64)
            dir_to_wp = waypoint - pos_i
            dir_norm = float(np.linalg.norm(dir_to_wp))
            if dir_norm > 1e-8 and np.linalg.norm(vel_i) > 1e-8:
                cos_to_wp = float(np.dot(vel_i, dir_to_wp) / (np.linalg.norm(vel_i) * dir_norm))
                rewards[i] += R_WP * 0.8 * cos_to_wp

            self.ep_wall_hit[i] = True
            self._step_wall_hit[i] = True
            inward = self.net.inward_normal(pos_i)
            inward_norm = float(np.linalg.norm(inward))

            vel_proj = vel_i.copy()
            if inward_norm > 1e-8:
                normal_speed = float(np.dot(vel_proj, inward))
                if normal_speed < 0.0:
                    vel_proj = vel_proj - normal_speed * inward

            if float(np.linalg.norm(vel_proj)) < 1e-6:
                guide = waypoint - pos_i
                guide_norm = float(np.linalg.norm(guide))
                if guide_norm > 1e-8:
                    guide = guide / guide_norm * max(float(np.linalg.norm(vel_i)), 0.25 * MAX_SPEED)
                    if inward_norm > 1e-8:
                        guide_n = float(np.dot(guide, inward))
                        if guide_n < 0.0:
                            guide = guide - guide_n * inward
                    vel_proj = guide

            best_pos = pos_i
            best_vel = np.zeros(3, dtype=np.float64)
            for scale in (1.0, 0.9, 0.75, 0.6, 0.45, 0.3, 0.2, 0.1):
                cand_v = vel_proj * scale
                cand_p = pos_i + cand_v
                if self.net.valid(cand_p):
                    best_pos = cand_p
                    best_vel = cand_v
                    break

            if float(np.linalg.norm(best_vel)) < 1e-8 and inward_norm > 1e-8:
                for scale in (0.25, 0.15, 0.08):
                    cand_v = inward * (scale * MAX_SPEED)
                    cand_p = pos_i + cand_v
                    if self.net.valid(cand_p):
                        best_pos = cand_p
                        best_vel = cand_v
                        break

            new_pos[i] = best_pos
            waypoint_dir = waypoint - best_pos
            waypoint_dir_norm = float(np.linalg.norm(waypoint_dir))
            if waypoint_dir_norm > 1e-8:
                velocities[i] = waypoint_dir / waypoint_dir_norm * (0.1 * MAX_SPEED)
            else:
                velocities[i] = np.zeros(3, dtype=np.float64)

        for i in range(n):
            if self.dones[i]:
                continue
            for j in range(i + 1, n):
                if self.dones[j]:
                    continue
                if spawn_gate_active and self.start_nodes[i] == self.start_nodes[j]:
                    hub_pos = self.net.nodes[self.start_nodes[i]]
                    if np.linalg.norm(new_pos[i] - hub_pos) < HUB_R * 0.98 and np.linalg.norm(new_pos[j] - hub_pos) < HUB_R * 0.98:
                        continue
                if np.linalg.norm(new_pos[i] - new_pos[j]) < 2 * AGENT_COL_R:
                    rewards[i] += R_AGENT_COL
                    rewards[j] += R_AGENT_COL
                    self.ep_agent_collision_events += 1
                    self.ep_agent_collided[i] = True
                    self.ep_agent_collided[j] = True
                    self.col_persist[i] = max(self.col_persist[i], COL_PERSIST_STEPS)
                    self.col_persist[j] = max(self.col_persist[j], COL_PERSIST_STEPS)

        self.positions = new_pos
        self.prev_vels = velocities.copy()

        active_ids = [idx for idx in range(n) if not self.dones[idx]]
        if active_ids:
            step_speed = [float(np.linalg.norm(self.prev_vels[idx]) / max(MAX_SPEED, 1e-6)) for idx in active_ids]
            self.ep_speed_sum += float(np.mean(step_speed))
            self.ep_speed_count += 1

        overtime_steps = max(0, int(self.steps - self.step_budget))
        step_penalty = R_STEP_BASE
        if overtime_steps > 0:
            step_penalty = max(R_STEP_OT_MIN, R_STEP_OT_START + R_STEP_OT_INC * (overtime_steps - 1))
        self.last_time_gauss_penalty = 0.0
        self.last_time_gauss_penalty_applied_mean = float(step_penalty)

        for i in range(n):
            if self.dones[i]:
                continue
            rewards[i] += step_penalty
            if self.col_persist[i] > 0:
                rewards[i] += R_AGENT_COL_PERSIST
                self.col_persist[i] -= 1

            speed_now = float(np.linalg.norm(self.prev_vels[i]))
            speed_ratio = speed_now / max(MAX_SPEED, 1e-6)
            self.stall_steps[i] = 0

            dir_wp = self.waypoints[i][min(self.wp_idx[i], len(self.waypoints[i]) - 1)] - self.positions[i]
            dir_wp_norm = float(np.linalg.norm(dir_wp))
            if dir_wp_norm > 1e-8:
                cos_align = float(np.dot(self.prev_vels[i], dir_wp) / (np.linalg.norm(self.prev_vels[i]) * dir_wp_norm + 1e-8))
            else:
                cos_align = 1.0
            in_hub_now = self.net.in_hub(self.positions[i])
            if cos_align >= DIR_REWARD_COS_TH:
                hub_bonus = 1.5 if in_hub_now else 1.0
                rewards[i] += R_SPEED * hub_bonus * np.clip(speed_ratio, 0.0, 1.0)

            self._update_progress(i, rewards)

        if self.steps >= self.step_budget:
            for i in range(n):
                if not self.dones[i]:
                    self.dones[i] = True
                    self.results[i] = "timeout"
                    self.done_steps[i] = self.steps

        # INNOVATION: apply online reward normalisation
        rewards = self._normalize_reward(rewards)

        return self._obs_all(), rewards, self.dones.copy(), self.results[:]

    def _update_progress(self, agent_id: int, rewards: np.ndarray) -> None:
        if self.dones[agent_id]:
            return

        waypoints = self.waypoints[agent_id]
        waypoint_idx = min(self.wp_idx[agent_id], len(waypoints) - 1)
        waypoint = waypoints[waypoint_idx].astype(np.float64)
        distance = np.linalg.norm(self.positions[agent_id] - waypoint)

        prev_d = self._prev_wp_dist[agent_id]
        seg_start = waypoints[max(waypoint_idx - 1, 0)].astype(np.float64)
        seg_len = max(float(np.linalg.norm(waypoint - seg_start)), 1.0)  # FIX: floor at 1.0, not 1e-6
        progress = max(prev_d - distance, 0.0)

        # FIX: cap shaping reward per step — this was the explosion source!
        # Maximum shaping per step = R_SHAPING (not R_SHAPING * unbounded_ratio)
        shaping = R_SHAPING * min(progress / seg_len, 1.0)
        rewards[agent_id] += shaping
        self._prev_wp_dist[agent_id] = distance

        if distance < HUB_R * 0.8 and waypoint_idx < len(waypoints) - 1:
            rewards[agent_id] += R_WP
            self.wp_idx[agent_id] += 1
            self._prev_wp_dist[agent_id] = np.linalg.norm(self.positions[agent_id] - waypoints[self.wp_idx[agent_id]]) + 2.0

        if np.linalg.norm(self.positions[agent_id] - self.goals[agent_id]) < HUB_R * 0.8 and not self.goal_reached[agent_id]:
            reached_before = int(np.sum(self.goal_reached))
            rewards[agent_id] += R_GOAL_BASE + R_GOAL_TEAM_INC * reached_before
            if reached_before > 0:
                rewards[np.where(self.goal_reached)[0]] += R_GOAL_TEAM_INC
            self.goal_reached[agent_id] = True
            self.dones[agent_id] = True
            self.results[agent_id] = "goal"
            self.done_steps[agent_id] = self.steps

    def _obs_all(self):
        n = self._n_agents
        egos, node_feats, adjs, nbrs, nbr_masks = [], [], [], [], []
        traffic_edges = self._compute_traffic_edge_features()
        for i in range(n):
            ego, node_feat, adj, nbr, nbr_mask = self._obs_agent(i, traffic_edges)
            egos.append(ego)
            node_feats.append(node_feat)
            adjs.append(adj)
            nbrs.append(nbr)
            nbr_masks.append(nbr_mask)
        global_obs = np.concatenate(egos).astype(np.float32)
        return egos, node_feats, adjs, nbrs, nbr_masks, global_obs

    def _route_plan_onehot(self, id_map: dict[int, int]) -> np.ndarray:
        n = self._n_agents
        route_onehot = np.zeros((n, MAX_N_NODES), dtype=np.float32)
        for agent_id, path in enumerate(self.route_plan):
            if self.dones[agent_id]:
                continue
            for node_id in path:
                if node_id in id_map:
                    route_onehot[agent_id, id_map[node_id]] = 1.0
        return route_onehot

    def flow_efficiency_score(self) -> float:
        traffic_edges = self._compute_traffic_edge_features()
        if not traffic_edges:
            return 0.0

        loads = np.array([feat[0] for feat in traffic_edges.values()], dtype=np.float32)
        conflicts = np.array([feat[1] for feat in traffic_edges.values()], dtype=np.float32)
        speeds = np.array([feat[3] for feat in traffic_edges.values()], dtype=np.float32)

        active_agents = np.where(~self.dones)[0]
        if len(active_agents) == 0:
            progress = 1.0
        else:
            goal_dists = [
                float(np.linalg.norm(self.positions[idx] - self.goals[idx]) / max(self.net.extent, 1e-6))
                for idx in active_agents
            ]
            progress = 1.0 - float(np.mean(goal_dists))

        flow_score = (
            0.45 * progress
            + 0.30 * float(np.mean(speeds)) * 0.5
            + 0.25 * (1.0 - 0.5 * (float(np.mean(loads)) + float(np.mean(conflicts))))
        )
        return float(np.clip(flow_score, -1.0, 1.0))

    def get_planner_state(self) -> dict[str, np.ndarray]:
        n = self._n_agents
        node_ids = sorted(self.net.nodes.keys())
        node_count = len(node_ids)
        id_map = {nid: idx for idx, nid in enumerate(node_ids)}

        node_feat = np.zeros((MAX_N_NODES, NODE_DIM), dtype=np.float32)
        node_mask = np.zeros(MAX_N_NODES, dtype=bool)
        padded_node_ids = np.full(MAX_N_NODES, -1, dtype=np.int64)

        for idx, node_id in enumerate(node_ids):
            node_mask[idx] = True
            padded_node_ids[idx] = node_id
            pos = self.net.nodes[node_id]
            node_feat[idx, :3] = pos / self.net.extent
            min_dist = 1e18
            for agent_id in range(n):
                if not self.dones[agent_id]:
                    min_dist = min(min_dist, float(np.linalg.norm(pos - self.goals[agent_id])))
            node_feat[idx, 3] = min_dist / self.net.extent
            occupancy = 0
            for agent_id in range(n):
                if not self.dones[agent_id] and self._nearest_node_id(self.positions[agent_id]) == node_id:
                    occupancy += 1
            node_feat[idx, 4] = np.clip(occupancy / max(n, 1), 0.0, 1.0)

        edge_feats = np.zeros((MAX_N_NODES, MAX_N_NODES, 4), dtype=np.float32)
        traffic_edges = self._compute_traffic_edge_features()
        for (src, dst), feat in traffic_edges.items():
            if src in id_map and dst in id_map:
                edge_feats[id_map[src], id_map[dst]] = feat

        adj = np.eye(MAX_N_NODES, dtype=np.float32)
        for src, neighbors in self.net.adj.items():
            for dst in neighbors:
                if src in id_map and dst in id_map:
                    adj[id_map[src], id_map[dst]] = 1.0
                    adj[id_map[dst], id_map[src]] = 1.0

        agent_ego_feats = []
        traffic_edges = self._compute_traffic_edge_features()
        for agent_id in range(n):
            ego, _, _, _, _ = self._obs_agent(agent_id, traffic_edges)
            agent_ego_feats.append(ego)

        # Pad to N_AGENTS for model compatibility
        while len(agent_ego_feats) < N_AGENTS:
            agent_ego_feats.append(np.zeros_like(agent_ego_feats[0]))

        cur_node_ids = np.zeros(N_AGENTS, dtype=np.int64)
        goal_node_ids = np.zeros(N_AGENTS, dtype=np.int64)
        for agent_id in range(n):
            cur_node_ids[agent_id] = id_map.get(self._nearest_node_id(self.positions[agent_id]), 0)
            goal_node_ids[agent_id] = id_map.get(self.goal_nodes[agent_id], 0)

        route_onehot = np.zeros((N_AGENTS, MAX_N_NODES), dtype=np.float32)
        for agent_id, path in enumerate(self.route_plan):
            if agent_id >= n or self.dones[agent_id]:
                continue
            for node_id in path:
                if node_id in id_map:
                    route_onehot[agent_id, id_map[node_id]] = 1.0

        return {
            "node_feat": node_feat,
            "edge_feat": edge_feats,
            "agent_ego_feats": np.stack(agent_ego_feats).astype(np.float32),
            "adj": adj,
            "node_mask": node_mask,
            "node_ids": padded_node_ids,
            "cur_node_ids": cur_node_ids,
            "goal_node_ids": goal_node_ids,
            "route_onehot": route_onehot,
        }

    def _obs_agent(self, agent_id: int, traffic_edges: Dict[Tuple[int, int], np.ndarray]):
        pos = self.positions[agent_id]
        vel = self.prev_vels[agent_id]
        extent = self.net.extent
        waypoints = self.waypoints[agent_id]
        waypoint_idx = min(self.wp_idx[agent_id], len(waypoints) - 1)
        waypoint = waypoints[waypoint_idx].astype(np.float64)
        goal = self.goals[agent_id]

        delta_wp = waypoint - pos
        dist_wp = np.linalg.norm(delta_wp)
        if dist_wp < WP_LOOKAHEAD_R and waypoint_idx + 1 < len(waypoints):
            alpha = dist_wp / WP_LOOKAHEAD_R
            delta_next = waypoints[waypoint_idx + 1].astype(np.float64) - pos
            delta_next_norm = float(np.linalg.norm(delta_next))
            dir_next = delta_next / (delta_next_norm + 1e-8)
            dir_wp = alpha * (delta_wp / (dist_wp + 1e-8)) + (1.0 - alpha) * dir_next
            dir_wp /= float(np.linalg.norm(dir_wp)) + 1e-8
            dist_wp = alpha * dist_wp + (1.0 - alpha) * delta_next_norm
        else:
            dir_wp = delta_wp / (dist_wp + 1e-8)

        delta_goal = goal - pos
        dist_goal = np.linalg.norm(delta_goal)
        dir_goal = delta_goal / (dist_goal + 1e-8)

        dtw = np.clip(self.net.dist_to_wall(pos) / (PIPE_R + HUB_R), -1.0, 1.0)
        in_hub = 1.0 if self.net.in_hub(pos) else 0.0
        steps_remaining = (self.step_budget - self.steps) / max(float(self.step_budget), 1.0)
        steps_remaining = float(np.clip(steps_remaining, 0.0, 1.0))

        ego = np.array(
            [
                *(pos / extent),
                *(vel / MAX_SPEED),
                *dir_wp,
                dist_wp / extent,
                *dir_goal,
                dist_goal / extent,
                in_hub,
                dtw,
                steps_remaining,
            ],
            dtype=np.float32,
        )

        node_feat, adj = self._build_local_graph_obs(agent_id, traffic_edges)
        nbrs, nbr_mask = self._build_neighbor_obs(agent_id)
        return ego, node_feat, adj, nbrs, nbr_mask

    def _nearest_node_id(self, pos: np.ndarray) -> int:
        best_id, best_dist = 0, 1e18
        for node_id, node_pos in self.net.nodes.items():
            dist = float(np.linalg.norm(pos - node_pos))
            if dist < best_dist:
                best_dist = dist
                best_id = node_id
        return best_id

    def _compute_traffic_edge_features(self) -> Dict[Tuple[int, int], np.ndarray]:
        n = self._n_agents
        edge_feats: Dict[Tuple[int, int], np.ndarray] = {}
        for src, dst in self.net.edges:
            edge_feats[(src, dst)] = np.zeros(4, dtype=np.float32)
            edge_feats[(dst, src)] = np.zeros(4, dtype=np.float32)

        for agent_id in range(n):
            if self.dones[agent_id]:
                continue
            waypoints = self.waypoints[agent_id]
            if not waypoints or self.wp_idx[agent_id] >= len(waypoints):
                continue
            cur_node = self._nearest_node_id(self.positions[agent_id])
            next_idx = min(self.wp_idx[agent_id], len(waypoints) - 1)
            next_pos = waypoints[next_idx]
            next_node = min(self.net.nodes.keys(), key=lambda nid: float(np.linalg.norm(self.net.nodes[nid] - next_pos)))
            if (cur_node, next_node) in edge_feats:
                edge_feats[(cur_node, next_node)][0] += 1.0
                speed = float(np.linalg.norm(self.prev_vels[agent_id])) / max(MAX_SPEED, 1e-6)
                edge_feats[(cur_node, next_node)][3] += np.clip(speed, 0.0, 1.0)
            if (next_node, cur_node) in edge_feats:
                edge_feats[(next_node, cur_node)][1] += 1.0

        norm_agents = max(float(n), 1.0)
        smoothed: Dict[Tuple[int, int], np.ndarray] = {}
        for edge, feat in edge_feats.items():
            src, dst = edge
            load = np.clip(np.log1p(feat[0]) / np.log1p(norm_agents), 0.0, 1.0)
            conflict = np.clip(np.log1p(feat[1]) / np.log1p(norm_agents), 0.0, 1.0)
            speed = np.clip(feat[3] / max(feat[0], 1.0), 0.0, 1.0)
            reverse = edge_feats.get((dst, src), np.zeros(4, dtype=np.float32))
            flow_fwd = feat[0]
            flow_rev = reverse[0]
            flow_bias = (flow_fwd - flow_rev) / (flow_fwd + flow_rev + 1e-6)
            flow_bias = 0.5 * (flow_bias + 1.0)
            raw = np.array([load, conflict, flow_bias, speed], dtype=np.float32)
            prev = self._edge_feature_cache.get(edge, raw)
            cur = self._edge_feat_ema * prev + (1.0 - self._edge_feat_ema) * raw
            smoothed[edge] = cur

        self._edge_feature_cache = smoothed
        return smoothed

    def _decode_flow_path(
        self,
        transition_probs: np.ndarray,
        start_idx: int,
        goal_idx: int,
        node_ids: np.ndarray,
    ) -> list[int]:
        idx_to_node = {idx: int(node_id) for idx, node_id in enumerate(node_ids) if node_id >= 0}
        node_to_idx = {node_id: idx for idx, node_id in idx_to_node.items()}
        if start_idx not in idx_to_node or goal_idx not in idx_to_node:
            return []

        path_idx = [start_idx]
        visited = {start_idx}
        cur_idx = start_idx
        max_hops = max(len(idx_to_node) + 2, 4)

        for _ in range(max_hops):
            if cur_idx == goal_idx:
                break
            cur_node = idx_to_node[cur_idx]
            neighbor_idx = [node_to_idx[nbr] for nbr in self.net.adj[cur_node] if nbr in node_to_idx]
            if not neighbor_idx:
                break

            ranked = sorted(
                neighbor_idx,
                key=lambda idx: float(transition_probs[cur_idx, idx]),
                reverse=True,
            )
            next_idx = None
            for cand_idx in ranked:
                if cand_idx == goal_idx or cand_idx not in visited:
                    next_idx = cand_idx
                    break
            if next_idx is None:
                next_idx = ranked[0]

            path_idx.append(next_idx)
            if next_idx == goal_idx:
                break
            if next_idx in visited:
                break
            visited.add(next_idx)
            cur_idx = next_idx

        actual_path = [idx_to_node[idx] for idx in path_idx]
        if actual_path[-1] != idx_to_node[goal_idx]:
            fallback = self.net.bfs(actual_path[-1], idx_to_node[goal_idx])
            if fallback and len(fallback) > 1:
                actual_path.extend(fallback[1:])
        return actual_path

    def apply_flow_subgoal(
        self,
        agent_id: int,
        transition_probs: np.ndarray,
        subgoal_idx: int,
        node_ids: np.ndarray,
    ) -> float:
        idx_to_node = {idx: int(node_id) for idx, node_id in enumerate(node_ids) if node_id >= 0}
        node_to_idx = {node_id: idx for idx, node_id in idx_to_node.items()}
        cur_node = self._nearest_node_id(self.positions[agent_id])
        goal_node = self.goal_nodes[agent_id]

        if cur_node not in node_to_idx or goal_node not in node_to_idx or subgoal_idx not in idx_to_node:
            fallback = self.net.bfs(cur_node, goal_node) or [cur_node]
            self.route_plan[agent_id] = fallback
            self.waypoints[agent_id] = [self.net.nodes[node].copy() for node in fallback]
            self.wp_idx[agent_id] = min(1, len(fallback) - 1)
            return 0.0

        cur_idx = node_to_idx[cur_node]
        goal_idx = node_to_idx[goal_node]
        path_to_subgoal = self._decode_flow_path(transition_probs, cur_idx, subgoal_idx, node_ids)
        subgoal_node = idx_to_node[subgoal_idx]
        path_to_goal = self._decode_flow_path(transition_probs, node_to_idx.get(subgoal_node, goal_idx), goal_idx, node_ids)

        if not path_to_subgoal:
            full_path = self.net.bfs(cur_node, goal_node) or [cur_node]
        else:
            full_path = path_to_subgoal
            if path_to_goal:
                full_path = full_path + path_to_goal[1:]

        if full_path[-1] != goal_node:
            fallback = self.net.bfs(full_path[-1], goal_node)
            if fallback and len(fallback) > 1:
                full_path = full_path + fallback[1:]

        if len(full_path) < 2:
            full_path = self.net.bfs(cur_node, goal_node) or [cur_node]

        old_len = max(len(self.route_plan[agent_id]), 1)
        new_len = max(len(full_path), 1)
        len_gain = (old_len - new_len) / float(old_len)

        self.route_plan[agent_id] = full_path
        self.waypoints[agent_id] = [self.net.nodes[node].copy() for node in full_path]
        self.wp_idx[agent_id] = min(1, len(full_path) - 1)
        return float(len_gain)

    def _get_global_planner_state(self):
        state = self.get_planner_state()
        return state["node_feat"], state["edge_feat"], state["agent_ego_feats"], state["adj"], state["node_ids"]

    def _build_local_graph_obs(self, agent_id: int, traffic_edges: Dict[Tuple[int, int], np.ndarray]):
        pos = self.positions[agent_id]
        goal = self.goals[agent_id]
        extent = self.net.extent
        n = self._n_agents

        node_items = sorted(self.net.nodes.items(), key=lambda item: float(np.linalg.norm(item[1] - pos)))[:LOCAL_TOPK]
        node_ids = [node_id for node_id, _ in node_items]
        id_to_local = {node_id: idx for idx, node_id in enumerate(node_ids)}

        cur_node = self._nearest_node_id(pos)
        goal_node = self.goal_nodes[agent_id]

        occupancy = np.zeros(LOCAL_TOPK, dtype=np.float32)
        for other_id in range(n):
            if self.dones[other_id]:
                continue
            other_node = self._nearest_node_id(self.positions[other_id])
            if other_node in id_to_local:
                occupancy[id_to_local[other_node]] += 1.0
        occupancy = np.clip(occupancy / max(n, 1), 0.0, 1.0)

        feats = np.zeros((LOCAL_TOPK, NODE_DIM), dtype=np.float32)
        for idx, (node_id, node_pos) in enumerate(node_items):
            rel = (node_pos - pos) / extent
            dist_goal = np.linalg.norm(goal - node_pos) / extent
            is_cur = 1.0 if node_id == cur_node else 0.0
            is_goal = 1.0 if node_id == goal_node else 0.0
            feats[idx] = np.array([rel[0], rel[1], rel[2], dist_goal, occupancy[idx], is_cur, is_goal], dtype=np.float32)

        adj = np.eye(LOCAL_TOPK, dtype=np.float32)
        for src, dst in self.net.edges:
            if src in id_to_local and dst in id_to_local:
                local_src = id_to_local[src]
                local_dst = id_to_local[dst]
                adj[local_src, local_dst] = 1.0
                adj[local_dst, local_src] = 1.0

        return feats, adj

    def _build_neighbor_obs(self, agent_id: int):
        n = self._n_agents
        pos_i = self.positions[agent_id]
        vel_i = self.prev_vels[agent_id]
        extent = self.net.extent

        rows = []
        for other_id in range(n):
            if other_id == agent_id or self.dones[other_id]:
                continue
            delta_pos = self.positions[other_id] - pos_i
            dist = float(np.linalg.norm(delta_pos))
            if dist > NBR_RADIUS:
                continue
            delta_vel = self.prev_vels[other_id] - vel_i
            rows.append(
                (
                    dist,
                    np.array([*(delta_pos / extent), *(delta_vel / MAX_SPEED), dist / extent], dtype=np.float32),
                )
            )

        rows.sort(key=lambda item: item[0])
        nbrs = np.zeros((MAX_NBR, NBR_DIM), dtype=np.float32)
        mask = np.ones(MAX_NBR, dtype=bool)
        for idx, (_, vec) in enumerate(rows[:MAX_NBR]):
            nbrs[idx] = vec
            mask[idx] = False
        return nbrs, mask
