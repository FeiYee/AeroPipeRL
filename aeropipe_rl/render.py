"""3D rendering utilities for GUI visualization."""
import math
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

from aeropipe_rl.config import (
    AGENT_COLORS,
    HUB_R,
    PIPE_R,
    W,
    H,
)


class Camera:
    """Orbit camera for 3D navigation."""
    def __init__(self):
        self.angle_y = 0.3
        self.angle_x = 0.3
        self.distance = 60.0
        self.drag = False
        self.last_pos = (0, 0)

    def event(self, ev):
        if ev.type == MOUSEBUTTONDOWN:
            if ev.button == 1:
                self.drag = True
                self.last_pos = ev.pos
            elif ev.button == 4:
                self.distance = max(10.0, self.distance * 0.9)
            elif ev.button == 5:
                self.distance = min(200.0, self.distance * 1.1)
        elif ev.type == MOUSEBUTTONUP:
            if ev.button == 1:
                self.drag = False
        elif ev.type == MOUSEMOTION:
            if self.drag:
                dx, dy = ev.pos[0] - self.last_pos[0], ev.pos[1] - self.last_pos[1]
                self.angle_y += dx * 0.005
                self.angle_x += dy * 0.005
                self.angle_x = max(-math.pi/2 + 0.01, min(math.pi/2 - 0.01, self.angle_x))
                self.last_pos = ev.pos

    def apply(self, target):
        glTranslatef(0, 0, -self.distance)
        glRotatef(self.angle_x * 180/math.pi, 1, 0, 0)
        glRotatef(self.angle_y * 180/math.pi, 0, 1, 0)
        glTranslatef(-target[0], -target[1], -target[2])


class HUD:
    """2D heads-up display for rendering status information."""
    def __init__(self):
        self.font = pygame.font.SysFont('Consolas', 16)

    def render_text(self, text, x, y, color=(1.0, 1.0, 1.0)):
        text_surface = self.font.render(text, True, (int(color[0]*255), int(color[1]*255), int(color[2]*255)))
        text_data = pygame.image.tostring(text_surface, "RGBA", True)
        glWindowPos2i(x, y)
        glDrawPixels(text_surface.get_width(), text_surface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)

    def render(self, trainer, mode, env, last_results):
        # 切换到正交投影渲染2D内容
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, W, 0, H, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # 状态文本
        y = H - 25
        self.render_text(f"Mode: {mode.upper()}", 10, y)
        y -= 20
        self.render_text(f"Agents: {sum(not d for d in env.dones)}/{len(env.dones)} active", 10, y)
        y -= 20
        self.render_text(f"Step: {env.step_count}/{env.step_budget}", 10, y)
        y -= 20

        # 结果统计
        success = sum(r == 'goal' for r in last_results if r)
        wall = sum(r == 'wall' for r in last_results if r)
        collision = sum(r == 'collision' for r in last_results if r)
        timeout = sum(r == 'timeout' for r in last_results if r)
        if success + wall + collision + timeout > 0:
            self.render_text(f"Last run: S:{success} W:{wall} C:{collision} T:{timeout}", 10, y)
            y -= 20
            sr = success / (success + wall + collision + timeout) * 100
            self.render_text(f"Success rate: {sr:.1f}%", 10, y)
            y -= 20

        # 恢复3D渲染状态
        glEnable(GL_DEPTH_TEST)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)


def draw_pipe_network(
    net,
    starts=None,
    goals=None,
    trails=None,
    waypoints_list=None,
    attn_lines=None
):
    """Draw the 3D pipe network and related elements."""
    if not hasattr(net, 'q') or net.q is None:
        net.q = gluNewQuadric()

    # 绘制管道和节点
    glLineWidth(4.0)
    glBegin(GL_LINES)
    for a, b in net.edges:
        p1, p2 = net.nodes[a], net.nodes[b]
        glColor4f(0.15, 0.15, 0.35, 1.0)
        glVertex3f(*p1)
        glVertex3f(*p2)
    glEnd()
    glLineWidth(1.0)

    # 绘制节点球体
    for nid, pos in net.nodes.items():
        glPushMatrix()
        glTranslatef(*pos)
        if starts and nid in starts:
            glColor4f(0.2, 1.0, 0.2, 0.7)
        elif goals and nid in goals:
            glColor4f(1.0, 0.2, 0.2, 0.7)
        else:
            glColor4f(0.2, 0.2, 0.4, 0.3)
        gluSphere(net.q, HUB_R, 16, 16)
        glPopMatrix()

    # 绘制路径轨迹
    if trails:
        glLineWidth(2.0)
        for i, trail in enumerate(trails):
            if len(trail) < 2:
                continue
            c = AGENT_COLORS[i % len(AGENT_COLORS)]
            glBegin(GL_LINE_STRIP)
            glColor4f(c[0], c[1], c[2], 0.6)
            for pos in trail:
                glVertex3f(*pos)
            glEnd()
        glLineWidth(1.0)

    # 绘制路点
    if waypoints_list:
        for i, waypoints in enumerate(waypoints_list):
            if not waypoints:
                continue
            c = AGENT_COLORS[i % len(AGENT_COLORS)]
            for wp in waypoints:
                glPushMatrix()
                glTranslatef(*wp)
                glColor4f(c[0], c[1], c[2], 0.4)
                gluSphere(net.q, 0.2, 8, 8)
                glPopMatrix()

    # 绘制注意力连线
    if attn_lines:
        glLineWidth(2.0)
        glBegin(GL_LINES)
        for p0, p1, alpha in attn_lines:
            glColor4f(1.0, 1.0, 0.0, alpha * 0.8)
            glVertex3f(*p0)
            glVertex3f(*p1)
        glEnd()
        glLineWidth(1.0)
