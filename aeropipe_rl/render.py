"""3D rendering utilities for GUI visualization."""

from __future__ import annotations

import math

import pygame
from pygame.locals import MOUSEBUTTONDOWN, MOUSEBUTTONUP, MOUSEMOTION
from OpenGL.GL import (
    GL_BLEND,
    GL_DEPTH_TEST,
    GL_MODELVIEW,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_PROJECTION,
    GL_RGBA,
    GL_SRC_ALPHA,
    GL_UNSIGNED_BYTE,
    glBlendFunc,
    glColor4f,
    glDisable,
    glDrawPixels,
    glEnable,
    glLoadIdentity,
    glMatrixMode,
    glOrtho,
    glPopMatrix,
    glPushMatrix,
    glRectf,
    glWindowPos2i,
)
from OpenGL.GLU import gluLookAt

from aeropipe_rl.config import H, W


class Camera:
    """Original watch-mode orbit camera."""

    def __init__(self) -> None:
        self.az = 45.0
        self.el = 28.0
        self.dist = 45.0
        self.dragging = False
        self.last_xy = (0, 0)

    def event(self, ev) -> None:
        if ev.type == MOUSEBUTTONDOWN:
            if ev.button == 1:
                self.dragging = True
                self.last_xy = ev.pos
            elif ev.button == 4:
                self.dist = max(5.0, self.dist - 2.5)
            elif ev.button == 5:
                self.dist = min(150.0, self.dist + 2.5)
        elif ev.type == MOUSEBUTTONUP and ev.button == 1:
            self.dragging = False
        elif ev.type == MOUSEMOTION and self.dragging:
            dx = ev.pos[0] - self.last_xy[0]
            dy = ev.pos[1] - self.last_xy[1]
            self.az += dx * 0.45
            self.el = max(-85.0, min(85.0, self.el + dy * 0.35))
            self.last_xy = ev.pos

    def apply(self, target) -> None:
        az = math.radians(self.az)
        el = math.radians(self.el)
        eye = (
            target[0] + self.dist * math.cos(el) * math.cos(az),
            target[1] + self.dist * math.cos(el) * math.sin(az),
            target[2] + self.dist * math.sin(el),
        )
        gluLookAt(
            eye[0],
            eye[1],
            eye[2],
            target[0],
            target[1],
            target[2],
            0.0,
            0.0,
            1.0,
        )


class HUD:
    """HUD styled after the original watch GUI."""

    def __init__(self) -> None:
        pygame.font.init()
        try:
            self.fB = pygame.font.SysFont("consolas", 18, bold=True)
            self.fS = pygame.font.SysFont("consolas", 15)
        except Exception:
            self.fB = pygame.font.Font(None, 22)
            self.fS = pygame.font.Font(None, 18)

    def _enter(self) -> None:
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

    def _exit(self) -> None:
        glEnable(GL_DEPTH_TEST)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    def text(self, msg: str, x: int, y: int, color=(235, 240, 255), big: bool = False) -> None:
        font = self.fB if big else self.fS
        surf = font.render(msg, True, color)
        data = pygame.image.tostring(surf, "RGBA", True)
        glWindowPos2i(x, y)
        glDrawPixels(surf.get_width(), surf.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, data)

    def rect(self, x1: float, y1: float, x2: float, y2: float, color) -> None:
        glColor4f(*color)
        glRectf(x1, y1, x2, y2)

    def bar(self, x: int, y: int, w: int, h: int, frac: float, fg, bg=(0.08, 0.10, 0.16, 0.85)) -> None:
        frac = max(0.0, min(1.0, frac))
        self.rect(x, y, x + w, y + h, bg)
        self.rect(x, y, x + w * frac, y + h, fg)

    def render(self, trainer, mode, env, last_results) -> None:
        self._enter()
        panel_x = W - 320
        panel_y = H - 505
        panel_w = 300
        panel_h = 470
        self.rect(panel_x, panel_y, panel_x + panel_w, panel_y + panel_h, (0.04, 0.06, 0.10, 0.78))
        self.rect(panel_x, panel_y + panel_h - 4, panel_x + panel_w, panel_y + panel_h, (0.14, 0.60, 1.00, 0.85))

        active = int((~env.dones).sum())
        done_results = [r for r in last_results if r]
        success = sum(r == "goal" for r in done_results)
        wall = sum(r == "wall" for r in done_results)
        collision = sum(r == "collision" for r in done_results)
        timeout = sum(r == "timeout" for r in done_results)
        denom = max(len(done_results), 1)
        success_rate = success / denom
        progress = env.step_count / max(env.step_budget, 1)

        y = H - 58
        self.text("AeroPipeRL WATCH", panel_x + 14, y, (220, 240, 255), big=True)
        y -= 30
        self.text(f"Mode: {mode.upper()}", panel_x + 14, y)
        y -= 22
        self.text(f"Agents: {active}/{len(env.dones)} active", panel_x + 14, y)
        y -= 22
        self.text(f"Step: {env.step_count}/{env.step_budget}", panel_x + 14, y)
        y -= 28
        self.text("Episode Progress", panel_x + 14, y)
        y -= 16
        self.bar(panel_x + 14, y, panel_w - 28, 12, progress, (0.12, 0.72, 1.00, 0.90))
        y -= 30
        self.text("Last Episode", panel_x + 14, y)
        y -= 22
        self.text(f"Goal: {success}", panel_x + 14, y, (120, 255, 160))
        y -= 22
        self.text(f"Wall: {wall}", panel_x + 14, y, (255, 120, 120))
        y -= 22
        self.text(f"Collision: {collision}", panel_x + 14, y, (255, 195, 90))
        y -= 22
        self.text(f"Timeout: {timeout}", panel_x + 14, y, (190, 190, 210))
        y -= 28
        self.text("Success Rate", panel_x + 14, y)
        y -= 16
        self.bar(panel_x + 14, y, panel_w - 28, 12, success_rate, (0.20, 1.00, 0.35, 0.90))
        y -= 30
        if hasattr(trainer, "ep"):
            self.text(f"Trainer EP: {int(trainer.ep)}", panel_x + 14, y)
            y -= 22
        self.text(f"Nodes: {len(env.net.nodes)}  Edges: {len(env.net.edges)}", panel_x + 14, y)
        y -= 22
        self.text("Mouse drag: orbit   Wheel: zoom", panel_x + 14, y, (170, 185, 210))
        y -= 18
        self.text("ESC/Q: quit   R: regenerate", panel_x + 14, y, (170, 185, 210))
        self._exit()


def draw_pipe_network(
    net,
    starts=None,
    goals=None,
    trails=None,
    waypoints_list=None,
    attn_lines=None,
):
    """Compatibility wrapper around the network-owned renderer."""
    net.draw(
        starts=starts,
        goals=goals,
        trails=trails,
        waypoints_list=waypoints_list,
        attn_lines=attn_lines,
    )
