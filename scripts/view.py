"""
Graphical, real-time viewer for the commons gridworld (pygame).

Draws the environment in the style of Figure 2 of the proposal:
  * green patch cells with red apples; protected apples have a gold ring
  * a patch with no apples left is shaded brown (at lambda = 1 it never recovers)
  * T = teacher (blue), S = student (orange), with an arrow for the last move
  * the student's field of view as a shaded square
  * a dashed line when the student can observe the teacher (social channel)
  * "+1" when an agent harvests; a side panel with live statistics

By default both agents are SCRIPTED. With --student-q a TRAINED student is
loaded from its saved Q-table and acts on its learned values (as in evaluation).
A single episode is illustration, not evidence.

Controls:
  SPACE  pause / resume          RIGHT  one step (while paused)
  UP     faster                  DOWN   slower
  R      new episode (next seed) S      save screenshot to results/
  V      start / stop recording an MP4 video to results/videos/
  ESC    quit (or close the window; a recording in progress is saved)

Recording: the video gets ONE FRAME PER ENVIRONMENT STEP and plays at
--video-fps (default 8 steps per second), however fast you watch. So you can
speed the viewer up with UP to record a long stretch quickly, and pausing does
not add frozen time to the video. The red REC badge is not in the video.
Needs:  pip install imageio imageio-ffmpeg

Usage (virtual environment active, from the project folder):
  python scripts/view.py
  python scripts/view.py --lam 0.0
  python scripts/view.py --lam 1.0 --restraint-k 3 --steps 500
  python scripts/view.py --teacher mixed --rho 0.5 --student restrained
  python scripts/view.py --record results/videos/clip.mp4 --record-steps 300
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pygame

from commons_env import CommonsHarvestEnv
from commons_env.env import DOWN, LEFT, RIGHT, UP
from commons_env.metrics import restraint_rate
from commons_env.scripted import GreedyPolicy, TeacherPolicy, RandomPolicy, RestrainedPolicy


class LearnedPolicy:
    """A TRAINED student loaded from a saved Q-table (.npz). It acts greedily
    (exploration off), exactly as in the frozen evaluation, unless explore=True."""

    def __init__(self, path, explore=False, seed=0, wall_mask=True):
        from agents.q_learner import QLearner
        self.q = QLearner.load(path, seed=seed)
        self.q.freeze()
        self.explore, self.wall_mask = explore, wall_mask

    def act_from_obs(self, obs, env, agent):
        from agents.features import state_of, valid_actions
        mask = valid_actions(obs["window"]) if self.wall_mask else None
        return self.q.act(state_of(obs, (env.height, env.width)), greedy=not self.explore, mask=mask)

class Recorder:
    """Writes viewer frames to an MP4 file (H.264), one frame per call to add()."""

    def __init__(self, path, fps):
        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise SystemExit("Recording needs two extra packages:  pip install imageio imageio-ffmpeg") from exc
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(str(self.path), fps=fps, codec="libx264", quality=8,
                                         pixelformat="yuv420p", macro_block_size=1)
        self.frames = 0

    def add(self, surface):
        w, h = surface.get_size()
        to_bytes = getattr(pygame.image, "tobytes", None) or pygame.image.tostring
        frame = np.frombuffer(to_bytes(surface, "RGB"), dtype=np.uint8).reshape(h, w, 3)
        self.writer.append_data(frame[: h - h % 2, : w - w % 2])  # H.264 needs even sizes
        self.frames += 1

    def close(self):
        self.writer.close()
        print(f"saved {self.path}  ({self.frames} frames)")


# --------------------------------------------------------------------------- #
# Look and feel
# --------------------------------------------------------------------------- #
CELL = 64  # pixels per grid cell
PANEL_W = 330  # width of the statistics panel
MARGIN = 16

COL = {
    "bg": (245, 242, 235),
    "floor": (236, 231, 219),
    "grid": (220, 213, 198),
    "patch": (196, 224, 176),
    "patch_dead": (196, 170, 140),
    "apple": (206, 48, 48),
    "apple_hi": (240, 120, 110),
    "leaf": (70, 140, 60),
    "stem": (100, 70, 40),
    "protected": (235, 180, 30),
    "teacher": (45, 105, 190),
    "student": (235, 130, 35),
    "view": (235, 130, 35, 45),
    "view_edge": (235, 130, 35),
    "link": (120, 60, 160),
    "text": (40, 40, 45),
    "muted": (120, 120, 125),
    "panel": (255, 255, 255),
    "panel_edge": (215, 210, 200),
    "plus": (30, 140, 60),
    "shadow": (185, 178, 165),
}

ARROW = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}


def make_policy(kind: str, rho: float, seed: int):
    if kind == "restrained":
        return RestrainedPolicy(seed=seed)
    if kind == "greedy":
        return GreedyPolicy(seed=seed)
    if kind == "mixed":
        return TeacherPolicy(rho, seed=seed)
    return RandomPolicy(seed=seed)


class Viewer:
    def __init__(self, env: CommonsHarvestEnv, labels: dict[str, str], descr: dict[str, str]):
        self.env = env
        self.labels = labels  # agent id -> "T" / "S"
        self.descr = descr  # agent id -> policy description
        self.teacher, self.student = env.possible_agents
        pygame.init()
        pygame.display.set_caption("Commons gridworld: social transmission of restraint")
        self.grid_w = env.width * CELL
        self.grid_h = env.height * CELL
        w = MARGIN * 3 + self.grid_w + PANEL_W
        h = MARGIN * 2 + max(self.grid_h, 680)
        self.screen = pygame.display.set_mode((w, h))
        self.ox, self.oy = MARGIN, MARGIN  # grid origin
        names = "segoeui,helveticaneue,arial,dejavusans"
        self.f_big = pygame.font.SysFont(names, 22, bold=True)
        self.f_mid = pygame.font.SysFont(names, 17, bold=True)
        self.f_small = pygame.font.SysFont(names, 15)
        self.f_agent = pygame.font.SysFont(names, 26, bold=True)
        self.overlay = pygame.Surface((self.grid_w, self.grid_h), pygame.SRCALPHA)

    # ---------------- helpers ----------------
    def cell_rect(self, y, x):
        return pygame.Rect(self.ox + x * CELL, self.oy + y * CELL, CELL, CELL)

    def centre(self, y, x):
        r = self.cell_rect(y, x)
        return r.centerx, r.centery

    def text(self, s, font, colour, pos):
        self.screen.blit(font.render(s, True, colour), pos)

    # ---------------- drawing ----------------
    def draw_apple(self, cx, cy, protected, scale=1.0):
        rad = CELL * 0.26 * scale
        if protected:
            pygame.draw.circle(self.screen, COL["protected"], (cx, cy + 2), rad + 7 * scale,
                               width=max(int(4 * scale), 2))
        pygame.draw.circle(self.screen, COL["apple"], (cx, cy + 3 * scale), rad)
        pygame.draw.circle(self.screen, COL["apple_hi"], (cx - rad * 0.35, cy - rad * 0.2), rad * 0.28)
        pygame.draw.line(self.screen, COL["stem"], (cx, cy - rad + 3 * scale),
                         (cx + 2 * scale, cy - rad - 5 * scale), max(int(3 * scale), 1))
        pygame.draw.ellipse(self.screen, COL["leaf"],
                            (cx + 3 * scale, cy - rad - 7 * scale, 11 * scale, 6 * scale))

    def draw_agent(self, agent, action, harvested):
        y, x = self.env.agent_position(agent)
        cx, cy = self.centre(y, x)
        colour = COL["teacher"] if self.labels[agent] == "T" else COL["student"]
        rad = CELL * 0.36
        pygame.draw.circle(self.screen, COL["shadow"], (cx + 2, cy + 3), rad)  # shadow
        pygame.draw.circle(self.screen, colour, (cx, cy), rad)
        pygame.draw.circle(self.screen, (255, 255, 255), (cx, cy), rad, width=3)
        lab = self.f_agent.render(self.labels[agent], True, (255, 255, 255))
        self.screen.blit(lab, lab.get_rect(center=(cx, cy)))
        if action in ARROW:  # small arrow showing the last move
            dx, dy = ARROW[action]
            tip = (cx + dx * (rad + 12), cy + dy * (rad + 12))
            base = (cx + dx * (rad + 3), cy + dy * (rad + 3))
            px, py = -dy * 6, dx * 6
            pygame.draw.polygon(
                self.screen, colour,
                [tip, (base[0] + px, base[1] + py), (base[0] - px, base[1] - py)],
            )
        if harvested:
            plus = self.f_mid.render("+1", True, COL["plus"])
            self.screen.blit(plus, (cx + rad - 4, cy - rad - 14))

    def draw_grid(self, obs, actions, infos):
        env = self.env
        apples = env.apple_grid()
        prot = env.protected_apple_grid()
        counts = env.patch_counts()

        pygame.draw.rect(self.screen, COL["floor"], (self.ox, self.oy, self.grid_w, self.grid_h))
        for y in range(env.height):
            for x in range(env.width):
                r = self.cell_rect(y, x)
                if env.patch_mask[y, x]:
                    dead = counts[env.patch_id[y, x]] == 0
                    pygame.draw.rect(self.screen, COL["patch_dead"] if dead else COL["patch"], r)
                pygame.draw.rect(self.screen, COL["grid"], r, width=1)

        # student's field of view (translucent square, clipped to the grid)
        self.overlay.fill((0, 0, 0, 0))
        sy, sx = env.agent_position(self.student)
        v = env.social_radius  # region in which the student observes the teacher
        y0, x0 = max(sy - v, 0), max(sx - v, 0)
        y1, x1 = min(sy + v, env.height - 1), min(sx + v, env.width - 1)
        vr = pygame.Rect(x0 * CELL, y0 * CELL, (x1 - x0 + 1) * CELL, (y1 - y0 + 1) * CELL)
        pygame.draw.rect(self.overlay, COL["view"], vr)
        self.screen.blit(self.overlay, (self.ox, self.oy))
        pygame.draw.rect(self.screen, COL["view_edge"], vr.move(self.ox, self.oy), width=2)

        for y in range(env.height):
            for x in range(env.width):
                if apples[y, x]:
                    self.draw_apple(*self.centre(y, x), protected=bool(prot[y, x]))

        # dashed line when the student can observe the teacher
        if obs and obs[self.student]["other_visible"]:
            a = pygame.Vector2(self.centre(sy, sx))
            b = pygame.Vector2(self.centre(*env.agent_position(self.teacher)))
            d = b - a
            n = max(int(d.length() // 10), 1)
            for i in range(0, n, 2):
                pygame.draw.line(self.screen, COL["link"], a + d * (i / n), a + d * ((i + 1) / n), 3)

        for agent in env.possible_agents:
            harvested = bool(infos.get(agent, {}).get("harvested", False))
            self.draw_agent(agent, actions.get(agent), harvested)

        pygame.draw.rect(self.screen, COL["text"], (self.ox, self.oy, self.grid_w, self.grid_h), width=2)

    def draw_panel(self, paused, fps, seed, done):
        env, s = self.env, self.env.episode_stats
        px = self.ox + self.grid_w + MARGIN
        panel = pygame.Rect(px, self.oy, PANEL_W, self.screen.get_height() - 2 * MARGIN)
        pygame.draw.rect(self.screen, COL["panel"], panel, border_radius=10)
        pygame.draw.rect(self.screen, COL["panel_edge"], panel, width=1, border_radius=10)
        x, y = px + 16, self.oy + 14

        self.text("Commons gridworld", self.f_big, COL["text"], (x, y)); y += 32
        status = "EPISODE OVER (R: new)" if done else ("PAUSED" if paused else "running")
        self.text(f"step {s['steps']} / {env.max_steps}   [{status}]", self.f_small, COL["muted"], (x, y)); y += 22
        self.text(f"lambda = {env.lam}   restraint_k = {env.restraint_k}   seed = {seed}",
                  self.f_small, COL["muted"], (x, y)); y += 22
        alive = int((env.patch_counts() > 0).sum())
        self.text(f"apples {int(env.apple_grid().sum())}   patches alive {alive}/{env.n_patches}",
                  self.f_small, COL["muted"], (x, y)); y += 30

        for agent in env.possible_agents:
            colour = COL["teacher"] if self.labels[agent] == "T" else COL["student"]
            pygame.draw.circle(self.screen, colour, (x + 9, y + 11), 9)
            self.text(f"{self.labels[agent]}  {self.descr[agent]}", self.f_mid, COL["text"], (x + 24, y)); y += 26
            res = restraint_rate(s, agent)
            res_txt = "n/a" if math.isnan(res) else f"{res:.2f}"
            self.text(f"return {s['returns'][agent]:.0f}    harvests {s['harvests'][agent]}",
                      self.f_small, COL["text"], (x + 24, y)); y += 20
            self.text(f"restraint {res_txt}  (encounters {s['restraint_encounters'][agent]},"
                      f" steps beside {s['restraint_opportunities'][agent]})",
                      self.f_small, COL["text"], (x + 24, y)); y += 30

        surv = s["survival_time"]
        self.text(f"survival time: {surv if surv is not None else 'not depleted'}",
                  self.f_small, COL["text"], (x, y)); y += 34

        self.text("Legend", self.f_mid, COL["text"], (x, y)); y += 30
        self.draw_apple(x + 15, y + 9, protected=False, scale=0.6)
        self.text("apple (+1)", self.f_small, COL["text"], (x + 40, y)); y += 30
        self.draw_apple(x + 15, y + 9, protected=True, scale=0.6)
        self.text("protected apple (restraint)", self.f_small, COL["text"], (x + 40, y)); y += 30
        pygame.draw.rect(self.screen, COL["patch_dead"], (x + 4, y, 22, 18))
        self.text("empty patch", self.f_small, COL["text"], (x + 40, y)); y += 26
        pygame.draw.rect(self.screen, COL["view_edge"], (x + 4, y, 22, 18), width=2)
        self.text("student observes teacher here", self.f_small, COL["text"], (x + 40, y)); y += 26
        pygame.draw.line(self.screen, COL["link"], (x + 2, y + 9), (x + 28, y + 9), 3)
        self.text("student observes teacher", self.f_small, COL["text"], (x + 40, y)); y += 34

        self.text("Controls", self.f_mid, COL["text"], (x, y)); y += 26
        for line in ("SPACE pause    RIGHT one step",
                     f"UP / DOWN speed  (now {fps} steps/s)",
                     "R new episode    S screenshot",
                     "V start / stop recording video",
                     "ESC quit"):
            self.text(line, self.f_small, COL["muted"], (x, y)); y += 20

    def draw(self, obs, actions, infos, paused, fps, seed, done):
        """Draw a frame without showing it (call show() afterwards)."""
        self.screen.fill(COL["bg"])
        self.draw_grid(obs, actions, infos)
        self.draw_panel(paused, fps, seed, done)

    def draw_rec_badge(self, frames, video_fps):
        """Red REC badge, drawn after the frame is captured so it is not in the video."""
        x = self.ox + self.grid_w + MARGIN + 16  # bottom-left of the side panel
        y = self.screen.get_height() - MARGIN - 44
        pygame.draw.rect(self.screen, (255, 235, 235), (x, y, 134, 28), border_radius=14)
        pygame.draw.circle(self.screen, (210, 30, 30), (x + 16, y + 14), 7)
        self.text(f"REC  {frames / video_fps:5.1f} s", self.f_mid, (180, 20, 20), (x + 30, y + 4))

    def show(self):
        pygame.display.flip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--restraint-k", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=6, help="steps per second")
    ap.add_argument("--teacher", default="restrained", choices=["restrained", "greedy", "mixed", "random"])
    ap.add_argument("--student", default="greedy", choices=["restrained", "greedy", "mixed", "random"])
    ap.add_argument("--rho", type=float, default=0.75, help="restraint probability for 'mixed'")
    ap.add_argument("--student-q", default=None,
                    help="path to a TRAINED student's Q-table (.npz); replaces the scripted student")
    ap.add_argument("--explore", action="store_true", help="trained student explores (softmax) instead of greedy")
    ap.add_argument("--no-wall-mask", dest="wall_mask", action="store_false",
                    help="for students trained without the wall mask")
    ap.add_argument("--social", action="store_true",
                    help="show the social channel as on (value-shaping students were trained with it on)")
    ap.add_argument("--screenshot", default=None, help="save one frame to this file, then exit")
    ap.add_argument("--screenshot-step", type=int, default=1, help="step at which --screenshot is taken")
    ap.add_argument("--video-fps", type=int, default=8, help="playback speed of recorded videos (steps per second)")
    ap.add_argument("--record", default=None,
                    help="record from the first step to this .mp4 file (V still stops it early)")
    ap.add_argument("--record-steps", type=int, default=None,
                    help="with --record: stop recording and quit after this many steps")
    args = ap.parse_args()

    env = CommonsHarvestEnv(
        lam=args.lam, max_steps=args.steps, restraint_k=args.restraint_k,
        agent_ids=("teacher", "student"),
    )
    labels = {"teacher": "T", "student": "S"}
    descr = {
        "teacher": f"teacher: {args.teacher}" + (f" (rho={args.rho})" if args.teacher == "mixed" else ""),
        "student": ("student: TRAINED (loaded)" if args.student_q
                    else f"student: {args.student} (scripted)"),
    }
    viewer = Viewer(env, labels, descr)
    clock = pygame.time.Clock()

    seed, fps = args.seed, args.fps
    paused, done, running = False, False, True

    def new_episode(sd):
        pols = {
            "teacher": make_policy(args.teacher, args.rho, 1000 + sd),
            "student": (LearnedPolicy(args.student_q, explore=args.explore, seed=sd, wall_mask=args.wall_mask) if args.student_q
                        else make_policy(args.student, args.rho, 2000 + sd)),
        }
        if args.student_q:
            env.set_social_channel("student", args.social)
        obs, infos = env.reset(seed=sd)
        return pols, obs, infos

    pols, obs, infos = new_episode(seed)
    actions: dict = {}
    shots = Path("results")
    recorder = None
    capture = False  # True when the current frame should go into the video

    def start_recording(path=None):
        if path is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = shots / "videos" / f"rec_lam{env.lam}_seed{seed}_t{env.episode_stats['steps']}_{stamp}.mp4"
        print(f"recording to {path}  (press V to stop)")
        return Recorder(path, args.video_fps)

    if args.record:
        recorder, capture = start_recording(args.record), True

    while running:
        step_once = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT:
                    step_once = True
                elif event.key == pygame.K_UP:
                    fps = min(fps + 2, 60)
                elif event.key == pygame.K_DOWN:
                    fps = max(fps - 2, 1)
                elif event.key == pygame.K_r:
                    seed += 1
                    pols, obs, infos = new_episode(seed)
                    actions, done = {}, False
                    capture = recorder is not None
                elif event.key == pygame.K_v:
                    if recorder is None:
                        recorder, capture = start_recording(), True
                    else:
                        recorder.close()
                        recorder = None
                elif event.key == pygame.K_s:
                    shots.mkdir(exist_ok=True)
                    path = shots / f"frame_lam{env.lam}_seed{seed}_t{env.episode_stats['steps']}.png"
                    pygame.image.save(viewer.screen, str(path))
                    print(f"saved {path}")

        if not done and (not paused or step_once):
            actions = {a: (pols[a].act_from_obs(obs[a], env, a) if hasattr(pols[a], "act_from_obs")
                           else pols[a].act(env, a)) for a in env.agents}
            obs, _, _, _, infos = env.step(actions)
            done = not env.agents
            capture = True

        viewer.draw(obs, actions, infos, paused, fps, seed, done)
        if recorder is not None:
            if capture:  # one frame per environment step (plus the first frame)
                recorder.add(viewer.screen)
                capture = False
            viewer.draw_rec_badge(recorder.frames, args.video_fps)
            if args.record and (done or (args.record_steps and env.episode_stats["steps"] >= args.record_steps)):
                running = False  # --record mode: stop at --record-steps or at the end of the episode
        viewer.show()

        if args.screenshot and (env.episode_stats["steps"] >= args.screenshot_step or done):
            pygame.image.save(viewer.screen, args.screenshot)
            print(f"saved {args.screenshot}")
            running = False

        clock.tick(fps)

    if recorder is not None:
        recorder.close()
    pygame.quit()


if __name__ == "__main__":
    main()
