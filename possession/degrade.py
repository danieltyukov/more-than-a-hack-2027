"""Make clean DFL tracking look like the competition's automatic tracking.

The DFL matches are corrected multi-camera tracking: every player, the ball on every frame, ball
height included. The competition match is one wide camera and an uncorrected tracker. A model
trained on the clean data would lean on things that are simply not there at test time, so each
training copy is degraded in the ways the competition data is:

  - the ball goes missing in long stretches (no gaps under ~3 s, the provider bridges those),
    mostly while the ball is dead and otherwise in crowded moments, where it is occluded
  - a ball in the air is seen where the camera ray through it meets the grass, not below it
  - players drop out of the tracking for a few seconds at a time
  - positions carry a little slowly varying noise

Every copy draws its own amounts of each, so the models cannot fit one particular level.
"""

import numpy as np
from scipy.signal import lfilter

from .atd import Tracking

FPS = 25


def _smooth_noise(rng, shape, sd, tau_frames=12):
    """AR(1) noise with the given marginal sd, correlated over ~tau frames."""
    a = np.exp(-1.0 / tau_frames)
    e = rng.normal(0.0, sd * np.sqrt(1 - a * a), size=shape)
    e[0] = rng.normal(0.0, sd, size=shape[1:])
    return lfilter([1.0], [1.0, -a], e, axis=0).astype(np.float32)


def _runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def draw_params(rng):
    return {
        "player_sd": rng.uniform(0.1, 0.3),
        "ball_sd": rng.uniform(0.15, 0.4),
        "dropout_per_min": rng.uniform(0.2, 0.8),
        "dead_hidden": rng.uniform(0.6, 0.9),
        "alive_hidden": rng.uniform(0.08, 0.24),
        "cam_back": rng.uniform(15.0, 45.0),
        "cam_height": rng.uniform(12.0, 30.0),
    }


def degrade(t, labels, rng, params=None):
    p = params or draw_params(rng)
    n, k = t.players.shape[:2]
    length, width = t.pitch
    players = t.players + _smooth_noise(rng, t.players.shape, p["player_sd"])
    ball = t.ball.copy()

    # aerial balls land on the ground plane along the camera ray
    z = np.nan_to_num(t.ball_z, nan=0.0)
    cam = np.array([length / 2.0, width + p["cam_back"]], dtype=np.float32)
    lift = np.clip(z / np.maximum(p["cam_height"] - z, 1.0), 0.0, 2.0)[:, None]
    ball = ball + lift * (ball - cam)
    ball = ball + _smooth_noise(rng, ball.shape, p["ball_sd"], tau_frames=4)

    # player dropouts, a few seconds each
    minutes = n / (FPS * 60.0)
    for j in range(k):
        for _ in range(rng.poisson(p["dropout_per_min"] * minutes)):
            start = rng.integers(0, n)
            dur = int(np.clip(rng.lognormal(np.log(3 * FPS), 0.7), FPS, 30 * FPS))
            players[start:start + dur, j] = np.nan

    hidden = np.zeros(n, dtype=bool)
    # stoppages: the ball is out of shot or in someone's hands for most of them
    dead = labels == 2
    for a, b in _runs(dead):
        if rng.random() < p["dead_hidden"]:
            s = a + int(rng.uniform(0, 1.5) * FPS)
            e = b - int(rng.uniform(0, 0.8) * FPS)
            if e > s:
                hidden[s:e] = True
    # live play: occlusions where the ball is crowded
    alive = (labels == 0) | (labels == 1)
    d = np.linalg.norm(t.players - t.ball[:, None, :], axis=2)
    crowd = np.nansum(d < 3.0, axis=1).astype(float)
    weight = np.where(alive, crowd ** 2 + 0.2, 0.0)
    if weight.sum() > 0:
        weight /= weight.sum()
        target = p["alive_hidden"] * alive.sum()
        tries = 0
        while (hidden & alive).sum() < target and tries < 2000:
            s = rng.choice(n, p=weight)
            dur = int(np.clip(rng.lognormal(np.log(5 * FPS), 0.5), 3 * FPS, 30 * FPS))
            hidden[max(0, s - dur // 3):s + dur] = True
            tries += 1
    # the provider bridges gaps shorter than ~3 s, so those stay visible
    for a, b in _runs(hidden):
        if b - a < 3 * FPS:
            hidden[a:b] = False
    ball[hidden] = np.nan

    speed = np.where(np.isfinite(players[..., 0]), t.speed, np.nan).astype(np.float32)
    out = Tracking(players.astype(np.float32), speed, ball.astype(np.float32),
                   np.full(n, np.nan, dtype=np.float32), t.team, t.pitch, t.blocks)
    return out, p
