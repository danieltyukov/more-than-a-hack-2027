"""Per-frame geometry computed once from the tracking.

Everything is a flat array over video frames so the later stages can stay vectorised. Distances are
in metres, speeds in m/s, and NaN means "not observable on this frame", never zero.
"""

import warnings

import numpy as np

from .atd import AWAY, HOME

FPS = 25

# Gaps in the ball track up to this long are bridged by linear interpolation. A ball crossing a
# 0.3 s gap moves at most ~10 m, so the straight line is a fair guess; beyond that it is not.
SHORT_GAP = 8

# Half-width, in frames, of the central difference used for ball velocity. Tracker jitter is a few
# tens of centimetres, so a one-frame difference is mostly noise at 25 fps.
VELOCITY_HALF_WINDOW = 2


def runs_of(mask):
    """[(start, stop)] half-open runs where mask is True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def bridge_short_gaps(xy, max_gap=SHORT_GAP):
    """Linear interpolation across interior gaps of at most max_gap frames. Returns (xy, filled)."""
    out = xy.copy()
    filled = np.zeros(len(xy), dtype=bool)
    seen = np.isfinite(xy[:, 0])
    for start, stop in runs_of(~seen):
        if start == 0 or stop == len(xy) or stop - start > max_gap:
            continue
        a, b = out[start - 1], out[stop]
        w = (np.arange(1, stop - start + 1) / float(stop - start + 1))[:, None]
        out[start:stop] = a + (b - a) * w
        filled[start:stop] = True
    return out, filled


def central_velocity(xy, half=VELOCITY_HALF_WINDOW, fps=FPS):
    """(n, 2) velocity in m/s, NaN unless both ends of the window are observed."""
    v = np.full_like(xy, np.nan)
    v[half:-half] = (xy[2 * half:] - xy[:-2 * half]) * (fps / (2.0 * half))
    return v


def nearest(players, mask, point):
    """Distance from point to the nearest tracked player in mask, and that player's slot."""
    d = np.linalg.norm(players[:, mask] - point[:, None, :], axis=2)
    slots = np.flatnonzero(mask)
    has = np.isfinite(d).any(axis=1)
    idx = np.zeros(len(point), dtype=np.int64)
    idx[has] = np.nanargmin(d[has], axis=1)
    dist = np.full(len(point), np.nan, dtype=np.float32)
    dist[has] = d[has, idx[has]]
    return dist, np.where(has, slots[idx], -1)


class Features:
    """extra_ball, if given, fills frames where the tactical tracker lost the ball (the aligned
    broadcast ball, see broadcast.py). ball_seen keeps meaning "the tactical tracker saw it"."""

    def __init__(self, tracking, extra_ball=None):
        t = tracking
        self.n = t.n
        self.pitch = t.pitch
        self.ball_seen = np.isfinite(t.ball[:, 0])
        ball = t.ball
        if extra_ball is not None:
            ball = np.where(self.ball_seen[:, None], t.ball, extra_ball)
        self.ball, self.ball_bridged = bridge_short_gaps(ball)
        self.ball_known = np.isfinite(self.ball[:, 0])
        self.ball_z = t.ball_z

        self.ball_vel = central_velocity(self.ball)
        self.ball_speed = np.linalg.norm(self.ball_vel, axis=1)

        home, away = t.team_mask(HOME), t.team_mask(AWAY)
        self.d_home, self.slot_home = nearest(t.players, home, self.ball)
        self.d_away, self.slot_away = nearest(t.players, away, self.ball)

        tracked = np.isfinite(t.players[..., 0])
        self.n_home = tracked[:, home].sum(axis=1)
        self.n_away = tracked[:, away].sum(axis=1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # frames with nobody tracked
            speed = np.where(tracked, t.speed, np.nan)
            self.mean_speed = np.nanmean(speed, axis=1)
            self.mean_speed_home = np.nanmean(speed[:, home], axis=1)
            self.mean_speed_away = np.nanmean(speed[:, away], axis=1)
            self.centroid_home = np.nanmean(t.players[:, home], axis=1)
            self.centroid_away = np.nanmean(t.players[:, away], axis=1)

        length, width = t.pitch
        bx, by = self.ball[:, 0], self.ball[:, 1]
        with np.errstate(invalid="ignore"):
            self.ball_out = (bx < 0) | (bx > length) | (by < 0) | (by > width)
