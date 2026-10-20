"""Instant cues that pin down where a stoppage starts and ends.

The dead-ball score in deadball.py says *that* the game is stopped, but its evidence is smeared
over a couple of seconds: players keep jogging after the whistle and only accelerate after the
restart. These detectors fire on single frames -- the ball being struck after sitting still, the
ball crossing a line -- and the decoder uses them to make a change of state cheap exactly there.
"""

import numpy as np

from .smooth import window_mean

FPS = 25


def restart_kicks(f, still_speed=1.5, kick_speed=4.0, before=25, after=4, reach=2.0):
    """1.0 on frames where a ball that was still, or unseen, is struck with a player beside it.

    "Still" covers a ball placed for a free kick or a goal kick; "unseen" covers a throw-in, where
    the tracker usually loses the ball in the thrower's hands.
    """
    n = f.n
    speed = f.ball_speed
    known = np.isfinite(speed)
    slow = np.where(known, speed < still_speed, True)        # unseen counts as not moving
    # share of the previous `before` frames that were still or unseen
    cs = np.concatenate(([0], np.cumsum(slow)))
    idx = np.arange(n)
    lo = np.maximum(0, idx - before)
    calm_before = (cs[idx] - cs[lo]) / np.maximum(idx - lo, 1)
    # ball moving fast over the next `after` frames
    fast = np.where(known, speed > kick_speed, False).astype(float)
    cf = np.concatenate(([0], np.cumsum(fast)))
    hi = np.minimum(n, idx + after)
    moving_after = (cf[hi] - cf[idx]) / np.maximum(hi - idx, 1)
    near = np.fmin(f.d_home, f.d_away) < reach
    kick = (calm_before > 0.8) & (moving_after > 0.75) & near & known
    # keep the first frame of each burst
    first = kick & ~np.concatenate(([False], kick[:-1]))
    return first.astype(float)


def line_crossings(f, margin=0.3):
    """1.0 on the first frame the ball is seen past a line after being seen inside."""
    length, width = f.pitch
    x, y = f.ball[:, 0], f.ball[:, 1]
    with np.errstate(invalid="ignore"):
        past = np.fmax.reduce([-x, x - length, -y, y - width]) > margin
        inside = np.fmax.reduce([-x, x - length, -y, y - width]) < 0
    known = np.isfinite(x)
    last_inside = np.maximum.accumulate(np.where(inside & known, np.arange(f.n), -1))
    crossing = past & ~np.concatenate(([False], past[:-1])) & (last_inside >= 0)
    # the line was crossed between the last inside frame and this one; mark the last inside frame
    out = np.zeros(f.n)
    out[last_inside[crossing] + 1] = 1.0
    return out
