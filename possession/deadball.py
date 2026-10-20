"""How dead the ball looks on each frame, as a log-odds score.

Four things give a stoppage away:

  - the players slow to a walk within a couple of seconds and stay there until the restart
  - the ball leaves the pitch or is picked up, and the tracker stops seeing it
  - a ball seen past a line is out, whatever the players are doing
  - the TV director cuts away from the wide shot to a close-up or a replay

The weights are a logistic regression of the public five minutes on those features, refit with
scripts/fit_deadball.py. A handful of weights is about as much as five minutes can support.
"""

import numpy as np

from .broadcast import NO_PITCH, OTHER_VIEW
from .smooth import frames_since, frames_until, window_frac, window_mean

SPEED_HALF_WINDOW = 50       # +/- 2 s
STOPPED_SPEED = 1.6          # m/s, mean player speed where live and stopped play meet
OUT_MARGIN = 0.5             # metres past a line before a ball counts as out
SHOT_HALF_WINDOW = 25

WEIGHTS = {
    "intercept": -2.738,
    "slow": 1.335,           # per m/s below STOPPED_SPEED
    "missing": 2.225,        # ball not seen on this frame
    "out": 5.819,            # share of the surrounding second with the ball past a line
    "no_pitch": 4.144,       # share of the surrounding second the broadcast shows no pitch
    "other_view": -2.567,    # broadcast shows the pitch but not the live picture we track
}


def metres_outside(ball, pitch):
    """How far past the nearest line the ball is; negative inside the pitch, NaN if unknown."""
    length, width = pitch
    x, y = ball[:, 0], ball[:, 1]
    return np.fmax.reduce([-x, x - length, -y, y - width])


def features(f, shot=None):
    ms = window_mean(f.mean_speed, SPEED_HALF_WINDOW)
    ms = np.where(np.isfinite(ms), ms, STOPPED_SPEED)
    gap = np.minimum(frames_since(f.ball_seen), frames_until(f.ball_seen))
    with np.errstate(invalid="ignore"):
        out = metres_outside(f.ball, f.pitch) > OUT_MARGIN
    if shot is None:
        shot = np.full(f.n, -1, dtype=np.int8)
    return {
        "slow": STOPPED_SPEED - ms,
        "missing": (gap > 0).astype(float),
        "out": window_frac(out, 25),
        "no_pitch": window_frac(shot == NO_PITCH, SHOT_HALF_WINDOW),
        "other_view": window_frac(shot == OTHER_VIEW, SHOT_HALF_WINDOW),
    }


def score(f, shot=None, weights=None):
    w = dict(WEIGHTS, **(weights or {}))
    z = np.full(f.n, w["intercept"])
    for name, value in features(f, shot).items():
        z += w[name] * value
    return z
