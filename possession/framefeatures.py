"""One row of features per frame, the same for the competition match and for the DFL matches.

Nothing here uses a player's identity: the competition tracking has none, only 46 anonymous track
slots that swap and fragment. Everything is said in terms of "the nearest home player", "how many
away players within 5 m", "the home side's shape", and the same quantities a second or two either
side of the frame, because possession is decided by what happens around a moment rather than in it.
"""

import numpy as np

from . import shape
from .atd import AWAY, HOME
from .deadball import metres_outside
from .smooth import frames_since, frames_until, window_frac, window_mean

REACH = 1.5
CLIP = 750                      # frames; "more than 30 s ago" is all the same


def _past_mean(x, w):
    """Mean over the w frames before each frame, ignoring NaN."""
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    cs = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    cn = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(x))
    lo = np.maximum(0, i - w)
    n = cn[i] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (cs[i] - cs[lo]) / n, np.nan)


def _future_mean(x, w):
    return _past_mean(np.asarray(x)[::-1], w)[::-1]


def _kernel(d, reach=REACH):
    with np.errstate(invalid="ignore"):
        k = np.exp(-0.5 * (d / reach) ** 2)
    return np.where(np.isfinite(k), k, 0.0)


def _ball_player_geometry(t, ball):
    """Per team: nearest and second-nearest distance, players within 2 m and 5 m."""
    out = {}
    d_all = np.linalg.norm(t.players - ball[:, None, :], axis=2)
    for side, name in ((HOME, "home"), (AWAY, "away")):
        d = d_all[:, t.team == side]
        d = np.where(np.isfinite(d), d, np.inf)
        part = np.partition(d, 1, axis=1)[:, :2]
        part = np.where(np.isfinite(part), part, np.nan)
        out["d1_" + name] = part[:, 0]
        out["d2_" + name] = part[:, 1]
        out["n2_" + name] = (d < 2.0).sum(axis=1).astype(float)
        out["n5_" + name] = (d < 5.0).sum(axis=1).astype(float)
    return out


def _slot_velocity(players, half=2, fps=25):
    v = np.full_like(players, np.nan)
    v[half:-half] = (players[2 * half:] - players[:-2 * half]) * (fps / (2.0 * half))
    return v


def _ball_dynamics(f):
    v = f.ball_vel
    a = np.full(f.n, np.nan)
    a[3:-3] = np.linalg.norm(v[6:] - v[:-6], axis=1) / (6 / 25.0)
    turn = np.full(f.n, np.nan)
    v0, v1 = v[:-6], v[6:]
    dot = np.sum(v0 * v1, axis=1)
    norms = np.linalg.norm(v0, axis=1) * np.linalg.norm(v1, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        turn[3:-3] = np.degrees(np.arccos(np.clip(dot / norms, -1, 1)))
    return a, turn


def _relative_speed(t, f, slots):
    vel = _slot_velocity(t.players)
    ok = slots >= 0
    rel = np.full(f.n, np.nan)
    pv = vel[np.arange(f.n)[ok], slots[ok]]
    rel[ok] = np.linalg.norm(f.ball_vel[ok] - pv, axis=1)
    return rel


def _home_attack_x(t, x):
    """Ball x measured towards the goal the home side attacks."""
    signs = shape.attack_signs(t)
    out = np.full(t.n, np.nan)
    for half, first, last in t.blocks:
        sl = slice(first, last + 1)
        out[sl] = x[sl] if signs[half][HOME] > 0 else t.pitch[0] - x[sl]
    return out


def compute(t, f):
    """(X, names) for every frame of tracking t, with f = Features(t)."""
    cols = {}
    geo = _ball_player_geometry(t, f.ball)
    cols.update(geo)
    cols["near_diff"] = geo["d1_home"] - geo["d1_away"]
    cols["ball_speed"] = f.ball_speed
    acc, turn = _ball_dynamics(f)
    cols["ball_acc"] = acc
    cols["ball_turn"] = turn
    cols["rel_home"] = _relative_speed(t, f, f.slot_home)
    cols["rel_away"] = _relative_speed(t, f, f.slot_away)

    length, width = t.pitch
    cols["ball_x_home"] = _home_attack_x(t, f.ball[:, 0])
    cols["ball_y_mid"] = np.abs(f.ball[:, 1] - width / 2.0)
    with np.errstate(invalid="ignore"):
        cols["ball_outside"] = metres_outside(f.ball, t.pitch)
        out = cols["ball_outside"] > 0.5
    cols["out_1s"] = window_frac(out, 25)
    cols["out_past_2s"] = _past_mean(out.astype(float), 50)

    cols["seen"] = f.ball_seen.astype(float)
    cols["since_seen"] = np.minimum(frames_since(f.ball_seen), CLIP).astype(float)
    cols["until_seen"] = np.minimum(frames_until(f.ball_seen), CLIP).astype(float)
    cols["seen_1s"] = window_frac(f.ball_seen, 25)
    cols["seen_4s"] = window_frac(f.ball_seen, 100)

    for name, x in (("speed", f.mean_speed), ("speed_home", f.mean_speed_home),
                    ("speed_away", f.mean_speed_away)):
        cols[name] = x
        cols[name + "_2s"] = window_mean(x, 50)
        cols[name + "_past2s"] = _past_mean(x, 50)
        cols[name + "_next2s"] = _future_mean(x, 50)
    cols["n_home"] = f.n_home.astype(float)
    cols["n_away"] = f.n_away.astype(float)

    touch = _kernel(f.d_home) - _kernel(f.d_away)
    touch = np.where(f.ball_known, touch, np.nan)
    for w in (6, 25, 75):
        cols["touch_past%d" % w] = _past_mean(touch, w)
        cols["touch_next%d" % w] = _future_mean(touch, w)

    clear = shape.clear_control(f)
    for side, name in ((1, "home"), (-1, "away")):
        mask = clear == side
        cols["since_ctrl_" + name] = np.minimum(frames_since(mask), CLIP).astype(float)
        cols["until_ctrl_" + name] = np.minimum(frames_until(mask), CLIP).astype(float)
    cols["since_ctrl_diff"] = cols["since_ctrl_home"] - cols["since_ctrl_away"]
    cols["until_ctrl_diff"] = cols["until_ctrl_home"] - cols["until_ctrl_away"]

    sh = shape.shape_features(t)
    names = list(cols) + ["shape%02d" % i for i in range(sh.shape[1])]
    x = np.column_stack([np.asarray(cols[k], dtype=np.float32) for k in cols] + [sh])
    return x, names
