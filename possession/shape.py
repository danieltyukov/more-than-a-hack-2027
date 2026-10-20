"""Team shape: what the players say about who has the ball when the ball itself is missing.

The ball is missing on 41% of graded frames, and a turnover inside one of those gaps leaves the
touch evidence with nothing to say. The players still know: the side with the ball spreads out
and moves forward, the side without it gets compact, drops and presses. These features describe
that for each side, in its own attacking direction.

A classifier on these features alone, trained on the frames where one player clearly has the ball
at his feet, called the possessing side on 83% of the public window's live frames without ever
seeing the ball. They are now part of the learned models' inputs (framefeatures.py).
"""

import warnings

import numpy as np

from .atd import AWAY, HOME

VEL_HALF = 5                 # frames each side for player velocities (0.2 s)
OWN_REACH = 1.0              # metres: the player "has" the ball
CLEAR_GAP = 2.5              # metres: nobody from the other side within this
MAX_BALL_SPEED = 6.0         # m/s: not a pass in flight


def attack_signs(tracking):
    """{half: {team: +1 if attacking towards x = length, else -1}} from where the keepers stand."""
    out = {}
    for half, first, last in tracking.blocks:
        sl = slice(first, last + 1)
        lows = {}
        for side in (HOME, AWAY):
            x = tracking.players[sl][:, tracking.team == side, 0]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                lows[side] = np.nanmean(np.nanmin(x, axis=1))
        home_left = lows[HOME] < lows[AWAY]
        out[half] = {HOME: 1 if home_left else -1, AWAY: -1 if home_left else 1}
    return out


def _team_features(pos, vel, speed, sign, length):
    """Shape of one team, in its own attacking direction. pos (n, k, 2)."""
    x = pos[..., 0] if sign > 0 else length - pos[..., 0]
    y = pos[..., 1]
    vx = vel[..., 0] * sign
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        xs = np.sort(x, axis=1)                       # NaN sorts last
        n = np.isfinite(x).sum(axis=1)
        last_outfield = np.where(n >= 2, xs[:, 1], np.nan)
        return [np.nanmean(x, axis=1), np.nanstd(x, axis=1), np.nanstd(y, axis=1),
                np.nanmax(x, axis=1) - last_outfield, last_outfield, np.nanmax(x, axis=1),
                np.nanmean(vx, axis=1), np.nanmean(speed, axis=1),
                np.nanpercentile(speed, 90, axis=1), n.astype(float)]


def _marking(pos_a, pos_b):
    """Mean over team a of the distance to the nearest player of team b."""
    d = np.linalg.norm(pos_a[:, :, None, :] - pos_b[:, None, :, :], axis=3)
    d = np.where(np.isfinite(d), d, np.inf).min(axis=2)
    d[~np.isfinite(d)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(d, axis=1), np.nanmedian(d, axis=1)


def shape_features(tracking):
    """(n, m) feature matrix, NaN outside the two halves."""
    t = tracking
    n = t.n
    length = t.pitch[0]
    vel = np.full_like(t.players, np.nan)
    vel[VEL_HALF:-VEL_HALF] = (t.players[2 * VEL_HALF:] - t.players[:-2 * VEL_HALF]) * (25.0 / (2 * VEL_HALF))
    home, away = t.team == HOME, t.team == AWAY
    signs = attack_signs(t)

    feats = None
    for half, first, last in t.blocks:
        sl = slice(first, last + 1)
        ph, pa = t.players[sl][:, home], t.players[sl][:, away]
        fh = _team_features(ph, vel[sl][:, home], t.speed[sl][:, home], signs[half][HOME], length)
        fa = _team_features(pa, vel[sl][:, away], t.speed[sl][:, away], signs[half][AWAY], length)
        mh = _marking(ph, pa)
        ma = _marking(pa, ph)
        cols = fh + fa + [a - b for a, b in zip(fh, fa)] + list(mh) + list(ma) + \
            [mh[0] - ma[0], mh[1] - ma[1]]
        block = np.column_stack(cols).astype(np.float32)
        if feats is None:
            feats = np.full((n, block.shape[1]), np.nan, dtype=np.float32)
        feats[sl] = block
    return feats


def clear_control(f):
    """+1 where a home player clearly has the ball, -1 for away, 0 otherwise."""
    with np.errstate(invalid="ignore"):
        slow = np.where(np.isfinite(f.ball_speed), f.ball_speed < MAX_BALL_SPEED, False)
        home = (f.d_home < OWN_REACH) & (f.d_away > CLEAR_GAP) & slow & f.ball_seen
        away = (f.d_away < OWN_REACH) & (f.d_home > CLEAR_GAP) & slow & f.ball_seen
    return home.astype(np.int8) - away.astype(np.int8)
