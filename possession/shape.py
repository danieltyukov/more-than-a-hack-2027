"""Who has the ball, judged from the players alone.

The ball is missing on 41% of graded frames, and a turnover inside one of those gaps leaves the
touch evidence with nothing to say about when it happened. The players still know: the side with
the ball spreads out and moves forward, the side without it gets compact, drops and presses.

This module learns that from the match itself. Frames where one player clearly has the ball at his
feet (and nobody from the other side is close) label themselves; a gradient-boosted classifier on
team-shape features, with no ball information at all, is fitted on those frames and then applied
everywhere. The halves are cross-fitted -- the model scoring the second half was trained on the
first and vice versa -- so no frame is scored by a model that saw its own label.
"""

import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

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


def fit_predict(tracking, f, seed=0):
    """P(home has the ball) per frame from team shape alone, cross-fitted by half. NaN outside
    the halves."""
    x = shape_features(tracking)
    y = clear_control(f)
    blocks = tracking.blocks
    prob = np.full(tracking.n, np.nan)
    for i, (_, first, last) in enumerate(blocks):
        train = np.zeros(tracking.n, dtype=bool)
        for j, (_, a, b) in enumerate(blocks):
            if j != i:
                train[a:b + 1] = True
        train &= y != 0
        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                               l2_regularization=1.0, random_state=seed)
        model.fit(x[train], (y[train] > 0).astype(int))
        sl = slice(first, last + 1)
        prob[sl] = model.predict_proba(x[sl])[:, 1]
    return prob
