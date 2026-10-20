"""Line the broadcast recording up with the tactical one, and read two things off it.

The two videos are not synchronised and no mapping is provided, but they record the same match in
real time. Live broadcast pictures therefore sit at a fixed offset from the tactical video, one
offset per half (the halftime break was cut differently), and replays and close-ups simply fail to
match anything. The offsets are found here by sliding the broadcast ball track along the tactical
one, then checked frame by frame against the players.

What the broadcast gives back:

  shot     what the TV director is showing at each tactical frame. Directors cut to close-ups and
           replays when the ball is dead, so this is a dead-ball signal that owes nothing to the
           tracking at all.
  ball     the broadcast camera's ball, on frames where the broadcast shows live play. It is
           zoomed in on the action, so it sometimes sees the ball when the wide camera does not.
"""

import warnings

import numpy as np
import pandas as pd

from games import GAME, asset_paths

from . import atd

# shot classes, per tactical frame
NO_BROADCAST, LIVE, OTHER_VIEW, NO_PITCH = -1, 0, 1, 2

MATCH_DIST = 1.5           # metres, median player mismatch below which a frame is live
MIN_PLAYERS = 4
COARSE = (-20000, 60000, 5)
REFINE_WINDOW = 1500       # broadcast frames per local offset estimate
REFINE_RADIUS = 3


def _ball_agreement(t_ball, b_ball, b_frames, offset, radius=1.0):
    j = b_frames + offset
    ok = (j >= 0) & (j < len(t_ball))
    d = np.linalg.norm(t_ball[j[ok]] - b_ball[b_frames[ok]], axis=1)
    d = d[np.isfinite(d)]
    return (np.mean(d < radius), len(d)) if len(d) > 50 else (0.0, len(d))


def find_offsets(tac, bro, n_halves=2):
    """The n best-supported offsets (tactical = broadcast + offset), at least a minute apart."""
    seen = np.flatnonzero(np.isfinite(bro.ball[:, 0]))
    sample = seen[::40]
    offsets = np.arange(*COARSE)
    score = np.array([_ball_agreement(tac.ball, bro.ball, sample, o, 2.0)[0] for o in offsets])
    picks = []
    for i in np.argsort(-score):
        if all(abs(offsets[i] - p) > 1500 for p in picks):
            picks.append(int(offsets[i]))
        if len(picks) == n_halves:
            break
    fine = []
    for p in picks:
        cands = np.arange(p - 10, p + 11)
        s = [_ball_agreement(tac.ball, bro.ball, seen, o)[0] for o in cands]
        fine.append(int(cands[int(np.argmax(s))]))
    return sorted(fine)


def _player_mismatch(tac, bro, offset_per_frame, chunk=20000):
    """Median distance from each broadcast player to the nearest same-team tactical player."""
    n = bro.n
    cost = np.full(n, np.nan)
    count = np.zeros(n, dtype=np.int64)
    for start in range(0, n, chunk):
        sl = slice(start, min(n, start + chunk))
        j = np.clip(np.arange(sl.start, sl.stop) + offset_per_frame[sl], 0, tac.n - 1)
        mins = []
        for side in (atd.HOME, atd.AWAY):
            pb = bro.players[sl][:, bro.team == side]
            pt = tac.players[j][:, tac.team == side]
            d = np.linalg.norm(pb[:, :, None, :] - pt[:, None, :, :], axis=3)
            mins.append(_nanmin(d, axis=2))
        m = np.concatenate(mins, axis=1)
        count[sl] = np.isfinite(m).sum(axis=1)
        cost[sl] = _nanmedian(m, axis=1)
    return cost, count


def _nanmin(a, axis):
    out = np.where(np.isfinite(a), a, np.inf).min(axis=axis)
    return np.where(np.isfinite(out), out, np.nan)


def _nanmedian(a, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(a, axis=axis)


def _assign_halves(tac, bro, offsets):
    """Offset per broadcast frame: one change point between consecutive offsets."""
    per = []
    for o in offsets:
        c, k = _player_mismatch(tac, bro, np.full(bro.n, o))
        per.append(np.where((k >= MIN_PLAYERS) & np.isfinite(c), np.minimum(c, 5.0), 2.5))
    per = np.array(per)
    assign = np.zeros(bro.n, dtype=np.int64)
    start = 0
    for h in range(len(offsets) - 1):
        a, b = per[h, start:], per[h + 1, start:]
        total = np.cumsum(a) + (b.sum() - np.cumsum(b))
        split = start + int(np.argmin(total)) + 1
        assign[split:] = h + 1
        start = split
    return np.asarray(offsets)[assign]


def _refine(tac, bro, base):
    """Small local corrections to the offset, window by window, from the ball tracks."""
    out = base.copy()
    seen = np.isfinite(bro.ball[:, 0])
    for start in range(0, bro.n, REFINE_WINDOW):
        idx = np.arange(start, min(bro.n, start + REFINE_WINDOW))
        idx = idx[seen[idx]]
        if len(idx) < 200:
            continue
        o0 = int(np.median(base[idx]))
        cands = np.arange(o0 - REFINE_RADIUS, o0 + REFINE_RADIUS + 1)
        res = [_ball_agreement(tac.ball, bro.ball, idx, o) for o in cands]
        best = int(np.argmax([r[0] for r in res]))
        if res[best][1] > 100 and res[best][0] > 0.3:
            out[start:start + REFINE_WINDOW] = cands[best]
    return out


class Broadcast:
    def __init__(self, tactical, game=GAME, data_dir=None):
        bro = atd.load(game, data_dir, source="broadcast")
        hom = pd.read_csv(asset_paths(game, data_dir, "broadcast")["homography"],
                          usecols=["frame", "valid"])
        valid = np.zeros(bro.n, dtype=bool)
        valid[hom["frame"].values] = hom["valid"].values == 1

        self.offsets = find_offsets(tactical, bro)
        offset = _refine(tactical, bro, _assign_halves(tactical, bro, self.offsets))
        cost, count = _player_mismatch(tactical, bro, offset)
        live = (count >= MIN_PLAYERS) & (cost < MATCH_DIST)

        n = tactical.n
        target = np.arange(bro.n) + offset
        ok = (target >= 0) & (target < n)
        self.shot = np.full(n, NO_BROADCAST, dtype=np.int8)
        self.shot[target[ok]] = np.where(live, LIVE, np.where(valid, OTHER_VIEW, NO_PITCH))[ok]

        self.ball = np.full((n, 2), np.nan, dtype=np.float32)
        use = ok & live & np.isfinite(bro.ball[:, 0])
        self.ball[target[use]] = bro.ball[use]
