"""Tracking in, one label per video frame out."""

import numpy as np

from games import GAME, half_video_blocks

from . import atd, broadcast, control, deadball, segment
from .features import Features

DEFAULTS = {
    "touch_gain": 1.0,       # weight of one frame of touch evidence
    "switch_ha": 6.0,        # price of a turnover
    "switch_dead": 12.0,     # price of entering or leaving a stoppage
    "dead_gain": 1.0,        # weight of the dead-ball log-odds
    "restart_gain": 1.0,     # weight of "who is standing over the dead ball"
    "switch_restarter": 4.0, # price of changing our mind about who restarts
}


def label(features, params=None, shot=None, game=GAME):
    p = dict(DEFAULTS, **(params or {}))
    n = features.n
    e = control.evidence(features) * p["touch_gain"]
    dead = deadball.score(features, shot) * p["dead_gain"]

    r = control.restart_evidence(features) * p["restart_gain"]

    score = np.zeros((n, 4))
    score[:, segment.S_H] = e
    score[:, segment.S_A] = -e
    score[:, segment.S_DH] = dead + r
    score[:, segment.S_DA] = dead - r
    cost = segment.cost_matrix(p["switch_ha"], p["switch_dead"], p["switch_restarter"])

    out = np.full(n, "X", dtype="<U1")
    for _, first, last in half_video_blocks(game):
        path = segment.viterbi(score[first:last + 1], cost)
        out[first:last + 1] = segment.LETTERS[segment.STATE_LABEL[path]]
    return out


def run(params=None, game=GAME, data_dir=None, use_broadcast=True):
    tracking = atd.load(game, data_dir)
    shot = broadcast.Broadcast(tracking, game, data_dir).shot if use_broadcast else None
    return label(Features(tracking), params, shot, game)


def write_submission(path, labels):
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    return path
