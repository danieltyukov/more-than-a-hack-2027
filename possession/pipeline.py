"""Tracking in, one label per video frame out."""

import numpy as np

from games import GAME, half_video_blocks

from . import atd, control, deadball, segment
from .features import Features

DEFAULTS = {
    "touch_gain": 1.0,       # weight of one frame of touch evidence
    "switch_ha": 6.0,        # price of a turnover
    "switch_dead": 12.0,     # price of entering or leaving a stoppage
}


def label(features, params=None, game=GAME):
    p = dict(DEFAULTS, **(params or {}))
    n = features.n
    e = control.evidence(features) * p["touch_gain"]
    dead = deadball.score(features)

    score = np.zeros((n, 3))
    score[:, segment.H] = e
    score[:, segment.A] = -e
    score[:, segment.D] = dead

    cost = np.array([[0, p["switch_ha"], p["switch_dead"]],
                     [p["switch_ha"], 0, p["switch_dead"]],
                     [p["switch_dead"], p["switch_dead"], 0]])

    out = np.full(n, "X", dtype="<U1")
    for _, first, last in half_video_blocks(game):
        path = segment.viterbi(score[first:last + 1], cost)
        out[first:last + 1] = segment.LETTERS[path]
    return out


def run(params=None, game=GAME, data_dir=None):
    return label(Features(atd.load(game, data_dir)), params, game)


def write_submission(path, labels):
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    return path
