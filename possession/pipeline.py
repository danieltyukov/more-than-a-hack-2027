"""Tracking in, one label per video frame out.

Two ways of scoring frames share the decoder:

  learned   the DFL-trained classifiers in learned.py, plus the broadcast shot as extra dead-ball
            evidence. This is what writes the submission.
  rules     the hand-built touch and dead-ball scores (control.py, deadball.py). Kept as the
            reference the learned version has to beat.
"""

import json
import os

import numpy as np

from games import GAME

from . import atd, broadcast, control, deadball, framefeatures, learned, segment
from .features import Features
from .smooth import window_frac

RULE_DEFAULTS = {
    "touch_gain": 1.0,       # weight of one frame of touch evidence
    "switch_ha": 12.0,       # price of a turnover
    "switch_dead": 25.0,     # price of entering or leaving a stoppage
    "dead_gain": 2.0,        # weight of the dead-ball log-odds
    "restart_gain": 2.0,     # weight of "who is standing over the dead ball"
    "switch_restarter": 4.0, # price of changing our mind about who restarts
}


def shot_features(shot):
    return {"no_pitch": window_frac(shot == broadcast.NO_PITCH, 25),
            "other_view": window_frac(shot == broadcast.OTHER_VIEW, 25)}


def letters(states):
    out = np.full(len(states), "X", dtype="<U1")
    played = states >= 0
    out[played] = segment.LETTERS[states[played]]
    return out


def rule_scores(f, params=None, shot=None):
    p = dict(RULE_DEFAULTS, **(params or {}))
    e = control.evidence(f) * p["touch_gain"]
    dead = deadball.score(f, shot) * p["dead_gain"]
    r = control.restart_evidence(f) * p["restart_gain"]
    score = np.zeros((f.n, 4))
    score[:, segment.S_H] = e
    score[:, segment.S_A] = -e
    score[:, segment.S_DH] = dead + r
    score[:, segment.S_DA] = dead - r
    return score


def rule_label(f, blocks, params=None, shot=None):
    p = dict(RULE_DEFAULTS, **(params or {}))
    cost = segment.cost_matrix(p["switch_ha"], p["switch_dead"], p["switch_restarter"])
    score = rule_scores(f, p, shot)
    states = np.full(f.n, -1, dtype=np.int8)
    for _, first, last in blocks:
        states[first:last + 1] = segment.STATE_LABEL[segment.viterbi(score[first:last + 1], cost)]
    return letters(states)


def first_stage_sets(folder=learned.MODEL_DIR):
    """Folders of first-stage models: models/first/* if there are several, else models/ itself."""
    nested = os.path.join(folder, "first")
    if os.path.isdir(nested):
        return [os.path.join(nested, d) for d in sorted(os.listdir(nested))]
    return [folder]


def load_settings(folder=learned.MODEL_DIR):
    """(decoder costs, broadcast weights or None). The second stage brings its own costs."""
    name = "stack.json" if learned.has_stack(folder) else "decoder.json"
    with open(os.path.join(folder, name)) as handle:
        decoder = json.load(handle)["decoder"]
    path = os.path.join(folder, "broadcast.json")
    weights = None
    if os.path.isfile(path):
        with open(path) as handle:
            weights = json.load(handle)
    return decoder, weights


def probabilities(t, folder=learned.MODEL_DIR):
    """(p_dead, p_home, p_restart) per frame: every first-stage set, averaged, then the second
    stage if there is one."""
    x, _ = framefeatures.compute(t, Features(t))
    probs = []
    for path in first_stage_sets(folder):
        models = learned.load(path)
        expected = models["dead"].n_features_in_
        if x.shape[1] != expected:
            raise RuntimeError("the models in {} expect {} features, framefeatures gives {}: "
                               "retrain with scripts/train.py".format(path, expected, x.shape[1]))
        probs.append(learned.predict(models, x))
    p = tuple(learned.average([pr[k] for pr in probs]) for k in range(3))
    if learned.has_stack(folder):
        p = learned.predict(learned.load(folder, "stack_"), learned.stack_features(*p))
    return p


def dead_offset(p_dead, shot, weights):
    """Turn the DFL dead-ball log-odds into the calibrated one with the broadcast added."""
    if weights is None or shot is None:
        return None
    logit = np.log(p_dead) - np.log1p(-p_dead)
    z = weights["intercept"] + (weights["model"] - 1.0) * logit
    for name, value in shot_features(shot).items():
        z = z + weights[name] * value
    return z


# How much earlier than decoded to put each stoppage's start and end, in frames. On the DFL
# matches the decoder finds both to within a few frames, but on this tracking the ball usually
# vanishes as it goes out and the players and the TV director react a second or two later, so
# the decoded stoppages come out late: all eight stoppage edges in the public window were, by
# 0.3 to 4.3 s. The shift used is a conservative part of that.
STOPPAGE_LEAD = {"onset": 25, "restart": 12}

# First live video frame of each half, as published in DATA.md and FAQ.md (and checked by
# starter/load_smart_tagging.py). The first half is graded from its kickoff; the second is graded
# from 110808 but kicks off at 113742, and the 2934 frames in between are all dead.
KICKOFFS = {"BAR-ATH": {1: 11185, 2: 113742}}


def learned_label(t, shot=None, folder=learned.MODEL_DIR, game=GAME):
    decoder, weights = load_settings(folder)
    p_dead, p_home, p_restart = probabilities(t, folder)
    scores = learned.emissions(p_dead, p_home, p_restart, dead_offset(p_dead, shot, weights))
    states = learned.decode(scores, t.blocks, decoder, KICKOFFS.get(game))
    states = learned.shift_stoppages(states, t.blocks, STOPPAGE_LEAD["onset"],
                                     STOPPAGE_LEAD["restart"], KICKOFFS.get(game))
    return letters(states)


def run(game=GAME, data_dir=None, use_broadcast=True):
    t = atd.load(game, data_dir)
    shot = broadcast.Broadcast(t, game, data_dir).shot if use_broadcast else None
    return learned_label(t, shot)


def write_submission(path, labels):
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    return path
