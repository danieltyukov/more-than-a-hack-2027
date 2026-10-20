"""Per-frame models trained on the DFL matches, and the decoder that turns them into labels.

Three gradient-boosted classifiers over framefeatures.compute():

  dead     is the ball dead on this frame?
  team     if it is alive, does home have it?
  restart  if it is dead, will home have it when play restarts?

Their log-probabilities are the emissions of the four-state decoder in segment.py. The fitted
models are committed under models/ so that writing a submission needs no training and gives the
same file every time; scripts/train.py rebuilds them from the public data.
"""

import os
import pickle

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from . import segment
from .atd import ROOT

MODEL_DIR = os.path.join(ROOT, "models")
EPS = 1e-4
KICKOFF_LIVE = 25

GBM = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=200,
           l2_regularization=1.0, early_stopping=False)

# chosen by leave-one-match-out spellF1 on the degraded DFL matches (scripts/train.py)
DECODER = {
    "switch_ha": 4.0,
    "switch_dead": 8.0,
    "switch_restarter": 1.0,
    "temper": 1.0,
}


def fit(x, labels, seed=0):
    """labels: per-row 0 H, 1 A, 2 D, plus next_home for dead rows (1 if home restarts)."""
    y, next_home = labels
    models = {}
    models["dead"] = HistGradientBoostingClassifier(random_state=seed, **GBM).fit(x, y == 2)
    alive = y != 2
    models["team"] = HistGradientBoostingClassifier(random_state=seed, **GBM).fit(
        x[alive], y[alive] == 0)
    dead = (y == 2) & (next_home >= 0)
    models["restart"] = HistGradientBoostingClassifier(random_state=seed, **GBM).fit(
        x[dead], next_home[dead] == 1)
    return models


def predict(models, x):
    """(p_dead, p_home_if_alive, p_home_restarts) per row."""
    return tuple(np.clip(models[k].predict_proba(x)[:, 1], EPS, 1 - EPS)
                 for k in ("dead", "team", "restart"))


STACK_OFFSETS = (-50, -25, -12, -6, -3, 0, 3, 6, 12, 25, 50)


def _shifted(x, k):
    out = np.full_like(x, np.nan)
    if k > 0:
        out[:-k] = x[k:]
    elif k < 0:
        out[-k:] = x[:k]
    else:
        out[:] = x
    return out


def _side_mean(x, w, future):
    if future:
        return _side_mean(x[::-1], w, False)[::-1]
    cs = np.concatenate(([0.0], np.cumsum(x)))
    i = np.arange(len(x))
    lo = np.maximum(0, i - w)
    return (cs[i] - cs[lo]) / np.maximum(i - lo, 1)


def stack_features(p_dead, p_home, p_restart):
    """Second-stage inputs: the first-stage log-odds around each frame."""
    cols = []
    for p in (p_dead, p_home, p_restart):
        logit = np.log(p) - np.log1p(-p)
        cols += [_shifted(logit, k) for k in STACK_OFFSETS]
        cols += [_side_mean(logit, w, f) for w in (12, 50) for f in (False, True)]
    dead = np.log(p_dead) - np.log1p(-p_dead)
    pad = np.pad(dead, 25, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(pad, 51)
    cols += [windows.min(axis=1), windows.max(axis=1)]
    return np.column_stack(cols).astype(np.float32)


def average(probs):
    """Average probabilities from several model sets in log-odds."""
    z = np.mean([np.log(p) - np.log1p(-p) for p in probs], axis=0)
    return np.clip(1.0 / (1.0 + np.exp(-z)), EPS, 1 - EPS)


def emissions(p_dead, p_home, p_restart, dead_offset=None, temper=1.0):
    """(n, 4) log-scores for the hidden states of segment.py.

    dead_offset, if given, is added to the dead log-odds before splitting it into probabilities;
    it is how evidence the DFL models never saw (the broadcast shot) gets in.
    """
    logit = np.log(p_dead) - np.log1p(-p_dead)
    if dead_offset is not None:
        logit = logit + dead_offset
    log_dead = -np.logaddexp(0.0, -logit)
    log_alive = -np.logaddexp(0.0, logit)
    s = np.empty((len(p_dead), 4))
    s[:, segment.S_H] = log_alive + np.log(p_home)
    s[:, segment.S_A] = log_alive + np.log1p(-p_home)
    s[:, segment.S_DH] = log_dead + np.log(p_restart)
    s[:, segment.S_DA] = log_dead + np.log1p(-p_restart)
    return s * temper


def decode(scores, blocks, params=None, dead_before=None):
    """dead_before: {half: kickoff frame} -- the ball is dead from the start of that half's block
    up to the kickoff and alive for the second after it, whatever the scores say."""
    p = dict(DECODER, **(params or {}))
    cost = segment.cost_matrix(p["switch_ha"], p["switch_dead"], p["switch_restarter"])
    out = np.full(len(scores), -1, dtype=np.int8)
    for half, first, last in blocks:
        block = scores[first:last + 1] * p["temper"]
        kickoff = (dead_before or {}).get(half)
        if kickoff is not None and first <= kickoff <= last:
            k = kickoff - first
            block = block.copy()
            block[:k, segment.S_H] = -segment.NEVER
            block[:k, segment.S_A] = -segment.NEVER
            # the kickoff puts the ball in play
            block[k:k + KICKOFF_LIVE, segment.S_DH] = -segment.NEVER
            block[k:k + KICKOFF_LIVE, segment.S_DA] = -segment.NEVER
        path = segment.viterbi(block, cost)
        out[first:last + 1] = segment.STATE_LABEL[path]
    return out


def shift_stoppages(states, blocks, onset_lead, restart_lead, fixed=None):
    """Move every predicted stoppage earlier: its start by onset_lead frames, its end by
    restart_lead. A stoppage never takes more than half of the play before it, so no possession
    disappears, and one that would shrink below a second keeps its end. fixed: {half: frame},
    stoppages ending there (the kickoff) keep their end.
    """
    out = states.copy()
    for half, first, last in blocks:
        seg = states[first:last + 1]
        edges = np.flatnonzero(np.diff(seg) != 0) + 1
        bounds = np.concatenate(([0], edges, [len(seg)]))
        runs = [(first + bounds[i], first + bounds[i + 1], seg[bounds[i]])
                for i in range(len(bounds) - 1)]
        for i, (s, e, lab) in enumerate(runs):
            if lab != segment.D:
                continue
            if i > 0 and runs[i - 1][2] != segment.D:
                ps, pe, _ = runs[i - 1]
                lead = min(onset_lead, (pe - ps) // 2)
                out[s - lead:s] = segment.D
            ends_at_kickoff = fixed and fixed.get(half) == e
            if i + 1 < len(runs) and not ends_at_kickoff and e - s - restart_lead >= 25:
                out[e - restart_lead:e] = runs[i + 1][2]
    return out


def next_restarter(labels):
    """For dead frames, 1 if home has the ball when play next resumes, 0 if away, -1 if never."""
    out = np.full(len(labels), -1, dtype=np.int8)
    upcoming = -1
    for i in range(len(labels) - 1, -1, -1):
        lab = labels[i]
        if lab == 0:
            upcoming = 1
        elif lab == 1:
            upcoming = 0
        elif lab == 2:
            out[i] = upcoming
        else:
            upcoming = -1
    return out


def save(models, folder=MODEL_DIR):
    os.makedirs(folder, exist_ok=True)
    for name, model in models.items():
        with open(os.path.join(folder, name + ".pkl"), "wb") as handle:
            pickle.dump(model, handle, protocol=4)


def load(folder=MODEL_DIR, prefix=""):
    # these are the repository's own files written by save(); never point this at anything else
    models = {}
    for name in ("dead", "team", "restart"):
        with open(os.path.join(folder, prefix + name + ".pkl"), "rb") as handle:
            models[name] = pickle.load(handle)
    return models


def has_stack(folder=MODEL_DIR):
    return os.path.isfile(os.path.join(folder, "stack_dead.pkl"))
