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


def decode(scores, blocks, params=None):
    p = dict(DECODER, **(params or {}))
    cost = segment.cost_matrix(p["switch_ha"], p["switch_dead"], p["switch_restarter"])
    out = np.full(len(scores), -1, dtype=np.int8)
    for _, first, last in blocks:
        path = segment.viterbi(scores[first:last + 1] * p["temper"], cost)
        out[first:last + 1] = segment.STATE_LABEL[path]
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


def load(folder=MODEL_DIR):
    # these are the repository's own files written by save(); never point this at anything else
    models = {}
    for name in ("dead", "team", "restart"):
        with open(os.path.join(folder, name + ".pkl"), "rb") as handle:
            models[name] = pickle.load(handle)
    return models
