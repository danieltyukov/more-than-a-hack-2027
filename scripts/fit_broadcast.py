"""Fit how much the broadcast shot adds to the DFL dead-ball model, on the public ground truth.

    python scripts/fit_broadcast.py

The DFL matches have no broadcast, so the learned dead-ball model never sees the shot type. This
fits a three-feature logistic regression on the public five minutes (the learned models' final
dead log-odds, and the share of the surrounding second the broadcast shows no pitch or another
view) and writes the weights to models/broadcast.json. Three weights and an intercept on 7500
frames. Rerun it whenever models/ changes.
"""

import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from possession import atd, broadcast, learned  # noqa: E402
from possession.pipeline import probabilities, shot_features  # noqa: E402

GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
L2 = 1.0


def main():
    t = atd.load()
    p_dead = probabilities(t)[0]
    shot = broadcast.Broadcast(t).shot
    gt = pm.label_codes(pm.read_label_csv(GT))
    graded = gt != pm.X

    logit = np.log(p_dead) - np.log1p(-p_dead)
    sf = shot_features(shot)
    cols = np.column_stack([logit] + [sf[k] for k in sorted(sf)])[graded]
    y = (gt[graded] == pm.D).astype(float)

    def loss(beta):
        z = beta[0] + cols @ beta[1:]
        return (np.sum(np.logaddexp(0.0, z) - y * z) + L2 * np.sum(beta[2:] ** 2)) / len(y)

    beta = minimize(loss, np.r_[0.0, 1.0, np.zeros(len(sf))], method="BFGS").x
    weights = {"intercept": beta[0], "model": beta[1]}
    weights.update({k: b for k, b in zip(sorted(sf), beta[2:])})
    with open(os.path.join(learned.MODEL_DIR, "broadcast.json"), "w") as handle:
        json.dump({k: round(float(v), 4) for k, v in weights.items()}, handle, indent=1)
    z = beta[0] + cols @ beta[1:]
    print("in-sample dead accuracy {:.3f} (model alone {:.3f})".format(
        float(np.mean((z > 0) == (y > 0))), float(np.mean((cols[:, 0] > 0) == (y > 0)))))
    print(weights)


if __name__ == "__main__":
    main()
