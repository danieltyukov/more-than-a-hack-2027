"""Refit the dead-ball weights in possession/deadball.py from the public ground truth.

    python scripts/fit_deadball.py

Prints a WEIGHTS block to paste over the one in deadball.py. Uses only the published five
minutes; nothing else in the repository is fitted to a ground truth.
"""

import os
import sys

import numpy as np
from scipy.optimize import minimize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from possession import atd, broadcast, deadball  # noqa: E402
from possession.features import Features  # noqa: E402

GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
L2 = 1.0


def main():
    tracking = atd.load()
    f = Features(tracking)
    shot = broadcast.Broadcast(tracking).shot
    feats = deadball.features(f, shot)
    names = list(feats)

    gt = pm.label_codes(pm.read_label_csv(GT))
    graded = gt != pm.X
    y = (gt[graded] == pm.D).astype(float)
    x = np.column_stack([feats[k] for k in names])[graded]

    def loss(beta):
        z = beta[0] + x @ beta[1:]
        return (np.sum(np.logaddexp(0.0, z) - y * z) + L2 * np.sum(beta[1:] ** 2)) / len(y)

    beta = minimize(loss, np.zeros(len(names) + 1), method="BFGS").x
    z = beta[0] + x @ beta[1:]
    print("frames {}  dead {}  in-sample accuracy {:.3f}".format(
        len(y), int(y.sum()), float(np.mean((z > 0) == (y > 0)))))
    print("WEIGHTS = {")
    print('    "intercept": {:.3f},'.format(beta[0]))
    for name, b in zip(names, beta[1:]):
        print('    "{}": {:.3f},'.format(name, b))
    print("}")


if __name__ == "__main__":
    main()
