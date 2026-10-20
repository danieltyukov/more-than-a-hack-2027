"""Viterbi decoding of per-frame scores into H / A / D runs.

The ranking metric scores possessions as whole intervals, so the decoder is where most of the
score is won or lost: a per-frame argmax flickers and every flicker is a false possession. Here a
change of state has a fixed price, and a run only changes label when the evidence summed over the
frames since the change pays for it.
"""

import numpy as np

H, A, D = 0, 1, 2
LETTERS = np.array(["H", "A", "D"])


def viterbi(score, switch_cost):
    """Best state path for per-frame log-scores.

    score        (n, 3) log-score of H, A, D on each frame (higher is better)
    switch_cost  (3, 3) price of moving from state i to state j, zero on the diagonal

    Returns an int8 array of states.
    """
    n = len(score)
    back = np.zeros((n, 3), dtype=np.int8)
    best = score[0].astype(np.float64).copy()
    cost = np.asarray(switch_cost, dtype=np.float64)
    for t in range(1, n):
        cand = best[:, None] - cost              # cand[i, j]: come from i, arrive in j
        prev = cand.argmax(axis=0)
        best = cand[prev, np.arange(3)] + score[t]
        back[t] = prev
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(best.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path
