"""Viterbi decoding of per-frame scores into H / A / D runs.

The ranking metric scores possessions as whole intervals, so the decoder is where most of the
score is won or lost: a per-frame argmax flickers and every flicker is a false possession. Here a
change of state has a price, and a run only changes label when the evidence summed over the frames
since the change pays for it.

Dead ball is two hidden states, not one: dead with home to restart and dead with away to restart.
Each can only be left into its own team's possession. That is what lets the player standing over
the ball during a stoppage -- the thrower, the corner taker, the goalkeeper at a goal kick -- vote
on who has the ball once play resumes, instead of the restart being decided by whatever happens in
the first second after it.
"""

import numpy as np
from numba import njit

H, A, D = 0, 1, 2
LETTERS = np.array(["H", "A", "D"])

# hidden states
S_H, S_A, S_DH, S_DA = 0, 1, 2, 3
STATE_LABEL = np.array([H, A, D, D], dtype=np.int8)
NEVER = 1e9


def viterbi(score, switch_cost):
    """Best state path for per-frame log-scores.

    score        (n, k) log-score of each hidden state on each frame (higher is better)
    switch_cost  (k, k) price of moving from state i to state j, zero on the diagonal

    Returns an int8 array of hidden states. Ties go to staying in the current state.
    """
    return _viterbi(np.ascontiguousarray(score, dtype=np.float64),
                    np.ascontiguousarray(switch_cost, dtype=np.float64))


@njit(cache=True)
def _viterbi(score, cost):
    n, k = score.shape
    back = np.zeros((n, k), dtype=np.int8)
    best = score[0].copy()
    new = np.empty(k)
    for t in range(1, n):
        for j in range(k):
            arg = j
            val = best[j]                        # staying wins ties
            for i in range(k):
                c = best[i] - cost[i, j]
                if c > val:
                    val = c
                    arg = i
            new[j] = val + score[t, j]
            back[t, j] = arg
        best[:] = new
    path = np.empty(n, dtype=np.int8)
    path[n - 1] = np.argmax(best)
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def cost_matrix(switch_ha, switch_dead, switch_restarter):
    c = np.full((4, 4), NEVER)
    np.fill_diagonal(c, 0.0)
    c[S_H, S_A] = c[S_A, S_H] = switch_ha
    for alive in (S_H, S_A):
        c[alive, S_DH] = c[alive, S_DA] = switch_dead
    c[S_DH, S_H] = c[S_DA, S_A] = switch_dead
    c[S_DH, S_DA] = c[S_DA, S_DH] = switch_restarter
    return c
