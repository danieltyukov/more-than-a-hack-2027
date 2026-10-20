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

    Returns an int8 array of hidden states.
    """
    n, k = score.shape
    back = np.zeros((n, k), dtype=np.int8)
    best = score[0].astype(np.float64).copy()
    cost = np.asarray(switch_cost, dtype=np.float64)
    cols = np.arange(k)
    for t in range(1, n):
        cand = best[:, None] - cost              # cand[i, j]: come from i, arrive in j
        prev = cand.argmax(axis=0)
        best = cand[prev, cols] + score[t]
        back[t] = prev
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(best.argmax())
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
