"""Unit tests that need no match data: pytest tests/"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from possession import broadcast, degrade, learned, segment, smooth  # noqa: E402
from possession.atd import Tracking  # noqa: E402
from possession.features import bridge_short_gaps  # noqa: E402

H, A, D = segment.H, segment.A, segment.D


def scores_for(sequence, strength=1.0):
    """(n, 4) scores that favour the given hidden states frame by frame."""
    s = np.zeros((len(sequence), 4))
    s[np.arange(len(sequence)), sequence] = strength
    return s


def test_viterbi_follows_strong_evidence():
    seq = [segment.S_H] * 50 + [segment.S_A] * 50
    path = segment.viterbi(scores_for(seq), segment.cost_matrix(5.0, 5.0, 1.0))
    assert (path[:50] == segment.S_H).all() and (path[50:] == segment.S_A).all()


def test_viterbi_ignores_a_short_blip():
    seq = [segment.S_H] * 50 + [segment.S_A] * 3 + [segment.S_H] * 50
    path = segment.viterbi(scores_for(seq), segment.cost_matrix(5.0, 5.0, 1.0))
    assert (path == segment.S_H).all()


def test_dead_state_restarts_only_with_its_own_side():
    cost = segment.cost_matrix(5.0, 5.0, 1.0)
    assert cost[segment.S_DH, segment.S_A] >= segment.NEVER
    assert cost[segment.S_DA, segment.S_H] >= segment.NEVER
    # dead with home to restart, then strong away evidence: the decoder has to pass through
    # dead-away rather than jump straight to away
    seq = [segment.S_DH] * 30 + [segment.S_A] * 30
    path = segment.viterbi(scores_for(seq, 3.0), cost)
    assert segment.S_H not in path[25:35]


def test_next_restarter():
    labels = np.array([H, D, D, A, A, D, D, H, 3, D], dtype=np.int8)
    out = learned.next_restarter(labels)
    assert out[1] == 0 and out[2] == 0          # away restarts after the first stoppage
    assert out[5] == 1 and out[6] == 1          # home after the second
    assert out[9] == -1                         # nothing follows the last one
    assert (out[[0, 3, 4, 7, 8]] == -1).all()   # only dead frames get a value


def test_shift_stoppages_moves_both_edges_earlier():
    states = np.array([H] * 100 + [D] * 100 + [A] * 100, dtype=np.int8)
    out = learned.shift_stoppages(states, [(1, 0, 299)], onset_lead=20, restart_lead=10)
    assert (out[80:190] == D).all()
    assert (out[190:] == A).all() and (out[:80] == H).all()


def test_shift_stoppages_never_removes_a_possession():
    states = np.array([A] * 50 + [H] * 10 + [D] * 100 + [A] * 50, dtype=np.int8)
    out = learned.shift_stoppages(states, [(1, 0, 209)], onset_lead=40, restart_lead=0)
    assert (out == H).sum() == 5


def test_shift_stoppages_keeps_the_kickoff():
    states = np.array([D] * 100 + [H] * 100, dtype=np.int8)
    out = learned.shift_stoppages(states, [(2, 0, 199)], 10, 10, fixed={2: 100})
    assert (out == states).all()


def test_bridge_short_gaps():
    xy = np.arange(20, dtype=np.float32).repeat(2).reshape(-1, 2)
    xy[5:8] = np.nan        # short, bridged
    xy[10:19] = np.nan      # longer than max_gap, left alone
    out, filled = bridge_short_gaps(xy, max_gap=4)
    assert np.allclose(out[5:8, 0], [5, 6, 7])
    assert filled[5:8].all() and np.isnan(out[10:19]).all()


def test_window_statistics():
    x = np.array([1.0, np.nan, 3.0, 5.0])
    assert np.allclose(smooth.window_mean(x, 1), [1.0, 2.0, 4.0, 4.0])
    mask = np.array([True, False, False, True, False])
    assert list(smooth.frames_since(mask)) == [0, 1, 2, 0, 1]
    assert list(smooth.frames_until(mask)[:4]) == [0, 2, 1, 0]


def synthetic_match(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 0.3, size=(n, 2))
    ball = (np.cumsum(steps, axis=0) % [100.0, 60.0] + [2.5, 4.0]).astype(np.float32)
    players = (ball[:, None, :] + rng.normal(0, 8, size=(n, 22, 2))).astype(np.float32)
    team = np.array([0] * 11 + [1] * 11, dtype=np.int8)
    labels = np.zeros(n, dtype=np.int8)
    labels[1000:1400] = D
    labels[3000:3600] = A
    t = Tracking(players, np.full((n, 22), 2.0, np.float32), ball, np.zeros(n, np.float32), team,
                 (105.0, 68.0), [(1, 0, n - 1)])
    return t, labels


def test_degrade_does_not_freeze_the_dead_ball():
    t, labels = synthetic_match()
    t.ball[1000:1400] = t.ball[999]     # the DFL habit: a frozen ball for the whole stoppage
    params = dict(degrade.draw_params(np.random.default_rng(1)), dead_hidden=0.0, alive_hidden=0.0)
    noisy, _ = degrade.degrade(t, labels, np.random.default_rng(2), params)
    step = np.linalg.norm(np.diff(noisy.ball[1000:1400], axis=0), axis=1)
    assert np.nanmax(step) > 0.0
    # and no jump to the restart spot
    assert np.linalg.norm(noisy.ball[1400] - noisy.ball[1399]) < 2.0


def test_degrade_leaves_no_short_gaps():
    t, labels = synthetic_match()
    noisy, _ = degrade.degrade(t, labels, np.random.default_rng(3))
    hidden = np.isnan(noisy.ball[:, 0])
    edges = np.diff(np.concatenate(([0], hidden.astype(np.int8), [0])))
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    assert (lengths >= 75).all()


@pytest.mark.parametrize("offset", [300, 1700])
def test_broadcast_offset_is_recovered(offset):
    tac, _ = synthetic_match(n=8000, seed=4)
    n = 5000
    bro_ball = tac.ball[offset:offset + n].copy()
    bro_ball[1000:1500] = np.nan        # a replay: no usable ball
    bro = Tracking(tac.players[offset:offset + n], tac.speed[offset:offset + n], bro_ball,
                   np.zeros(n, np.float32), tac.team, tac.pitch)
    old = broadcast.COARSE
    broadcast.COARSE = (-500, 2500, 5)
    try:
        found = broadcast.find_offsets(tac, bro, n_halves=1)
    finally:
        broadcast.COARSE = old
    assert found == [offset]
