"""How dead the ball looks on each frame, as a log-score.

The one signal that survives every kind of stoppage is the players: whatever stopped the game,
twenty-two people slow to a walk within a couple of seconds and stay there until the restart.
"""

import numpy as np

from .smooth import window_mean

# Mean speed of the tracked players, m/s, below which play looks stopped. Live play in the public
# window averages 2.2-3.5 m/s per segment and stoppages 1.1-1.5.
STOPPED_SPEED = 1.6
SPEED_HALF_WINDOW = 50


def score(f, stopped_speed=STOPPED_SPEED, gain=2.0, half=SPEED_HALF_WINDOW):
    ms = window_mean(f.mean_speed, half)
    ms = np.where(np.isfinite(ms), ms, stopped_speed)
    return gain * (stopped_speed - ms)
