"""Per-frame evidence of which team is touching the ball.

Positive values favour home, negative favour away, zero means the frame says nothing (no ball, or
nobody near it). It is evidence, not a decision: a pass flying past an opponent puts him close to
the ball for a frame or two without changing possession, and the decoder is what stops that from
becoming a turnover.
"""

import numpy as np

# Reach of a player, in metres. Ball and player positions each carry tracker error of a few tens
# of centimetres, so the kernel is wider than a real foot.
REACH = 1.2

# A ball faster than this is in flight rather than at someone's feet, so being near it says little.
FLIGHT_SPEED = 9.0


def touch_kernel(d, reach=REACH):
    with np.errstate(invalid="ignore"):
        k = np.exp(-0.5 * (d / reach) ** 2)
    return np.where(np.isfinite(k), k, 0.0)


def evidence(f, reach=REACH, flight_speed=FLIGHT_SPEED):
    e = touch_kernel(f.d_home, reach) - touch_kernel(f.d_away, reach)
    speed = np.where(np.isfinite(f.ball_speed), f.ball_speed, 0.0)
    weight = 1.0 / (1.0 + np.exp((speed - flight_speed) / 1.5))
    return np.where(f.ball_known, e * weight, 0.0)
