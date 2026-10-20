"""Frame-level ball possession from the automatic tracking of one match.

The pipeline, in the order it runs:

    atd        raw tracking text -> numpy arrays (cached)
    features   per-frame geometry: ball, nearest players, team shape, motion
    deadball   alive / dead
    control    who is touching the ball, frame by frame
    segment    turn the noisy per-frame evidence into whole possessions

`python -m possession` writes a submission. See README.md.
"""
