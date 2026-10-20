"""Smart Tagging intervals turned into "this team had the ball" hints.

Every phase is tagged with the team performing it. For the attacking phases that is the team in
possession; for the defending phases it is the team without the ball. Set pieces are restarts, so
their team is the one about to have the ball, except throw-ins, which come with no team at all.
"""

import os
import sys

import numpy as np

from games import GAME

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "starter"))
import load_smart_tagging  # noqa: E402

ATTACKING = {"build up", "progression", "creating chances", "attacking transition", "long ball",
             "ball in final third", "ball in the box", "players in the box", "goals"}
DEFENDING = {"defending in attacking third", "defending in middle third",
             "defending in defensive third", "defensive transition"}
HOME_NAME = "barcelona"


def _side(team_tag):
    team = (team_tag or "").strip().lower()
    if team in ("", "n/a"):
        return None
    return 0 if team == HOME_NAME else 1


def load(game=GAME, data_dir=None):
    """[(code, start_frame, end_frame, side_in_possession, type)] with side 0 home, 1 away, or None."""
    out = []
    for event in load_smart_tagging.load(game, data_dir):
        code = (event["code"] or "").strip().lower()
        side = _side(event["tags"].get("Team"))
        if code in DEFENDING and side is not None:
            side = 1 - side
        elif code not in ATTACKING and code != "set pieces":
            continue
        kind = (event["tags"].get("Type") or "").strip().lower() or None
        out.append((code, event["start_frame"], event["end_frame"], side, kind))
    return out


def possession_prior(n, events):
    """(n,) votes: +1 per home-possession phase covering the frame, -1 per away one."""
    votes = np.zeros(n)
    for code, start, stop, side, _ in events:
        if side is None or code == "set pieces":
            continue
        votes[max(0, start):min(n, stop)] += 1 if side == 0 else -1
    return votes
