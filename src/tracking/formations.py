"""Load user-marked pre-snap position points and match them to tracked players."""
import json
from pathlib import Path

import numpy as np

from src.tracking.tracking import inferred_head_box


ALLOWED_POSITIONS = {"QB", "RB", "WR", "TE", "DB", "LB"}


def load_formation_map(path):
    """Validate a versioned JSON map keyed by clip stem."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(data, dict) or data.get("version") != 1 or
            not isinstance(data.get("plays"), dict)):
        raise ValueError("Formation map requires version 1 and a plays object.")
    for clip, play in data["plays"].items():
        if (not isinstance(play, dict) or
                not isinstance(play.get("anchor_frame"), int) or
                isinstance(play.get("anchor_frame"), bool)):
            raise ValueError(f"Formation {clip!r} requires an integer anchor_frame.")
        if play["anchor_frame"] < 0:
            raise ValueError(f"Formation {clip!r} anchor_frame must be nonnegative.")
        slots = play.get("slots")
        if not isinstance(slots, list) or not slots:
            raise ValueError(f"Formation {clip!r} requires at least one slot.")
        if len(slots) > 22:
            raise ValueError(f"Formation {clip!r} cannot contain more than 22 slots.")
        names = set()
        for slot in slots:
            if not isinstance(slot, dict):
                raise ValueError(f"Formation {clip!r} slots must be objects.")
            name = slot.get("name")
            position = slot.get("position")
            point = slot.get("point")
            if not isinstance(name, str) or not name.strip() or name in names:
                raise ValueError(f"Formation {clip!r} slot names must be unique and nonempty.")
            names.add(name)
            if not isinstance(position, str) or position.upper() not in ALLOWED_POSITIONS:
                raise ValueError(f"Unsupported position {position!r} in formation {clip!r}.")
            if (not isinstance(point, list) or len(point) != 2 or
                    any(not isinstance(value, (int, float)) or not 0 <= value <= 1
                        for value in point)):
                raise ValueError(f"Slot {name!r} requires normalized [x, y] point values from 0 to 1.")
        radius = play.get("max_distance", .06)
        if not isinstance(radius, (int, float)) or not 0 < radius <= .2:
            raise ValueError(f"Formation {clip!r} max_distance must be in (0, 0.2].")
    return data["plays"]


def match_formation_slots(observations, play, frame_shape, team_names=None):
    """Assign at most one tracked identity to each nearby, unambiguous slot."""
    height, width = frame_shape[:2]
    radius = float(play.get("max_distance", .06))
    candidates = []
    slot_options = {index: [] for index in range(len(play["slots"]))}
    for index, slot in enumerate(play["slots"]):
        target = np.asarray((slot["point"][0] * width, slot["point"][1] * height))
        expected_team = slot.get("team")
        for item in observations:
            if item.track_id is None:
                continue
            if expected_team:
                actual_team = (team_names or {}).get(item.team, item.team)
                if actual_team.casefold() != str(expected_team).casefold():
                    continue
            box = inferred_head_box(item.box) if item.source == "body_support" else item.box
            center = np.asarray(((box[0]+box[2])/2, (box[1]+box[3])/2))
            distance = float(np.linalg.norm(center-target) / max(width, height))
            if distance <= radius:
                candidates.append((distance, index, item.track_id))
                slot_options[index].append((distance, item.track_id))

    ambiguous = set()
    for index, options in slot_options.items():
        options.sort()
        if len(options) > 1 and options[1][0] - options[0][0] < .006:
            ambiguous.add(index)

    assignments = {}
    used_slots = set()
    for distance, index, identity in sorted(candidates):
        if index in ambiguous or index in used_slots or identity in assignments:
            continue
        assignments[identity] = play["slots"][index]
        used_slots.add(index)
    return assignments, [play["slots"][index] for index in range(len(play["slots"]))
                         if index not in used_slots]