"""LOS-relative geometry. Coordinates must share an explicitly chosen unit system."""
from dataclasses import asdict, dataclass
import math


SHELL_FOLDERS = {"0_high": "zero_high", "1_high": "one_high",
                 "2_high": "two_high", "unknown": "unknown"}
ROTATION_FOLDERS = {"stay_1": "stay_1", "stay_2": "stay_2",
                    "2_to_1": "two_to_one", "1_to_2": "one_to_two",
                    "2_to_0": "two_to_zero", "unknown_rotation": "unknown_rotation"}


@dataclass(frozen=True)
class ShellResult:
    shell: str
    deep_defenders: int | None
    confidence: float
    reason: str


def classify_shell(defenders, *, complete=False, depth_threshold=10.0,
                   lateral_limit=18.0, middle_limit=6.0, margin=1.0):
    """Input: defensive points {lateral, depth} relative to field center and LOS.

    Defaults are yards. For pixels, supply thresholds in the same pixel basis.
    `complete` certifies visibility AND reliable defensive detections, not merely
    a settled camera. Wide corners are excluded by the lateral gate. This is a
    heuristic, not position recognition or calibrated confidence.
    """
    settings = (depth_threshold, lateral_limit, middle_limit, margin)
    if not all(math.isfinite(v) and v > 0 for v in settings):
        raise ValueError("Geometry thresholds must be finite and positive")
    if middle_limit >= lateral_limit or margin >= depth_threshold:
        raise ValueError("Require middle_limit < lateral_limit and margin < depth_threshold")
    points = [(float(p["lateral"]), float(p["depth"])) for p in defenders]
    if not all(math.isfinite(x) and math.isfinite(y) for x, y in points):
        raise ValueError("Defender coordinates must be finite")
    if complete is not True or not points:
        return ShellResult("unknown", None, 0.0, "Incomplete visibility or unverified defensive detections")
    if any(abs(y - depth_threshold) <= margin and abs(x) <= lateral_limit + margin
           or y >= depth_threshold and abs(abs(x) - lateral_limit) <= margin
           for x, y in points):
        return ShellResult("unknown", None, 0.0, "Defender lies near a geometry threshold")
    deep = [(x, y) for x, y in points if y > depth_threshold and abs(x) < lateral_limit]
    count = len(deep)
    if count == 0:
        return ShellResult("0_high", 0, .75, "No central deep defender in a verified complete view")
    if count == 1 and abs(deep[0][0]) <= middle_limit:
        return ShellResult("1_high", 1, .75, "One deep middle candidate")
    if count == 2 and min(x for x, _ in deep) < -margin and max(x for x, _ in deep) > margin:
        return ShellResult("2_high", 2, .75, "Two deep candidates split across field center")
    return ShellResult("unknown", count, 0.0, "Ambiguous alignment or optional three-high structure")


def analyze_play(play):
    """Pre/post observations are independent; no claim of identity tracking."""
    geometry = play.get("geometry", {})
    results = []
    for phase in ("pre_snap", "post_snap"):
        observation = play.get(phase, {})
        results.append(classify_shell(observation.get("defenders", []),
                                     complete=observation.get("complete", False), **geometry))
    pre, post = results
    rotation = "unknown_rotation"
    if "unknown" not in (pre.shell, post.shell):
        a, b = pre.shell[0], post.shell[0]
        candidate = f"stay_{a}" if a == b else f"{a}_to_{b}"
        if candidate in ROTATION_FOLDERS:
            rotation = candidate
    return dict(play_id=play["play_id"], pre_snap_shell=pre.shell,
                post_snap_shell=post.shell, rotation=rotation,
                deep_defenders_pre_snap=pre.deep_defenders,
                deep_defenders_post_snap=post.deep_defenders,
                confidence=min(pre.confidence, post.confidence) if rotation != "unknown_rotation" else 0.0,
                evidence={"pre_snap": asdict(pre), "post_snap": asdict(post)})


def analysis_window(snap, fps, frame_count, pre=1.0, post=2.5):
    """Return [start, stop) original-clip frame indexes; never invent snap time."""
    if not all(math.isfinite(v) for v in (snap, fps, pre, post)):
        raise ValueError("Window values must be finite")
    if fps <= 0 or frame_count <= 0 or snap <= 0 or snap >= frame_count / fps or pre <= 0 or post <= 0:
        raise ValueError("Snap must be inside the clip; window durations and FPS must be positive")
    return max(0, math.ceil((snap - pre) * fps)), min(frame_count, math.ceil((snap + post) * fps))
