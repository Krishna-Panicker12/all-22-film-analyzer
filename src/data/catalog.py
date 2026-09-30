"""Flat clip storage with explicit game identity and legacy-path resolution."""
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
PASS_DIR=ROOT/'data/plays/passes'


def clip_name(game_id,play_id):
    for value in (game_id,play_id):
        if not re.fullmatch(r'[A-Za-z0-9_-]+',value) or '__' in value:
            raise ValueError('Game/play IDs must contain letters, digits, single underscores or hyphens')
    return f'{game_id}__{play_id}.mp4'


def game_identity(clip,explicit=None):
    path=Path(clip)
    if explicit: return str(explicit).casefold()
    if '__' in path.stem: return path.stem.split('__',1)[0].casefold()
    return str(path.parent).casefold()


def resolve_clip(path):
    path=Path(path)
    if not path.is_absolute(): path=ROOT/path
    if path.exists(): return path
    candidate=PASS_DIR/clip_name(path.parent.name,path.stem)
    return candidate if candidate.exists() else path
