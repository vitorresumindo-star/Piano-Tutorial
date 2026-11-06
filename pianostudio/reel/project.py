"""Reel project files: load/resolve configuration, prepare the song (hands,
fingering, timing) and render the audio track.

A project is a folder with a `project.json` (see projects/*/project.json and
README "Cinematic piano reels"). Shot times may be given in bars of the MIDI
file ("bar": 7) or seconds ("time": 12.5); both refer to the music timeline.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..audio import SR, NoteEvent, PedalSpan, Piano, apply_reverb, normalize, write_wav
from ..fingering import assign_fingering
from ..midi_io import load_midi
from ..model import Note, Song
from .layout import Layout, fit_range

DEFAULTS = {
    "title": "Song title",
    "subtitle": "Film title",
    "midi": "song.mid",
    "width": 1080,
    "height": 1920,
    "fps": 60,
    "leadIn": 0.4,           # seconds of falling notes before the first note sounds
    "tail": 2.4,             # seconds after the last key release (let the pedal ring)
    "layout": {"footageRatio": 0.38, "keyRatio": 0.142, "bodyRatio": 0.112, "range": "auto", "margin": 1},
    "notes": {"fallSeconds": 2.2, "colorMode": "pitch", "palette": {}, "handColors": ["#4CCB72", "#FFA630"],
              "labels": True, "keyLabels": "all", "spelling": "common", "barLines": True},
    "hands": {"skin": None, "outline": None},
    "titleStyle": {"y": 0.27, "size": 104, "subSize": 56, "revealAt": 0.25, "revealSeconds": 1.5},
    "audio": {"reverb": 0.30, "reverbSeconds": 2.4, "targetLufs": -14.0, "volume": 1.0, "fadeOut": 1.6},
    "footage": {"transition": 0.6, "grain": 0.012, "vignette": 0.32, "shots": []},
    "output": {"dir": None, "name": None, "crf": 17, "shareCrf": 21, "preset": "medium"},
}


def deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_project(path: str | Path) -> dict:
    path = Path(path)
    if path.is_dir():
        path = path / "project.json"
    cfg = deep_merge(DEFAULTS, json.loads(path.read_text()))
    cfg["_dir"] = str(path.parent.resolve())
    cfg["_file"] = str(path.resolve())
    return cfg


def resolve(cfg: dict, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else Path(cfg["_dir"]) / q


class MissingMidi(FileNotFoundError):
    pass


@dataclass
class PreparedReel:
    cfg: dict
    song: Song
    notes: list[Note]               # times already shifted by leadIn (video timeline)
    layout: Layout
    duration: float                 # seconds of video
    frames: int
    shots: list[dict]
    bar_times: list[float]
    issues: list = field(default_factory=list)


def bar_time(song: Song, bar: float) -> float:
    """Seconds (music timeline) at the start of 1-based `bar` (fractional bars allowed)."""
    downs = [t for t, is_down in song.beat_times(until=song.duration + 60) if is_down]
    b = bar - 1
    i = int(math.floor(b))
    if i < 0:
        return 0.0
    if i + 1 < len(downs):
        return downs[i] + (downs[i + 1] - downs[i]) * (b - i)
    return downs[-1] + (b - (len(downs) - 1)) * (downs[-1] - downs[-2])


def prepare(cfg: dict) -> PreparedReel:
    midi = resolve(cfg, cfg["midi"])
    if not midi.exists():
        raise MissingMidi(
            f"MIDI file not found: {midi}\n"
            f"Place the (licensed) solo-piano MIDI arrangement there, ideally with two tracks named "
            f"'Right Hand' and 'Left Hand', then run the render again.")
    song = load_midi(midi)
    issues = assign_fingering(song)
    lead = float(cfg["leadIn"])
    # optional excerpt of a longer arrangement: {"fromBar": 9, "toBar": 33} or {"from": 12.0, "to": 80.0}
    ex = cfg.get("excerpt") or {}
    t0 = bar_time(song, float(ex["fromBar"])) if "fromBar" in ex else float(ex.get("from", 0.0))
    t1 = bar_time(song, float(ex["toBar"])) if "toBar" in ex else float(ex.get("to", 1e9))
    notes = []
    for n in song.notes:
        if not (t0 - 1e-6 <= n.start < t1 - 1e-6):
            continue
        m = copy.copy(n)
        m.start += lead - t0
        m.end = min(m.end, t1 + float(ex.get("ringOut", 0.0))) + lead - t0
        notes.append(m)
    if not notes:
        raise ValueError("No notes in the selected excerpt.")
    if ex:
        from ..model import PedalEvent
        ped = [PedalEvent(e.time - t0, e.down, e.tick) for e in song.pedal if t0 <= e.time < t1]
        down_before = song.pedal_down_at(t0 - 1e-6)
        if down_before:
            ped.insert(0, PedalEvent(0.0, True))
        song.pedal = ped
    song_offset = t0
    lc = cfg["layout"]
    if lc.get("range", "auto") == "auto":
        lo, hi = fit_range([n.pitch for n in notes], int(lc.get("margin", 1)))
    else:
        lo, hi = lc["range"]
    lay = Layout.build(int(cfg["width"]), int(cfg["height"]), lo, hi, lc["footageRatio"], lc["keyRatio"], lc["bodyRatio"])
    last_off = max(n.end for n in notes)
    duration = last_off + float(cfg["tail"])
    fps = int(cfg["fps"])
    frames = int(math.ceil(duration * fps))
    duration = frames / fps
    # shots: resolve start times, end = next start
    downs = [t - song_offset + lead for t, d in song.beat_times(until=song.duration + 30) if d]
    downs = [t for t in downs if 0 <= t <= duration]
    shots = []
    specs = cfg["footage"]["shots"]
    for k, sp in enumerate(specs):
        s = dict(sp)
        if "bar" in s:          # bar number of the original score
            s["start"] = bar_time(song, float(s["bar"])) - song_offset + lead
        elif "time" in s:       # seconds on the video timeline
            s["start"] = float(s["time"])
        else:                   # spread evenly, snapped to the nearest bar line
            nominal = duration * k / max(len(specs), 1)
            s["start"] = min(downs, key=lambda d: abs(d - nominal)) if (downs and k) else nominal
        if "clip" in s:
            s["clip"] = str(resolve(cfg, s["clip"]))
        shots.append(s)
    shots.sort(key=lambda s: s["start"])
    if shots:
        shots[0]["start"] = 0.0
    for a, b in zip(shots, shots[1:]):
        a["end"] = b["start"]
    if shots:
        shots[-1]["end"] = duration
    return PreparedReel(cfg, song, notes, lay, duration, frames, shots, downs, issues)


def render_audio(p: PreparedReel, out_wav: Path, piano: Piano | None = None) -> dict:
    piano = piano or Piano()
    acfg = p.cfg["audio"]
    lead = float(p.cfg["leadIn"])
    events = [NoteEvent(n.start, n.end, n.pitch, n.velocity) for n in p.notes]
    pedal, cur = [], None
    for e in p.song.pedal:          # (already relative to the excerpt start)
        if e.down and cur is None:
            cur = e.time + lead
        elif not e.down and cur is not None:
            pedal.append(PedalSpan(cur, e.time + lead))
            cur = None
    if cur is not None:
        pedal.append(PedalSpan(cur, 1e9))
    audio = piano.render(events, pedal, p.duration + 3.0)
    audio = apply_reverb(audio, float(acfg["reverb"]), float(acfg["reverbSeconds"]))
    audio = audio[: int(round(p.duration * SR))] * float(acfg.get("volume", 1.0))
    audio, stats = normalize(audio, float(acfg["targetLufs"]))
    f = int(0.01 * SR)
    audio[:f] *= np.linspace(0, 1, f)[:, None]
    fo = int(float(acfg.get("fadeOut", 1.6)) * SR)
    audio[-fo:] *= (np.linspace(1, 0, fo) ** 2)[:, None]
    write_wav(out_wav, audio)
    stats.update({"instrument": piano.name, "sample_rate": SR, "duration": round(len(audio) / SR, 4)})
    return stats
