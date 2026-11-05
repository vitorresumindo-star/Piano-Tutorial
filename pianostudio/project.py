"""Project = MIDI file + configuration + manual corrections.

``prepare()`` runs the whole analysis chain (parse -> hand/finger -> animation
-> validation) and ``export_scene()`` writes what the renderer needs.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import geometry as G
from .animation import Bake, bake_song, bake_to_bytes, FLOATS_PER_HAND, KEY_DOWN_TIME, KEY_UP_TIME
from .audio import CREDIT, NoteEvent, PedalSpan, Piano, SR, apply_reverb, click, normalize, write_wav
from .fingering import assign_fingering
from .midi_io import load_midi
from .model import Song, LEFT, RIGHT
from .program import Program, build_program
from .validation import validate

ROOT = Path(__file__).resolve().parent.parent
THEMES_DIR = ROOT / "themes"

VIDEO_PRESETS = {
    "1080p30": {"width": 1920, "height": 1080, "fps": 30},
    "1080p60": {"width": 1920, "height": 1080, "fps": 60},
    "4k30": {"width": 3840, "height": 2160, "fps": 30},
    "4k60": {"width": 3840, "height": 2160, "fps": 60},
    "shorts": {"width": 1080, "height": 1920, "fps": 30},
    "preview": {"width": 1280, "height": 720, "fps": 30},
}

DIFFICULTY_PRESETS = {
    "beginner": {"travelTime": 4.0, "showNoteNames": True, "showFingers": True},
    "intermediate": {"travelTime": 3.0, "showNoteNames": False, "showFingers": False},
    "advanced": {"travelTime": 2.2, "showNoteNames": False, "showFingers": False},
}

DEFAULT_CONFIG = {
    "title": "Untitled",
    "composer": "",
    "kicker": "Piano Tutorial",
    "midi": "",
    "theme": "midnight_studio",
    "colors": None,                      # None -> theme colours
    "hands": {"skinTone": "light", "sleeve": True, "sleeveColor": "#101014"},
    "video": {"preset": "1080p30", **VIDEO_PRESETS["1080p30"], "crf": 18, "x264preset": "medium"},
    "notes": {"travelTime": 3.0, "glow": 1.0, "showNoteNames": False, "showFingers": False,
              "difficultyPreset": "intermediate", "sparks": 1.0},
    "structure": {"intro": True, "introSeconds": 5.0, "introText": "", "practice": False, "practiceRate": 0.6,
                  "practiceBars": 8, "practiceStart": None, "practiceEnd": None, "practiceRepeats": 1,
                  "countdown": True, "outro": True, "outroSeconds": 4.5, "progressBar": True, "showLegend": True},
    "audio": {"reverb": 0.28, "reverbSeconds": 2.1, "volume": 1.0, "metronome": False, "targetLufs": -14.0,
              "releaseResonance": True},
    "branding": {"channel": "", "tagline": "", "watermark": "", "outroLine": "Thank you for playing along"},
    "range": "auto",
    "difficulty": "auto",
    "overrides": {},
    "debug": False,
    "strict": True,
    "approved": False,
    "cinematic": False,
}


def deep_merge(base: dict, extra: dict | None) -> dict:
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def list_themes() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(THEMES_DIR.glob("*.json"))]


def load_theme(theme_id: str) -> dict:
    p = THEMES_DIR / f"{theme_id}.json"
    if not p.exists():
        p = THEMES_DIR / "midnight_studio.json"
    return json.loads(p.read_text())


def normalize_config(cfg: dict) -> dict:
    cfg = deep_merge(DEFAULT_CONFIG, cfg)
    v = cfg["video"]
    if v.get("preset") in VIDEO_PRESETS and not cfg.get("_customVideo"):
        v.update(VIDEO_PRESETS[v["preset"]])
    return cfg


@dataclass
class Prepared:
    cfg: dict
    song: Song
    bake: Bake
    report: dict
    program: Program
    fingering_issues: list


def apply_overrides(song: Song, overrides: dict) -> None:
    by_id = {n.id: n for n in song.notes}
    for k, o in (overrides or {}).items():
        n = by_id.get(int(k))
        if not n:
            continue
        if o.get("hand") in (LEFT, RIGHT):
            n.hand, n.hand_source = o["hand"], "user"
        if o.get("finger") in (1, 2, 3, 4, 5):
            n.finger, n.finger_source = int(o["finger"]), "user"


def prepare(cfg: dict) -> Prepared:
    cfg = normalize_config(cfg)
    song = load_midi(cfg["midi"])
    apply_overrides(song, cfg.get("overrides"))
    fing = assign_fingering(song)
    bake = bake_song(song)
    report = validate(song, bake, fing, strict=cfg.get("strict", True), approved=cfg.get("approved", False))
    program = build_program(song, cfg)
    return Prepared(cfg, song, bake, report, program, fing)


def estimate_difficulty(song: Song) -> str:
    if not song.notes:
        return "Beginner"
    dur = max(song.duration, 1.0)
    nps = len(song.notes) / dur
    from .midi_io import group_onsets
    groups = group_onsets(song.notes)
    max_chord = max(len(g) for g in groups)
    score = nps * 0.9 + max(0, max_chord - 2) * 0.8
    for h in (LEFT, RIGHT):
        hn = sorted(song.notes_for(h), key=lambda n: n.start)
        leaps = sum(1 for a, b in zip(hn, hn[1:]) if abs(b.pitch - a.pitch) > 9 and b.start - a.start < 0.3)
        score += leaps / dur * 6
    if score < 3.2:
        return "Beginner"
    if score < 7.0:
        return "Intermediate"
    return "Advanced"


def visible_range(song: Song, mode: str = "auto") -> tuple[int, int]:
    if mode == "full" or not song.notes:
        return G.LOWEST_MIDI, G.HIGHEST_MIDI
    lo = min(n.pitch for n in song.notes)
    hi = max(n.pitch for n in song.notes)
    lo, hi = lo - 3, hi + 3
    while hi - lo < 40:
        lo -= 1
        hi += 1
    lo = max(G.LOWEST_MIDI, lo)
    hi = min(G.HIGHEST_MIDI, hi)
    while G.is_black(lo):
        lo -= 1
    while G.is_black(hi):
        hi += 1
    return max(G.LOWEST_MIDI, lo), min(G.HIGHEST_MIDI, hi)


def scene_data(p: Prepared, preview: bool = False) -> dict:
    cfg = p.cfg
    theme = load_theme(cfg["theme"])
    colors = cfg.get("colors") or {"R": theme["notes"]["R"], "L": theme["notes"]["L"]}
    nc = cfg["notes"]
    preset = DIFFICULTY_PRESETS.get(nc.get("difficultyPreset", ""), {})
    difficulty = cfg["difficulty"] if cfg.get("difficulty") not in (None, "", "auto") else estimate_difficulty(p.song)
    lo, hi = visible_range(p.song, cfg.get("range", "auto"))
    notes = []
    for n in sorted(p.song.notes, key=lambda n: (n.start, n.pitch)):
        on, off = p.bake.press.get(n.id, (n.start, n.end))
        notes.append({"id": n.id, "p": n.pitch, "t0": round(n.start, 6), "t1": round(n.end, 6),
                      "on": round(on, 6), "off": round(off, 6), "h": n.hand or "?", "f": n.finger or 0,
                      "v": n.velocity, "fs": n.finger_source, "hs": n.hand_source})
    return {
        "version": 1,
        "geometry": G.scene_geometry(),
        "song": {"title": cfg["title"], "composer": cfg.get("composer", ""), "kicker": cfg.get("kicker", "Piano Tutorial"),
                 "difficulty": difficulty, "introText": cfg["structure"].get("introText", ""),
                 "duration": p.song.duration},
        "notes": notes,
        "pedal": [[e.time, e.down] for e in p.song.pedal],
        "bake": {"t0": p.bake.t0, "rate": p.bake.rate, "count": len(p.bake.times), "floatsPerHand": FLOATS_PER_HAND,
                 "visible": {h: bool(p.bake.hands[h].visible) for h in (RIGHT, LEFT)}},
        "range": {"lo": lo, "hi": hi},
        "program": p.program.to_dict(),
        "theme": theme,
        "constants": {"KEY_DOWN_TIME": KEY_DOWN_TIME, "KEY_UP_TIME": KEY_UP_TIME},
        "credits": CREDIT,
        "options": {
            "colors": colors,
            "hands": cfg["hands"],
            "travelTime": float(nc.get("travelTime", preset.get("travelTime", 3.0))),
            "glow": float(nc.get("glow", 1.0)),
            "showNoteNames": bool(nc.get("showNoteNames", False)),
            "showFingers": bool(nc.get("showFingers", False)),
            "sparks": float(nc.get("sparks", 1.0)),
            "progressBar": bool(cfg["structure"].get("progressBar", True)),
            "showLegend": bool(cfg["structure"].get("showLegend", True)),
            "branding": cfg.get("branding", {}),
            "debug": bool(cfg.get("debug", False)),
            "msaa": 0 if preview else 4,
            "cinematic": bool(cfg.get("cinematic", False)),
        },
    }


def export_scene(p: Prepared, out_dir: Path, preview: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = scene_data(p, preview)
    (out_dir / "perf.json").write_text(json.dumps(data))
    (out_dir / "bake.bin").write_bytes(bake_to_bytes(p.bake))
    return data


# ---------------------------------------------------------------- audio
def render_program_audio(p: Prepared, out_wav: Path, piano: Piano | None = None) -> dict:
    piano = piano or Piano()
    prog, song, acfg = p.program, p.song, p.cfg["audio"]
    total = prog.duration
    mix = np.zeros((int(math.ceil(total * SR)) + SR, 2), np.float32)
    clicks = []
    for seg in prog.segments:
        if seg.type in ("performance", "practice") and seg.audio:
            events, pedal = [], []
            for n in song.notes:
                if seg.songStart - 1e-9 <= n.start < seg.songEnd:
                    t0 = seg.video_time(n.start) - seg.start
                    t1 = seg.video_time(n.end) - seg.start
                    events.append(NoteEvent(t0, t1, n.pitch, n.velocity))
            downs = [e for e in song.pedal]
            cur = None
            for e in downs:
                if e.down and cur is None:
                    cur = e.time
                elif not e.down and cur is not None:
                    pedal.append(PedalSpan(seg.video_time(cur) - seg.start, seg.video_time(e.time) - seg.start))
                    cur = None
            if cur is not None:
                pedal.append(PedalSpan(seg.video_time(cur) - seg.start, 1e9))
            seg_len = seg.end - seg.start
            audio = piano.render(events, pedal, seg_len + 4.0, acfg.get("releaseResonance", True))
            # fade out ringing notes at the end of the segment (except the final performance tail)
            n_end = int(seg_len * SR)
            fade = int(0.6 * SR)
            audio[n_end:] = 0
            if fade < n_end:
                audio[n_end - fade:n_end] *= np.linspace(1, 0, fade)[:, None] ** 2
            i0 = int(round(seg.start * SR))
            i1 = min(len(mix), i0 + len(audio))
            mix[i0:i1] += audio[: i1 - i0]
            if acfg.get("metronome"):
                for bt, down in song.beat_times():
                    if seg.songStart <= bt < seg.songEnd:
                        clicks.append((seg.video_time(bt), down))
        elif seg.type == "countdown":
            n = max(1, seg.count or 3)
            step = (seg.end - seg.start - 0.3) / n
            for k in range(n):
                clicks.append((seg.start + k * step, k == 0))
    mix = apply_reverb(mix, float(acfg.get("reverb", 0.28)), float(acfg.get("reverbSeconds", 2.1)))
    for t, down in clicks:
        c = click(1568.0 if down else 1046.0, down)
        i0 = int(round(t * SR))
        mix[i0:i0 + len(c)] += c[: max(0, len(mix) - i0)]
    mix = mix[: int(round(total * SR))]
    mix *= float(acfg.get("volume", 1.0))
    mix, stats = normalize(mix, float(acfg.get("targetLufs", -14.0)))
    # gentle fade at the very start/end
    f = int(0.02 * SR)
    mix[:f] *= np.linspace(0, 1, f)[:, None]
    mix[-f:] *= np.linspace(1, 0, f)[:, None]
    write_wav(out_wav, mix)
    stats.update({"instrument": piano.name, "sample_rate": SR, "duration": round(len(mix) / SR, 4)})
    return stats


def metadata_draft(p: Prepared, data: dict) -> dict:
    cfg = p.cfg
    title, composer = cfg["title"], cfg.get("composer", "")
    diff = data["song"]["difficulty"]
    chapters = []
    names = {"intro": "Intro", "performance": "Full performance", "countdown": "Practice mode",
             "practice": "Slow practice", "outro": "Outro"}
    seen = set()
    for s in p.program.segments:
        nm = names[s.type]
        if nm in seen:
            continue
        seen.add(nm)
        m, sec = divmod(int(s.start), 60)
        chapters.append(f"{m}:{sec:02d} {nm}")
    if chapters and not chapters[0].startswith("0:00"):
        chapters.insert(0, "0:00 Start")
    channel = cfg.get("branding", {}).get("channel", "")
    desc = [f"Learn to play {title}{' by ' + composer if composer else ''} on piano — {diff.lower()} level tutorial "
            f"with falling notes and animated hands.", "",
            "Right hand and left hand are shown in different colours. "
            + ("Finger numbers: 1 = thumb … 5 = little finger." if cfg["notes"].get("showFingers") or cfg["structure"].get("practice") else ""),
            "", "Chapters:"] + chapters + ["",
            "Fingering is a computer-generated suggestion reviewed for playability; adapt it to your hand.",
            CREDIT + ".", ""]
    if channel:
        desc.append(f"Subscribe to {channel} for more tutorials.")
    return {
        "title": f"{title}{' – ' + composer if composer else ''} | Piano Tutorial ({diff})"[:100],
        "description": "\n".join(desc),
        "tags": [t for t in ["piano tutorial", "piano", title, composer, f"{diff.lower()} piano", "falling notes",
                             "synthesia style", "how to play", "piano lesson", "sheet music alternative"] if t],
        "chapters": chapters,
        "category": "Music",
        "madeForKids": False,
        "license_notes": [CREDIT, "Make sure you have the rights to the composition/arrangement you publish."],
    }
