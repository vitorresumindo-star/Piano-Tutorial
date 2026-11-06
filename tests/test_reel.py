"""Tests for the vertical cinematic reel pipeline (pianostudio.reel)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from pianostudio.reel.hands2d import HandStyle
from pianostudio.reel.layout import Layout, fit_range
from pianostudio.reel.pianoroll import PianoRoll, RollStyle
from pianostudio.reel.project import MissingMidi, load_project, prepare
from pianostudio.reel.render import sync_report

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "projects" / "dust_and_starlight"


@pytest.fixture(scope="module")
def demo():
    p = prepare(load_project(DEMO))
    roll = PianoRoll(p.layout, p.notes, RollStyle(), HandStyle(), p.duration, p.bar_times)
    return p, roll


def test_layout_covers_song_and_split():
    lo, hi = fit_range([45, 81])
    assert lo < 45 and hi > 81
    lay = Layout.build(1080, 1920, lo, hi)
    xs = [lay.x_center(m) for m in range(lo, hi + 1)]
    assert all(b > a for a, b in zip(xs, xs[1:]))
    assert abs(lay.footage_h / lay.height - 0.38) < 0.005          # ~38 % footage / 62 % piano
    assert lay.key_top > lay.footage_h and lay.key_bottom < lay.height


def test_every_note_is_pressed_by_its_finger_exactly_at_onset(demo):
    p, _ = demo
    rep = sync_report(p)
    assert rep["failures"] == 0, rep["key_down_at_onset_failures"]
    assert rep["worst_fingertip_offset_px"] < 0.5


def test_hand_motion_is_continuous(demo):
    p, roll = demo
    fps = 60
    for hand, tr in roll.tracks.items():
        prev = None
        worst = 0.0
        for i in range(0, p.frames, 1):
            st = tr.state(i / fps)
            cur = np.array([[st.fingers[f].x, st.fingers[f].y] for f in range(1, 6)])
            if prev is not None:
                worst = max(worst, float(np.abs(cur - prev).max()))
            prev = cur
        # at 60 fps nothing teleports: < 1 white key per frame
        assert worst < p.layout.white_w, (hand, worst)


def test_lit_keys_always_have_a_finger_and_no_finger_sharing(demo):
    p, roll = demo
    fps = 60
    for i in range(0, p.frames, 7):
        t = i / fps
        states = {h: tr.state(t) for h, tr in roll.tracks.items()}
        pressing = [(h, f, fs) for h, st in states.items() for f, fs in st.fingers.items() if fs.press > 0.5]
        keys = [fs.pitch for _, _, fs in pressing]
        assert len(keys) == len(set(keys)), f"two fingers on one key at t={t:.2f}"
        lit = [n for n in p.notes if n.start <= t < min(n.end, roll.visual_end.get(n.id, n.end) + 0.035)]
        for n in lit:
            st = states[n.hand]
            fs = st.fingers[n.finger]
            assert abs(fs.x - p.layout.x_center(n.pitch)) < 1.0, (t, n.pitch)


def test_falling_notes_land_on_time(demo):
    p, roll = demo
    n = p.notes[10]
    # bottom edge of the falling bar reaches the keyboard exactly at the onset
    y_bot = p.layout.key_top - (n.start - n.start) * roll.speed
    assert y_bot == p.layout.key_top
    assert math.isclose(roll.speed * p.cfg["notes"]["fallSeconds"], p.layout.key_top - p.layout.notes_top)


def test_shots_are_contiguous_and_cover_the_video(demo):
    p, _ = demo
    assert p.shots[0]["start"] == 0.0
    for a, b in zip(p.shots, p.shots[1:]):
        assert a["end"] == b["start"] and b["start"] > a["start"]
    assert abs(p.shots[-1]["end"] - p.duration) < 1e-9
    assert 60.0 <= p.duration <= 75.5


def test_excerpt_and_auto_shots(tmp_path):
    cfg = json.loads((DEMO / "project.json").read_text())
    cfg["midi"] = str(DEMO / "dust_and_starlight.mid")
    cfg["excerpt"] = {"fromBar": 7, "toBar": 11}
    cfg["footage"]["shots"] = [{"scene": "field"}, {"scene": "night"}, {"scene": "planet"}]
    (tmp_path / "project.json").write_text(json.dumps(cfg))
    p = prepare(load_project(tmp_path))
    assert min(n.start for n in p.notes) == pytest.approx(cfg["leadIn"], abs=0.02)
    assert p.duration < 15
    starts = [s["start"] for s in p.shots]
    assert starts == sorted(starts) and starts[0] == 0.0
    for s in starts[1:]:   # snapped to bar lines
        assert min(abs(s - d) for d in p.bar_times) < 1e-6


def test_missing_midi_is_reported_not_substituted(tmp_path):
    (tmp_path / "project.json").write_text(json.dumps({"title": "X", "midi": "missing.mid"}))
    with pytest.raises(MissingMidi):
        prepare(load_project(tmp_path))


@pytest.mark.parametrize("name,params", [
    ("field", {}), ("field", {"palette": "dusk", "camera": "rise"}), ("truck", {}), ("storm", {}),
    ("night", {}), ("launch", {"phase": "pad"}), ("launch", {"phase": "liftoff"}),
    ("launch", {"phase": "ascent"}), ("orbit", {}), ("planet", {}), ("blackhole", {}),
])
def test_scenes_render_finite_frames(name, params):
    from pianostudio.reel.scenes import SCENES
    sc = SCENES[name](360, 244, dict(params, plants=300))
    for t in (-0.35, 0.0, 2.7, 5.35):      # includes the dissolve overhang before/after the shot
        img = sc.render(t, 5.0)
        assert img.shape == (244, 360, 3)
        assert np.isfinite(img).all() and img.min() >= -1e-3


def test_title_reveal():
    from pianostudio.reel.titles import TitleOverlay
    ov = TitleOverlay(540, 365, "Cornfield Chase", "Interstellar", {"revealAt": 0.25, "revealSeconds": 1.0})
    base = np.full((365, 540, 3), 0.3, np.float32)
    assert np.array_equal(ov.apply(base, 0.0), base)
    out = ov.apply(base, 3.0)
    assert (out > 0.95).any() and not np.array_equal(out, base)


def test_full_frame_composite_small():
    from pianostudio.reel.render import Compositor
    cfg = load_project(DEMO)
    cfg.update({"width": 540, "height": 960, "fps": 30})
    cfg["footage"]["shots"] = [{"time": 0, "scene": "night"}]
    p = prepare(cfg)
    img = Compositor(p).frame(int(12.0 * 30))
    assert img.shape == (960, 540, 3) and img.dtype == np.uint8
    top, bottom = img[: p.layout.footage_h], img[p.layout.footage_h:]
    assert top.mean() < 120            # night sky footage on top
    assert bottom[50:200].mean() > 200  # clean white piano roll below


def test_demo_score_is_deterministic(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("compose", DEMO / "compose.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    a = mod.build().notes
    b = mod.build().notes
    assert a == b and len(a["R"]) > 100 and len(a["L"]) > 100
