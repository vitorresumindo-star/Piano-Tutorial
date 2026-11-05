"""Automated QA for Piano Studio: MIDI parsing, timing, key mapping, fingering,
key depression, kinematic contact, collisions, audio/video sync, export."""
from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import mido
import numpy as np
import pytest

from pianostudio import geometry as G
from pianostudio.animation import (KEY_DOWN_TIME, KEY_UP_TIME, bake_song, fk_fingertips, key_depth_profile)
from pianostudio.audio import SR, NoteEvent, Piano, integrated_loudness, normalize
from pianostudio.fingering import assign_fingering
from pianostudio.midi_io import load_midi
from pianostudio.program import build_program, frame_count, frame_time
from pianostudio.project import normalize_config, prepare, render_program_audio
from pianostudio.validation import validate

ROOT = Path(__file__).resolve().parent.parent
MIDI = ROOT / "examples" / "midi"


# ------------------------------------------------------------------ helpers
def make_midi(path: Path, tracks, tempo_changes=((0, 120),), tpb=480, ts=(4, 4), pedal=()):
    """tracks: list of (name, [(start_beat, dur_beats, pitch, vel)])"""
    mid = mido.MidiFile(type=1, ticks_per_beat=tpb)
    meta = mido.MidiTrack()
    last = 0
    meta.append(mido.MetaMessage("time_signature", numerator=ts[0], denominator=ts[1], time=0))
    for beat, bpm in tempo_changes:
        t = int(beat * tpb)
        meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=t - last))
        last = t
    mid.tracks.append(meta)
    for name, notes in tracks:
        tr = mido.MidiTrack()
        tr.append(mido.MetaMessage("track_name", name=name, time=0))
        ev = []
        for s, d, p, v in notes:
            ev.append((int(s * tpb), 1, mido.Message("note_on", note=p, velocity=v)))
            ev.append((int((s + d) * tpb), 0, mido.Message("note_off", note=p, velocity=0)))
        for beat, down in pedal:
            ev.append((int(beat * tpb), 2, mido.Message("control_change", control=64, value=127 if down else 0)))
        ev.sort(key=lambda e: (e[0], e[1]))
        last = 0
        for t, _, m in ev:
            tr.append(m.copy(time=t - last))
            last = t
        mid.tracks.append(tr)
    mid.save(str(path))
    return path


@pytest.fixture(scope="module")
def minuet():
    s = load_midi(MIDI / "minuet_in_g.mid")
    issues = assign_fingering(s)
    return s, issues, bake_song(s)


# ------------------------------------------------------------------ MIDI parsing
def test_parse_notes_pitches_velocities(tmp_path):
    f = make_midi(tmp_path / "a.mid", [("Right Hand", [(0, 1, 60, 80), (1, 1, 64, 90), (2, 2, 67, 100)])])
    s = load_midi(f)
    assert [n.pitch for n in s.notes] == [60, 64, 67]
    assert [n.velocity for n in s.notes] == [80, 90, 100]
    assert s.notes[0].start == pytest.approx(0.0)
    assert s.notes[1].start == pytest.approx(0.5)        # 120 bpm -> 0.5 s per beat
    assert s.notes[2].end == pytest.approx(2.0)


def test_tempo_changes_exact_no_drift(tmp_path):
    # 4 beats @ 60 bpm (4 s), then 120 bpm: beat 4 = 4.0 s, beat 1004 = 4 + 500 s
    notes = [(b, 0.5, 60 + (b % 12), 64) for b in range(0, 1008, 4)]
    f = make_midi(tmp_path / "t.mid", [("Piano", notes)], tempo_changes=((0, 60), (4, 120)))
    s = load_midi(f)
    by_start = {round(n.start_tick / 480): n for n in s.notes}
    assert by_start[0].start == pytest.approx(0.0, abs=1e-12)
    assert by_start[4].start == pytest.approx(4.0, abs=1e-12)
    assert by_start[1004].start == pytest.approx(4.0 + 1000 * 0.5, abs=1e-9)
    assert len(s.tempos) == 2 and s.tempos[1].time == pytest.approx(4.0)


def test_time_signature_and_beats(tmp_path):
    f = make_midi(tmp_path / "ts.mid", [("Piano", [(0, 6, 60, 64)])], ts=(3, 4))
    s = load_midi(f)
    assert (s.time_signatures[0].numerator, s.time_signatures[0].denominator) == (3, 4)
    beats = s.beat_times()
    assert [db for _, db in beats[:7]] == [True, False, False, True, False, False, True]


def test_sustain_pedal_events(tmp_path):
    f = make_midi(tmp_path / "p.mid", [("Piano", [(0, 1, 60, 64), (2, 1, 62, 64)])], pedal=[(0.1, True), (1.9, False)])
    s = load_midi(f)
    assert [(round(e.time, 3), e.down) for e in s.pedal] == [(0.05, True), (0.95, False)]
    assert s.pedal_down_at(0.5) and not s.pedal_down_at(1.2)


def test_hands_from_track_names(tmp_path):
    f = make_midi(tmp_path / "h.mid", [("Left Hand", [(0, 1, 48, 64)]), ("Right Hand", [(0, 1, 72, 64)])])
    s = load_midi(f)
    hands = {n.pitch: (n.hand, n.hand_source) for n in s.notes}
    assert hands[48] == ("L", "midi-track") and hands[72] == ("R", "midi-track")


def test_hand_split_heuristic_flags_review():
    s = load_midi(MIDI / "single_track_test.mid")
    assert all(n.hand in ("L", "R") and n.hand_source == "heuristic" for n in s.notes)
    assert any(e["code"] == "hands-heuristic" for e in s.parse_log)
    low = [n for n in s.notes if n.pitch < 48]
    assert all(n.hand == "L" for n in low)


def test_chords_grouped(minuet):
    s, _, _ = minuet
    first = [n for n in s.notes if n.start < 0.01]
    assert sorted(n.pitch for n in first) == [55, 59, 62, 74]   # G3 B3 D4 chord + D5


def test_overlapping_same_pitch_is_resolved(tmp_path):
    mid = mido.MidiFile(ticks_per_beat=480)
    tr = mido.MidiTrack()
    tr += [mido.Message("note_on", note=60, velocity=70, time=0), mido.Message("note_on", note=60, velocity=70, time=240),
           mido.Message("note_off", note=60, velocity=0, time=240), mido.Message("note_off", note=60, velocity=0, time=240)]
    mid.tracks.append(tr)
    mid.save(str(tmp_path / "o.mid"))
    s = load_midi(tmp_path / "o.mid")
    assert len(s.notes) == 2
    assert s.notes[0].end <= s.notes[1].start
    assert any(e["code"] == "overlap-retrigger" for e in s.parse_log)


# ------------------------------------------------------------------ key mapping
def test_88_key_layout():
    whites = [m for m in range(21, 109) if not G.is_black(m)]
    assert len(whites) == 52 and len(range(21, 109)) == 88
    xs = [G.key_x(m) for m in range(21, 109)]
    assert all(b > a for a, b in zip(xs, xs[1:]))                 # strictly increasing
    for m in range(21, 109):
        k = G.key(m)
        if k.black:   # black key sits between its white neighbours
            assert G.key_x(m - 1) < k.x < G.key_x(m + 1)
    assert G.note_name(60) == "C4" and G.note_name(61) == "C#4" and G.note_name(21) == "A0"


def test_out_of_range_rejected():
    with pytest.raises(ValueError):
        G.key(20)
    with pytest.raises(ValueError):
        G.key(109)


# ------------------------------------------------------------------ fingering
def test_every_note_gets_hand_and_finger(minuet):
    s, issues, _ = minuet
    assert all(n.finger in (1, 2, 3, 4, 5) for n in s.notes)
    assert not [i for i in issues if i.severity == "error"]


def test_chord_fingers_monotonic(minuet):
    s, _, _ = minuet
    from pianostudio.midi_io import group_onsets
    for hand in ("L", "R"):
        for g in group_onsets(s.notes_for(hand)):
            g = sorted(g, key=lambda n: n.pitch)
            f = [n.finger for n in g]
            assert f == (sorted(f) if hand == "R" else sorted(f, reverse=True))
            assert len(set(f)) == len(f)


def test_user_override_is_respected(tmp_path):
    from pianostudio.project import apply_overrides
    s = load_midi(MIDI / "minuet_in_g.mid")
    target = s.notes_for("R")[1]
    apply_overrides(s, {str(target.id): {"finger": 3}})
    assign_fingering(s)
    assert target.finger == 3 and target.finger_source == "user"


def test_impossible_stretch_is_flagged(tmp_path):
    f = make_midi(tmp_path / "x.mid", [("Right Hand", [(0, 1, 60, 64), (0, 1, 79, 64)]),   # C4 + G5 (12th) in RH
                                       ("Left Hand", [(0, 1, 36, 64)])])
    s = load_midi(f)
    issues = assign_fingering(s)
    assert any(i.code == "unplayable-stretch" and i.severity == "error" for i in issues)


def test_too_many_notes_one_hand(tmp_path):
    f = make_midi(tmp_path / "six.mid", [("Right Hand", [(0, 1, p, 64) for p in (60, 62, 64, 65, 67, 69)]),
                                         ("Left Hand", [(0, 1, 36, 64)])])
    s = load_midi(f)
    issues = assign_fingering(s)
    assert any(i.code == "too-many-notes" for i in issues)


# ------------------------------------------------------------------ key depression
def test_key_depression_profile():
    t = np.array([0.9, 1.0 - KEY_DOWN_TIME, 1.0 - KEY_DOWN_TIME / 2, 1.0, 1.5, 2.0, 2.0 + KEY_UP_TIME / 2, 2.0 + KEY_UP_TIME, 2.2])
    d = key_depth_profile(t, 1.0, 2.0, 1.0)
    assert d[0] == 0 and d[1] == 0
    assert d[2] == pytest.approx(0.5)
    assert d[3] == pytest.approx(1.0) and d[4] == pytest.approx(1.0) and d[5] == pytest.approx(1.0)
    assert d[6] == pytest.approx(0.5)
    assert d[7] == pytest.approx(0.0) and d[8] == 0


def test_key_is_fully_down_exactly_at_onset(minuet):
    s, _, b = minuet
    for n in s.notes:
        on, off = b.press[n.id]
        assert on == pytest.approx(n.start)          # sound and key bottom coincide
        assert off > on


# ------------------------------------------------------------------ kinematics
def test_fingertips_touch_intended_keys(minuet):
    s, _, b = minuet
    for hand in ("L", "R"):
        hb = b.hands[hand]
        tips = fk_fingertips(hb.pose.astype(np.float64), hand)
        for n in s.notes_for(hand):
            on, off = b.press[n.id]
            i = int(round(((on + off) / 2 - b.t0) * b.rate))
            tip = tips[i, n.finger - 1]
            k = G.key(n.pitch)
            assert abs(tip[0] - k.x) <= k.width / 2, f"{G.note_name(n.pitch)} finger {n.finger} misses key in x"
            assert 0 <= tip[2] <= k.length
            skin = tip[1] - G.FINGERS[n.finger].tip_pad
            assert skin == pytest.approx(k.top - G.key_depth_at_contact(n.pitch), abs=0.3)


def test_fk_matches_targets_while_pressing(minuet):
    _, _, b = minuet
    for hand in ("L", "R"):
        hb = b.hands[hand]
        tips = fk_fingertips(hb.pose.astype(np.float64), hand)
        err = np.linalg.norm(tips - hb.targets, axis=2)
        assert err[hb.pressing >= 0].max() < 0.3


def test_joint_angles_within_anatomical_limits(minuet):
    from pianostudio.animation import FLEX1_LIMIT, FLEX2_LIMIT, YAW_LIMIT
    _, _, b = minuet
    for hand in ("L", "R"):
        P = b.hands[hand].pose
        for i, f in enumerate(range(1, 6)):
            a, t1, t2 = P[:, 7 + 4 * i], P[:, 8 + 4 * i], P[:, 9 + 4 * i]
            assert a.min() >= YAW_LIMIT[f][0] - 1e-5 and a.max() <= YAW_LIMIT[f][1] + 1e-5
            assert t1.min() >= FLEX1_LIMIT[f][0] - 1e-5 and t1.max() <= FLEX1_LIMIT[f][1] + 1e-5
            assert t2.min() >= FLEX2_LIMIT[0] - 1e-5 and t2.max() <= FLEX2_LIMIT[1] + 1e-5


def test_no_idle_finger_sinks_into_keys(minuet):
    _, _, b = minuet
    for hand in ("L", "R"):
        hb = b.hands[hand]
        tips = fk_fingertips(hb.pose.astype(np.float64), hand)[::4]
        pressing = hb.pressing[::4]
        for f in range(5):
            idle = pressing[:, f] < 0
            r = G.FINGERS[f + 1].tip_pad
            skin = tips[idle, f, 1] - r
            # never below the deepest key travel
            assert skin.min() > -G.KEY_DEPTH_WHITE - 0.35


def test_motion_is_smooth(minuet):
    _, _, b = minuet
    for hand in ("L", "R"):
        P = b.hands[hand].pose[:, :3]
        v = np.linalg.norm(np.diff(P, axis=0), axis=1) * b.rate      # cm/s
        assert v.max() < 400, "wrist teleports"


def test_validation_passes_for_demo(minuet):
    s, issues, b = minuet
    rep = validate(s, b, issues)
    assert rep["status"] == "pass", [i for i in rep["issues"] if i["severity"] != "info"]
    for k in ("finger_contact", "correct_key", "no_floating", "finger_conflicts", "key_mapping", "timing"):
        assert rep["checks"][k]["passed"], k


def test_validation_blocks_unplayable(tmp_path):
    f = make_midi(tmp_path / "x.mid", [("Right Hand", [(0, 1, 60, 64), (0, 1, 79, 64)]), ("Left Hand", [(0, 1, 36, 64)])])
    p = prepare({"midi": str(f), "title": "x"})
    assert p.report["status"] == "blocked"


# ------------------------------------------------------------------ program / sync
def test_frame_times_are_exact():
    assert frame_time(3, 30) == pytest.approx(0.1)
    assert frame_time(90000, 30) == 3000.0                 # no float accumulation after 50 min
    assert frame_count(66.8, 30) == 2004


def test_program_mapping_consistent(minuet):
    s, _, _ = minuet
    cfg = normalize_config({"midi": str(MIDI / "minuet_in_g.mid"), "structure": {"practice": True}})
    prog = build_program(s, cfg)
    types = [x.type for x in prog.segments]
    assert types == ["intro", "performance", "countdown", "practice", "outro"]
    for a, b in zip(prog.segments, prog.segments[1:]):
        assert a.end == pytest.approx(b.start)
    perf = prog.segments[1]
    n = s.notes[0]
    vt = perf.video_time(n.start)
    # the first note is heard after the falling-note travel time + lead-in
    assert vt - perf.start == pytest.approx(cfg["notes"]["travelTime"] + 1.0)
    prac = prog.segments[3]
    assert prac.rate == pytest.approx(0.6)
    assert prog.duration * prog.fps == pytest.approx(round(prog.duration * prog.fps))


def test_audio_onsets_align_with_note_times(tmp_path):
    """Synthesise the program audio and check that energy rises exactly where
    the video timeline says each note starts (within one video frame)."""
    f = make_midi(tmp_path / "s.mid", [("Right Hand", [(b, 0.5, 72, 100) for b in range(0, 8, 2)]),
                                       ("Left Hand", [(1 + b, 0.5, 48, 100) for b in range(0, 8, 2)])])
    cfg = {"midi": str(f), "title": "sync", "structure": {"intro": True, "introSeconds": 2.0, "outro": False},
           "audio": {"reverb": 0.0}}
    p = prepare(cfg)
    wav = tmp_path / "a.wav"
    render_program_audio(p, wav)
    from scipy.io import wavfile
    sr, x = wavfile.read(wav)
    x = x.astype(np.float32).mean(axis=1)
    env = np.convolve(np.abs(x), np.ones(64) / 64, mode="same")
    perf = next(sg for sg in p.program.segments if sg.type == "performance")
    frame = 1 / p.program.fps
    for n in p.song.notes:
        t = perf.video_time(n.start)
        i = int(t * sr)
        before = env[max(0, i - int(0.03 * sr)):max(1, i - int(0.005 * sr))].mean()
        after = env[i:i + int(0.02 * sr)].max()
        assert after > 4 * before + 1e-3, f"no onset at {t:.3f}s"
        # find actual onset: first sample exceeding 50% of local peak
        seg = env[i - int(frame * sr): i + int(frame * sr)]
        on = np.argmax(seg > 0.5 * seg.max()) - int(frame * sr)
        assert abs(on) / sr <= frame


def test_sustain_pedal_extends_notes():
    piano = Piano()
    ev = [NoteEvent(0.0, 0.2, 60, 90)]
    from pianostudio.audio import PedalSpan
    dry = piano.render(ev, [], 3.0)
    ped = piano.render(ev, [PedalSpan(0.0, 2.0)], 3.0)
    def rms(a, t0, t1):
        return float(np.sqrt((a[int(t0 * SR):int(t1 * SR)] ** 2).mean()))
    assert rms(ped, 1.0, 1.5) > 5 * rms(dry, 1.0, 1.5)


def test_velocity_changes_loudness():
    piano = Piano()
    soft = piano.render([NoteEvent(0, 1, 60, 30)], [], 1.5)
    loud = piano.render([NoteEvent(0, 1, 60, 110)], [], 1.5)
    assert np.abs(loud).max() > 2 * np.abs(soft).max()


def test_loudness_normalisation():
    t = np.arange(SR * 5) / SR
    x = np.stack([0.05 * np.sin(2 * math.pi * 440 * t)] * 2, -1).astype(np.float32)
    y, st = normalize(x, -14.0)
    assert integrated_loudness(y) == pytest.approx(-14.0, abs=0.6)
    assert st["true_peak_db"] <= -0.9


# ------------------------------------------------------------------ export integrity (needs node + chromium)
@pytest.mark.skipif(not shutil.which("node") or not shutil.which("ffmpeg"), reason="renderer toolchain missing")
def test_short_render_and_integrity(tmp_path):
    from pianostudio.export import produce
    cfg = {"midi": str(MIDI / "minuet_in_g.mid"), "title": "Integrity Test",
           "video": {"preset": "preview", "x264preset": "ultrafast"},
           "structure": {"intro": False, "outro": False}}
    res = produce(cfg, tmp_path, frames=(120, 132), thumbnail=False, workers=1)
    assert res["rendered"]
    integ = res["integrity"]
    assert integ["passed"], integ
    assert integ["frames"] == 12
