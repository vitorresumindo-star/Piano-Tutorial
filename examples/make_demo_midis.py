"""Generate the demonstration MIDI files shipped with the project.

* minuet_in_g.mid   -- Minuet in G major, BWV Anh. 114 (attributed to Christian
                       Petzold, c. 1725; public domain). Simple two-hand
                       arrangement made for this project.
* morning_light.mid -- An original étude written for this project (free to use),
                       with chords, black keys, sustain pedal and arpeggios.
* single_track_test.mid -- Same étude merged into one track (no hand info) to
                       exercise the automatic hand-splitting heuristic.
"""
from pathlib import Path

import mido

OUT = Path(__file__).resolve().parent / "midi"
N = {n: i for i, n in enumerate(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])}
N.update({"Db": 1, "Eb": 3, "Gb": 6, "Ab": 8, "Bb": 10})


def p(name: str) -> int:
    pitch, octave = name[:-1], int(name[-1])
    return 12 * (octave + 1) + N[pitch]


def track_from(events, name, tpb, channel=0):
    """events: list of (start_beat, dur_beats, [pitches], velocity)"""
    msgs = []
    for start, dur, pitches, vel in events:
        for pt in pitches:
            msgs.append((round(start * tpb), 1, mido.Message("note_on", note=p(pt) if isinstance(pt, str) else pt,
                                                              velocity=vel, channel=channel)))
            msgs.append((round((start + dur) * tpb) - 2, 0, mido.Message("note_off", note=p(pt) if isinstance(pt, str) else pt,
                                                                          velocity=0, channel=channel)))
    return msgs


def write(path, tracks, tpb, tempo_bpm, ts=(4, 4), pedal=None, title=""):
    mid = mido.MidiFile(type=1, ticks_per_beat=tpb)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("track_name", name=title, time=0))
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0))
    meta.append(mido.MetaMessage("time_signature", numerator=ts[0], denominator=ts[1], time=0))
    mid.tracks.append(meta)
    for name, msgs, extra in tracks:
        tr = mido.MidiTrack()
        tr.append(mido.MetaMessage("track_name", name=name, time=0))
        tr.append(mido.Message("program_change", program=0, time=0))
        allm = list(msgs) + list(extra or [])
        allm.sort(key=lambda m: (m[0], m[1]))
        last = 0
        for tick, _prio, msg in allm:
            tr.append(msg.copy(time=tick - last))
            last = tick
        mid.tracks.append(tr)
    path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(path))
    return path


def minuet():
    tpb = 480
    q, e, h, dh = 1.0, 0.5, 2.0, 3.0
    rh_bars = [
        [("D5", q), ("G4", e), ("A4", e), ("B4", e), ("C5", e)],
        [("D5", q), ("G4", q), ("G4", q)],
        [("E5", q), ("C5", e), ("D5", e), ("E5", e), ("F#5", e)],
        [("G5", q), ("G4", q), ("G4", q)],
        [("C5", q), ("D5", e), ("C5", e), ("B4", e), ("A4", e)],
        [("B4", q), ("C5", e), ("B4", e), ("A4", e), ("G4", e)],
        [("F#4", q), ("G4", e), ("A4", e), ("B4", e), ("G4", e)],
        [("A4", dh)],
        [("D5", q), ("G4", e), ("A4", e), ("B4", e), ("C5", e)],
        [("D5", q), ("G4", q), ("G4", q)],
        [("E5", q), ("C5", e), ("D5", e), ("E5", e), ("F#5", e)],
        [("G5", q), ("G4", q), ("G4", q)],
        [("C5", q), ("D5", e), ("C5", e), ("B4", e), ("A4", e)],
        [("B4", q), ("C5", e), ("B4", e), ("A4", e), ("G4", e)],
        [("A4", q), ("B4", e), ("A4", e), ("G4", e), ("F#4", e)],
        [("G4", dh)],
    ]
    lh_bars = [
        [(["G3", "B3", "D4"], h), (["A3"], q)],
        [(["B3"], dh)],
        [(["C4"], dh)],
        [(["B3"], dh)],
        [(["A3"], dh)],
        [(["G3"], dh)],
        [(["D4"], h), (["B3"], q)],
        [(["D4"], q), (["D3"], q), (["C4"], q)],
        [(["B3", "D4"], h), (["A3"], q)],
        [(["B3"], dh)],
        [(["C4"], dh)],
        [(["B3"], dh)],
        [(["A3"], dh)],
        [(["G3"], q), (["A3"], q), (["B3"], q)],
        [(["C4"], q), (["D4"], q), (["D3"], q)],
        [(["G2", "G3"], dh)],
    ]
    rh, lh = [], []
    beat = 0.0
    for i, bar in enumerate(rh_bars):
        b = i * 3.0
        for pitch, d in bar:
            accent = 78 if abs(b % 3.0) < 1e-9 else 66
            rh.append((b, d, [pitch], accent))
            b += d
    for i, bar in enumerate(lh_bars):
        b = i * 3.0
        for pitches, d in bar:
            lh.append((b, d, pitches, 56))
            b += d
    return write(OUT / "minuet_in_g.mid",
                 [("Right Hand", track_from(rh, "RH", tpb), None), ("Left Hand", track_from(lh, "LH", tpb), None)],
                 tpb, 116, (3, 4), title="Minuet in G major")


def morning_light(single_track=False):
    tpb = 480
    # chord progression (LH arpeggio patterns, 8th notes, 4/4)
    prog = [
        ["C3", "G3", "C4", "E4", "G4", "E4", "C4", "G3"],
        ["A2", "E3", "A3", "C4", "E4", "C4", "A3", "E3"],
        ["F2", "C3", "F3", "Ab3", "C4", "Ab3", "F3", "C3"],
        ["G2", "D3", "G3", "B3", "F4", "B3", "G3", "D3"],
        ["E2", "C3", "G3", "C4", "E4", "C4", "G3", "C3"],
        ["F2", "C3", "F3", "A3", "C4", "A3", "F3", "C3"],
        ["D3", "A3", "C4", "F4", "A4", "F4", "C4", "A3"],
        ["G2", "D3", "G3", "B3", "D4", "B3", "G3", "D3"],
    ]
    melody = [  # (beat offset within bar, dur, pitches, vel)
        [(0, 1.5, ["E5"], 74), (1.5, 0.5, ["D5"], 64), (2, 1, ["C5"], 66), (3, 1, ["G5"], 72)],
        [(0, 2, ["E5", "C5"], 70), (2, 1, ["D5"], 62), (3, 1, ["E5"], 66)],
        [(0, 1.5, ["F5", "C5"], 72), (1.5, 0.5, ["Eb5"], 62), (2, 1, ["F5"], 66), (3, 1, ["Ab5"], 74)],
        [(0, 3, ["G5", "D5", "B4"], 70), (3, 1, ["F5"], 60)],
        [(0, 1, ["E5"], 70), (1, 1, ["G5"], 68), (2, 1, ["C6"], 76), (3, 1, ["B5"], 66)],
        [(0, 1, ["A5", "F5"], 70), (1, 1, ["G5"], 62), (2, 1, ["F5"], 62), (3, 1, ["E5"], 60)],
        [(0, 1.5, ["D5", "A4"], 66), (1.5, 0.5, ["E5"], 60), (2, 1, ["F5"], 64), (3, 1, ["A5"], 68)],
        [(0, 2, ["G5", "D5", "B4"], 72), (2, 2, ["G5", "F5", "B4"], 64)],
    ]
    lh, rh, pedal = [], [], []
    for bar, pat in enumerate(prog):
        for i, pt in enumerate(pat):
            lh.append((bar * 4 + i * 0.5, 0.48, [pt], 52 + (8 if i == 0 else 0)))
        for off, d, pts, v in melody[bar]:
            rh.append((bar * 4 + off, d, pts, v))
        pedal.append((round((bar * 4) * tpb) + 10, 2, mido.Message("control_change", control=64, value=100)))
        pedal.append((round((bar * 4 + 4) * tpb) - 12, 0, mido.Message("control_change", control=64, value=0)))
    final = 8 * 4
    lh.append((final, 4, ["C2", "C3"], 60))
    rh.append((final, 4, ["C5", "E5", "G5", "C6"], 70))
    pedal.append((round(final * tpb) + 10, 2, mido.Message("control_change", control=64, value=100)))
    pedal.append((round((final + 4) * tpb), 0, mido.Message("control_change", control=64, value=0)))
    if single_track:
        msgs = track_from(lh + rh, "Piano", tpb)
        return write(OUT / "single_track_test.mid", [("Piano", msgs, pedal)], tpb, 76,
                     title="Morning Light (single track)")
    return write(OUT / "morning_light.mid",
                 [("Right Hand", track_from(rh, "RH", tpb), pedal), ("Left Hand", track_from(lh, "LH", tpb), None)],
                 tpb, 76, title="Morning Light")


if __name__ == "__main__":
    for f in (minuet(), morning_light(), morning_light(single_track=True)):
        print("wrote", f)
