""""Dust & Starlight" -- an original solo-piano piece written for this project.

Composed programmatically (not transcribed from any recording). It is the
stand-in score for the demo reel: a quiet, hypnotic right-hand ostinato, a
descending "lament" bass, a lyrical melody, a chordal climax and a return to
the opening figure.

    D minor, 4/4, quarter = 88 (ritardando in the last bars), ~70 seconds.

Run:  python3 projects/dust_and_starlight/compose.py
Writes dust_and_starlight.mid next to this file (track "Right Hand" and
track "Left Hand", sustain pedal on CC64).
"""
from __future__ import annotations

import random
from pathlib import Path

import mido

TPB = 480
EIGHTH = TPB // 2
BAR = TPB * 4
OUT = Path(__file__).with_name("dust_and_starlight.mid")

_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def p(name: str) -> int:
    """'A4' -> 69, 'Bb2' -> 46, 'C#5' -> 73."""
    letter, rest = name[0], name[1:]
    acc = 0
    while rest and rest[0] in "#b":
        acc += 1 if rest[0] == "#" else -1
        rest = rest[1:]
    return 12 * (int(rest) + 1) + _PC[letter] + acc


class Score:
    def __init__(self):
        self.notes = {"R": [], "L": []}   # (start_tick, dur_tick, pitch, velocity)
        self.pedal = []                    # (tick, value)
        self.rng = random.Random(2014)

    def add(self, hand, start, dur, pitches, vel, roll=0):
        if isinstance(pitches, (str, int)):
            pitches = [pitches]
        for i, pc in enumerate(pitches):
            pitch = p(pc) if isinstance(pc, str) else pc
            v = max(1, min(127, int(round(vel + self.rng.uniform(-3, 3)))))
            self.notes[hand].append((start + i * roll, dur - i * roll, pitch, v))

    def seq(self, hand, bar, items, base_vel, step=EIGHTH, legato=0.92, accents=None, start_eighth=0):
        """items: list of pitch names (or lists for chords, or None for rest), one per step."""
        t = (bar - 1) * BAR + start_eighth * EIGHTH
        for k, it in enumerate(items):
            if it is not None:
                acc = accents[k] if accents else 0
                self.add(hand, t, int(step * legato), it if isinstance(it, list) else [it], base_vel + acc)
            t += step

    def melody(self, hand, bar, items, base_vel, legato=0.96):
        """items: (pitches, beats[, vel_offset]) consecutively from the downbeat of `bar`."""
        t = (bar - 1) * BAR
        for it in items:
            pitches, beats = it[0], it[1]
            dv = it[2] if len(it) > 2 else 0
            d = int(beats * TPB)
            if pitches:
                self.add(hand, t, int(d * legato), pitches if isinstance(pitches, list) else [pitches], base_vel + dv)
            t += d

    def pedal_change(self, tick):
        self.pedal.append((tick + 6, 0))
        self.pedal.append((tick + 70, 127))


# ---------------------------------------------------------------- material
OST_DM = ["A4", "D5", "E5", "D5", "A4", "D5", "F5", "D5"]        # rocking right-hand ostinato
OST_SUS = ["A4", "D5", "E5", "D5", "A4", "C#5", "E5", "C#5"]     # A sus4 -> A major
OST_ACC = [6, 0, 3, -1, 3, 0, 4, -1]                             # gentle metric accents

LH_WAVE = {   # [root, fifth, octave, ninth, octave, fifth, octave, fifth]
    "Dm": ["D3", "A3", "D4", "E4", "D4", "A3", "D4", "A3"],
    "Dm/C": ["C3", "G3", "C4", "D4", "C4", "G3", "C4", "G3"],
    "Bb": ["Bb2", "F3", "Bb3", "C4", "Bb3", "F3", "Bb3", "F3"],
    "A": ["A2", "E3", "A3", "B3", "A3", "E3", "A3", "E3"],
}
WAVE_ACC = [8, 0, 2, 3, 1, -1, 2, -2]

LH_DRIVE = {  # climax: [root+fifth, octave, fifth, octave] x2
    "Bb": [["Bb2", "F3"], "Bb3", "F3", "Bb3"],
    "C": [["C3", "G3"], "C4", "G3", "C4"],
    "Dm": [["D3", "A3"], "D4", "A3", "D4"],
    "Dm/C": [["C3", "G3"], "C4", "G3", "C4"],
    "A": [["A2", "E3"], "A3", "E3", "A3"],
}
DRIVE_ACC = [10, -2, 0, -1, 6, -2, 0, -1]


def build() -> Score:
    s = Score()

    # ---- bars 1-2: the ostinato alone (pp) -----------------------------------
    for bar, vel in ((1, 44), (2, 47)):
        s.seq("R", bar, OST_DM, vel, accents=OST_ACC)

    # ---- bars 3-6: lament bass enters under the ostinato (p) -------------------
    cyc1 = [(3, "D3", "A3", OST_DM), (4, "C3", "A3", OST_DM), (5, "Bb2", "F3", OST_DM), (6, "A2", "E3", OST_SUS)]
    for i, (bar, root, fifth, ost) in enumerate(cyc1):
        s.seq("R", bar, ost, 49 + i, accents=OST_ACC)
        t = (bar - 1) * BAR
        s.add("L", t, int(BAR * 0.97), [root], 56 + i)
        s.add("L", t + 2 * TPB, int(2 * TPB * 0.95), [fifth], 47 + i)

    # ---- bars 7-10: melody over a flowing left hand (mp) ----------------------
    prog = ["Dm", "Dm/C", "Bb", "A"]
    for i, ch in enumerate(prog):
        s.seq("L", 7 + i, LH_WAVE[ch], 50 + i, accents=WAVE_ACC)
    s.melody("R", 7, [("A4", 2, -2), ("D5", 1), ("E5", 1, 2)], 68)
    s.melody("R", 8, [("F5", 2, 5), ("E5", 1), ("D5", 1, -2)], 68)
    s.melody("R", 9, [("D5", 2, 2), ("C5", 1, -1), ("Bb4", 1, -3)], 68)
    s.melody("R", 10, [("C#5", 3, 3), ("A4", 1, -4)], 68)

    # ---- bars 11-14: melody in thirds, crescendo (mf) ---------------------------
    for i, ch in enumerate(prog):
        s.seq("L", 11 + i, LH_WAVE[ch], 57 + 3 * i, accents=WAVE_ACC)
    s.melody("R", 11, [(["D5", "F5"], 2, 0), (["F5", "A5"], 1, 4), (["E5", "G5"], 1, 1)], 76)
    s.melody("R", 12, [(["A4", "F5"], 2, 3), (["C5", "E5"], 1), (["A4", "D5"], 1, -2)], 77)
    s.melody("R", 13, [(["Bb4", "D5"], 2, 2), (["D5", "F5"], 1, 4), (["C5", "E5"], 1, 2)], 80)
    s.melody("R", 14, [(["A4", "C#5"], 1, 0), (["C#5", "E5"], 1, 4), (["E5", "G5"], 2, 9)], 82)

    # ---- bars 15-20: climax (f - ff) ----------------------------------------
    climax_prog = ["Bb", "C", "Dm", "Dm/C", "Bb", "A"]
    for i, ch in enumerate(climax_prog):
        lvl = [80, 83, 86, 84, 88, 80][i]
        s.seq("L", 15 + i, LH_DRIVE[ch] * 2, lvl, accents=DRIVE_ACC)
    s.melody("R", 15, [(["F4", "Bb4", "D5"], 1, -6), (["Bb4", "D5", "F5"], 1, -2), (["D5", "F5", "A5"], 2, 6)], 96)
    s.melody("R", 16, [(["C5", "E5", "G5"], 3, 4), (["G4", "C5", "E5"], 1, -6)], 96)
    s.melody("R", 17, [(["A4", "D5", "F5"], 2, 4), (["A4", "E5"], 1, -4), (["A4", "D5"], 1, -8)], 98)
    s.melody("R", 18, [(["G4", "C5", "E5"], 2, -2), (["A4", "C5", "F5"], 1, 2), (["C5", "E5", "G5"], 1, 6)], 98)
    s.melody("R", 19, [(["D5", "F5", "A5"], 2, 10), (["A4", "D5", "F5"], 1, 0), (["F4", "A4", "D5"], 1, -6)], 100)
    s.melody("R", 20, [(["A4", "D5"], 2, -8), (["E4", "A4", "C#5"], 2, -14)], 98)

    # ---- bars 21-24: the opening figure returns (p -> pp), ritardando ------------
    s.seq("R", 21, OST_DM, 55, accents=OST_ACC)
    s.seq("R", 22, [["A4", "A5"]] + OST_DM[1:], 51, accents=OST_ACC)
    s.seq("R", 23, OST_SUS, 47, accents=OST_ACC)
    s.seq("R", 24, OST_DM[:7] + [None], 42, accents=OST_ACC)
    for bar, root, fifth, vel in ((21, "D3", "A3", 60), (22, "Bb2", "F3", 56), (23, "A2", "E3", 52), (24, "D3", "A3", 46)):
        t = (bar - 1) * BAR
        s.add("L", t, int(BAR * 0.97), [root], vel)
        s.add("L", t + 2 * TPB, int(2 * TPB * 0.95), [fifth], vel - 8)

    # ---- bar 25: final rolled D minor (add 9) chord, let ring ---------------------
    t = 24 * BAR
    s.add("L", t, int(BAR * 1.15), ["D3", "A3"], 54, roll=40)
    s.add("R", t + 80, int(BAR * 1.15) - 80, ["A4", "D5", "E5"], 50, roll=50)

    # ---- pedal: legato change on every harmony ----------------------------------
    s.pedal.append((0, 127))
    for bar in range(2, 26):
        s.pedal_change((bar - 1) * BAR)
    for bar in (6, 20, 23):                    # sus4 -> major within the bar
        s.pedal_change((bar - 1) * BAR + 2 * TPB)
    s.pedal.append((25 * BAR + int(BAR * 0.6), 0))

    # ---- light timing humanisation (never on downbeats) ------------------------
    for hand in s.notes:
        out = []
        for (st, du, pi, ve) in s.notes[hand]:
            if st % TPB:
                st += s.rng.randint(-4, 4)
            out.append((st, du, pi, ve))
        s.notes[hand] = out
    return s


def tempo_map():
    """(tick, bpm): steady 88, then ritardando in bars 23-25."""
    return [(0, 88.0), (22 * BAR, 82.0), (22 * BAR + 2 * TPB, 78.0), (23 * BAR, 74.0),
            (23 * BAR + 2 * TPB, 69.0), (24 * BAR, 64.0)]


def write(score: Score, path: Path = OUT) -> Path:
    mid = mido.MidiFile(type=1, ticks_per_beat=TPB)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("track_name", name="Dust & Starlight", time=0))
    meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    meta.append(mido.MetaMessage("key_signature", key="Dm", time=0))
    last = 0
    for tick, bpm in tempo_map():
        meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=tick - last))
        last = tick
    mid.tracks.append(meta)
    for hand, name, ch in (("R", "Right Hand", 0), ("L", "Left Hand", 1)):
        tr = mido.MidiTrack()
        tr.append(mido.MetaMessage("track_name", name=name, time=0))
        tr.append(mido.Message("program_change", program=0, channel=ch, time=0))
        evs = []
        for st, du, pi, ve in score.notes[hand]:
            evs.append((st, 1, mido.Message("note_on", note=pi, velocity=ve, channel=ch)))
            evs.append((st + du, 0, mido.Message("note_off", note=pi, velocity=0, channel=ch)))
        if hand == "L":   # pedal lives on the left-hand track
            for tick, val in score.pedal:
                evs.append((tick, 0, mido.Message("control_change", control=64, value=val, channel=ch)))
        evs.sort(key=lambda e: (e[0], e[1]))
        last = 0
        for tick, _o, msg in evs:
            tr.append(msg.copy(time=tick - last))
            last = tick
        mid.tracks.append(tr)
    mid.save(str(path))
    return path


if __name__ == "__main__":
    out = write(build())
    m = mido.MidiFile(str(out))
    print(f"wrote {out}  ({m.length:.2f} s)")
