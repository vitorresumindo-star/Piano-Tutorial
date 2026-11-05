"""MIDI import: notes, tempo map, time signatures, sustain pedal, hand detection."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import mido

from .geometry import LOWEST_MIDI, HIGHEST_MIDI
from .model import Note, PedalEvent, Song, TempoChange, TimeSignature, tick_to_seconds, LEFT, RIGHT

RH_PAT = re.compile(r"\b(right|rh|r\.?h\.?|treble|upper|melody|droite|rechts|destra|derecha|direita)\b", re.I)
LH_PAT = re.compile(r"\b(left|lh|l\.?h\.?|bass|lower|gauche|links|sinistra|izquierda|esquerda)\b", re.I)
CHORD_EPS = 0.035  # seconds: onsets closer than this form one chord


def load_midi(path: str | Path, fingering_sidecar: str | Path | None = None) -> Song:
    path = Path(path)
    mid = mido.MidiFile(str(path), clip=True)
    tpb = mid.ticks_per_beat
    log: list[dict] = []

    # ---- gather absolute-tick events from every track -------------------
    events = []  # (tick, order, track_idx, msg)
    track_names: list[str] = []
    for ti, track in enumerate(mid.tracks):
        tick = 0
        name = ""
        for order, msg in enumerate(track):
            tick += msg.time
            if msg.is_meta and msg.type == "track_name" and not name:
                name = msg.name
            events.append((tick, order, ti, msg))
        track_names.append(name)
    if mid.type == 2:
        log.append({"level": "warning", "code": "midi-type-2",
                    "message": "Type-2 MIDI (independent sequences) is treated as simultaneous tracks."})
    events.sort(key=lambda e: (e[0], e[2], e[1]))

    # ---- tempo map ---------------------------------------------------------
    raw_tempos = [(0, 500_000)]
    for tick, _o, _t, msg in events:
        if msg.is_meta and msg.type == "set_tempo":
            if tick == raw_tempos[-1][0]:
                raw_tempos[-1] = (tick, msg.tempo)
            else:
                raw_tempos.append((tick, msg.tempo))
    tempos: list[TempoChange] = []
    for tick, tempo in raw_tempos:
        if not tempos:
            tempos.append(TempoChange(tick, 0.0, tempo))
        else:
            p = tempos[-1]
            tempos.append(TempoChange(tick, p.time + (tick - p.tick) * p.tempo / (tpb * 1_000_000), tempo))

    def secs(tick: int) -> float:
        return tick_to_seconds(tick, tempos, tpb)

    # ---- time signatures -----------------------------------------------------
    sigs: list[TimeSignature] = []
    for tick, _o, _t, msg in events:
        if msg.is_meta and msg.type == "time_signature":
            ts = TimeSignature(tick, secs(tick), msg.numerator, msg.denominator)
            if sigs and sigs[-1].tick == tick:
                sigs[-1] = ts
            else:
                sigs.append(ts)
    if not sigs or sigs[0].tick != 0:
        sigs.insert(0, TimeSignature(0, 0.0, 4, 4))

    # ---- notes + pedal -------------------------------------------------------
    active: dict[tuple[int, int, int], list[tuple[int, int]]] = defaultdict(list)
    raw_notes: list[dict] = []
    pedal_raw: list[tuple[int, bool]] = []
    last_tick = 0
    for tick, _o, ti, msg in events:
        last_tick = max(last_tick, tick)
        if msg.is_meta:
            continue
        if msg.type == "note_on" and msg.velocity > 0:
            k = (ti, msg.channel, msg.note)
            if active[k]:
                st, vel = active[k].pop(0)
                raw_notes.append(dict(pitch=msg.note, st=st, et=tick, vel=vel, track=ti, ch=msg.channel))
                log.append({"level": "info", "code": "overlap-retrigger",
                            "message": f"Note {msg.note} re-struck before release at tick {tick}; previous note shortened."})
            active[k].append((tick, msg.velocity))
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            k = (ti, msg.channel, msg.note)
            if active[k]:
                st, vel = active[k].pop(0)
                raw_notes.append(dict(pitch=msg.note, st=st, et=tick, vel=vel, track=ti, ch=msg.channel))
        elif msg.type == "control_change" and msg.control == 64:
            pedal_raw.append((tick, msg.value >= 64))
    for (ti, ch, pitch), lst in active.items():
        for st, vel in lst:
            raw_notes.append(dict(pitch=pitch, st=st, et=last_tick, vel=vel, track=ti, ch=ch))
            log.append({"level": "warning", "code": "unterminated-note",
                        "message": f"Note {pitch} at tick {st} never released; closed at end of file."})

    # Collapse cross-track/channel duplicates of the same pitch at the same tick
    raw_notes.sort(key=lambda n: (n["st"], n["pitch"], n["track"]))
    dedup = []
    seen = set()
    for n in raw_notes:
        key = (n["st"], n["pitch"])
        if key in seen:
            log.append({"level": "info", "code": "duplicate-note",
                        "message": f"Duplicate note {n['pitch']} at tick {n['st']} removed."})
            continue
        seen.add(key)
        dedup.append(n)

    notes: list[Note] = []
    for n in dedup:
        if not (LOWEST_MIDI <= n["pitch"] <= HIGHEST_MIDI):
            log.append({"level": "error", "code": "out-of-range",
                        "message": f"Note {n['pitch']} is outside the 88-key piano range and was skipped."})
            continue
        start, end = secs(n["st"]), secs(n["et"])
        if end - start < 0.03:
            end = start + 0.03
            log.append({"level": "info", "code": "short-note",
                        "message": f"Very short note {n['pitch']} at {start:.3f}s extended to 30 ms."})
        notes.append(Note(id=len(notes), pitch=n["pitch"], start=start, end=end, velocity=n["vel"],
                          start_tick=n["st"], end_tick=n["et"], track=n["track"], channel=n["ch"]))

    # Same pitch overlapping across tracks: truncate the earlier one
    by_pitch = defaultdict(list)
    for n in notes:
        by_pitch[n.pitch].append(n)
    for lst in by_pitch.values():
        for a, b in zip(lst, lst[1:]):
            if a.end > b.start:
                a.end = max(a.start + 0.03, b.start - 0.005)
                log.append({"level": "info", "code": "same-pitch-overlap",
                            "message": f"Overlapping repeats of {a.pitch} at {b.start:.3f}s: earlier note shortened."})

    pedal: list[PedalEvent] = []
    for tick, down in sorted(pedal_raw):
        if pedal and pedal[-1].down == down:
            continue
        if not pedal and not down:
            continue
        pedal.append(PedalEvent(secs(tick), down, tick))

    duration = max([n.end for n in notes] + [0.0])
    song = Song(notes=notes, ticks_per_beat=tpb, tempos=tempos, time_signatures=sigs, pedal=pedal,
                duration=duration, track_names=track_names, source_file=str(path), parse_log=log)
    assign_hands_from_midi(song)
    if fingering_sidecar is None:
        cand = path.with_suffix(".fingering.json")
        fingering_sidecar = cand if cand.exists() else None
    if fingering_sidecar:
        apply_fingering_sidecar(song, fingering_sidecar)
    return song


def assign_hands_from_midi(song: Song) -> None:
    notes = song.notes
    if not notes:
        return
    tracks_with_notes = sorted({n.track for n in notes})
    track_hand: dict[int, str] = {}
    for t in tracks_with_notes:
        name = song.track_names[t] if t < len(song.track_names) else ""
        if RH_PAT.search(name) and not LH_PAT.search(name):
            track_hand[t] = RIGHT
        elif LH_PAT.search(name) and not RH_PAT.search(name):
            track_hand[t] = LEFT
    if len(tracks_with_notes) >= 2 and len(track_hand) == len(tracks_with_notes) and len(set(track_hand.values())) == 2:
        for n in notes:
            n.hand, n.hand_source = track_hand[n.track], "midi-track"
        return
    if len(tracks_with_notes) == 2:
        a, b = tracks_with_notes
        avg = {t: sum(n.pitch for n in notes if n.track == t) / sum(1 for n in notes if n.track == t)
               for t in (a, b)}
        hi = a if avg[a] >= avg[b] else b
        for n in notes:
            n.hand, n.hand_source = (RIGHT if n.track == hi else LEFT), "midi-track"
        song.parse_log.append({"level": "info", "code": "hands-by-track-pitch",
                               "message": "Two note tracks found; the higher track was assigned to the right hand."})
        return
    channels = sorted({n.channel for n in notes})
    if len(tracks_with_notes) == 1 and len(channels) == 2:
        avg = {c: sum(n.pitch for n in notes if n.channel == c) / sum(1 for n in notes if n.channel == c)
               for c in channels}
        hi = max(channels, key=lambda c: avg[c])
        for n in notes:
            n.hand, n.hand_source = (RIGHT if n.channel == hi else LEFT), "midi-channel"
        return
    split_hands_heuristic(notes)
    song.parse_log.append({"level": "warning", "code": "hands-heuristic",
                           "message": "No hand information in the MIDI file: hands were split automatically. "
                                      "Please review the left/right assignment."})


def group_onsets(notes: list[Note], eps: float = CHORD_EPS) -> list[list[Note]]:
    groups: list[list[Note]] = []
    for n in sorted(notes, key=lambda n: (n.start, n.pitch)):
        if groups and n.start - groups[-1][0].start <= eps:
            groups[-1].append(n)
        else:
            groups.append([n])
    return groups


def split_hands_heuristic(notes: list[Note]) -> None:
    """Greedy split with running hand centres. Picks, for each chord, the split
    index minimising hand span, distance to each hand's recent position and
    the 5-notes-per-hand limit."""
    from .geometry import key_x
    lc, rc = key_x(48), key_x(67)  # C3 / G4 start positions
    for g in group_onsets(notes):
        g.sort(key=lambda n: n.pitch)
        xs = [key_x(n.pitch) for n in g]
        best, best_cost = 0, float("inf")
        for split in range(len(g) + 1):  # notes[:split] -> L
            L, R = xs[:split], xs[split:]
            if len(L) > 5 or len(R) > 5:
                continue
            cost = 0.0
            for side, c in ((L, lc), (R, rc)):
                if side:
                    span = side[-1] - side[0]
                    cost += max(0.0, span - 16.5) * 3 + abs(sum(side) / len(side) - c) * 0.6
            if L and R and L[-1] > R[0]:
                cost += 100
            # mild bias: middle C region goes to whichever hand is closer
            if best_cost > cost:
                best, best_cost = split, cost
        for i, n in enumerate(g):
            n.hand = LEFT if i < best else RIGHT
            n.hand_source = "heuristic"
        L, R = xs[:best], xs[best:]
        if L:
            lc = 0.6 * lc + 0.4 * (sum(L) / len(L))
        if R:
            rc = 0.6 * rc + 0.4 * (sum(R) / len(R))


def apply_fingering_sidecar(song: Song, sidecar: str | Path) -> None:
    """Sidecar format: {"notes": [{"id": 3, "finger": 2, "hand": "R"}, ...]} or
    {"by_time": [{"pitch": 60, "start": 1.25, "finger": 1}]}"""
    data = json.loads(Path(sidecar).read_text())
    by_id = {n.id: n for n in song.notes}
    for e in data.get("notes", []):
        n = by_id.get(e["id"])
        if n:
            if "finger" in e:
                n.finger, n.finger_source = int(e["finger"]), "midi"
            if "hand" in e:
                n.hand, n.hand_source = e["hand"], "midi-track"
    for e in data.get("by_time", []):
        for n in song.notes:
            if n.pitch == e["pitch"] and abs(n.start - e["start"]) < 0.02:
                n.finger, n.finger_source = int(e["finger"]), "midi"
                if "hand" in e:
                    n.hand = e["hand"]
