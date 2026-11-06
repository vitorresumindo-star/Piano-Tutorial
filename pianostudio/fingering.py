"""Automatic fingering suggestion.

Approach: a beam search over onset groups (chords) per hand. Every candidate
state assigns fingers to the notes of a chord in strictly monotonic order
(RH: low->high = 1->5, LH: low->high = 5->1). Costs come from a *hand
position model*: each finger has a natural offset from the hand centre, so a
state implies a hand position P and a stretch penalty; transitions pay for hand
movement (scaled by available time), awkward crossings, repeated fingers on
different keys, thumbs on black keys and fingers still holding sounding notes.

The result is a plausible, physically-checked suggestion -- NOT a guarantee of
professional pianist fingering. Difficult choices are reported so a human can
review them (see validation.py and the UI fingering editor).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from .geometry import WHITE_W, SPAN_COMFORT, SPAN_MAX, is_black, key_x, note_name
from .midi_io import group_onsets
from .model import Note, Song, LEFT, RIGHT

NATURAL = {1: -2.05, 2: -1.0, 3: 0.0, 4: 1.0, 5: 2.0}   # in white-key widths (RH)
TOLERANCE = {1: 2.4, 2: 1.0, 3: 0.9, 4: 0.8, 5: 1.1}    # comfortable deviation (white widths)
DIFFICULT_COST = 14.0
RECENT = 0.35            # s: a just-released finger still constrains crossings
CROSS_RECENT_COST = 9.0
IMPOSSIBLE_COST = 40.0


@dataclass
class FingeringIssue:
    note_ids: list[int]
    time: float
    hand: str
    severity: str        # "error" | "warning"
    code: str
    message: str

    def to_dict(self):
        return self.__dict__.copy()


@dataclass
class _Beam:
    cost: float
    choices: list
    active: tuple  # ((end, pitch, finger, x), ...)
    local: list = field(default_factory=list)


def side_sign(hand: str) -> int:
    return 1 if hand == RIGHT else -1


def _pair_span_ok(f1: int, x1: float, f2: int, x2: float) -> tuple[float, bool]:
    """Returns (penalty, impossible) for two simultaneously-held fingers."""
    a, b = sorted((f1, f2))
    if a == b:
        return 0.0, False
    span = abs(x2 - x1)
    comf, mx = SPAN_COMFORT[(a, b)], SPAN_MAX[(a, b)]
    if span <= comf:
        return 0.0, False
    if span <= mx:
        return 3.0 * (span - comf) / WHITE_W, False
    return IMPOSSIBLE_COST + 5.0 * (span - mx) / WHITE_W, True


def _states(group: list[Note], hand: str) -> list[tuple[int, ...]]:
    k = len(group)
    if k > 5:
        return []
    fingers = (1, 2, 3, 4, 5) if hand == RIGHT else (5, 4, 3, 2, 1)
    out = []
    for combo in combinations(range(5), k):
        st = tuple(fingers[i] for i in combo)  # aligned with group sorted by pitch
        ok = True
        for n, f in zip(group, st):
            if n.finger is not None and n.finger_source in ("midi", "user") and n.finger != f:
                ok = False
                break
        if ok:
            out.append(st)
    return out


def _state_cost(xs, fingers, s, held):
    """Stretch cost of a full hand configuration (new notes + still-held notes)."""
    pts = list(zip(xs, fingers)) + [(h[3], h[2]) for h in held]
    P = sum(x - s * NATURAL[f] * WHITE_W for x, f in pts) / len(pts)
    cost = 0.0
    for x, f in pts:
        dev = (x - s * NATURAL[f] * WHITE_W - P) / (TOLERANCE[f] * WHITE_W)
        cost += max(0.0, abs(dev) - 1.0) ** 2 * 2.0 + 0.15 * dev * dev
    impossible = False
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            pen, imp = _pair_span_ok(pts[i][1], pts[i][0], pts[j][1], pts[j][0])
            cost += pen
            impossible |= imp
    return cost, P, impossible


def finger_hand(notes: list[Note], hand: str, beam_width: int = 16) -> list[FingeringIssue]:
    s = side_sign(hand)
    issues: list[FingeringIssue] = []
    groups = group_onsets(notes)
    beams = [_Beam(0.0, [], (), [])]
    prevP: dict[int, float] = {}
    for gi, g in enumerate(groups):
        g.sort(key=lambda n: n.pitch)
        onset = g[0].start
        xs = [key_x(n.pitch) for n in g]
        states = _states(g, hand)
        if not states:
            if len(g) > 5:
                issues.append(FingeringIssue([n.id for n in g], onset, hand, "error", "too-many-notes",
                                             f"{len(g)} simultaneous notes assigned to one hand (max 5)."))
                g = g[:5] if hand == LEFT else g[-5:]
                xs = [key_x(n.pitch) for n in g]
                states = _states(g, hand)
            if not states:  # conflicting fixed fingers -> relax constraints
                issues.append(FingeringIssue([n.id for n in g], onset, hand, "error", "fixed-finger-conflict",
                                             "Fixed fingers for this chord are not in a playable order."))
                for n in g:
                    if n.finger_source in ("midi", "user"):
                        n.finger_source = "conflict"
                states = _states(g, hand)
            groups[gi] = g
        cand: dict = {}
        for b in beams:
            pitches = [n.pitch for n in g]
            held = tuple(h for h in b.active if h[0] > onset + 0.04 and h[1] not in pitches)
            # fingers that are still down or only just lifted: crossing over/under them is unnatural
            recent = tuple(h for h in b.active if onset - RECENT <= h[0] <= onset + 0.04 and h[1] not in pitches)
            last = b.choices[-1] if b.choices else None
            last_g = groups[gi - 1] if gi > 0 else None
            gap = (onset - last_g[0].start) if last_g else 1.0
            gap = max(gap, 0.04)
            for st in states:
                c, P, imp = _state_cost(xs, st, s, held)
                # held-finger conflicts
                held_f = {h[2] for h in held}
                for f in st:
                    if f in held_f:
                        c += 25.0
                # non-thumb finger crossings against fingers still on / just off their keys
                for n, f in zip(g, st):
                    if f == 1:
                        continue
                    for h in held + recent:
                        pf = h[2]
                        if pf == 1 or pf == f:
                            continue
                        if (f - pf) * s * (key_x(n.pitch) - h[3]) < 0:
                            c += CROSS_RECENT_COST
                # black keys
                for n, f in zip(g, st):
                    if is_black(n.pitch):
                        if f == 1:
                            c += 2.2 if len(g) == 1 else 1.0
                        elif f == 5:
                            c += 0.6
                if last is not None:
                    pP = b.local[-1][1]
                    move = abs(P - pP) / WHITE_W
                    c += move * (0.35 + 0.12 / gap)
                    prev_notes = list(zip(last_g, last))
                    for n, f in zip(g, st):
                        for pn, pf in prev_notes:
                            dx = s * (key_x(n.pitch) - key_x(pn.pitch))
                            if pn.pitch == n.pitch:
                                c += 0.0 if pf == f else 0.25
                                continue
                            if pf == f:
                                c += 3.0 + (1.2 / gap if gap < 0.6 else 0.0)
                                continue
                            df = f - pf
                            crossing = (df > 0 and dx < 0) or (df < 0 and dx > 0)
                            if crossing:
                                if 1 in (f, pf) and {f, pf} & {2, 3, 4}:
                                    c += 1.3 + 0.08 / gap
                                    if 1 == f and is_black(n.pitch):
                                        c += 2.0
                                else:
                                    c += 12.0
                key_ = (st, tuple(sorted((h[2], h[1]) for h in held)))
                total = b.cost + c
                if key_ not in cand or cand[key_].cost > total:
                    new_active = held + recent + tuple((n.end, n.pitch, f, x) for n, f, x in zip(g, st, xs))
                    cand[key_] = _Beam(total, b.choices + [st], new_active, b.local + [(c, P, imp)])
        beams = sorted(cand.values(), key=lambda b: b.cost)[:beam_width]
    if not beams or not groups:
        return issues
    best = beams[0]
    for g, st, (c, P, imp) in zip(groups, best.choices, best.local):
        for n, f in zip(g, st):
            if n.finger_source not in ("midi", "user"):
                n.finger, n.finger_source = f, "auto"
        names = ", ".join(note_name(n.pitch) for n in g)
        if imp or c >= IMPOSSIBLE_COST:
            issues.append(FingeringIssue([n.id for n in g], g[0].start, hand, "error", "unplayable-stretch",
                                         f"{names}: stretch or finger conflict exceeds what one hand can play."))
        elif c >= DIFFICULT_COST:
            issues.append(FingeringIssue([n.id for n in g], g[0].start, hand, "warning", "difficult-passage",
                                         f"{names}: awkward fingering/position change (cost {c:.1f}); please review."))
    return issues


def assign_fingering(song: Song, beam_width: int = 16) -> list[FingeringIssue]:
    issues: list[FingeringIssue] = []
    for hand in (RIGHT, LEFT):
        hn = song.notes_for(hand)
        if hn:
            issues += finger_hand(hn, hand, beam_width)
    unassigned = [n for n in song.notes if n.hand is None]
    if unassigned:
        issues.append(FingeringIssue([n.id for n in unassigned], unassigned[0].start, "?", "error",
                                     "missing-hand", f"{len(unassigned)} notes have no hand assignment."))
    return issues
