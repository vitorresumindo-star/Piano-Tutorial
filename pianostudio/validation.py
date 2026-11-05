"""Machine-checked validation of a prepared song.

Separates two kinds of results:
  * machine-checked accuracy (key mapping, contact, timing, collisions)
  * items that need human review (auto fingering/hand split, difficult passages)
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from . import geometry as G
from .animation import Bake, fk_fingertips, KEY_DOWN_TIME
from .fingering import FingeringIssue
from .model import Song, LEFT, RIGHT

CONTACT_TOL = 0.30      # cm between solved fingertip and target while pressing
FLOAT_TOL = 0.30        # cm: fingertip skin vs. depressed key surface
COLLIDE_FRAC = 0.80     # adjacent fingertips closer than 80% of the summed radii


def _issue(severity, code, message, time=None, note_ids=None, hand=None, category="machine"):
    return {"severity": severity, "code": code, "message": message, "time": None if time is None else round(time, 3),
            "note_ids": note_ids or [], "hand": hand, "category": category}


def validate(song: Song, bake: Bake, fing_issues: list[FingeringIssue], strict: bool = True,
             approved: bool = False) -> dict:
    issues: list[dict] = []
    checks: dict[str, dict] = {}

    # ---- parse log -------------------------------------------------------
    for e in song.parse_log:
        sev = e["level"]
        cat = "review" if e["code"] == "hands-heuristic" else "machine"
        issues.append(_issue("warning" if sev == "warning" else ("error" if sev == "error" else "info"),
                             e["code"], e["message"], category=cat))
    if not song.notes:
        issues.append(_issue("error", "empty", "The MIDI file contains no playable notes."))

    # ---- note -> key mapping ----------------------------------------------
    bad_range = [n for n in song.notes if not (G.LOWEST_MIDI <= n.pitch <= G.HIGHEST_MIDI)]
    checks["key_mapping"] = {"passed": not bad_range, "notes": len(song.notes),
                             "detail": "every MIDI pitch maps to exactly one of the 88 keys"}

    # ---- hands & fingers ---------------------------------------------------
    no_hand = [n.id for n in song.notes if n.hand not in (LEFT, RIGHT)]
    no_finger = [n.id for n in song.notes if n.finger not in (1, 2, 3, 4, 5)]
    if no_hand:
        issues.append(_issue("error", "missing-hand", f"{len(no_hand)} notes have no hand assignment.", note_ids=no_hand))
    if no_finger:
        issues.append(_issue("error", "missing-finger", f"{len(no_finger)} notes have no finger assignment.",
                             note_ids=no_finger))
    for fi in fing_issues:
        issues.append(_issue(fi.severity, fi.code, fi.message, fi.time, fi.note_ids, fi.hand,
                             category="review" if fi.severity == "warning" else "machine"))
    # same finger pressing two different keys at the same time
    conflicts = []
    for hand in (LEFT, RIGHT):
        hn = sorted(song.notes_for(hand), key=lambda n: n.start)
        for i, a in enumerate(hn):
            for b in hn[i + 1:]:
                if b.start >= a.end - 0.04:
                    break
                if a.finger == b.finger and a.pitch != b.pitch and a.finger is not None:
                    conflicts.append((a, b))
    for a, b in conflicts[:50]:
        issues.append(_issue("error", "finger-conflict",
                             f"Finger {a.finger} ({a.hand}) holds {G.note_name(a.pitch)} while also assigned "
                             f"{G.note_name(b.pitch)}.", b.start, [a.id, b.id], a.hand))
    checks["finger_conflicts"] = {"passed": not conflicts, "count": len(conflicts)}

    # ---- kinematic contact checks -------------------------------------------
    contact_fail, wrong_key, floating = [], [], []
    tips_cache = {}
    for hand in (RIGHT, LEFT):
        hb = bake.hands[hand]
        if not hb.visible:
            continue
        tips = fk_fingertips(hb.pose.astype(np.float64), hand)
        tips_cache[hand] = tips
        err = np.linalg.norm(tips - hb.targets, axis=2)
        for n in song.notes_for(hand):
            if n.finger not in (1, 2, 3, 4, 5):
                continue
            on, off = bake.press[n.id]
            i0 = int(np.ceil((on - bake.t0) * bake.rate))
            i1 = int(np.floor((off - bake.t0) * bake.rate))
            if i1 <= i0:
                i1 = i0 + 1
            f = n.finger - 1
            e = err[i0:i1, f]
            k = G.key(n.pitch)
            r = G.FINGERS[n.finger].tip_pad
            tip = tips[i0:i1, f]
            if e.size and e.max() > CONTACT_TOL:
                contact_fail.append((n, float(e.max())))
            # contact point must be over the intended key
            cx = tip[:, 0]
            cz = tip[:, 2]
            inside_x = np.abs(cx - k.x) <= k.width / 2 + 0.05
            inside_z = (cz >= 0) & (cz <= k.length)
            if not k.black:
                # must not be on a neighbouring black key's area
                on_black = (cz < G.BLACK_LEN) & np.array([G.surface_top(x, z) > 0 for x, z in zip(cx[::6], cz[::6])]).repeat(6)[: len(cx)]
                inside_x &= ~on_black
            if tip.size and not (inside_x & inside_z).all():
                wrong_key.append(n)
            # fingertip skin touches the depressed key (no floating / no sinking)
            depth = G.key_depth_at_contact(n.pitch)
            skin = tip[:, 1] - r
            target = k.top - depth
            mid = slice(len(skin) // 4, max(len(skin) // 4 + 1, 3 * len(skin) // 4))
            if skin.size and np.abs(skin[mid] - target).max() > FLOAT_TOL:
                floating.append(n)
    for n, e in contact_fail[:40]:
        issues.append(_issue("error", "contact", f"{G.note_name(n.pitch)} ({n.hand}{n.finger}): fingertip misses the key "
                                                 f"by {e:.2f} cm.", n.start, [n.id], n.hand))
    for n in wrong_key[:40]:
        issues.append(_issue("error", "wrong-key", f"{G.note_name(n.pitch)} ({n.hand}{n.finger}): contact point leaves the "
                                                   f"intended key.", n.start, [n.id], n.hand))
    for n in floating[:40]:
        issues.append(_issue("error", "floating", f"{G.note_name(n.pitch)} ({n.hand}{n.finger}): fingertip not on the key "
                                                  f"surface while pressing.", n.start, [n.id], n.hand))
    fingered = sum(1 for n in song.notes if n.finger in (1, 2, 3, 4, 5))
    checks["finger_contact"] = {"passed": not contact_fail, "notes_checked": fingered, "failures": len(contact_fail),
                                "tolerance_cm": CONTACT_TOL}
    checks["correct_key"] = {"passed": not wrong_key, "failures": len(wrong_key)}
    checks["no_floating"] = {"passed": not floating, "failures": len(floating), "tolerance_cm": FLOAT_TOL}

    # ---- collisions (sampled at 30 Hz) --------------------------------------
    step = max(1, bake.rate // 30)
    coll = Counter()
    for hand, tips in tips_cache.items():
        t = tips[::step]
        for f in range(4):
            ra, rb = G.FINGERS[f + 1].tip_pad, G.FINGERS[f + 2].tip_pad
            d = np.linalg.norm(t[:, f] - t[:, f + 1], axis=1)
            bad = np.where(d < COLLIDE_FRAC * (ra + rb))[0]
            if len(bad):
                coll[(hand, f + 1, f + 2)] += len(bad)
                sustained = len(bad) * step / bake.rate >= 0.25
                issues.append(_issue("warning" if sustained else "info", "finger-collision",
                                     f"{hand} fingers {f + 1}-{f + 2} come very close ({len(bad)} frames); check visually.",
                                     float(bake.times[bad[0] * step]), hand=hand, category="review"))
    hand_close = 0
    if len(tips_cache) == 2:
        a, b = tips_cache[RIGHT][::step], tips_cache[LEFT][::step]
        d = np.linalg.norm(a[:, :, None, :] - b[:, None, :, :], axis=3).min(axis=(1, 2))
        bad = np.where(d < 1.6)[0]
        hand_close = len(bad)
        if hand_close:
            issues.append(_issue("warning", "hands-overlap", f"Left and right hands come within 1.6 cm "
                                                             f"({hand_close} frames); check for intersections.",
                                 float(bake.times[bad[0] * step]), category="review"))
    checks["finger_collisions"] = {"passed": all(c * step / bake.rate < 0.25 for c in coll.values()), "pairs": {f"{h}{a}-{b}": c for (h, a, b), c in coll.items()}}
    checks["hand_overlap"] = {"passed": hand_close == 0, "frames": hand_close}

    # ---- timing ----------------------------------------------------------
    tempos_ok = all(b.time >= a.time for a, b in zip(song.tempos, song.tempos[1:]))
    order_ok = all(n.end > n.start for n in song.notes)
    checks["timing"] = {"passed": tempos_ok and order_ok, "tempo_changes": len(song.tempos),
                        "detail": "note times derived from tick positions through the tempo map (no accumulation)"}

    # ---- review items -------------------------------------------------------
    auto = sum(1 for n in song.notes if n.finger_source == "auto")
    if auto:
        issues.append(_issue("info", "auto-fingering",
                             f"{auto} of {len(song.notes)} fingers were suggested automatically. Automatic fingering "
                             f"is plausible but not guaranteed to match professional pianist fingering.",
                             category="review"))

    errors = [i for i in issues if i["severity"] == "error"]
    review = [i for i in issues if i["category"] == "review" and i["severity"] == "warning"]
    if errors:
        status = "blocked"
    elif review and strict and not approved:
        status = "needs_review"
    else:
        status = "pass"
    return {
        "status": status,
        "strict": strict,
        "approved": approved,
        "summary": {"notes": len(song.notes), "errors": len(errors), "review_warnings": len(review),
                    "auto_fingered": auto, "duration": round(song.duration, 3)},
        "checks": checks,
        "issues": issues,
        "machine_checked": ["key_mapping", "finger_conflicts", "finger_contact", "correct_key", "no_floating",
                            "finger_collisions", "hand_overlap", "timing"],
        "needs_human_review": ["fingering quality (musical/pedagogical judgement)",
                               "left/right split when inferred heuristically",
                               "visual realism of hand motion (watch the debug preview)"],
    }


def report_markdown(rep: dict, title: str = "") -> str:
    lines = [f"# Validation report{': ' + title if title else ''}", "",
             f"**Status: {rep['status'].upper()}**  ",
             f"Notes: {rep['summary']['notes']} · errors: {rep['summary']['errors']} · "
             f"review warnings: {rep['summary']['review_warnings']} · auto-fingered: {rep['summary']['auto_fingered']}", "",
             "## Machine-checked", ""]
    for k in rep["machine_checked"]:
        c = rep["checks"].get(k, {})
        lines.append(f"- {'✅' if c.get('passed') else '❌'} `{k}` — " +
                     ", ".join(f"{a}: {b}" for a, b in c.items() if a != "passed"))
    lines += ["", "## Needs human review", ""] + [f"- {x}" for x in rep["needs_human_review"]]
    lines += ["", "## Issues", ""]
    if not rep["issues"]:
        lines.append("None.")
    for i in rep["issues"]:
        t = f" @ {i['time']:.2f}s" if i.get("time") is not None else ""
        lines.append(f"- **{i['severity']}** `{i['code']}`{t}: {i['message']}")
    return "\n".join(lines) + "\n"
