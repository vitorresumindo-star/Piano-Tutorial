"""Hand choreography + inverse kinematics.

Given a fingered Song, this module produces a *baked* animation sampled at
``BAKE_RATE`` Hz: for each hand, the wrist transform, a splay parameter and
4 joint angles per finger.  The renderer only applies these numbers through
forward kinematics (same conventions as :func:`fk_fingertips`), so Python is
the single source of truth and everything is deterministic and testable.

Conventions (hand frame, RH; LH mirrors x):
  wrist rotation  R = Ry(yaw) @ Rx(pitch) @ Rz(roll)          (Euler 'YXZ')
  finger base     B_f = wrist + R @ (base_f with x scaled by splay)
  finger frame    R_f = R @ Rbase_f @ Ry(a) ; then per joint Rx(-theta_i) and
                  a bone of length L_i along local -z.
Flexion angles are positive when the finger curls downward.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import geometry as G
from .model import Song, Note, LEFT, RIGHT
from .fingering import NATURAL, side_sign

BAKE_RATE = 120          # samples per second of song time
PRE_ROLL = 2.0           # seconds baked before t=0
POST_ROLL = 2.5
KEY_DOWN_TIME = 0.035    # key travels from surface to bottom before the onset
KEY_UP_TIME = 0.045
LIFT_TIME = 0.085
FLOATS_PER_HAND = 7 + 5 * 4
SPLAY_K = {1: 0.0, 2: 1.0, 3: 0.0, 4: 1.0, 5: 1.0}
BODY_X = G.key_x(62)

YAW_LIMIT = {1: (-0.95, 0.95), 2: (-0.60, 0.60), 3: (-0.50, 0.50), 4: (-0.55, 0.55), 5: (-0.72, 0.72)}
FLEX1_LIMIT = {1: (-0.95, 1.2), 2: (-0.45, 1.55), 3: (-0.45, 1.55), 4: (-0.45, 1.55), 5: (-0.45, 1.55)}
REACH_X = {1: 3.6, 2: 1.5, 3: 1.2, 4: 1.3, 5: 1.9}   # lateral wrist tolerance per pressing finger (cm)
REACH_Z = {1: 2.6, 2: 2.8, 3: 3.0, 4: 2.8, 5: 2.4}
FLEX2_LIMIT = (0.0, 1.95)


# ---------------------------------------------------------------------------
# rotation helpers (vectorised: angles are arrays of shape (N,))
# ---------------------------------------------------------------------------
def _rx(a):
    c, s = np.cos(a), np.sin(a)
    z, o = np.zeros_like(a), np.ones_like(a)
    return np.stack([np.stack([o, z, z], -1), np.stack([z, c, -s], -1), np.stack([z, s, c], -1)], -2)


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    z, o = np.zeros_like(a), np.ones_like(a)
    return np.stack([np.stack([c, z, s], -1), np.stack([z, o, z], -1), np.stack([-s, z, c], -1)], -2)


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    z, o = np.zeros_like(a), np.ones_like(a)
    return np.stack([np.stack([c, -s, z], -1), np.stack([s, c, z], -1), np.stack([z, z, o], -1)], -2)


def euler_yxz(y, x, z):
    return _ry(y) @ _rx(x) @ _rz(z)


def finger_base_local(f: int, s: int, splay):
    spec = G.FINGERS[f]
    bx, by, bz = spec.base
    sx = s * bx * (1.0 + np.asarray(splay) * SPLAY_K[f])
    return np.stack([sx, np.full_like(sx, by), np.full_like(sx, bz)], -1)


def finger_base_rot(f: int, s: int):
    ey, ex, ez = G.FINGERS[f].base_euler
    return euler_yxz(np.array(s * ey), np.array(ex), np.array(s * ez))


def planar_chain(f: int, t1, t2):
    """Tip offset (u forward, v up) of the finger chain for flex angles."""
    spec = G.FINGERS[f]
    L1, L2, L3 = spec.lengths
    t3 = spec.dip_ratio * t2
    a1, a2, a3 = t1, t1 + t2, t1 + t2 + t3
    u = L1 * np.cos(a1) + L2 * np.cos(a2) + L3 * np.cos(a3)
    v = -(L1 * np.sin(a1) + L2 * np.sin(a2) + L3 * np.sin(a3))
    return u, v


def fk_fingertips(pose: np.ndarray, hand: str) -> np.ndarray:
    """pose: (N, FLOATS_PER_HAND) -> fingertip centres (N, 5, 3) in world space."""
    s = side_sign(hand)
    P = pose[:, 0:3]
    R = euler_yxz(pose[:, 3], pose[:, 4], pose[:, 5])
    splay = pose[:, 6]
    out = np.zeros((pose.shape[0], 5, 3))
    for i, f in enumerate(range(1, 6)):
        a, t1, t2, _t3 = (pose[:, 7 + 4 * i + k] for k in range(4))
        B = P + np.einsum("nij,nj->ni", R, finger_base_local(f, s, splay))
        u, v = planar_chain(f, t1, t2)
        local = np.stack([-u * np.sin(a), v, -u * np.cos(a)], -1)
        Rb = finger_base_rot(f, s)
        out[:, i] = B + np.einsum("nij,nj->ni", R @ Rb, local)
    return out


def solve_finger_ik(f: int, s: int, R, B, T):
    """Vectorised analytic/bisection IK. Returns (a, t1, t2, t3, err)."""
    Rb = finger_base_rot(f, s)
    Rf = R @ Rb
    d = np.einsum("nji,nj->ni", Rf, T - B)          # Rf^T (T - B)
    a = np.arctan2(-d[:, 0], -d[:, 2])
    a = np.clip(a, *YAW_LIMIT[f])
    h = np.hypot(d[:, 0], d[:, 2])
    dy = d[:, 1]
    D = np.hypot(h, dy)
    lo = np.full_like(D, FLEX2_LIMIT[0])
    hi = np.full_like(D, FLEX2_LIMIT[1])
    for _ in range(28):  # r(t2) decreases monotonically with t2
        mid = 0.5 * (lo + hi)
        u, v = planar_chain(f, 0.0, mid)
        r = np.hypot(u, v)
        too_long = r > D
        lo = np.where(too_long, mid, lo)
        hi = np.where(too_long, hi, mid)
    t2 = 0.5 * (lo + hi)
    u, v = planar_chain(f, 0.0, t2)
    chain_angle = np.arctan2(-v, u)          # below horizontal when t1 = 0
    target_angle = np.arctan2(-dy, h)
    t1 = np.clip(target_angle - chain_angle, *FLEX1_LIMIT[f])
    t3 = G.FINGERS[f].dip_ratio * t2
    return a, t1, t2, t3


# ---------------------------------------------------------------------------
# time-curve helpers
# ---------------------------------------------------------------------------
def _ease(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (x * (x * 6 - 15) + 10)      # minimum-jerk (quintic smoothstep)


def _idx(times, t0, t):
    return int(np.clip(math.floor((t - t0) * BAKE_RATE), 0, len(times)))


def _smooth(arr, seconds):
    n = max(1, int(seconds * BAKE_RATE))
    if n <= 1:
        return arr
    k = np.hanning(2 * n + 1)
    k /= k.sum()
    pad = np.pad(arr, (n, n), mode="edge")
    return np.convolve(pad, k, mode="valid")


def visual_offsets(fingered: list[Note], hand: str) -> dict[int, float]:
    """Visual release time per note.  When the next notes of the same hand
    cannot be reached while this key is still held (a leap), the key is
    released a moment early so the hand can travel -- what a pianist does,
    relying on the pedal for legato.  Audio keeps the exact MIDI durations."""
    s = side_sign(hand)
    nat = {f: s * NATURAL[f] * G.WHITE_W for f in range(1, 6)}
    onsets = sorted({n.start for n in fingered})
    out = {}
    for n in fingered:
        off = n.end
        vx = G.key_x(n.pitch) - nat[n.finger]
        for t in onsets:
            if t <= n.start + 1e-6 or t >= off + 0.12:
                continue
            nxt = [m for m in fingered if abs(m.start - t) < 1e-6]
            conflict = any(abs((G.key_x(m.pitch) - nat[m.finger]) - vx) > REACH_X[n.finger] + REACH_X[m.finger]
                           or m.finger == n.finger for m in nxt)
            if conflict:
                dist = max(abs((G.key_x(m.pitch) - nat[m.finger]) - vx) for m in nxt)
                travel = float(np.clip(0.05 + dist / 160.0, 0.06, 0.16))
                off = min(off, max(n.start + 0.45 * (t - n.start), t - travel))
                break
        out[n.id] = max(off, n.start + 0.04)
    return out


def press_interval(n: Note, next_same_finger_start: float | None, off: float | None = None) -> tuple[float, float]:
    """Visual key-down interval. Repeated strikes by the same finger need time to lift."""
    off = n.end if off is None else off
    if next_same_finger_start is not None:
        off = min(off, next_same_finger_start - (KEY_UP_TIME + 0.03))
    off = max(off, n.start + 0.04)
    return n.start, off


def key_depth_profile(t, on, off, depth):
    """Depth of a key over times t for one press (shared formula with the JS renderer)."""
    down = np.clip((t - (on - KEY_DOWN_TIME)) / KEY_DOWN_TIME, 0, 1)
    up = np.clip((t - off) / KEY_UP_TIME, 0, 1)
    return depth * np.minimum(down, 1 - up) * ((t >= on - KEY_DOWN_TIME) & (t <= off + KEY_UP_TIME))


@dataclass
class HandBake:
    hand: str
    pose: np.ndarray        # (N, FLOATS_PER_HAND)
    targets: np.ndarray     # (N, 5, 3) desired fingertip centres
    pressing: np.ndarray    # (N, 5) note id being pressed (-1 none)
    visible: bool


@dataclass
class Bake:
    t0: float
    rate: int
    times: np.ndarray
    hands: dict
    press: dict             # note id -> (on, off)


def rest_tip_offsets() -> dict[int, np.ndarray]:
    """Fingertip centre of each finger relative to the wrist at the rest pose (RH)."""
    out = {}
    for f in range(1, 6):
        t1, t2, _ = G.REST_FLEX[f]
        u, v = planar_chain(f, np.array([t1]), np.array([t2]))
        Rb = finger_base_rot(f, 1)
        loc = np.array([0.0, v[0], -u[0]])
        out[f] = np.array(G.FINGERS[f].base) + (Rb @ loc).reshape(3)
    return out


REST_TIPS = rest_tip_offsets()


def _contact_point(n: Note, f: int, hand_x: float, s: int) -> tuple[float, float]:
    k = G.key(n.pitch)
    natural = hand_x + s * NATURAL[f] * G.WHITE_W
    lim = k.width * 0.22
    x = float(np.clip(natural, k.x - lim, k.x + lim))
    return x, G.contact_z(n.pitch, f)


def bake_song(song: Song) -> Bake:
    t0 = -PRE_ROLL
    N = int(math.ceil((song.duration + POST_ROLL - t0) * BAKE_RATE)) + 1
    times = t0 + np.arange(N) / BAKE_RATE
    hands = {}
    press: dict[int, tuple[float, float]] = {}
    for hand in (RIGHT, LEFT):
        notes = sorted(song.notes_for(hand), key=lambda n: (n.start, n.pitch))
        hb = _bake_hand(hand, notes, times, t0, press)
        hands[hand] = hb
    # notes with no hand/finger still need press intervals for keys
    for n in song.notes:
        press.setdefault(n.id, (n.start, max(n.end, n.start + 0.04)))
    return Bake(t0, BAKE_RATE, times, hands, press)


def _bake_hand(hand: str, notes: list[Note], times, t0, press) -> HandBake:
    s = side_sign(hand)
    N = len(times)
    nat_x = {f: s * NATURAL[f] * G.WHITE_W for f in range(1, 6)}
    default_x = G.key_x(67 if hand == RIGHT else 50)
    fingered = [n for n in notes if n.finger in (1, 2, 3, 4, 5)]
    visible = bool(fingered)

    # ---- 1. hand position: weighted blend of every engaged finger ------------
    # Each note pulls the hand towards the position where its finger sits
    # naturally on the key; the pull ramps up while the finger prepares, stays
    # while the key is held and fades after release.  Held notes therefore keep
    # the hand within reach while the next notes are approached.
    VOFF = visual_offsets(fingered, hand)
    WS = np.zeros(N)
    WX = np.zeros(N)
    WZ = np.zeros(N)
    prev_on = {}
    onsets = sorted({round(n.start, 4) for n in fingered})
    for n in fingered:
        f = n.finger
        on, off = n.start, VOFF[n.id]
        k = onsets.index(round(on, 4))
        gap_prev = on - onsets[k - 1] if k > 0 else 1.0
        prep = float(np.clip(gap_prev * 0.6, 0.05, 0.22))
        vx = G.key_x(n.pitch) - nat_x[f]
        vz = G.contact_z(n.pitch, f) - REST_TIPS[f][2]
        a, b, c, d = on - prep, on - KEY_DOWN_TIME - 0.01, off, off + KEY_UP_TIME
        ia, ib, ic, id_ = (_idx(times, t0, v) for v in (a, b, c, d))
        w = np.zeros(N)
        # approaching notes pull gently; keys being held dominate (weight 12)
        w[ia:ib] = _ease((times[ia:ib] - a) / max(b - a, 1e-6))
        w[ib:ic] = 12.0
        w[ic:id_] = 12.0 * (1.0 - _ease((times[ic:id_] - c) / (d - c)))
        w *= 1.0 + 0.6 * (f in (1, 5))     # outer fingers constrain the hand most
        WS += w
        WX += w * vx
        WZ += w * vz
    X = np.full(N, np.nan)
    Z = np.full(N, np.nan)
    ok = WS > 1e-4
    X[ok] = WX[ok] / WS[ok]
    Z[ok] = WZ[ok] / WS[ok]
    X = _ffill(X, default_x)
    Z = _ffill(Z, 24.0)
    # Feasibility: while keys are held, keep the wrist inside the region from
    # which every pressing finger can reach its key.
    XLO, XHI = np.full(N, -np.inf), np.full(N, np.inf)
    ZLO, ZHI = np.full(N, -np.inf), np.full(N, np.inf)
    for n in fingered:
        f = n.finger
        ia, ib = _idx(times, t0, n.start - KEY_DOWN_TIME - 0.02), _idx(times, t0, VOFF[n.id])
        vx = G.key_x(n.pitch) - nat_x[f]
        vz = G.contact_z(n.pitch, f) - REST_TIPS[f][2]
        XLO[ia:ib] = np.maximum(XLO[ia:ib], vx - REACH_X[f])
        XHI[ia:ib] = np.minimum(XHI[ia:ib], vx + REACH_X[f])
        ZLO[ia:ib] = np.maximum(ZLO[ia:ib], vz - REACH_Z[f])
        ZHI[ia:ib] = np.minimum(ZHI[ia:ib], vz + REACH_Z[f])
    for _ in range(3):
        X = _smooth(X, 0.03)
        Z = _smooth(Z, 0.05)
        X = _clamp_feasible(X, XLO, XHI)
        Z = _clamp_feasible(Z, ZLO, ZHI)
    # splay: how much wider than natural the simultaneously engaged fingers are
    SPL = np.zeros(N)
    from .midi_io import group_onsets
    for g in group_onsets(fingered):
        on = g[0].start
        active = [n for n in fingered if n.start <= on + 0.035 and VOFF[n.id] > on + 0.03]
        if len(active) > 1:
            lo = min(active, key=lambda n: n.pitch)
            hi = max(active, key=lambda n: n.pitch)
            need = (G.key_x(hi.pitch) - G.key_x(lo.pitch)) - abs(nat_x[hi.finger] - nat_x[lo.finger])
            if need > 0:
                end_t = max(VOFF[n.id] for n in active)
                i0, i1 = _idx(times, t0, on - 0.15), _idx(times, t0, end_t + 0.05)
                SPL[i0:i1] = np.maximum(SPL[i0:i1], min(0.35, need / 10.0))
    SPL = _smooth(SPL, 0.12)
    LIFT = np.zeros(N)
    for k in range(1, len(onsets)):
        gap = onsets[k] - max([VOFF[n.id] for n in fingered if n.start <= onsets[k - 1] + 1e-6] + [onsets[k - 1]])
        if gap > 1.0:  # breathe: lift the hand between phrases
            tp = onsets[k] - gap
            ia2, ib2 = _idx(times, t0, tp + 0.25), _idx(times, t0, onsets[k] - 0.25)
            if ib2 > ia2:
                ph = np.linspace(0, math.pi, ib2 - ia2)
                LIFT[ia2:ib2] = np.maximum(LIFT[ia2:ib2], np.sin(ph) * min(1.6, 0.6 * gap))
    Z = np.clip(Z, 19.5, 27.5)

    # ---- 2. per-finger fingertip targets ---------------------------------------
    by_f = {f: [n for n in fingered if n.finger == f] for f in range(1, 6)}
    TX = np.zeros((N, 5))
    TZ = np.zeros((N, 5))
    LY = np.zeros((N, 5))       # lift relative to key surface (contact point height)
    SURF = np.zeros((N, 5))     # surface height used for the lift
    PRESSING = np.full((N, 5), -1, dtype=np.int32)
    activity = np.zeros((N, 5))
    for i, f in enumerate(range(1, 6)):
        r = G.FINGERS[f].tip_pad
        idle_x = X + nat_x[f]
        idle_z = Z + REST_TIPS[f][2]
        tx, tz = idle_x.copy(), idle_z.copy()
        lift = np.full(N, G.HOVER)
        contact_mask = np.zeros(N, bool)
        surf_contact = np.zeros(N)
        lst = by_f[f]
        prev_release_end = -1e9
        for j, n in enumerate(lst):
            nxt = lst[j + 1].start if j + 1 < len(lst) else None
            on, off = press_interval(n, nxt, VOFF[n.id])
            press[n.id] = (on, off)
            ih = _idx(times, t0, on)
            cx, cz = _contact_point(n, f, float(X[min(ih, N - 1)]), s)
            avail = on - prev_release_end
            prep = float(np.clip(avail * 0.85, 0.06, 0.38))
            desc = min(0.085, prep * 0.55)
            touch = on - KEY_DOWN_TIME
            a0, d0 = on - prep, touch - desc
            ia0, id0 = _idx(times, t0, a0), _idx(times, t0, d0)
            j0 = min(ia0, N - 1)
            rsx, rsz = tx[j0] - X[j0], tz[j0] - Z[j0]     # start, relative to the moving hand
            if id0 > ia0:
                w = _ease((times[ia0:id0] - a0) / max(d0 - a0, 1e-6))
                tx[ia0:id0] = (X[ia0:id0] + rsx) * (1 - w) + cx * w
                tz[ia0:id0] = (Z[ia0:id0] + rsz) * (1 - w) + cz * w
            r_end = off + KEY_UP_TIME + LIFT_TIME
            ir = _idx(times, t0, r_end)
            iup_ = _idx(times, t0, off + KEY_UP_TIME)
            tx[id0:iup_] = cx
            tz[id0:iup_] = cz
            # vertical profile
            itouch, ion, ioff = _idx(times, t0, touch), _idx(times, t0, on), _idx(times, t0, off)
            iup = _idx(times, t0, off + KEY_UP_TIME)
            depth = G.key_depth_at_contact(n.pitch)
            if itouch > id0:
                w = (times[id0:itouch] - d0) / max(touch - d0, 1e-6)
                lift[id0:itouch] = G.HOVER * (1 - np.clip(w, 0, 1) ** 1.6)
            tt = times[itouch:iup]
            lift[itouch:iup] = -key_depth_profile(tt, on, off, depth)
            if ir > iup:
                w = _ease((times[iup:ir] - (off + KEY_UP_TIME)) / LIFT_TIME)
                lift[iup:ir] = G.HOVER * w
            contact_mask[id0:ir] = True
            surf_contact[id0:ir] = G.key(n.pitch).top
            PRESSING[ion:ioff, i] = n.id
            activity[itouch:iup, i] = 1.0
            # return towards idle after release (next note overwrites from its a0)
            back = 0.3
            r0 = off + KEY_UP_TIME
            ib = _idx(times, t0, r0 + back)
            if ib > iup_:
                w = _ease((times[iup_:ib] - r0) / back)
                jr = min(iup_, N - 1)
                rcx, rcz = cx - X[jr], cz - Z[jr]           # keep the finger attached to the hand
                tx[iup_:ib] = X[iup_:ib] + rcx + (nat_x[f] - rcx) * w
                tz[iup_:ib] = Z[iup_:ib] + rcz + (REST_TIPS[f][2] - rcz) * w
            prev_release_end = off + KEY_UP_TIME
        # surface under idle/approach positions (hover above black keys when needed)
        surf = np.where(contact_mask, surf_contact, 0.0)
        idle_pts = ~contact_mask
        if idle_pts.any():
            surf_idle = np.array([G.surface_top(x, z) for x, z in zip(tx[idle_pts][::4], tz[idle_pts][::4])])
            surf_full = np.repeat(surf_idle, 4)[: idle_pts.sum()]
            surf[idle_pts] = _smooth_masked(surf_full)
        TX[:, i], TZ[:, i], LY[:, i], SURF[:, i] = tx, tz, lift + r, surf

    # ---- 2b. keep neighbouring fingertips apart (order + minimum gap) ----------
    pressing_any = PRESSING >= 0
    contact = np.zeros((N, 5), bool)
    for i in range(5):
        contact[:, i] = LY[:, i] < G.FINGERS[i + 1].tip_pad + 0.25   # at/near key surface
    for _ in range(4):
        for i in range(1, 4):            # pairs 2-3, 3-4, 4-5 (thumb may legitimately cross)
            a, b = i, i + 1
            sep = 0.82 * (G.FINGERS[a + 1].tip_pad + G.FINGERS[b + 1].tip_pad) + 0.55
            gap = s * (TX[:, b] - TX[:, a])
            short = np.maximum(0.0, sep - gap)
            if not short.any():
                continue
            fa, fb = contact[:, a], contact[:, b]
            move_a = np.where(fa & ~fb, 0.0, np.where(fb & ~fa, 1.0, np.where(fa & fb, 0.0, 0.5)))
            move_b = np.where(fa & ~fb, 1.0, np.where(fb & ~fa, 0.0, np.where(fa & fb, 0.0, 0.5)))
            TX[:, a] -= s * short * move_a
            TX[:, b] += s * short * move_b
        # thumb vs index: when both hover, keep them apart; when crossing, lift the hovering one
        d = np.hypot(TX[:, 0] - TX[:, 1], TZ[:, 0] - TZ[:, 1])
        lim = 0.82 * (G.FINGERS[1].tip_pad + G.FINGERS[2].tip_pad) + 0.4
        close = d < lim
        idle0, idle1 = ~contact[:, 0], ~contact[:, 1]
        LY[:, 0] += np.where(close & idle0 & ~idle1, 0.9, 0.0) * (lim - d).clip(0) / lim * 2
        LY[:, 1] += np.where(close & idle1 & ~idle0, 0.9, 0.0) * (lim - d).clip(0) / lim * 2
        both = close & idle0 & idle1
        TX[:, 0] -= s * np.where(both, (lim - d).clip(0) * 0.5, 0.0)
        TX[:, 1] += s * np.where(both, (lim - d).clip(0) * 0.5, 0.0)
    for i in range(5):
        TX[:, i] = _smooth(TX[:, i], 0.03) * (~contact[:, i]) + TX[:, i] * contact[:, i]

    # ---- 3. wrist orientation & height ------------------------------------------
    act = np.stack([_smooth(activity[:, i], 0.15) for i in range(5)], -1)
    yaw = -0.42 * np.arctan2(X - BODY_X, 58.0) + s * 0.04
    roll = s * (0.10 * act[:, 0] - 0.13 * act[:, 4])
    pitch = 0.05 + 0.03 * act.mean(axis=1)
    PY = G.WRIST_HEIGHT + LIFT - 0.35 * act.mean(axis=1) + 0.25 * np.maximum(0, 1 - Z / 24.0)
    pose = np.zeros((N, FLOATS_PER_HAND), dtype=np.float64)
    pose[:, 0], pose[:, 1], pose[:, 2] = X, PY, Z
    pose[:, 3], pose[:, 4], pose[:, 5], pose[:, 6] = yaw, pitch, roll, SPL
    R = euler_yxz(yaw, pitch, roll)
    # X/Z were solved for an unrotated hand; rotate about the fingertip line
    # instead of the wrist so the fingers stay where they were planned.
    A = np.array([0.0, -4.5, REST_TIPS[3][2]])
    comp = A[None, :] - np.einsum("nij,j->ni", R, A)
    pose[:, 0] += comp[:, 0]
    pose[:, 2] += comp[:, 2]
    W = pose[:, 0:3]
    targets = np.zeros((N, 5, 3))
    for i, f in enumerate(range(1, 6)):
        T = np.stack([TX[:, i], SURF[:, i] + LY[:, i], TZ[:, i]], -1)
        targets[:, i] = T
        B = W + np.einsum("nij,nj->ni", R, finger_base_local(f, s, SPL))
        a, t1, t2, t3 = solve_finger_ik(f, s, R, B, T)
        pose[:, 7 + 4 * i: 11 + 4 * i] = np.stack([a, t1, t2, t3], -1)
    return HandBake(hand, pose.astype(np.float32), targets.astype(np.float32), PRESSING, visible)


def _clamp_feasible(v, lo, hi):
    lo2, hi2 = np.minimum(lo, hi), np.maximum(lo, hi)
    with np.errstate(invalid="ignore"):
        mid = 0.5 * (lo + hi)
    empty = lo > hi                     # impossible stretch: centre between limits
    out = np.clip(v, lo2, hi2)
    out[empty] = mid[empty]
    return out


def _ffill(a, default):
    a = a.copy()
    valid = ~np.isnan(a)
    if not valid.any():
        return np.full_like(a, default)
    idx = np.where(valid, np.arange(len(a)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = a[idx]
    first = np.argmax(valid)
    out[:first] = a[first]
    return out


def _smooth_masked(a):
    if len(a) < 5:
        return a
    n = 4
    k = np.ones(2 * n + 1) / (2 * n + 1)
    return np.maximum(a, np.convolve(np.pad(a, (n, n), mode="edge"), k, mode="valid"))


def bake_to_bytes(b: Bake) -> bytes:
    """Interleaved float32: per sample [RH pose (27), LH pose (27)]."""
    arr = np.concatenate([b.hands[RIGHT].pose, b.hands[LEFT].pose], axis=1).astype("<f4")
    return arr.tobytes()
