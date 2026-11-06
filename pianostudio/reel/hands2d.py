"""Illustrated hands for the reel: articulated choreography + overhead drawing.

Model (right hand; units are white-key widths W, x to the right, y towards the
player, z up from the white-key surface):

* Each finger has three phalanges with adult proportions (thumb: two) and a
  knuckle (MCP) at a fixed place on the palm, raised above the keys. The DIP
  joint bends with the PIP joint (DIP = 0.75 x PIP), as real fingers do.
* Every frame, a planar inverse-kinematics solve bends each finger so its tip
  reaches its target in 3D. The overhead camera then shows the curl as
  foreshortening: a hovering finger curls and looks shorter, and a finger
  reaching up to a black key straightens instead of stretching.
* The palm is not fixed. Each frame, its position, yaw and finger spread are
  fitted (weighted least squares) to the keys the hand is about to play, is
  playing, or just played, with a prior that keeps it calm. The whole hand
  travels to the music and fingers stay roughly parallel instead of fanning
  out from a fixed point. Resting fingers ride along in the hand frame,
  curved, hovering above the keys.
* Timing: each finger reaches its key early, the key is fully down exactly at
  the note onset, and the finger releases before a leap so it never teleports.
  Everything is a pure function of time, so frames can render in parallel.

Drawing (cairo): tapered jointed fingers with knuckle highlights,
slope-dependent shading, foreshortened nails and creases. The palm is drawn
over the finger bases, with knuckle bumps and faint tendons. The thumb comes
off the side of the palm. A soft drop shadow sits under the hand.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cairo
import cv2
import numpy as np

from ..geometry import SPAN_COMFORT, WHITE_W as KEY_CM, is_black
from ..midi_io import group_onsets
from ..model import LEFT, RIGHT, Note
from .layout import Layout

# ----------------------------------------------------------------- hand model (RH)
# natural fingertip position in the hand frame (relative to the middle knuckle, z ignored)
NAT = {1: (-2.02, -1.65), 2: (-0.98, -3.00), 3: (0.0, -3.09), 4: (0.95, -3.00), 5: (1.85, -2.29)}
# knuckles (MCP) on the palm: (x, y, z)
MCP = {1: (-1.26, 0.66, 0.80), 2: (-0.84, 0.06, 1.62), 3: (0.0, 0.0, 1.68), 4: (0.80, 0.08, 1.60),
       5: (1.52, 0.34, 1.42)}
# phalanx lengths (proximal, middle, distal) from adult hand proportions / 2.35 cm
PHAL = {1: (1.42, 1.12, 0.0), 2: (1.96, 1.11, 0.81), 3: (2.13, 1.28, 0.85), 4: (2.00, 1.19, 0.85),
        5: (1.62, 0.85, 0.77)}
# widths at MCP, PIP, DIP, tip
WID = {1: (0.92, 0.86, 0.80, 0.80), 2: (0.80, 0.74, 0.66, 0.66), 3: (0.82, 0.76, 0.68, 0.68),
       4: (0.76, 0.70, 0.62, 0.62), 5: (0.66, 0.60, 0.54, 0.54)}
DEV = {1: 2.4, 2: 1.2, 3: 1.05, 4: 1.0, 5: 1.3}   # max comfortable lateral reach from the natural spot
# palm outline in the hand frame (x, y); y grows towards the wrist
PALM = [(-1.18, 0.18), (-0.84, -0.12), (0.0, -0.18), (0.80, -0.10), (1.50, 0.12), (1.84, 0.50),
        (1.88, 1.40), (1.70, 2.70), (1.40, 3.90), (1.50, 5.4), (1.62, 8.0), (-1.10, 8.0),
        (-0.98, 5.4), (-0.92, 3.90), (-1.20, 2.95), (-1.60, 1.95), (-1.56, 1.15), (-1.34, 0.55)]
HOVER = 0.50          # fingertip height of a resting finger
# how much hand spread moves each resting finger (outer fingers reach, middle fingers stay curled in)
SPREAD_K = {1: 1.0, 2: 0.45, 3: 0.3, 4: 0.45, 5: 1.0}
BLACK_Z = 0.45        # black-key top above the white-key surface
KEY_DIP = 0.22        # key travel shown by the pressing fingertip
CURL = 0.60           # DIP flex as a fraction of PIP flex

PRE = 0.065           # s: finger descends during [onset-PRE, onset]; key bottoms out at the onset
REL = 0.07            # s: release / lift time
SAMPLE_HZ = 240


def _smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def _minjerk(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * u * (10 - 15 * u + 6 * u * u)


def contact_y(lay: Layout, pitch: int, finger: int) -> float:
    """Where a fingertip meets the key (screen y): fingers play just in front of the
    black keys, the thumb and pinky nearer the front edge; black keys near their front."""
    W = lay.white_w
    if is_black(pitch):
        front = lay.key_top + lay.black_h
        return front - (0.30 if finger == 1 else 0.55) * W
    return lay.key_bottom - {1: 0.62, 2: 1.95, 3: 2.05, 4: 1.95, 5: 1.25}[finger] * W


@dataclass
class Press:
    on: float
    off: float        # visual release
    x: float
    y: float
    z: float
    note: Note


@dataclass
class FingerState:
    x: float          # fingertip (screen px)
    y: float
    press: float      # 0 = hovering, 1 = key fully down
    pitch: int | None
    z: float = 0.0    # fingertip height (W)
    joints: list = field(default_factory=list)   # [(x, y, z) screen px / W] knuckle .. tip


@dataclass
class HandState:
    side: int         # +1 right hand, -1 left hand
    palm_x: float
    palm_y: float
    fingers: dict     # f -> FingerState
    theta: float = 0.0
    spread: float = 1.0
    visible: float = 1.0


class _IK:
    """Planar finger IK by table lookup: PIP angle b from the knuckle->tip distance."""

    def __init__(self, L):
        self.L = L
        b = np.linspace(0.0, 1.85, 400)
        L1, L2, L3 = L
        vx = L1 + L2 * np.cos(b) + L3 * np.cos((1 + CURL) * b)
        vy = -(L2 * np.sin(b) + L3 * np.sin((1 + CURL) * b))
        r = np.hypot(vx, vy)
        order = np.argsort(r)
        self.r, self.b = r[order], b[order]
        self.phi = np.arctan2(vy, vx)[order]
        self.rmax = float(r.max())

    def solve(self, d: float, dz: float):
        """d: horizontal distance knuckle->tip (along the finger), dz: tip z - knuckle z.
        Returns joint (h, z) offsets for PIP, DIP, tip in the finger plane."""
        r = math.hypot(d, dz)
        scale = 1.0
        if r > self.rmax:
            scale = r / self.rmax
            r = self.rmax
        b = float(np.interp(r, self.r, self.b))
        phi = float(np.interp(r, self.r, self.phi))
        a = math.atan2(dz, d) - phi
        L1, L2, L3 = (l * scale for l in self.L)
        j1 = (L1 * math.cos(a), L1 * math.sin(a))
        j2 = (j1[0] + L2 * math.cos(a - b), j1[1] + L2 * math.sin(a - b))
        if L3 > 0:
            j3 = (j2[0] + L3 * math.cos(a - (1 + CURL) * b), j2[1] + L3 * math.sin(a - (1 + CURL) * b))
            return [j1, j2, j3]
        return [j1, j2]


class HandTrack:
    def __init__(self, notes: list[Note], hand: str, lay: Layout, duration: float, home_x: float | None = None):
        self.hand = hand
        self.side = 1 if hand == RIGHT else -1
        self.lay = lay
        self.W = W = lay.white_w
        self.notes = sorted(notes, key=lambda n: (n.start, n.pitch))
        self.duration = duration
        s = self.side
        self.ik = {f: _IK(PHAL[f]) for f in range(1, 6)}
        self.rest_y = lay.key_bottom + 1.05 * W   # middle knuckle at rest, just behind the key slip

        # ---- presses per finger ------------------------------------------
        self.presses: dict[int, list[Press]] = {f: [] for f in range(1, 6)}
        for n in self.notes:
            f = n.finger or 3
            z = BLACK_Z if is_black(n.pitch) else 0.0
            self.presses[f].append(Press(n.start, n.end, lay.x_center(n.pitch), contact_y(lay, n.pitch, f), z, n))
        onsets_by_pitch: dict[int, list[float]] = {}
        for n in self.notes:
            onsets_by_pitch.setdefault(n.pitch, []).append(n.start)
        for f, ps in self.presses.items():
            ps.sort(key=lambda p: p.on)
            for i, p in enumerate(ps):
                off = p.note.end
                if i + 1 < len(ps):
                    nxt = ps[i + 1]
                    dist = abs(nxt.x - p.x) + abs(nxt.y - p.y)
                    # leave time to travel to the next key (released early, as before a leap)
                    travel = 0.0 if dist < 1.0 else min(0.24, max(0.08, dist / (14.0 * W)))
                    off = min(off, nxt.on - PRE - REL - travel - 0.005)
                # another finger re-striking this key: lift before it descends
                later = [o for o in onsets_by_pitch[p.note.pitch] if o > p.on + 1e-6]
                if later:
                    off = min(off, later[0] - PRE - 0.01)
                p.off = max(off, p.on + 0.04)
        allp = sorted(((p.on, f, p) for f, ps in self.presses.items() for p in ps), key=lambda e: e[0])
        ons = [e[0] for e in allp]
        for f, ps in self.presses.items():
            for p in ps:
                j = int(np.searchsorted(ons, p.on + 1e-6))
                while j < len(allp) and allp[j][0] <= p.off + 0.25:
                    qon, g, q = allp[j]
                    j += 1
                    if g == f:
                        continue
                    comfy = SPAN_COMFORT[tuple(sorted((f, g)))] / KEY_CM * W * 1.05
                    span = abs(q.x - p.x)
                    if span > comfy:
                        travel = min(0.20, max(0.08, (span - comfy) / (12.0 * W)))
                        p.off = max(p.on + 0.06, min(p.off, qon - PRE - travel))
        self._on = {f: np.array([p.on for p in ps]) for f, ps in self.presses.items()}
        self.visual_end = {p.note.id: p.off for ps in self.presses.values() for p in ps}

        # ---- palm prior: smoothed hand position per onset group -------------
        groups = group_onsets(self.notes) if self.notes else []
        times, raw = [], []
        for g in groups:
            t = g[0].start
            xs = [lay.x_center(n.pitch) - s * NAT[n.finger or 3][0] * W for n in g]
            times.append(t)
            raw.append(sum(xs) / len(xs))
        n_s = int(math.ceil((duration + 1.0) * SAMPLE_HZ)) + 1
        if home_x is None:
            home_x = raw[0] if raw else lay.width * (0.68 if s > 0 else 0.32)
        px = np.full(n_s, float(home_x))
        for t, x in zip(times, raw):
            px[int(t * SAMPLE_HZ):] = x
        px = cv2.GaussianBlur(px.reshape(1, -1).astype(np.float64), (0, 0), sigmaX=0.45 * SAMPLE_HZ).ravel()
        # shift earlier so the prior anticipates rather than lags
        lead = int(0.25 * SAMPLE_HZ)
        px = np.concatenate([px[lead:], np.full(lead, px[-1])])
        self.palm_samples = px
        self.home_x = home_x

    # ------------------------------------------------------------------
    def palm_prior(self, t: float) -> float:
        i = t * SAMPLE_HZ
        i0 = int(min(max(math.floor(i), 0), len(self.palm_samples) - 2))
        a = min(max(i - i0, 0.0), 1.0)
        return float(self.palm_samples[i0] * (1 - a) + self.palm_samples[i0 + 1] * a)

    def finger_at(self, f: int, t: float):
        """-> (key_x, key_y, key_z, rest_weight, press, pitch, fit_weight).
        rest_weight 0 = at/near its key, 1 = relaxed in the hand frame.
        fit_weight: how strongly this finger's key steers the palm (wider window)."""
        ps = self.presses[f]
        if not ps:
            return None, None, 0.0, 1.0, 0.0, None, 0.0
        on = self._on[f]
        i = int(np.searchsorted(on - PRE, t, side="right")) - 1
        cur = ps[i] if i >= 0 else None
        nxt = ps[i + 1] if i + 1 < len(ps) else None
        fit_prev = 0.0 if cur is None else (1.0 if t <= cur.off else 1.0 - float(_smooth((t - cur.off) / 0.18)))
        fit_next = 0.0 if nxt is None else float(_smooth((t - (nxt.on - PRE - 0.45)) / 0.36))
        fit_w = max(fit_prev, fit_next)
        if cur is not None and t <= cur.off + REL:
            if t < cur.on:
                pr = float(_smooth((t - (cur.on - PRE)) / PRE))
            elif t <= cur.off:
                pr = 1.0
            else:
                pr = 1.0 - float(_smooth((t - cur.off) / REL))
            return cur.x, cur.y, cur.z, 0.0, pr, cur.note.pitch, 1.0
        t_free = cur.off + REL if cur else -1e9
        t_need = nxt.on - PRE - 0.015 if nxt else 1e9
        if cur and nxt:
            G = max(t_need - t_free, 1e-3)
            u = float(_minjerk((t - t_free) / G))
            kx, ky, kz = (cur.x + (nxt.x - cur.x) * u, cur.y + (nxt.y - cur.y) * u, cur.z + (nxt.z - cur.z) * u)
            relax = min(1.0, max(0.0, (G - 0.20) / 0.35))
            # a released finger lifts and rides with the hand straight away
            w = relax * float(_smooth((t - t_free) / 0.22) * _smooth((t_need - 0.10 - t) / 0.24))
        elif nxt:
            kx, ky, kz = nxt.x, nxt.y, nxt.z
            w = float(_smooth((t_need - 0.06 - t) / 0.30))
        else:
            kx, ky, kz = cur.x, cur.y, cur.z
            w = float(_smooth((t - t_free) / 0.30))
        return kx, ky, kz, w, 0.0, None, fit_w

    def _fit_palm(self, raw, t):
        """Weighted least squares for palm (x, y), yaw theta and finger spread s, in W units:
        tip_f ~ P + R(theta) diag(s, 1) NAT_f (theta linearised), with a calm prior."""
        W, s = self.W, self.side
        px0, py0 = self.palm_prior(t) / W, self.rest_y / W
        # unknowns u = [px, py, theta, s]
        A, b = [], []
        lam = (0.35, 1.2, 9.0, 5.0)
        for k, (val, l) in enumerate(zip((px0, py0, 0.0, 1.0), lam)):
            row = [0.0] * 4
            row[k] = math.sqrt(l)
            A.append(row)
            b.append(math.sqrt(l) * val)
        for f, r in raw.items():
            kx, ky, kz, w, pr, pitch, fw = r
            if kx is None or fw <= 1e-4:
                continue
            e = math.sqrt(fw)
            nx, ny = s * NAT[f][0], NAT[f][1]
            if f == 1:
                e *= 0.6          # the thumb is the most mobile finger
            A.append([e, 0.0, -e * ny, e * nx])
            b.append(e * kx / W)
            A.append([0.0, e, e * nx, 0.0])
            b.append(e * (ky / W - ny))       # tip_y = py + theta * nx + ny
        sol, *_ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
        px, py, th, sp = sol
        th = float(np.clip(th, -0.30, 0.30))
        sp = float(np.clip(sp, 0.85, 1.65))
        py = float(np.clip(py, py0 - 1.6, py0 + 0.4))
        return px * W, py * W, th, sp

    def frame_point(self, P, th, sp, x, y, spread_k=1.0):  # noqa: D401
        """Hand-frame point (W units, RH convention) -> screen px."""
        W, s = self.W, self.side
        xx = s * x * (1 + (sp - 1) * spread_k)
        c, si = math.cos(th), math.sin(th)
        return P[0] + (c * xx - si * y) * W, P[1] + (si * xx + c * y) * W

    def state(self, t: float) -> HandState:
        W, s = self.W, self.side
        raw = {f: self.finger_at(f, t) for f in range(1, 6)}
        px, py, th, sp = self._fit_palm(raw, t)
        P = (px, py)
        fingers = {}
        idle = 0.04 * W
        for f in range(1, 6):
            kx, ky, kz, w, pr, pitch, fw = raw[f]
            nxf, nyf = NAT[f]
            fx, fy = self.frame_point(P, th, sp, nxf, nyf + (0.42 if f > 1 else 0.15), spread_k=SPREAD_K[f])
            fx += idle * math.sin(t * 1.3 + f * 1.7)          # a little life in resting fingers
            fy += idle * 0.6 * math.sin(t * 0.9 + f * 2.3)
            if kx is None:
                x, y, z_key = fx, fy, 0.0
            else:
                x, y = kx + (fx - kx) * w, ky + (fy - ky) * w
                z_key = kz * (1 - w)
            z = z_key - KEY_DIP * pr + (1 - pr) * (0.28 + (HOVER - 0.28) * w)
            fingers[f] = FingerState(float(x), float(y), pr, pitch, float(z))
        self._separate(fingers, raw)
        # knuckles from the palm, then IK for every finger
        for f in range(1, 6):
            mx, my, mz = MCP[f]
            kxs, kys = self.frame_point(P, th, sp, mx, my, spread_k=0.3)
            fs = fingers[f]
            dx, dy = fs.x - kxs, fs.y - kys
            d = math.hypot(dx, dy) / W
            ux, uy = (dx / (d * W), dy / (d * W)) if d > 1e-6 else (0.0, -1.0)
            pts = self.ik[f].solve(max(d, 1e-3), fs.z - mz)
            hs, prev = [], 0.0
            for k_, (h, zz) in enumerate(pts):          # keep the projection advancing towards the tip
                lo = prev + 0.12 * PHAL[f][k_] * max(0.3, 1 - prev / max(d, 1e-3))
                h = min(max(h, lo), d)
                hs.append((h, zz))
                prev = h
            joints = [(kxs, kys, mz)] + [(kxs + ux * h * W, kys + uy * h * W, mz + zz) for h, zz in hs]
            joints[-1] = (fs.x, fs.y, fs.z)       # the tip lands exactly on its target
            fs.joints = joints
        return HandState(s, px, py, fingers, th, sp)

    def _separate(self, fingers, raw):
        """Relaxed fingertips 2-5 keep their order and spacing (pressing fingers never move)."""
        s, gap = self.side, 0.74 * self.W
        mob = {f: (1.0 - fingers[f].press) ** 3 * raw[f][3] for f in range(1, 6)}
        order = [2, 3, 4, 5]
        for _ in range(8):
            for a, b in zip(order, order[1:]):
                fa, fb = fingers[a], fingers[b]
                d = (fb.x - fa.x) * s
                if d >= gap:
                    continue
                ma, mb = mob[a], mob[b]
                tot = ma + mb
                if tot <= 1e-6:
                    continue
                need = (gap - d) * min(1.0, tot / 0.5)
                fa.x -= s * need * ma / tot
                fb.x += s * need * mb / tot


# ======================================================================= drawing
@dataclass
class HandStyle:
    skin: tuple = (0.965, 0.800, 0.665)
    skin_light: tuple = (1.0, 0.895, 0.80)
    skin_shade: tuple = (0.855, 0.640, 0.515)
    outline: tuple = (0.470, 0.300, 0.230)
    nail: tuple = (0.995, 0.890, 0.860)
    crease: tuple = (0.700, 0.470, 0.370)
    shadow_alpha: float = 0.24
    outline_w: float = 2.2

    @staticmethod
    def from_hex(skin: str | None = None, outline: str | None = None) -> "HandStyle":
        st = HandStyle()
        if skin:
            r, g, b = hex_rgb(skin)
            st.skin = (r, g, b)
            st.skin_light = tuple(min(1.0, c * 1.07 + 0.05) for c in (r, g, b))
            st.skin_shade = tuple(c * 0.86 for c in (r, g, b))
            st.crease = tuple(c * 0.72 for c in (r, g, b))
        if outline:
            st.outline = hex_rgb(outline)
        return st


def hex_rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def _mix(a, b, t):
    return tuple(a[i] * (1 - t) + b[i] * t for i in range(3))


def _catmull(ctx, pts, closed=True):
    n = len(pts)
    ctx.move_to(*pts[0])
    rng = range(n) if closed else range(n - 1)
    for i in rng:
        p0, p1, p2, p3 = pts[(i - 1) % n], pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        ctx.curve_to(c1[0], c1[1], c2[0], c2[1], p2[0], p2[1])
    if closed:
        ctx.close_path()


class HandGeometry:
    """Resolves a HandState into drawable shapes (pixel coordinates)."""

    def base_dir(self, f):
        return self.thumb_in if f == 1 else self.wrist

    def __init__(self, st: HandState, W: float):
        self.st, self.W, self.s = st, W, st.side
        c, si = math.cos(st.theta), math.sin(st.theta)

        def fp(x, y):
            xx = st.side * x
            return st.palm_x + (c * xx - si * y) * W, st.palm_y + (si * xx + c * y) * W

        self.fp = fp
        self.palm = [fp(x, y) for x, y in PALM]
        self.fingers = {}
        for f in range(1, 6):
            fs = st.fingers[f]
            js = fs.joints
            wid = WID[f]
            if f == 1:
                wid = (wid[0], wid[1], wid[3])
            self.fingers[f] = (js, [w * W for w in wid], fs.press)
        self.wrist = (-si, c)      # unit vector along the hand towards the wrist
        k1 = self.fingers[1][0][0]
        inner = fp(-0.55, 1.75)
        dx, dy = inner[0] - k1[0], inner[1] - k1[1]
        L = math.hypot(dx, dy) or 1.0
        self.thumb_in = (dx / L, dy / L)


def _finger_outline(ctx, js, ws, W, base_ext=0.55, wrist=(0.0, 1.0)):
    """Tapered polyline through the joints with a rounded fingertip."""
    pts = [(j[0], j[1]) for j in js]
    # extend the base under the palm, towards the wrist (never out of the far side of the hand)
    base = (pts[0][0] + wrist[0] * base_ext * W, pts[0][1] + wrist[1] * base_ext * W)
    pts = [base] + pts
    ws = [ws[0]] + list(ws)
    # narrow the shafts between the joints (from the knuckle on)
    P2, W2 = [pts[0], pts[1]], [ws[0], ws[1]]
    for i in range(2, len(pts)):
        m = ((pts[i - 1][0] + pts[i][0]) / 2, (pts[i - 1][1] + pts[i][1]) / 2)
        if math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]) > 0.25 * W:
            P2.append(m)
            W2.append((ws[i - 1] + ws[i]) / 2 * 0.90)
        P2.append(pts[i])
        W2.append(ws[i])
    pts, ws = P2, W2
    dirs = []
    for i in range(len(pts) - 1):
        ddx, ddy = pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]
        l = math.hypot(ddx, ddy)
        dirs.append((ddx / l, ddy / l) if l > 1e-3 else (dirs[-1] if dirs else (0.0, -1.0)))
    normals = []
    for i in range(len(pts)):
        if i == 0:
            d = dirs[0]
        elif i == len(pts) - 1:
            d = dirs[-1]
        else:
            a, b = dirs[i - 1], dirs[i]
            d = (a[0] + b[0], a[1] + b[1])
            l = math.hypot(*d)
            d = (d[0] / l, d[1] / l) if l > 1e-6 else b
        normals.append((-d[1], d[0]))
    left = [(p[0] + n[0] * w / 2, p[1] + n[1] * w / 2) for p, n, w in zip(pts, normals, ws)]
    right = [(p[0] - n[0] * w / 2, p[1] - n[1] * w / 2) for p, n, w in zip(pts, normals, ws)]
    tip = pts[-1]
    r = ws[-1] / 2
    n = normals[-1]
    ctx.move_to(*left[0])
    for i in range(1, len(left)):
        # smooth the sides through the joints
        p0, p1 = left[i - 1], left[i]
        ctx.curve_to(p0[0] + (p1[0] - p0[0]) * 0.35, p0[1] + (p1[1] - p0[1]) * 0.35,
                     p0[0] + (p1[0] - p0[0]) * 0.65, p0[1] + (p1[1] - p0[1]) * 0.65, p1[0], p1[1])
    a0 = math.atan2(n[1], n[0])
    ctx.arc_negative(tip[0], tip[1], r, a0, a0 - math.pi)
    for i in range(len(right) - 2, -1, -1):
        p0, p1 = right[i + 1], right[i]
        ctx.curve_to(p0[0] + (p1[0] - p0[0]) * 0.35, p0[1] + (p1[1] - p0[1]) * 0.35,
                     p0[0] + (p1[0] - p0[0]) * 0.65, p0[1] + (p1[1] - p0[1]) * 0.65, p1[0], p1[1])
    ctx.close_path()
    return pts, normals, dirs


def _fill_finger(ctx, js, ws, press, W, style: HandStyle, wrist=(0.0, 1.0)):
    pts, normals, dirs = _finger_outline(ctx, js, ws, W, wrist=wrist)
    path = ctx.copy_path()
    # lengthwise light: knuckle tops (high) catch the light, steep phalanges turn away
    k, tip = pts[1], pts[-1]
    g = cairo.LinearGradient(pts[0][0], pts[0][1], tip[0], tip[1])
    total = math.hypot(tip[0] - pts[0][0], tip[1] - pts[0][1]) or 1.0

    def frac(p):
        return min(1.0, max(0.0, ((p[0] - pts[0][0]) * (tip[0] - pts[0][0]) + (p[1] - pts[0][1]) * (tip[1] - pts[0][1])) / total ** 2))

    sk, li, sh = style.skin, style.skin_light, style.skin_shade
    g.add_color_stop_rgb(0.0, *sk)
    for i in range(1, len(js) - 1):           # PIP / DIP: highlight on the joint, shade on the steep side
        f = frac((js[i][0], js[i][1]))
        g.add_color_stop_rgb(max(0.0, f - 0.06), *_mix(sk, li, 0.35))
        g.add_color_stop_rgb(f, *li)
        z0, z1 = js[i][2], js[i + 1][2]
        hlen = math.hypot(js[i + 1][0] - js[i][0], js[i + 1][1] - js[i][1]) / W
        steep = min(1.0, abs(z1 - z0) / max(hlen + abs(z1 - z0), 1e-3))
        f2 = frac((js[i + 1][0], js[i + 1][1]))
        g.add_color_stop_rgb(min(1.0, (f + f2) / 2), *_mix(sk, sh, 0.15 + 0.55 * steep))
    g.add_color_stop_rgb(1.0, *_mix(sk, sh, 0.25 + 0.25 * press))
    ctx.set_source(g)
    ctx.fill()
    # cross shading for roundness (darker edges, light from the upper left)
    ctx.append_path(path)
    ctx.save()
    ctx.clip()
    for i in range(len(pts) - 1):
        p0, p1 = pts[i], pts[i + 1]
        n = normals[i]
        w = ws[min(i + 1, len(ws) - 1)] if i else ws[0]
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        cg = cairo.LinearGradient(mx + n[0] * w * 0.6, my + n[1] * w * 0.6, mx - n[0] * w * 0.6, my - n[1] * w * 0.6)
        cg.add_color_stop_rgba(0.0, *sh, 0.55)
        cg.add_color_stop_rgba(0.3, *sh, 0.0)
        cg.add_color_stop_rgba(0.62, *li, 0.18)
        cg.add_color_stop_rgba(1.0, *sh, 0.6)
        ctx.set_source(cg)
        d = dirs[i]
        L = math.hypot(p1[0] - p0[0], p1[1] - p0[1]) + w
        ctx.save()
        ctx.translate(mx, my)
        ctx.rotate(math.atan2(d[1], d[0]))
        ctx.rectangle(-L / 2 - 2, -w, L + 4, 2 * w)
        ctx.restore()
        ctx.fill()
    ctx.restore()
    return pts, normals, dirs


def _finger_details(ctx, js, ws, press, W, style: HandStyle, thumb=False):
    # creases on the joints (PIP / DIP or thumb IP)
    for i in range(1, len(js) - 1):
        a, b = js[i - 1], js[i + 1]
        d = (b[0] - a[0], b[1] - a[1])
        L = math.hypot(*d) or 1.0
        d = (d[0] / L, d[1] / L)
        n = (-d[1], d[0])
        c = js[i]
        w = ws[i] * 0.42
        ctx.set_source_rgba(*style.crease, 0.45)
        ctx.set_line_width(max(1.0, W * 0.03))
        for off in (-0.05, 0.05):
            ox, oy = c[0] + d[0] * off * W, c[1] + d[1] * off * W
            ctx.move_to(ox + n[0] * w, oy + n[1] * w)
            ctx.curve_to(ox + n[0] * w * 0.3 + d[0] * W * 0.04, oy + n[1] * w * 0.3 + d[1] * W * 0.04,
                         ox - n[0] * w * 0.3 + d[0] * W * 0.04, oy - n[1] * w * 0.3 + d[1] * W * 0.04,
                         ox - n[0] * w, oy - n[1] * w)
            ctx.stroke()
        # joint highlight (the bent joint is the highest point)
        ctx.save()
        ctx.translate(c[0] - d[0] * 0.04 * W, c[1] - d[1] * 0.04 * W)
        ctx.rotate(math.atan2(d[1], d[0]))
        ctx.scale(ws[i] * 0.22, ws[i] * 0.16)
        ctx.arc(0, 0, 1, 0, 2 * math.pi)
        ctx.restore()
        ctx.set_source_rgba(1, 1, 1, 0.20)
        ctx.fill()
    # nail on the last phalanx, foreshortened by its slope
    p0, p1 = js[-2], js[-1]
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    hl = math.hypot(dx, dy)
    d = (dx / hl, dy / hl) if hl > 1e-3 else (0.0, -1.0)
    wt = ws[-1]
    nl = max(wt * 0.24, min(wt * 0.52, hl * 0.42))
    nw = wt * (0.30 if not thumb else 0.34)
    cx, cy = p1[0] - d[0] * (nl * 0.85 + wt * 0.05), p1[1] - d[1] * (nl * 0.85 + wt * 0.05)
    ang = math.atan2(d[1], d[0])
    ctx.save()
    ctx.translate(cx, cy)
    ctx.rotate(ang)
    ctx.scale(nl, nw)
    ctx.arc(0, 0, 1, 0, 2 * math.pi)
    ctx.restore()
    ctx.set_source_rgba(*style.nail, 0.96)
    ctx.fill_preserve()
    ctx.set_source_rgba(*style.crease, 0.55)
    ctx.set_line_width(max(0.9, W * 0.022))
    ctx.stroke()
    ctx.save()
    ctx.translate(cx + d[0] * nl * 0.2 - d[1] * nw * 0.3, cy + d[1] * nl * 0.2 + d[0] * nw * 0.3)
    ctx.rotate(ang)
    ctx.scale(nl * 0.38, nw * 0.25)
    ctx.arc(0, 0, 1, 0, 2 * math.pi)
    ctx.restore()
    ctx.set_source_rgba(1, 1, 1, 0.55)
    ctx.fill()
    # weight on the key: the pressing fingertip pad darkens slightly
    if press > 0.01:
        rg = cairo.RadialGradient(p1[0], p1[1], 0, p1[0], p1[1], wt * 0.7)
        rg.add_color_stop_rgba(0, *style.skin_shade, 0.0)
        rg.add_color_stop_rgba(0.6, *style.skin_shade, 0.35 * press)
        rg.add_color_stop_rgba(1, *style.skin_shade, 0.0)
        ctx.set_source(rg)
        ctx.arc(p1[0], p1[1], wt * 0.7, 0, 2 * math.pi)
        ctx.fill()


def hand_silhouette(ctx, geo: HandGeometry):
    for f in range(1, 6):
        js, ws, press = geo.fingers[f]
        _finger_outline(ctx, js, ws, geo.W, wrist=geo.base_dir(f))
        ctx.fill()
    _catmull(ctx, geo.palm)
    ctx.fill()


def draw_hand(ctx, geo: HandGeometry, style: HandStyle):
    W = geo.W
    ow = style.outline_w * max(1.0, W / 45)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    # 1) outer outline of the whole hand (inner parts get covered by the fills)
    ctx.set_source_rgb(*style.outline)
    ctx.set_line_width(ow * 2)
    for f in range(1, 6):
        js, ws, press = geo.fingers[f]
        _finger_outline(ctx, js, ws, W, wrist=geo.base_dir(f))
        ctx.stroke()
    _catmull(ctx, geo.palm)
    ctx.stroke()
    # 2) fingers, far side first, each with a thin outline so neighbours stay separate
    order = [1, 5, 4, 3, 2]
    for f in order:
        js, ws, press = geo.fingers[f]
        _fill_finger(ctx, js, ws, press, W, style, wrist=geo.base_dir(f))
        _finger_outline(ctx, js, ws, W, wrist=geo.base_dir(f))
        ctx.set_source_rgb(*style.outline)
        ctx.set_line_width(ow)
        ctx.stroke()
        if f == 1:   # the thumb can tuck under the fingers: its details go before theirs
            _finger_details(ctx, js, ws, press, W, style, thumb=True)
    # 3) back of the hand over the finger bases
    pc = geo.fp(0.15, 1.3)
    rg = cairo.RadialGradient(pc[0] - geo.s * 0.4 * W, pc[1] - 0.7 * W, 0.4 * W, pc[0], pc[1], 3.6 * W)
    rg.add_color_stop_rgb(0.0, *style.skin_light)
    rg.add_color_stop_rgb(0.5, *style.skin)
    rg.add_color_stop_rgb(1.0, *style.skin_shade)
    _catmull(ctx, geo.palm)
    ctx.set_source(rg)
    ctx.fill()
    ctx.save()
    _catmull(ctx, geo.palm)
    ctx.clip()
    # knuckle bumps: highlight on top, soft shade just behind
    for f in (2, 3, 4, 5):
        k = geo.fingers[f][0][0]
        w = geo.fingers[f][1][0]
        ctx.save()
        ctx.translate(k[0], k[1] + 0.32 * W)
        ctx.scale(w * 0.55, w * 0.32)
        ctx.arc(0, 0, 1, 0, 2 * math.pi)
        ctx.restore()
        ctx.set_source_rgba(*style.skin_shade, 0.16)
        ctx.fill()
        ctx.save()
        ctx.translate(k[0] - geo.s * 0.04 * W, k[1] + 0.05 * W)
        ctx.scale(w * 0.34, w * 0.20)
        ctx.arc(0, 0, 1, 0, 2 * math.pi)
        ctx.restore()
        ctx.set_source_rgba(1, 1, 1, 0.22)
        ctx.fill()
    # crease where the thumb joins the hand
    c0, c1, c2 = geo.fp(-1.18, 1.05), geo.fp(-1.02, 1.55), geo.fp(-1.12, 2.15)
    ctx.move_to(*c0)
    ctx.curve_to(c1[0], c1[1], c1[0], c1[1], c2[0], c2[1])
    ctx.set_source_rgba(*style.crease, 0.28)
    ctx.set_line_width(max(1.0, W * 0.03))
    ctx.stroke()
    ctx.restore()
    # 4) details on the fingers (the thumb's were drawn with it)
    for f in order[1:]:
        js, ws, press = geo.fingers[f]
        _finger_details(ctx, js, ws, press, W, style)


def draw_hand_shadow(ctx, geo: HandGeometry, style: HandStyle, canvas_w: int, canvas_h: int):
    """Blurred drop shadow, rendered in a cropped A8 buffer."""
    W = geo.W
    xs = [p[0] for p in geo.palm] + [j[0] for f in geo.fingers.values() for j in f[0]]
    ys = [p[1] for p in geo.palm] + [j[1] for f in geo.fingers.values() for j in f[0]]
    pad = int(W * 1.2)
    x0, y0 = int(max(0, min(xs) - pad)), int(max(0, min(ys) - pad))
    x1, y1 = int(min(canvas_w, max(xs) + pad)), int(min(canvas_h, max(ys) + pad))
    if x1 <= x0 or y1 <= y0:
        return
    w, h = x1 - x0, y1 - y0
    surf = cairo.ImageSurface(cairo.FORMAT_A8, w, h)
    c2 = cairo.Context(surf)
    c2.translate(-x0, -y0)
    c2.set_source_rgba(0, 0, 0, 1)
    hand_silhouette(c2, geo)
    surf.flush()
    stride = surf.get_stride()
    a = np.ndarray((h, stride), np.uint8, surf.get_data())
    a[:, :w] = cv2.GaussianBlur(a[:, :w], (0, 0), sigmaX=W * 0.18)
    surf.mark_dirty()
    ctx.save()
    ctx.set_source_rgba(0.12, 0.08, 0.06, style.shadow_alpha)
    ctx.mask_surface(surf, x0 + 0.12 * W, y0 + 0.30 * W)
    ctx.restore()
