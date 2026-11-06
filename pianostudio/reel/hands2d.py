"""Illustrated 2D hands for the reel: choreography + drawing.

Choreography (deterministic, from fingered notes):
  * every note carries hand + finger (MIDI/sidecar or the beam-search solver);
  * the palm glides between hand positions with minimum-jerk moves that finish
    just before the chord that needs them, with a dead-band so repeated
    figures don't make the hand wobble;
  * each finger reaches its key before the onset, the key is fully down exactly
    at the onset, held for the note, released, then the finger relaxes back to
    its natural place relative to the palm (or travels straight to its next key
    when that comes soon);
  * free fingers are kept in order and apart; the palm is clamped so every
    pressing finger stays within reach.

Drawing (cairo): overhead cartoon hands entering from the bottom of the frame,
soft gradient shading, outline, nails, joint creases and a blurred drop shadow.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cairo
import cv2
import numpy as np

from ..midi_io import group_onsets
from ..model import LEFT, RIGHT, Note
from .layout import Layout
from ..geometry import is_black

# ---- hand model (right hand, in white-key widths; x to the right, y down) -------
NAT = {1: -2.02, 2: -0.98, 3: 0.0, 4: 0.96, 5: 1.86}      # natural fingertip x offset from palm
DEV = {1: 2.4, 2: 1.2, 3: 1.05, 4: 1.0, 5: 1.3}          # max comfortable lateral deviation
KNUCKLE = {2: (-0.90, 0.10), 3: (0.0, 0.0), 4: (0.86, 0.10), 5: (1.60, 0.40)}
THUMB_BASE = (-1.30, 1.30)
WIDTH = {1: 0.96, 2: 0.86, 3: 0.88, 4: 0.82, 5: 0.71}    # finger width at the base
THUMB_DROP = 0.62                                          # thumb touches keys nearer the front edge
PALM = [(-1.36, 0.40), (-0.90, -0.16), (0.0, -0.24), (0.86, -0.16), (1.60, 0.12), (2.02, 0.55),
        (2.14, 1.45), (2.05, 2.6), (1.9, 4.4), (-1.40, 4.4), (-1.62, 2.7), (-1.86, 1.85), (-1.66, 1.0)]

PRE = 0.065      # s: finger descends during [onset-PRE, onset]; key bottoms out at the onset
REL = 0.07       # s: release / lift time
SAMPLE_HZ = 240


def _smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def _minjerk(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * u * (10 - 15 * u + 6 * u * u)


@dataclass
class Press:
    on: float
    off: float        # visual release
    x: float
    y: float
    note: Note


@dataclass
class FingerState:
    x: float
    y: float
    press: float      # 0 = hovering, 1 = key fully down
    pitch: int | None


@dataclass
class HandState:
    side: int         # +1 right hand, -1 left hand
    palm_x: float
    palm_y: float
    fingers: dict     # f -> FingerState
    visible: float = 1.0


class HandTrack:
    def __init__(self, notes: list[Note], hand: str, lay: Layout, duration: float, home_x: float | None = None):
        self.hand = hand
        self.side = 1 if hand == RIGHT else -1
        self.lay = lay
        self.W = W = lay.white_w
        self.notes = sorted(notes, key=lambda n: (n.start, n.pitch))
        self.duration = duration
        s = self.side

        # ---- presses per finger ------------------------------------------
        self.presses: dict[int, list[Press]] = {f: [] for f in range(1, 6)}
        for n in self.notes:
            f = n.finger or 3
            y = lay.tip_y(n.pitch) + (THUMB_DROP * W if f == 1 else 0.0)
            self.presses[f].append(Press(n.start, n.end, lay.x_center(n.pitch), y, n))
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
        self.visual_end = {p.note.id: p.off for ps in self.presses.values() for p in ps}
        self._on = {f: np.array([p.on for p in ps]) for f, ps in self.presses.items()}

        # ---- palm targets per onset group ----------------------------------
        groups = group_onsets(self.notes) if self.notes else []
        times, raw, lo_b, hi_b, blk = [], [], [], [], []
        for g in groups:
            t = g[0].start
            held = [n for n in self.notes if n.start < t - 0.02 and n.end > t + 0.05]
            eng = [(n, 1.0) for n in g] + [(n, 0.6) for n in held]
            xs = [(lay.x_center(n.pitch) - s * NAT[n.finger or 3] * W, w) for n, w in eng]
            raw.append(sum(x * w for x, w in xs) / sum(w for _, w in xs))
            lo = max(lay.x_center(n.pitch) - s * NAT[n.finger or 3] * W - 0.85 * DEV[n.finger or 3] * W for n in g)
            hi = min(lay.x_center(n.pitch) - s * NAT[n.finger or 3] * W + 0.85 * DEV[n.finger or 3] * W for n in g)
            lo_b.append(lo)
            hi_b.append(hi)
            times.append(t)
            blk.append(sum(1.0 for n, _ in eng if is_black(n.pitch)) / len(eng))
        times = np.array(times)
        raw = np.array(raw)
        if len(times):
            sm = []
            for k, t in enumerate(times):
                w = np.exp(-((times - t) / 0.9) ** 2) * (np.abs(times - t) < 2.0)
                v = float((w * raw).sum() / w.sum())
                if lo_b[k] <= hi_b[k]:
                    v = min(max(v, lo_b[k]), hi_b[k])
                sm.append(v)
            targets, cur = [], sm[0]
            for k, v in enumerate(sm):
                feasible = lo_b[k] - 1e-6 <= cur <= hi_b[k] + 1e-6 if lo_b[k] <= hi_b[k] else False
                if abs(v - cur) > 0.6 * W or not feasible:
                    cur = v
                targets.append(cur)
        else:
            targets = []
        self.group_times, self.targets = times, np.array(targets)
        if home_x is None:
            home_x = self.targets[0] if len(self.targets) else lay.width * (0.68 if s > 0 else 0.32)
        self.home_x = home_x

        # ---- sample the palm path ------------------------------------------
        n_s = int(math.ceil((duration + 1.0) * SAMPLE_HZ)) + 1
        ts = np.arange(n_s) / SAMPLE_HZ
        px = np.full(n_s, float(self.targets[0]) if len(self.targets) else home_x)
        for k in range(1, len(self.targets)):
            a, b = self.targets[k - 1], self.targets[k]
            if abs(b - a) < 1e-6:
                continue
            gap = times[k] - times[k - 1]
            te = times[k] - 0.035
            dur = min(0.45, max(0.07, 0.6 * gap))
            tsv = te - dur
            u = (ts - tsv) / dur
            m = ts >= tsv
            px[m] = a + (b - a) * _minjerk(u[m])
        # clamp so every pressing finger stays within reach, then smooth the result
        for f, ps in self.presses.items():
            for p in ps:
                i0, i1 = int((p.on - PRE) * SAMPLE_HZ), int(p.off * SAMPLE_HZ) + 1
                i0, i1 = max(i0, 0), min(i1, n_s)
                c = p.x - s * NAT[f] * W
                lo, hi = c - DEV[f] * W, c + DEV[f] * W
                px[i0:i1] = np.clip(px[i0:i1], lo, hi)
        px = cv2.GaussianBlur(px.reshape(1, -1).astype(np.float64), (0, 0), sigmaX=0.022 * SAMPLE_HZ).ravel()
        self.palm_samples = px
        # palm height: push forward a little when black keys are engaged
        py = np.zeros(n_s)
        for k, t in enumerate(times):
            py[int(t * SAMPLE_HZ):] = blk[k]
        py = cv2.GaussianBlur(py.reshape(1, -1), (0, 0), sigmaX=0.12 * SAMPLE_HZ).ravel()
        self.palm_y_samples = lay.knuckle_y - py * 0.42 * W

    # ------------------------------------------------------------------
    def palm_at(self, t: float) -> tuple[float, float]:
        i = t * SAMPLE_HZ
        i0 = int(min(max(math.floor(i), 0), len(self.palm_samples) - 2))
        a = min(max(i - i0, 0.0), 1.0)
        x = self.palm_samples[i0] * (1 - a) + self.palm_samples[i0 + 1] * a
        y = self.palm_y_samples[i0] * (1 - a) + self.palm_y_samples[i0 + 1] * a
        return float(x), float(y)

    def rest_tip(self, f: int, palm_x: float, palm_y: float) -> tuple[float, float]:
        W = self.W
        x = palm_x + self.side * NAT[f] * W
        y = self.lay.tip_y_white - (0.12 * W if f in (2, 3, 4) else 0.0) + (palm_y - self.lay.knuckle_y)
        if f == 1:
            y += (THUMB_DROP + 0.12) * W
        return x, y

    def finger_at(self, f: int, t: float):
        """-> (key_x, key_y, rest_weight, press, pitch). rest_weight 0 = at/near its key,
        1 = fully relaxed (position then comes from the hand frame, see state())."""
        ps = self.presses[f]
        if not ps:
            return None, None, 1.0, 0.0, None
        on = self._on[f]
        i = int(np.searchsorted(on - PRE, t, side="right")) - 1   # last press whose descent has begun
        cur = ps[i] if i >= 0 else None
        nxt = ps[i + 1] if i + 1 < len(ps) else None
        if cur is not None and t <= cur.off + REL:
            if t < cur.on:
                pr = _smooth((t - (cur.on - PRE)) / PRE)
            elif t <= cur.off:
                pr = 1.0
            else:
                pr = 1.0 - _smooth((t - cur.off) / REL)
            return cur.x, cur.y, 0.0, float(pr), cur.note.pitch
        t_free = cur.off + REL if cur else -1e9
        t_need = nxt.on - PRE - 0.015 if nxt else 1e9
        if cur and nxt:
            G = max(t_need - t_free, 1e-3)
            u = _minjerk((t - t_free) / G)              # travel spread over the whole free gap
            kx = cur.x + (nxt.x - cur.x) * u
            ky = cur.y + (nxt.y - cur.y) * u
            relax = min(1.0, max(0.0, (G - 0.25) / 0.5))
            w = relax * _smooth((t - t_free) / 0.24) * _smooth((t_need - 0.12 - t) / 0.26)
        elif nxt:
            kx, ky = nxt.x, nxt.y
            w = _smooth((t_need - 0.06 - t) / 0.30)
        else:
            kx, ky = cur.x, cur.y
            w = _smooth((t - t_free) / 0.30)
        return kx, ky, float(w), 0.0, None

    def state(self, t: float) -> HandState:
        """All quantities are continuous in t: anchors and mobilities are blended
        smoothly (no thresholds), so fingers never teleport between frames."""
        palm_x, palm_y = self.palm_at(t)
        s, W = self.side, self.W
        raw = {f: self.finger_at(f, t) for f in range(1, 6)}
        # how firmly each finger is anchored to a key (1 = pressing / on its way, 0 = relaxed)
        anchor = {f: (0.0 if r[0] is None else (1.0 - r[2])) for f, r in raw.items()}
        cur_x = {f: (r[0] if r[0] is not None else None) for f, r in raw.items()}
        fingers, mob = {}, {}
        for f in range(1, 6):
            kx, ky, w, pr, pitch = raw[f]
            rx, ry = self.rest_tip(f, palm_x, palm_y)
            # hand-frame estimate of where this finger would naturally be, from the other fingers
            num, den = 0.35 * rx, 0.35
            for a in range(1, 6):
                if a == f or cur_x[a] is None or anchor[a] <= 0:
                    continue
                wa = anchor[a] / (abs(a - f) ** 2) * (0.5 if 1 in (a, f) else 1.0)
                num += wa * (cur_x[a] + s * (NAT[f] - NAT[a]) * W)
                den += wa
            rx = num / den
            if kx is None:
                x, y = rx, ry
            else:
                x, y = kx + (rx - kx) * w, ky + (ry - ky) * w
            fingers[f] = FingerState(float(x), float(y), pr, pitch)
            mob[f] = (1.0 - pr) ** 3 * w   # only relaxed fingers yield; one on/heading to its key stays put
        self._separate(fingers, mob)
        return HandState(self.side, palm_x, palm_y, fingers)

    def _separate(self, fingers, mob):
        """Keep fingertips 2-5 ordered and apart; corrections are shared in proportion
        to each finger's mobility (pressing fingers have none)."""
        s, gap = self.side, 0.80 * self.W
        order = [2, 3, 4, 5]
        for _ in range(4):
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
        # a free thumb stays on its side of the index finger
        f1, f2 = fingers[1], fingers[2]
        d = (f2.x - f1.x) * s
        if d < 0.75 * self.W:
            f1.x -= s * (0.75 * self.W - d) * min(1.0, mob[1] / 0.7) ** 2


# ======================================================================= drawing
@dataclass
class HandStyle:
    skin: tuple = (0.976, 0.808, 0.663)
    skin_light: tuple = (1.0, 0.894, 0.788)
    skin_shade: tuple = (0.886, 0.651, 0.502)
    outline: tuple = (0.478, 0.290, 0.212)
    nail: tuple = (1.0, 0.902, 0.871)
    crease: tuple = (0.737, 0.478, 0.365)
    shadow_alpha: float = 0.26
    outline_w: float = 2.6

    @staticmethod
    def from_hex(skin: str | None = None, outline: str | None = None) -> "HandStyle":
        st = HandStyle()
        if skin:
            r, g, b = hex_rgb(skin)
            st.skin = (r, g, b)
            st.skin_light = tuple(min(1.0, c * 1.08 + 0.05) for c in (r, g, b))
            st.skin_shade = tuple(c * 0.88 for c in (r, g, b))
            st.crease = tuple(c * 0.74 for c in (r, g, b))
        if outline:
            st.outline = hex_rgb(outline)
        return st


def hex_rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


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

    def __init__(self, st: HandState, W: float):
        s = st.side
        self.st, self.W, self.s = st, W, s
        px, py = st.palm_x, st.palm_y
        self.palm = [(px + s * x * W, py + y * W) for x, y in PALM]
        self.fingers = {}
        for f in (2, 3, 4, 5):
            kx, ky = KNUCKLE[f]
            k = (px + s * kx * W, py + ky * W)
            fs = st.fingers[f]
            self.fingers[f] = (k, (fs.x, fs.y), WIDTH[f] * W, fs.press)
        bx, by = THUMB_BASE
        base = (px + s * bx * W, py + by * W)
        ts = st.fingers[1]
        self.thumb = (base, (ts.x, ts.y), WIDTH[1] * W, ts.press)

    @staticmethod
    def finger_path(ctx, k, tip, wb, press, W, base_ext=0.9):
        dx, dy = tip[0] - k[0], tip[1] - k[1]
        L = math.hypot(dx, dy) or 1.0
        d = (dx / L, dy / L)
        n = (-d[1], d[0])
        wt = wb * (0.86 + 0.04 * press)
        r = wt / 2
        b0 = (k[0] - d[0] * base_ext * W, k[1] - d[1] * base_ext * W)
        jm = 0.52
        m = (k[0] + dx * jm, k[1] + dy * jm)
        wm = (wb + wt) / 2 * 1.035
        ctx.move_to(b0[0] + n[0] * wb / 2, b0[1] + n[1] * wb / 2)
        ctx.line_to(k[0] + n[0] * wb / 2, k[1] + n[1] * wb / 2)
        ctx.curve_to(m[0] + n[0] * wm / 2 - d[0] * L * 0.12, m[1] + n[1] * wm / 2 - d[1] * L * 0.12,
                     m[0] + n[0] * wm / 2 + d[0] * L * 0.12, m[1] + n[1] * wm / 2 + d[1] * L * 0.12,
                     tip[0] + n[0] * r, tip[1] + n[1] * r)
        a0 = math.atan2(n[1], n[0])
        ctx.arc_negative(tip[0], tip[1], r, a0, a0 - math.pi)   # n = d rotated +90deg -> sweep through d
        ctx.curve_to(m[0] - n[0] * wm / 2 + d[0] * L * 0.12, m[1] - n[1] * wm / 2 + d[1] * L * 0.12,
                     m[0] - n[0] * wm / 2 - d[0] * L * 0.12, m[1] - n[1] * wm / 2 - d[1] * L * 0.12,
                     k[0] - n[0] * wb / 2, k[1] - n[1] * wb / 2)
        ctx.line_to(b0[0] - n[0] * wb / 2, b0[1] - n[1] * wb / 2)
        ctx.close_path()
        return d, n, L, wt

    def thumb_path(self, ctx):
        base, tip, wb, press = self.thumb
        W, s = self.W, self.s
        # bend point bulges outward (away from the fingers)
        mx, my = base[0] + (tip[0] - base[0]) * 0.5, base[1] + (tip[1] - base[1]) * 0.5
        dx, dy = tip[0] - base[0], tip[1] - base[1]
        L = math.hypot(dx, dy) or 1.0
        n = (-dy / L, dx / L)
        if n[0] * s > 0:
            n = (-n[0], -n[1])
        j = (mx + n[0] * 0.22 * W, my + n[1] * 0.22 * W)
        return j


def _fill_finger(ctx, k, tip, wb, press, W, style: HandStyle, base_ext=0.9):
    d, n, L, wt = HandGeometry.finger_path(ctx, k, tip, wb, press, W, base_ext)
    c = tip
    g = cairo.LinearGradient(c[0] + n[0] * wt * 0.62, c[1] + n[1] * wt * 0.62,
                             c[0] - n[0] * wt * 0.62, c[1] - n[1] * wt * 0.62)
    sk, li, sh = style.skin, style.skin_light, style.skin_shade
    g.add_color_stop_rgb(0.0, *sh)
    g.add_color_stop_rgb(0.32, *sk)
    g.add_color_stop_rgb(0.55, *li)
    g.add_color_stop_rgb(1.0, *sh)
    ctx.set_source(g)
    return d, n, L, wt


def _finger_details(ctx, k, tip, wt, press, W, style: HandStyle):
    dx, dy = tip[0] - k[0], tip[1] - k[1]
    L = math.hypot(dx, dy) or 1.0
    d = (dx / L, dy / L)
    n = (-d[1], d[0])
    ang = math.atan2(d[1], d[0])
    # joint creases
    ctx.set_source_rgba(*style.crease, 0.55)
    ctx.set_line_width(max(1.2, W * 0.035))
    for frac, wfrac in ((0.30, 0.42), (0.58, 0.46)):
        cx, cy = tip[0] - d[0] * L * frac, tip[1] - d[1] * L * frac
        hw = wt * wfrac / 2
        ctx.move_to(cx + n[0] * hw, cy + n[1] * hw)
        ctx.curve_to(cx + n[0] * hw * 0.3 + d[0] * W * 0.05, cy + n[1] * hw * 0.3 + d[1] * W * 0.05,
                     cx - n[0] * hw * 0.3 + d[0] * W * 0.05, cy - n[1] * hw * 0.3 + d[1] * W * 0.05,
                     cx - n[0] * hw, cy - n[1] * hw)
        ctx.stroke()
    # nail (shorter and closer to the tip when the finger curls into the key)
    nl = wt * (0.40 - 0.10 * press)
    nw = wt * 0.29
    off = wt * (0.30 - 0.12 * press)
    cx, cy = tip[0] - d[0] * off, tip[1] - d[1] * off
    ctx.save()
    ctx.translate(cx, cy)
    ctx.rotate(ang)
    ctx.scale(nl, nw)
    ctx.arc(0, 0, 1, 0, 2 * math.pi)
    ctx.restore()
    ctx.set_source_rgba(*style.nail, 0.95)
    ctx.fill_preserve()
    ctx.set_source_rgba(*style.crease, 0.6)
    ctx.set_line_width(max(1.0, W * 0.025))
    ctx.stroke()
    # nail highlight
    ctx.save()
    ctx.translate(cx + d[0] * nl * 0.25 - n[0] * nw * 0.25, cy + d[1] * nl * 0.25 - n[1] * nw * 0.25)
    ctx.rotate(ang)
    ctx.scale(nl * 0.35, nw * 0.28)
    ctx.arc(0, 0, 1, 0, 2 * math.pi)
    ctx.restore()
    ctx.set_source_rgba(1, 1, 1, 0.55)
    ctx.fill()


def _thumb_shape(ctx, geo: HandGeometry):
    base, tip, wb, press = geo.thumb
    j = geo.thumb_path(ctx)
    W = geo.W
    # two capsules: base->joint, joint->tip; drawn as one smooth outline via a polyline offset
    pts_c = [base, j, tip]
    ws = [wb * 1.18, wb * 1.02, wb * 0.86]
    left, right = [], []
    for i, p in enumerate(pts_c):
        if i == 0:
            v = (pts_c[1][0] - p[0], pts_c[1][1] - p[1])
        elif i == len(pts_c) - 1:
            v = (p[0] - pts_c[i - 1][0], p[1] - pts_c[i - 1][1])
        else:
            v = (pts_c[i + 1][0] - pts_c[i - 1][0], pts_c[i + 1][1] - pts_c[i - 1][1])
        L = math.hypot(*v) or 1.0
        nn = (-v[1] / L, v[0] / L)
        left.append((p[0] + nn[0] * ws[i] / 2, p[1] + nn[1] * ws[i] / 2))
        right.append((p[0] - nn[0] * ws[i] / 2, p[1] - nn[1] * ws[i] / 2))
    v = (tip[0] - j[0], tip[1] - j[1])
    L = math.hypot(*v) or 1.0
    d = (v[0] / L, v[1] / L)
    n = (-d[1], d[0])
    r = ws[-1] / 2
    ctx.move_to(*left[0])
    ctx.curve_to(left[0][0] * 0.4 + left[1][0] * 0.6, left[0][1] * 0.4 + left[1][1] * 0.6,
                 left[1][0], left[1][1], (left[1][0] + left[2][0]) / 2, (left[1][1] + left[2][1]) / 2)
    ctx.line_to(*left[2])
    a0 = math.atan2(n[1], n[0])
    ctx.arc_negative(tip[0], tip[1], r, a0, a0 - math.pi)
    ctx.line_to((right[1][0] + right[2][0]) / 2, (right[1][1] + right[2][1]) / 2)
    ctx.curve_to(right[1][0], right[1][1], right[0][0] * 0.4 + right[1][0] * 0.6,
                 right[0][1] * 0.4 + right[1][1] * 0.6, *right[0])
    ctx.close_path()
    return j, ws[-1]


def hand_silhouette(ctx, geo: HandGeometry):
    """Union of all hand parts as separate filled sub-paths (for shadows)."""
    _thumb_shape(ctx, geo)
    ctx.fill()
    _catmull(ctx, geo.palm)
    ctx.fill()
    for f in (2, 3, 4, 5):
        k, tip, wb, press = geo.fingers[f]
        HandGeometry.finger_path(ctx, k, tip, wb, press, geo.W)
        ctx.fill()


def draw_hand(ctx, geo: HandGeometry, style: HandStyle):
    W = geo.W
    ow = style.outline_w * max(1.0, W / 45)
    # 1) thick outline of every part (only the outer silhouette survives step 2)
    ctx.set_source_rgb(*style.outline)
    ctx.set_line_width(ow * 2)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    _thumb_shape(ctx, geo)
    ctx.stroke()
    _catmull(ctx, geo.palm)
    ctx.stroke()
    for f in (2, 3, 4, 5):
        k, tip, wb, press = geo.fingers[f]
        HandGeometry.finger_path(ctx, k, tip, wb, press, W)
        ctx.stroke()
    # 2) fills: thumb, palm, fingers
    j, twt = _thumb_shape(ctx, geo)
    base, ttip, twb, tpress = geo.thumb
    v = (ttip[0] - j[0], ttip[1] - j[1])
    L = math.hypot(*v) or 1.0
    n = (-v[1] / L, v[0] / L)
    g = cairo.LinearGradient(ttip[0] + n[0] * twt, ttip[1] + n[1] * twt, ttip[0] - n[0] * twt, ttip[1] - n[1] * twt)
    g.add_color_stop_rgb(0.0, *style.skin_shade)
    g.add_color_stop_rgb(0.4, *style.skin)
    g.add_color_stop_rgb(0.6, *style.skin_light)
    g.add_color_stop_rgb(1.0, *style.skin_shade)
    ctx.set_source(g)
    ctx.fill()
    cx = sum(p[0] for p in geo.palm[:6]) / 6
    cy = geo.st.palm_y + 1.4 * W
    rg = cairo.RadialGradient(cx - geo.s * 0.2 * W, cy - 0.6 * W, 0.3 * W, cx, cy, 3.4 * W)
    rg.add_color_stop_rgb(0.0, *style.skin_light)
    rg.add_color_stop_rgb(0.55, *style.skin)
    rg.add_color_stop_rgb(1.0, *style.skin_shade)
    _catmull(ctx, geo.palm)
    ctx.set_source(rg)
    ctx.fill()
    # thumb line where it meets the palm
    ctx.save()
    _catmull(ctx, geo.palm)
    ctx.clip()
    ctx.set_source_rgba(*style.crease, 0.35)
    ctx.set_line_width(max(1.2, W * 0.04))
    bx, by = base
    ctx.move_to(bx + geo.s * 0.25 * W, by + 1.6 * W)
    ctx.curve_to(bx + geo.s * 0.55 * W, by + 0.6 * W, bx + geo.s * 0.35 * W, by - 0.4 * W, j[0] + geo.s * 0.55 * W, j[1] - 0.1 * W)
    ctx.stroke()
    ctx.restore()
    order = (5, 4, 3, 2)
    for f in order:
        k, tip, wb, press = geo.fingers[f]
        d, nn, Lf, wt = _fill_finger(ctx, k, tip, wb, press, W, style)
        ctx.fill()
        # separation outline above the knuckle line
        ctx.save()
        ctx.rectangle(0, 0, 100000, k[1] - 0.18 * W)
        ctx.clip()
        HandGeometry.finger_path(ctx, k, tip, wb, press, W)
        ctx.set_source_rgb(*style.outline)
        ctx.set_line_width(ow)
        ctx.stroke()
        ctx.restore()
    # knuckle highlights
    for f in (2, 3, 4, 5):
        k = geo.fingers[f][0]
        ctx.save()
        ctx.translate(k[0], k[1] + 0.05 * W)
        ctx.scale(0.20 * W, 0.12 * W)
        ctx.arc(0, 0, 1, 0, 2 * math.pi)
        ctx.restore()
        ctx.set_source_rgba(1, 1, 1, 0.22)
        ctx.fill()
    # pressing fingertips read slightly darker (weight on the key)
    for k_, tip_, wb_, pr_ in [geo.fingers[f] for f in (2, 3, 4, 5)] + [geo.thumb]:
        if pr_ > 0.01:
            rg2 = cairo.RadialGradient(tip_[0], tip_[1], 0, tip_[0], tip_[1], wb_ * 0.75)
            rg2.add_color_stop_rgba(0, *style.skin_shade, 0.0)
            rg2.add_color_stop_rgba(0.55, *style.skin_shade, 0.42 * pr_)
            rg2.add_color_stop_rgba(1, *style.skin_shade, 0.0)
            ctx.set_source(rg2)
            ctx.arc(tip_[0], tip_[1], wb_ * 0.75, 0, 2 * math.pi)
            ctx.fill()
    # 3) details
    _finger_details(ctx, j, ttip, twt, tpress, W, style)
    for f in (2, 3, 4, 5):
        k, tip, wb, press = geo.fingers[f]
        _finger_details(ctx, k, tip, wb * (0.86 + 0.04 * press), press, W, style)


def draw_hand_shadow(ctx, geo: HandGeometry, style: HandStyle, canvas_w: int, canvas_h: int):
    """Blurred drop shadow, rendered in a cropped A8 buffer."""
    W = geo.W
    xs = [p[0] for p in geo.palm] + [geo.fingers[f][1][0] for f in (2, 3, 4, 5)] + [geo.thumb[1][0]]
    ys = [p[1] for p in geo.palm] + [geo.fingers[f][1][1] for f in (2, 3, 4, 5)] + [geo.thumb[1][1]]
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
    blur = cv2.GaussianBlur(a[:, :w], (0, 0), sigmaX=W * 0.16)
    a[:, :w] = blur
    surf.mark_dirty()
    ctx.save()
    ctx.set_source_rgba(0.12, 0.08, 0.06, style.shadow_alpha)
    ctx.mask_surface(surf, x0 + 0.10 * W, y0 + 0.24 * W)
    ctx.restore()
