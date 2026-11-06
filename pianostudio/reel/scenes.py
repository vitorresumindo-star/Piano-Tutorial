"""Original procedural cinematic scenes for the footage panel.

These are an *original* visual stand-in (no film footage): a rural golden-hour
cornfield, a pickup on a dirt road, a dust storm, a farm under the Milky Way,
a rocket launch, low orbit, a ringed giant and a black hole. Every scene is a
deterministic function of time, so renders are reproducible and can be split
across worker processes.

Each scene renders HDR float32 RGB at the panel size; the compositor in
cinema.py tonemaps, grades, adds grain/vignette and handles transitions.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .scenekit import (blur, bloom, camera, ease, fbm, fbm1d, grid, hash01, lerp, over, puff_sprite, radial,
                       rng, smoothstep, soft_poly, stars, vgrad)

SCENES: dict[str, type] = {}


def register(name):
    def deco(cls):
        SCENES[name] = cls
        cls.name = name
        return cls
    return deco


class Scene:
    grade = dict(lift=(0.0, 0.0, 0.0), gamma=(1.0, 1.0, 1.0), gain=(1.0, 1.0, 1.0), saturation=1.0,
                 contrast=1.0, exposure=1.0)

    def __init__(self, w: int, h: int, params: dict | None = None, seed: int = 1):
        self.w, self.h = w, h
        self.p = params or {}
        self.seed = seed

    def render(self, t: float, dur: float) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    # helpers --------------------------------------------------------------
    def view(self, layer: np.ndarray, M: np.ndarray, border=cv2.BORDER_REFLECT) -> np.ndarray:
        return cv2.warpAffine(layer, M, (self.w, self.h), flags=cv2.INTER_LINEAR, borderMode=border)

    def view_remap(self, layer: np.ndarray, M: np.ndarray, dx: np.ndarray | None = None,
                   dy: np.ndarray | None = None, border=cv2.BORDER_REFLECT) -> np.ndarray:
        Mi = cv2.invertAffineTransform(M)
        xx, yy = self._grid()
        mx = Mi[0, 0] * xx + Mi[0, 1] * yy + Mi[0, 2]
        my = Mi[1, 0] * xx + Mi[1, 1] * yy + Mi[1, 2]
        if dx is not None:
            mx = mx + dx
        if dy is not None:
            my = my + dy
        return cv2.remap(layer, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR, borderMode=border)

    def _grid(self):
        if not hasattr(self, "_g"):
            self._g = grid(self.h, self.w)
        return self._g


def _apply_affine(M, x, y):
    return M[0, 0] * x + M[0, 1] * y + M[0, 2], M[1, 0] * x + M[1, 1] * y + M[1, 2]


# =============================================================================== field
@register("field")
class Field(Scene):
    """Endless cornfield at golden hour (or dusk), backlit, swaying."""
    PALETTES = {
        "golden": dict(
            sky=[(0.0, (0.16, 0.24, 0.42)), (0.42, (0.55, 0.45, 0.48)), (0.78, (1.15, 0.72, 0.40)),
                 (1.0, (1.55, 1.05, 0.62))],
            sun=(9.0, 7.2, 4.8), glow=(1.6, 0.95, 0.45), haze=(1.0, 0.68, 0.42), dark=(0.028, 0.032, 0.010),
            leaf=(0.16, 0.19, 0.045), rim=(3.2, 2.1, 0.9), cloud_lit=(1.6, 0.85, 0.55), cloud_dark=(0.32, 0.27, 0.34),
            stars=0.0, tassel=(0.9, 0.65, 0.25)),
        "dusk": dict(
            sky=[(0.0, (0.025, 0.035, 0.10)), (0.45, (0.12, 0.10, 0.22)), (0.80, (0.62, 0.30, 0.26)),
                 (1.0, (1.10, 0.52, 0.28))],
            sun=(0.0, 0.0, 0.0), glow=(0.9, 0.38, 0.18), haze=(0.34, 0.18, 0.17), dark=(0.008, 0.007, 0.010),
            leaf=(0.035, 0.03, 0.028), rim=(0.55, 0.24, 0.13), cloud_lit=(0.9, 0.38, 0.30), cloud_dark=(0.10, 0.08, 0.14),
            stars=1.0, tassel=(0.35, 0.2, 0.12)),
    }
    grade = dict(lift=(0.02, 0.015, 0.03), gamma=(1.0, 1.0, 1.02), gain=(1.03, 1.0, 0.95), saturation=1.08,
                 contrast=1.06, exposure=1.0)

    def __init__(self, w, h, params=None, seed=1):
        super().__init__(w, h, params, seed)
        p = self.p
        self.pal = dict(self.PALETTES[p.get("palette", "golden")])
        self.CW, self.CH = int(w * 1.38), int(h * 1.28)
        self.hy = p.get("horizon", 0.47) * self.CH
        self.sun = (p.get("sun_x", 0.62) * self.CW, self.hy - p.get("sun_alt", 0.07) * self.CH)
        self._build_sky()
        self._build_field()
        r = rng(seed + 7)
        self.motes = r.random((110, 5)).astype(np.float32)
        self.rays = fbm1d(1440, 30, 4, seed + 3)

    # ---- sky -----------------------------------------------------------------
    def _build_sky(self):
        CW, CH, hy = self.CW, self.CH, self.hy
        pal = self.pal
        sky = vgrad(CH, CW, [(s[0] * hy / CH, s[1]) for s in pal["sky"]] + [(1.0, pal["sky"][-1][1])])
        sx, sy = self.sun
        d = radial(CH, CW, sx, sy, 1.0, 0.75)
        glow = np.array(pal["glow"], np.float32)
        sky += glow * (0.9 * np.exp(-d / 70.0) + 0.55 * np.exp(-d / 260.0) + 0.25 * np.exp(-d / 700.0))[..., None]
        # clouds: thin stratus streaks lit from below near the sun
        n = fbm(CH, CW, 260, 6, self.seed + 1, aspect=4.5)
        n2 = fbm(CH, CW, 90, 4, self.seed + 2, aspect=3.0)
        yy = np.arange(CH, dtype=np.float32)[:, None] / hy
        band = smoothstep(0.05, 0.25, yy) * (1 - smoothstep(0.62, 0.86, yy))
        dens = smoothstep(0.52, 0.80, n * 0.75 + n2 * 0.35) * band
        dens = np.broadcast_to(dens, (CH, CW))
        lit = np.exp(-radial(CH, CW, sx, sy, 1.0, 0.5) / 520.0)
        ccol = (np.array(pal["cloud_lit"], np.float32) * (0.25 + 1.4 * lit)[..., None] +
                np.array(pal["cloud_dark"], np.float32) * (1 - lit)[..., None])
        sky = over(sky, ccol, dens * 0.85)
        if pal["stars"]:
            st = stars(CH, CW, 2600, self.seed + 9)
            fade = (1 - smoothstep(0.15, 0.75, yy)) * (1 - dens)
            sky += st * fade[..., None] * 0.9 * pal["stars"]
        # sun disc
        if pal["sun"][0] > 0:
            disc = smoothstep(26, 21, radial(CH, CW, sx, sy))
            sky = over(sky, np.array(pal["sun"], np.float32), disc)
        self.sky = sky.astype(np.float32)

    # ---- field ---------------------------------------------------------------
    def _build_field(self):
        CW, CH, hy = self.CW, self.CH, self.hy
        pal = self.pal
        r = rng(self.seed + 11)
        img = np.zeros((CH, CW, 3), np.float32)
        alpha = np.zeros((CH, CW), np.float32)
        haze, dark, leaf = (np.array(pal[k], np.float32) for k in ("haze", "dark", "leaf"))
        # distant tree line + farm silhouettes on the horizon
        ridge = fbm1d(CW, 60, 4, self.seed + 5)
        clumps = smoothstep(0.45, 0.8, fbm1d(CW, 140, 3, self.seed + 6))
        top = hy - (3 + 10 * ridge + 22 * clumps * ridge)
        yy = np.arange(CH, dtype=np.float32)[:, None]
        tree = (yy >= top[None, :]).astype(np.float32)
        tree_col = haze * 0.62
        img = over(img, np.broadcast_to(tree_col, img.shape), tree)
        alpha = np.maximum(alpha, tree)
        if self.p.get("farm", True):
            self._farm_silhouette(img, alpha, self.p.get("farm_x", 0.22) * CW, hy, haze * 0.5)
        # ground base gradient below the horizon
        u = np.clip((yy - hy) / (CH - hy), 0, 1)
        fog = np.exp(-u * 13.0)[..., None]
        nz = fbm(CH, CW, 40, 4, self.seed + 12, aspect=6)
        base = dark * (1 - fog) + haze * 0.75 * fog
        base = base * (0.75 + 0.5 * nz[..., None])
        below = (yy >= hy).astype(np.float32)
        below = np.broadcast_to(below, (CH, CW))
        img = over(img, base, below)
        alpha = np.maximum(alpha, below)
        # corn plants, far -> near
        N = int(self.p.get("plants", 4200))
        us = np.sort(r.random(N) ** 1.9)
        for u0 in us:
            yb = hy + 3 + (CH * 1.12 - hy) * u0
            s = (yb - hy) / (CH - hy)
            if s < 0.012:
                continue
            x = r.random() * (CW + 80) - 40
            fogp = math.exp(-s * 13.0)
            mixc = r.random()
            col = (dark * (1 - mixc * 0.75) + leaf * mixc * 0.75) * (1 - fogp) + haze * 0.8 * fogp
            self._plant(img, alpha, x, yb, s, col, r, pal)
        # backlight rim on edges facing the sun
        sx, sy = self.sun
        mask = alpha.copy()
        vx, vy = sx - CW / 2, sy - CH
        L = math.hypot(vx, vy)
        dxp, dyp = 2.0 * vx / L, 2.0 * vy / L
        M = np.float32([[1, 0, -dxp], [0, 1, -dyp]])
        nb = cv2.warpAffine(mask, M, (CW, CH), borderMode=cv2.BORDER_REPLICATE)
        rim = np.clip(mask - nb, 0, 1)
        xs = np.arange(CW, dtype=np.float32)[None, :]
        prox = np.exp(-np.abs(xs - sx) / (0.42 * CW)) * (yy >= hy - 30)
        rows = 0.35 + 0.65 * np.clip((yy - hy) / (CH - hy), 0, 1)
        img += np.array(pal["rim"], np.float32) * (rim * prox * rows)[..., None]
        # translucent glow of leaves straight towards the sun
        glow = np.exp(-radial(CH, CW, sx, sy, 1.0, 1.6) / 380.0) * (yy > hy)
        img += np.array(pal["glow"], np.float32) * 0.35 * (glow * mask)[..., None]
        # shallow depth of field: soften the nearest plants
        near = smoothstep(hy + (CH - hy) * 0.55, CH, yy)[..., None]
        img = img * (1 - near) + blur(img, 2.6) * near
        alpha = alpha * (1 - near[..., 0]) + blur(alpha, 2.6) * near[..., 0]
        self.field = img.astype(np.float32)
        self.field_a = np.clip(alpha, 0, 1).astype(np.float32)

    def _farm_silhouette(self, img, alpha, x, hy, col):
        c = tuple(float(v) for v in col)
        s = self.CW / 1500
        # barn
        bw, bh = 46 * s, 30 * s
        soft_poly(img, [(x, hy), (x, hy - bh), (x + bw * 0.5, hy - bh - 16 * s), (x + bw, hy - bh), (x + bw, hy)], c, alpha)
        # silo
        soft_poly(img, [(x + bw + 4 * s, hy), (x + bw + 4 * s, hy - 58 * s), (x + bw + 9 * s, hy - 64 * s),
                        (x + bw + 16 * s, hy - 64 * s), (x + bw + 21 * s, hy - 58 * s), (x + bw + 21 * s, hy)], c, alpha)
        # farmhouse
        hx = x - 70 * s
        soft_poly(img, [(hx, hy), (hx, hy - 20 * s), (hx + 18 * s, hy - 33 * s), (hx + 36 * s, hy - 20 * s), (hx + 36 * s, hy)], c, alpha)
        # windmill
        wx = x + 120 * s
        soft_poly(img, [(wx - 5 * s, hy), (wx - 1 * s, hy - 50 * s), (wx + 1 * s, hy - 50 * s), (wx + 5 * s, hy)], c, alpha)
        for k in range(8):
            a = k * math.pi / 4 + 0.3
            cv2.line(img, (int(wx), int(hy - 50 * s)), (int(wx + math.cos(a) * 11 * s), int(hy - 50 * s + math.sin(a) * 11 * s)),
                     c, max(1, int(1.2 * s)), cv2.LINE_AA)

    def _plant(self, img, alpha, x, yb, s, col, r, pal):
        hgt = (300 + 80 * r.random()) * s
        lean = (r.random() - 0.5) * 18 * s
        c = tuple(float(v) for v in col)
        sw = max(1.0, 5.0 * s)
        top = (x + lean, yb - hgt)
        if s < 0.05:
            soft_poly(img, [(x - sw, yb), (top[0] - sw * 0.5, top[1]), (top[0] + sw * 0.5, top[1]), (x + sw, yb)], c, alpha)
            # two tiny leaves
            for side in (-1, 1):
                L = 40 * s
                soft_poly(img, [(x, yb - hgt * 0.5), (x + side * L, yb - hgt * 0.5 - L * 0.25),
                                (x + side * L * 1.1, yb - hgt * 0.5 + L * 0.1), (x, yb - hgt * 0.4)], c, alpha)
            return
        soft_poly(img, [(x - sw, yb), (top[0] - sw * 0.6, top[1]), (top[0] + sw * 0.6, top[1]), (x + sw, yb)], c, alpha)
        nl = 6
        side = 1 if r.random() > 0.5 else -1
        sx_sun = self.sun[0]
        prox = math.exp(-abs(x - sx_sun) / (0.30 * self.CW))
        rimc = tuple(float(v) * prox * (0.25 + 0.75 * min(1.0, s * 1.6)) for v in pal["rim"])
        for i in range(nl):
            a = 0.20 + 0.12 * i
            ax, ay = x + lean * a, yb - hgt * a
            L = (135 + 60 * r.random()) * s * (1.0 - 0.07 * i)
            Wd = (17 + 8 * r.random()) * s
            rise = 0.55 + 0.35 * r.random()
            droop = 1.0 + 0.5 * r.random()
            us = np.linspace(0, 1, 9)
            cxs = ax + side * L * us * (0.9 + 0.1 * r.random())
            cys = ay - L * (rise * us - droop * us * us)
            wv = Wd * np.sin(np.pi * np.clip(us * 0.95 + 0.05, 0, 1)) ** 0.7
            dx = np.gradient(cxs)
            dy = np.gradient(cys)
            nrm = np.sqrt(dx * dx + dy * dy) + 1e-6
            nx, ny = -dy / nrm, dx / nrm
            left = np.stack([cxs + nx * wv / 2, cys + ny * wv / 2], -1)
            right = np.stack([cxs - nx * wv / 2, cys - ny * wv / 2], -1)[::-1]
            lc = tuple(float(v) * (0.8 + 0.4 * r.random()) for v in col)
            soft_poly(img, np.concatenate([left, right]), lc, alpha)
            if s > 0.06 and prox > 0.05:
                up = left if left[:, 1].mean() < right[::-1][:, 1].mean() else right[::-1]
                pts = np.round(up[1:-1] * 4).astype(np.int32)
                cv2.polylines(img, [pts], False, rimc, max(1, int(round(1.6 * s + 0.6))), cv2.LINE_AA, shift=2)
            side = -side
        # tassel
        tc = tuple(float(a * 0.85 + b * 0.15 * prox) for a, b in zip(col, pal["tassel"]))
        for k in range(3):
            ang = -math.pi / 2 + (k - 1) * 0.30 + (r.random() - 0.5) * 0.2
            L = (20 + 12 * r.random()) * s
            cv2.line(img, (int(top[0] * 4), int(top[1] * 4)),
                     (int((top[0] + math.cos(ang) * L) * 4), int((top[1] + math.sin(ang) * L) * 4)),
                     tc, max(1, int(2.2 * s)), cv2.LINE_AA, shift=2)
            cv2.line(alpha, (int(top[0] * 4), int(top[1] * 4)),
                     (int((top[0] + math.cos(ang) * L) * 4), int((top[1] + math.sin(ang) * L) * 4)),
                     1.0, max(1, int(2.2 * s)), cv2.LINE_AA, shift=2)

    # ---- per frame -------------------------------------------------------------
    def render(self, t, dur):
        p = self.p
        u = ease(t / max(dur, 1e-3)) * 0.85 + (t / max(dur, 1e-3)) * 0.15
        mode = p.get("camera", "dolly")
        if mode == "dolly":
            z = lerp(1.0, 1.13, u)
            Mf = camera(self.CW, self.CH, self.w, self.h, z, 0.5 + 0.02 * u, lerp(0.53, 0.51, u))
            Ms = camera(self.CW, self.CH, self.w, self.h, lerp(1.0, 1.04, u), 0.5 + 0.01 * u, lerp(0.53, 0.52, u))
        elif mode == "pan":
            d = p.get("pan", 1)
            cx0, cx1 = (0.37, 0.63) if d > 0 else (0.63, 0.37)
            Mf = camera(self.CW, self.CH, self.w, self.h, 1.03, lerp(cx0, cx1, u), 0.52)
            Ms = camera(self.CW, self.CH, self.w, self.h, 1.03, lerp(0.5 + (cx0 - 0.5) * 0.35, 0.5 + (cx1 - 0.5) * 0.35, u), 0.52)
        else:  # rise
            Mf = camera(self.CW, self.CH, self.w, self.h, 1.04, 0.5, lerp(0.585, 0.50, u))
            Ms = camera(self.CW, self.CH, self.w, self.h, 1.04, 0.5, lerp(0.545, 0.505, u))
        sky = self.view(self.sky, Ms)
        # wind sway: stronger for near (lower) plants
        xx, yy = self._grid()
        hy_out = _apply_affine(Mf, 0, self.hy)[1]
        depth = np.clip((yy - hy_out) / (self.h - hy_out), 0, 1.3)
        wind = p.get("wind", 1.0)
        dx = wind * (1.2 + 7.0 * depth ** 1.6) * np.sin(t * 1.7 + xx * 0.011 + yy * 0.021 + np.sin(xx * 0.003 + t * 0.6) * 2.0)
        field = self.view_remap(self.field, Mf, dx)
        fa = self.view_remap(self.field_a, Mf, dx)
        img = over(sky, field, fa)
        # god rays
        sx, sy = _apply_affine(Ms, *self.sun)
        if self.pal["sun"][0] > 0 or p.get("rays", 0):
            ang = np.arctan2(yy - sy, xx - sx)
            idx = (((ang / (2 * np.pi)) + 0.5 + t * 0.004) * len(self.rays)).astype(np.int32) % len(self.rays)
            d = np.sqrt((xx - sx) ** 2 + (yy - sy) ** 2)
            rays = smoothstep(0.55, 0.92, self.rays[idx]) * np.exp(-d / 380.0) * (1 - 0.7 * fa)
            img += np.array(self.pal["glow"], np.float32) * (0.26 * rays)[..., None]
        # dust motes drifting in the light
        layer = np.zeros((self.h // 2, self.w // 2, 3), np.float32)
        m = self.motes
        for i in range(len(m)):
            x = ((m[i, 0] + t * (0.004 + 0.01 * m[i, 2])) % 1.0) * self.w
            y = (m[i, 1] * 0.85 + 0.12 - t * 0.006 * m[i, 3] + 0.01 * math.sin(t * 1.3 + i)) % 1.0 * self.h
            prox = math.exp(-math.hypot(x - sx, y - sy) / 380.0)
            rad = 1 + int(m[i, 4] * 3)
            b = (0.25 + 1.6 * prox) * (0.5 + 0.5 * math.sin(t * 2 + i * 1.7))
            cv2.circle(layer, (int(x / 2), int(y / 2)), rad, (b * 1.0, b * 0.8, b * 0.5), -1, cv2.LINE_AA)
        layer = cv2.resize(blur(layer, 1.2), (self.w, self.h))
        img += layer * (0.6 if self.pal["sun"][0] > 0 else 0.15)
        # subtle lens ghosts
        if self.pal["sun"][0] > 0:
            cx, cy = self.w / 2, self.h / 2
            for k, (f, rad, col) in enumerate(((0.55, 46, (0.12, 0.09, 0.04)), (1.8, 70, (0.06, 0.04, 0.05)))):
                gx, gy = sx + (cx - sx) * f, sy + (cy - sy) * f
                g = np.exp(-(((xx - gx) ** 2 + (yy - gy) ** 2) / (rad * rad)) ** 2)
                img += np.array(col, np.float32) * g[..., None]
        return img


# ============================================================ shared drawing helpers
def corn_plant(img, alpha, x, yb, s, col, r, rimc=None, tassel=None, leaves=6, side_view=False, stalk=True):
    """One corn plant (stalk, arching leaves, tassel) at base (x, yb), scale s."""
    hgt = (300 + 80 * r.random()) * s
    lean = (r.random() - 0.5) * 18 * s
    c = tuple(float(v) for v in col)
    sw = max(1.0, 5.0 * s)
    top = (x + lean, yb - hgt)
    if stalk:
        soft_poly(img, [(x - sw, yb), (top[0] - sw * 0.6, top[1]), (top[0] + sw * 0.6, top[1]), (x + sw, yb)], c, alpha)
    side = 1 if r.random() > 0.5 else -1
    for i in range(leaves):
        a = 0.20 + 0.12 * i
        ax, ay = x + lean * a, yb - hgt * a
        L = (135 + 60 * r.random()) * s * (1.0 - 0.07 * i)
        Wd = (17 + 8 * r.random()) * s
        rise = 0.55 + 0.35 * r.random()
        droop = 1.0 + 0.5 * r.random()
        us = np.linspace(0, 1, 9)
        cxs = ax + side * L * us * (0.9 + 0.1 * r.random())
        cys = ay - L * (rise * us - droop * us * us)
        wv = Wd * np.sin(np.pi * np.clip(us * 0.95 + 0.05, 0, 1)) ** 0.7
        dx, dy = np.gradient(cxs), np.gradient(cys)
        nrm = np.sqrt(dx * dx + dy * dy) + 1e-6
        nx, ny = -dy / nrm, dx / nrm
        left = np.stack([cxs + nx * wv / 2, cys + ny * wv / 2], -1)
        right = np.stack([cxs - nx * wv / 2, cys - ny * wv / 2], -1)[::-1]
        lc = tuple(float(v) * (0.8 + 0.4 * r.random()) for v in col)
        soft_poly(img, np.concatenate([left, right]), lc, alpha)
        if rimc is not None and s > 0.06:
            up = left if left[:, 1].mean() < right[::-1][:, 1].mean() else right[::-1]
            pts = np.round(up[1:-1] * 4).astype(np.int32)
            cv2.polylines(img, [pts], False, rimc, max(1, int(round(1.6 * s + 0.6))), cv2.LINE_AA, shift=2)
        side = -side
    if tassel is not None:
        for k in range(3):
            ang = -math.pi / 2 + (k - 1) * 0.30 + (r.random() - 0.5) * 0.2
            L = (20 + 12 * r.random()) * s
            p0 = (int(top[0] * 4), int(top[1] * 4))
            p1 = (int((top[0] + math.cos(ang) * L) * 4), int((top[1] + math.sin(ang) * L) * 4))
            cv2.line(img, p0, p1, tassel, max(1, int(2.2 * s)), cv2.LINE_AA, shift=2)
            cv2.line(alpha, p0, p1, 1.0, max(1, int(2.2 * s)), cv2.LINE_AA, shift=2)


def farm_silhouette(img, alpha, x, hy, col, scale=1.0, lit_windows=None):
    c = tuple(float(v) for v in col)
    s = scale
    bw, bh = 46 * s, 30 * s
    soft_poly(img, [(x, hy), (x, hy - bh), (x + bw * 0.5, hy - bh - 16 * s), (x + bw, hy - bh), (x + bw, hy)], c, alpha)
    soft_poly(img, [(x + bw + 4 * s, hy), (x + bw + 4 * s, hy - 58 * s), (x + bw + 9 * s, hy - 64 * s),
                    (x + bw + 16 * s, hy - 64 * s), (x + bw + 21 * s, hy - 58 * s), (x + bw + 21 * s, hy)], c, alpha)
    hx = x - 70 * s
    soft_poly(img, [(hx, hy), (hx, hy - 20 * s), (hx + 18 * s, hy - 33 * s), (hx + 36 * s, hy - 20 * s), (hx + 36 * s, hy)], c, alpha)
    wins = []
    if lit_windows is not None:
        for wx, wy in ((hx + 7 * s, hy - 14 * s), (hx + 22 * s, hy - 14 * s)):
            soft_poly(img, [(wx, wy), (wx + 6 * s, wy), (wx + 6 * s, wy + 6 * s), (wx, wy + 6 * s)], lit_windows)
            wins.append((wx + 3 * s, wy + 3 * s))
    wx = x + 120 * s
    soft_poly(img, [(wx - 5 * s, hy), (wx - 1 * s, hy - 50 * s), (wx + 1 * s, hy - 50 * s), (wx + 5 * s, hy)], c, alpha)
    for k in range(8):
        a = k * math.pi / 4 + 0.3
        cv2.line(img, (int(wx), int(hy - 50 * s)), (int(wx + math.cos(a) * 11 * s), int(hy - 50 * s + math.sin(a) * 11 * s)),
                 c, max(1, int(1.2 * s)), cv2.LINE_AA)
        if alpha is not None:
            cv2.line(alpha, (int(wx), int(hy - 50 * s)), (int(wx + math.cos(a) * 11 * s), int(hy - 50 * s + math.sin(a) * 11 * s)),
                     1.0, max(1, int(1.2 * s)), cv2.LINE_AA)
    return wins


def wrap_strip(draw_fn, w, h, period, seed):
    """Render a horizontally tileable RGBA strip by drawing each element at x and x +- period."""
    img = np.zeros((h, w, 3), np.float32)
    a = np.zeros((h, w), np.float32)
    draw_fn(img, a, seed)
    return img, a


def scroll(img, offset):
    """Horizontal wrap-around crop starting at `offset` (float px), same width as img."""
    w = img.shape[1]
    o = offset % w
    i = int(o)
    f = o - i
    a = np.roll(img, -i, axis=1)
    if f > 1e-3:
        b = np.roll(a, -1, axis=1)
        a = a * (1 - f) + b * f
    return a


# =============================================================================== truck
@register("truck")
class Truck(Scene):
    """Tracking shot: an old pickup on a dirt road between cornfields, kicking up dust."""
    grade = dict(lift=(0.02, 0.015, 0.03), gamma=(1.0, 1.0, 1.0), gain=(1.04, 1.0, 0.94), saturation=1.06,
                 contrast=1.05, exposure=1.0)

    def __init__(self, w, h, params=None, seed=2):
        super().__init__(w, h, params, seed)
        r = rng(seed)
        self.hy = 0.47 * h
        self.sun = (0.80 * w, 0.20 * h)
        sky = vgrad(h, w, [(0, (0.22, 0.30, 0.48)), (0.30, (0.62, 0.52, 0.52)), (0.47, (1.35, 0.92, 0.55)), (1.0, (1.35, 0.92, 0.55))])
        d = radial(h, w, *self.sun)
        sky += np.array((1.5, 0.9, 0.45), np.float32) * (0.9 * np.exp(-d / 60) + 0.5 * np.exp(-d / 240))[..., None]
        sky = over(sky, np.array((9, 7.5, 5), np.float32), smoothstep(20, 15, d))
        n = fbm(h, w, 200, 5, seed + 1, aspect=4)
        dens = smoothstep(0.58, 0.8, n) * (1 - smoothstep(0.2, 0.42, np.arange(h)[:, None] / h))
        sky = over(sky, np.array((1.3, 0.8, 0.6), np.float32), dens * 0.6)
        self.sky = sky
        Wt = 2 * w
        # far tree line strip
        far = np.zeros((h, Wt, 3), np.float32)
        far_a = np.zeros((h, Wt), np.float32)
        ridge = fbm(1, Wt, 70, 4, seed + 2, wrap_x=True)[0]
        top = self.hy - 6 - 16 * ridge
        yy = np.arange(h, dtype=np.float32)[:, None]
        m = (yy >= top[None, :]) & (yy < self.hy + 40)
        far[m] = (0.62, 0.42, 0.33)
        far_a[m] = 1
        self.far, self.far_a = far, far_a
        # corn wall behind the road
        CWt = 2 * w
        wall = np.zeros((h, CWt, 3), np.float32)
        wall_a = np.zeros((h, CWt), np.float32)
        base_y = 0.645 * h
        dark, leaf = np.array((0.035, 0.04, 0.012)), np.array((0.18, 0.21, 0.05))
        rimc = (2.6, 1.7, 0.75)
        cv2.rectangle(wall, (0, int(0.46 * h)), (CWt, int(base_y) + 2), (0.04, 0.045, 0.016), -1)
        cv2.rectangle(wall_a, (0, int(0.46 * h)), (CWt, int(base_y) + 2), 1.0, -1)
        for k in range(300):
            x = r.random() * CWt
            yb = base_y + r.random() * 6
            sc = 0.58 + 0.14 * r.random()
            col = dark * 0.5 + leaf * r.random() * 0.8
            for xx_ in (x, x - CWt, x + CWt):
                rr = np.random.default_rng(seed * 1000 + k)
                corn_plant(wall, wall_a, xx_, yb, sc, col, rr, rimc=tuple(v * (0.4 + 0.6 * math.exp(-abs(x - 0.8 * CWt) / CWt)) for v in rimc),
                           tassel=(0.5, 0.38, 0.15))
        self.wall, self.wall_a = wall, np.clip(wall_a, 0, 1)
        # road
        road = np.zeros((h, Wt, 3), np.float32)
        rn = fbm(h, Wt, 30, 4, seed + 3, aspect=3, wrap_x=True)
        ry0, ry1 = int(0.64 * h), int(0.84 * h)
        band = np.zeros((h, 1), np.float32)
        band[ry0:] = 1
        roadc = np.array((0.60, 0.44, 0.28), np.float32) * (0.78 + 0.45 * rn[..., None])
        tracks = np.exp(-((yy - 0.71 * h) / 4.0) ** 2) + np.exp(-((yy - 0.79 * h) / 5.5) ** 2)
        roadc = roadc * (1 - 0.3 * tracks[..., None])
        verge = np.array((0.20, 0.17, 0.08), np.float32) * (0.6 + 0.8 * fbm(h, Wt, 12, 4, seed + 8, aspect=0.4, wrap_x=True)[..., None])
        vmask = smoothstep(ry1 - 4, ry1 + 6, yy)[..., None]
        road = (roadc * (1 - vmask) + verge * vmask) * band[..., None]
        self.road, self.road_a = road.astype(np.float32), np.broadcast_to(band, (h, Wt)).astype(np.float32)
        # foreground corn (blurred, dark)
        fg = np.zeros((h, Wt, 3), np.float32)
        fg_a = np.zeros((h, Wt), np.float32)
        for k in range(12):
            x = r.random() * Wt
            for xx_ in (x, x - Wt, x + Wt):
                rr = np.random.default_rng(seed * 2000 + k)
                corn_plant(fg, fg_a, xx_, h * 1.42, 1.1 + 0.3 * r.random(), np.array((0.02, 0.022, 0.008)), rr,
                           rimc=(0.9, 0.6, 0.3), leaves=5, stalk=False)
        self.fg, self.fg_a = blur(fg, 5), blur(np.clip(fg_a, 0, 1), 5)
        self.dust_seed = seed + 50

    def _truck(self, img, x, y, t):
        """Pickup silhouette with rim light; returns rear-wheel position."""
        s = self.w / 1080 * 1.45
        bob = math.sin(t * 9.0) * 1.2 + math.sin(t * 5.3) * 0.8
        y = y + bob
        body = (0.09, 0.15, 0.16)
        P = lambda pts: [(x + px * s, y + py * s) for px, py in pts]
        # bed + cab (facing right)
        soft_poly(img, P([(-150, -18), (-150, -62), (-28, -62), (-26, -100), (22, -104), (58, -66), (110, -60), (118, -40),
                          (118, -18)]), body)
        soft_poly(img, P([(-18, -96), (18, -98), (48, -66), (-20, -64)]), (0.75, 0.58, 0.50))     # window glow
        soft_poly(img, P([(-150, -30), (118, -30), (118, -24), (-150, -24)]), (0.05, 0.08, 0.09))    # body crease
        soft_poly(img, P([(-160, -60), (-150, -66), (-150, -58), (-160, -54)]), body)
        soft_poly(img, P([(112, -50), (120, -50), (120, -42), (112, -42)]), (1.4, 1.2, 0.8))       # headlight
        soft_poly(img, P([(-156, -26), (124, -26), (126, -16), (-158, -16)]), (0.15, 0.15, 0.15))  # bumpers
        # rim light along the top edges
        rim = (2.4, 1.6, 0.8)
        pts = np.array(P([(-150, -62), (-28, -62), (-26, -100), (22, -104), (58, -66), (110, -60)]), np.float32)
        cv2.polylines(img, [np.round(pts * 4).astype(np.int32)], False, rim, 2, cv2.LINE_AA, shift=2)
        # wheels
        for wx in (-100, 70):
            cx, cy = x + wx * s, y - 16 * s
            cv2.circle(img, (int(cx * 4), int(cy * 4)), int(22 * s * 4), (0.02, 0.02, 0.02), -1, cv2.LINE_AA, shift=2)
            cv2.circle(img, (int(cx * 4), int(cy * 4)), int(10 * s * 4), (0.25, 0.22, 0.2), -1, cv2.LINE_AA, shift=2)
            for k in range(5):
                a = -t * 14 + k * 2 * math.pi / 5
                cv2.line(img, (int(cx * 4), int(cy * 4)), (int((cx + math.cos(a) * 9 * s) * 4), int((cy + math.sin(a) * 9 * s) * 4)),
                         (0.05, 0.05, 0.05), 2, cv2.LINE_AA, shift=2)
        return x - 100 * s, y - 8 * s, s

    def render(self, t, dur):
        w, h = self.w, self.h
        v = self.p.get("speed", 420.0)
        img = self.sky.copy()
        far = scroll(self.far, t * v * 0.04)[:, :w]
        img = over(img, far, scroll(self.far_a, t * v * 0.04)[:, :w])
        img = over(img, scroll(self.wall, t * v * 0.55)[:, :w], scroll(self.wall_a, t * v * 0.55)[:, :w])
        img = over(img, scroll(self.road, t * v)[:, :w], scroll(self.road_a, t * v)[:, :w])
        # truck shadow
        tx, ty = 0.47 * w + math.sin(t * 0.5) * 10, 0.80 * h
        sh = np.zeros((h, w), np.float32)
        cv2.ellipse(sh, (int(tx - 15), int(ty - 2)), (int(205 * w / 1080), 11), 0, 0, 360, 1.0, -1)
        img *= (1 - 0.45 * blur(sh, 5))[..., None]
        rx, ry, s = self._truck(img, tx, ty, t)
        # dust plume: particles emitted from the rear wheel, left behind as the truck moves on
        lay = np.zeros((h // 2, w // 2), np.float32)
        lit = np.zeros((h // 2, w // 2), np.float32)
        rate = 26.0
        k1 = int(t * rate)
        for k in range(max(0, k1 - int(3.2 * rate)), k1 + 1):
            age = t - k / rate
            if age < 0:
                continue
            hr = hash01(np.array([k, k + 1, k + 2, k + 3]), self.dust_seed)
            px = rx - age * v * (0.70 + 0.2 * hr[0]) - 10
            py = ry - age * (30 + 50 * hr[1]) - 60 * (1 - math.exp(-age * 1.6)) * (0.5 + hr[2])
            rad = (10 + 75 * (1 - math.exp(-age * 0.8))) * s * (0.7 + 0.6 * hr[3])
            a = math.exp(-age * 0.75) * min(1.0, age * 10) * 0.8
            if px + rad < -50:
                continue
            cv2.circle(lay, (int(px / 2), int(py / 2)), max(1, int(rad / 2)), a, -1, cv2.LINE_AA)
        lay = np.clip(blur(lay, 7), 0, 1)
        lay = cv2.resize(lay, (w, h))
        nz = fbm(h // 4, w // 4, 20, 3, int(t * 3) % 50 + 900)
        lay *= cv2.resize(0.65 + 0.7 * nz, (w, h))
        xx, yy = self._grid()
        glow = np.exp(-np.abs(xx - self.sun[0]) / 500.0)
        dust_col = np.array((1.7, 1.2, 0.72), np.float32) * (0.75 + 1.1 * glow[..., None])
        img = over(img, dust_col, np.clip(lay, 0, 0.92))
        img = over(img, scroll(self.fg, t * v * 1.9)[:, :w] * 0.8, scroll(self.fg_a, t * v * 1.9)[:, :w])
        # warm atmospheric glow from the low sun
        d = np.sqrt((xx - self.sun[0]) ** 2 + (yy - self.sun[1]) ** 2)
        img += np.array((0.5, 0.3, 0.12), np.float32) * np.exp(-d / 420)[..., None] * 0.6
        return img


# =============================================================================== storm
@register("storm")
class Storm(Scene):
    """A towering wall of dust rolls across the plains toward a lone farm."""
    grade = dict(lift=(0.03, 0.02, 0.01), gamma=(1.0, 0.98, 0.95), gain=(1.02, 0.97, 0.88), saturation=0.92,
                 contrast=1.08, exposure=1.0)

    def __init__(self, w, h, params=None, seed=3):
        super().__init__(w, h, params, seed)
        r = rng(seed)
        self.CW, self.CH = int(w * 1.3), int(h * 1.25)
        CW, CH = self.CW, self.CH
        self.hy = 0.66 * CH
        sky = vgrad(CH, CW, [(0, (0.42, 0.40, 0.42)), (0.5, (0.85, 0.66, 0.45)), (0.66, (1.05, 0.80, 0.52)), (1, (1.05, 0.8, 0.52))])
        self.sunp = (0.78 * CW, 0.25 * CH)
        d = radial(CH, CW, *self.sunp)
        sky += np.array((0.8, 0.6, 0.35), np.float32) * (0.6 * np.exp(-d / 90) + 0.4 * np.exp(-d / 300))[..., None]
        sky = over(sky, np.array((3.0, 2.6, 2.0), np.float32), smoothstep(22, 18, d))
        self.sky = sky
        # ground
        yy = np.arange(CH, dtype=np.float32)[:, None]
        g = fbm(CH, CW, 50, 5, seed + 1, aspect=8)
        u = np.clip((yy - self.hy) / (CH - self.hy), 0, 1)
        ground = np.array((0.32, 0.24, 0.15), np.float32) * (0.6 + 0.8 * g[..., None]) * (0.55 + 0.45 * (1 - u))[..., None]
        ground = over(ground, np.array((0.9, 0.7, 0.48), np.float32), np.exp(-u * 10)[..., None] * np.ones((1, CW, 1)) * 0.7)
        ga = (yy >= self.hy).astype(np.float32) * np.ones((1, CW), np.float32)
        land = ground * ga[..., None]
        farm_silhouette(land, ga, 0.70 * CW, self.hy + 1, (0.12, 0.09, 0.07), scale=1.5, lit_windows=(1.6, 1.0, 0.45))
        # fence posts receding
        for k in range(14):
            fx = 0.12 * CW + k * k * 5.5 + k * 22
            fy = self.hy + 120 - k * 8.5
            fh = 40 - k * 2.6
            if fh < 4:
                break
            cv2.line(land, (int(fx), int(fy)), (int(fx), int(fy - fh)), (0.06, 0.045, 0.03), max(1, int(4 - k * 0.25)), cv2.LINE_AA)
        self.land, self.land_a = land, ga
        # the dust wall: stacked billows
        WH = int(CH * 0.95)
        wall = np.zeros((WH, CW * 2, 3), np.float32)
        wa = np.zeros((WH, CW * 2), np.float32)
        base = WH - 10
        puffs = []
        for k in range(150):
            x = r.random() * CW * 1.9
            hgt = (0.35 + 0.65 * np.exp(-((x - 0.6 * CW) / (0.8 * CW)) ** 2)) * WH * 0.92
            y = base - r.random() ** 0.7 * hgt
            rad = int((0.10 + 0.12 * (y / base)) * CH * (0.6 + 0.6 * r.random()))
            puffs.append((y, x, rad))
        puffs.sort()
        cache = {}
        for y, x, rad in puffs:
            key = (rad // 6) * 6 + 6
            if key not in cache:
                cache[key] = puff_sprite(key, int(seed + key), light_dir=(-0.5, -0.85))
            shade, al = cache[key]
            n = shade.shape[0]
            x0, y0 = int(x - n / 2), int(y - n / 2)
            xs0, ys0 = max(0, -x0), max(0, -y0)
            x1, y1 = min(wall.shape[1], x0 + n), min(WH, y0 + n)
            if x1 <= max(x0, 0) or y1 <= max(y0, 0):
                continue
            sl = (slice(max(y0, 0), y1), slice(max(x0, 0), x1))
            sh = shade[ys0:ys0 + (y1 - max(y0, 0)), xs0:xs0 + (x1 - max(x0, 0))]
            aa = al[ys0:ys0 + (y1 - max(y0, 0)), xs0:xs0 + (x1 - max(x0, 0))]
            height = 1 - y / base
            dark = np.array((0.20, 0.14, 0.09), np.float32)
            litc = np.array((1.05, 0.80, 0.52), np.float32) * (0.6 + 0.6 * height)
            col = dark + (litc - dark) * np.clip(sh, 0, 1.2)[..., None]
            wall[sl] = wall[sl] * (1 - aa[..., None]) + col * aa[..., None]
            wa[sl] = np.maximum(wa[sl], aa)
        self.wall, self.wall_a = wall, wa
        self.WH = WH
        self.turb = fbm(CH // 2, CW, 60, 4, seed + 4, wrap_x=True)
        self.turb2 = fbm(CH // 2, CW, 60, 4, seed + 5, wrap_x=True)

    def render(self, t, dur):
        w, h, CW, CH = self.w, self.h, self.CW, self.CH
        u = t / max(dur, 1e-3)
        M = camera(CW, CH, w, h, lerp(1.0, 1.08, ease(u)), 0.52, 0.52)
        img = self.view(self.sky, M)
        # wall: advancing (scale + slide) with turbulent billowing
        adv = t * 0.06
        sc = 1.0 + adv * 0.6
        Mw = np.float32([[sc, 0, -0.35 * CW - CW * 0.25 * adv], [0, sc, self.hy + 8 - self.WH * sc]])
        Mt = M.copy()
        Mt = np.vstack([Mt, [0, 0, 1]]) @ np.vstack([Mw, [0, 0, 1]])
        Mt = Mt[:2]
        xx, yy = self._grid()
        ti = self.turb
        off = int(t * 25) % CW
        tdx = cv2.resize(np.roll(ti, -off, axis=1)[:, :CW], (w, h)) - 0.5
        tdy = cv2.resize(np.roll(self.turb2, off // 2, axis=1)[:, :CW], (w, h)) - 0.5
        wall = self.view_remap(self.wall, Mt, tdx * 26, tdy * 18, border=cv2.BORDER_CONSTANT)
        wa = self.view_remap(self.wall_a, Mt, tdx * 26, tdy * 18, border=cv2.BORDER_CONSTANT)
        # wall dims the sun as it rises
        img = over(img, wall, np.clip(wa, 0, 1))
        land = self.view(self.land, M)
        la = self.view(self.land_a, M)
        img = over(img, land, la)
        # airborne dust haze thickening to the left and over time
        haze = np.clip((1 - xx / w) * 0.35 + u * 0.12, 0, 0.6) * (0.4 + 0.6 * smoothstep(h * 0.2, h, yy))
        img = over(img, np.array((0.75, 0.56, 0.38), np.float32), haze)
        # streaking debris
        lay = np.zeros((h, w), np.float32)
        for k in range(70):
            hr = hash01(np.array([k * 3, k * 3 + 1, k * 3 + 2]), 77)
            x = ((hr[0] + t * (0.18 + 0.2 * hr[1])) % 1.2 - 0.1) * w
            y = (0.35 + 0.6 * hr[2]) * h
            cv2.line(lay, (int(x), int(y)), (int(x - 16 - 20 * hr[1]), int(y + 2)), 0.5, 1, cv2.LINE_AA)
        img += np.array((0.9, 0.7, 0.5), np.float32) * blur(lay, 0.8)[..., None] * 0.6
        return img


# =============================================================================== night
@register("night")
class Night(Scene):
    """The farm under the Milky Way; a figure on the truck looks up; a shooting star."""
    grade = dict(lift=(0.01, 0.015, 0.035), gamma=(1.0, 1.0, 1.05), gain=(0.98, 1.0, 1.08), saturation=1.05,
                 contrast=1.06, exposure=1.15)

    def __init__(self, w, h, params=None, seed=4):
        super().__init__(w, h, params, seed)
        self.CW, self.CH = int(w * 1.2), int(h * 1.35)
        CW, CH = self.CW, self.CH
        self.hy = 0.80 * CH
        sky = vgrad(CH, CW, [(0, (0.006, 0.010, 0.030)), (0.6, (0.02, 0.035, 0.08)), (0.8, (0.08, 0.09, 0.16)), (1, (0.08, 0.09, 0.16))])
        xx, yy = grid(CH, CW)
        # Milky Way band (diagonal)
        ang = math.radians(-58)
        dist = ((xx - CW * 0.55) * math.cos(ang) + (yy - CH * 0.35) * math.sin(ang))
        band = np.exp(-(dist / (CW * 0.16)) ** 2)
        n1 = fbm(CH, CW, 120, 6, seed + 1)
        n2 = fbm(CH, CW, 40, 5, seed + 2)
        glow = band * (0.4 + 0.9 * n1) * (0.6 + 0.4 * n2)
        lanes = smoothstep(0.55, 0.8, fbm(CH, CW, 70, 5, seed + 3)) * np.exp(-(dist / (CW * 0.06)) ** 2)
        glow = glow * (1 - 0.75 * lanes)
        core = np.exp(-(((xx - CW * 0.42) / (CW * 0.25)) ** 2 + ((yy - CH * 0.62) / (CH * 0.3)) ** 2))
        col = np.array((0.35, 0.38, 0.55), np.float32) + np.array((0.55, 0.40, 0.25), np.float32) * core[..., None]
        sky += col * glow[..., None] * 0.55
        st = stars(CH, CW, 9000, seed + 4, size_pow=4.0)
        dense = stars(CH, CW, 14000, seed + 5, size_pow=6.0) * band[..., None]
        sky += st * 0.9 + dense * 0.7
        sky += np.array((0.18, 0.12, 0.10), np.float32) * np.exp(-np.clip(self.hy - yy, 0, None) / 60)[..., None] * (yy < self.hy)[..., None]
        self.sky = sky.astype(np.float32)
        r = rng(seed + 9)
        self.twinkle = np.stack([r.random(140) * CW, r.random(140) * CH * 0.75, r.random(140), r.random(140)], -1)
        # ground silhouettes
        land = np.zeros((CH, CW, 3), np.float32)
        la = np.zeros((CH, CW), np.float32)
        ridge = fbm1d(CW, 18, 3, seed + 6)
        top = self.hy - 4 - 7 * ridge
        m = (yy >= top[None, :])
        land[m] = (0.006, 0.007, 0.012)
        la[m] = 1
        wins = farm_silhouette(land, la, 0.18 * CW, self.hy - 2, (0.004, 0.005, 0.009), scale=2.2, lit_windows=(3.0, 1.8, 0.7))
        self.windows = wins
        # truck + figure on the right, closer to camera
        tx, ty, s = 0.66 * CW, self.hy + 18, 1.25
        P = lambda pts: [(tx + px * s, ty + py * s) for px, py in pts]
        dark = (0.003, 0.004, 0.007)
        soft_poly(land, P([(-150, -18), (-150, -62), (-28, -62), (-26, -100), (22, -104), (58, -66), (110, -60), (118, -40), (118, 0), (-150, 0)]), dark, la)
        for wx in (-100, 70):
            cv2.circle(land, (int(tx + wx * s), int(ty - 16 * s)), int(22 * s), dark, -1, cv2.LINE_AA)
        # person standing in the truck bed, head tilted up
        bx, by = tx - 95 * s, ty - 62 * s
        soft_poly(land, [(bx - 9 * s, by), (bx - 7 * s, by - 48 * s), (bx - 11 * s, by - 50 * s), (bx - 9 * s, by - 78 * s),
                         (bx + 9 * s, by - 78 * s), (bx + 11 * s, by - 50 * s), (bx + 7 * s, by - 48 * s), (bx + 9 * s, by)], dark, la)
        cv2.ellipse(land, (int(bx + 2 * s), int(by - 88 * s)), (int(8 * s), int(10 * s)), -25, 0, 360, dark, -1, cv2.LINE_AA)
        cv2.ellipse(la, (int(bx + 2 * s), int(by - 88 * s)), (int(8 * s), int(10 * s)), -25, 0, 360, 1.0, -1, cv2.LINE_AA)
        # faint rim from the sky
        mask = la.copy()
        nb = cv2.warpAffine(mask, np.float32([[1, 0, 0], [0, 1, 2]]), (CW, CH))
        rim = np.clip(mask - nb, 0, 1)
        land += np.array((0.10, 0.12, 0.22), np.float32) * rim[..., None]
        self.land, self.land_a = land, la

    def render(self, t, dur):
        w, h, CW, CH = self.w, self.h, self.CW, self.CH
        u = ease(t / max(dur, 1e-3))
        cy = lerp(self.p.get("cy0", 0.60), self.p.get("cy1", 0.47), u)
        z = lerp(1.0, 1.05, u)
        M = camera(CW, CH, w, h, z, 0.5, cy)
        Ms = camera(CW, CH, w, h, z * 0.99, 0.5, lerp(cy, 0.5, 0.12))
        img = self.view(self.sky, Ms)
        # twinkling bright stars
        for x, y, ph, b in self.twinkle:
            ox, oy = _apply_affine(Ms, x, y)
            if 0 <= ox < w and 0 <= oy < h:
                k = 0.5 + 0.5 * math.sin(t * (2 + 3 * b) + ph * 20)
                c = 0.6 + 1.6 * b * k
                cv2.circle(img, (int(ox), int(oy)), 1, (c, c, c * 1.1), -1, cv2.LINE_AA)
        # shooting star
        ts = self.p.get("meteor", 0.55) * dur
        if ts <= t <= ts + 0.9:
            a = (t - ts) / 0.9
            x0, y0 = 0.25 * w, 0.12 * h
            x1, y1 = 0.62 * w, 0.42 * h
            hx, hy = x0 + (x1 - x0) * a, y0 + (y1 - y0) * a
            lay = np.zeros((h, w), np.float32)
            n_seg = 24
            for k in range(n_seg):
                f0, f1 = k / n_seg, (k + 1) / n_seg
                p0 = (hx - (x1 - x0) * 0.25 * f0, hy - (y1 - y0) * 0.25 * f0)
                p1 = (hx - (x1 - x0) * 0.25 * f1, hy - (y1 - y0) * 0.25 * f1)
                cv2.line(lay, (int(p0[0] * 4), int(p0[1] * 4)), (int(p1[0] * 4), int(p1[1] * 4)),
                         (1 - f0) ** 1.5 * math.sin(math.pi * a), 2, cv2.LINE_AA, shift=2)
            lay = blur(lay, 0.8) * 5
            img += np.array((1.0, 1.0, 1.2), np.float32) * lay[..., None]
        land = self.view(self.land, M)
        la = self.view(self.land_a, M)
        img = over(img, land, la)
        for (wx, wy) in self.windows:
            ox, oy = _apply_affine(M, wx, wy)
            d = radial(h, w, ox, oy)
            img += np.array((1.0, 0.55, 0.2), np.float32) * (0.5 * np.exp(-d / 18) + 0.12 * np.exp(-d / 70))[..., None]
        return img


# =============================================================================== launch
def rocket_shape(img, alpha, x, y_base, H, col, rim=None, shade=None):
    """Slender two-stage rocket silhouette standing on y_base, height H.
    shade=(light_rgb, dark_rgb) paints a cylindrical left-lit gradient over the body."""
    wd = H * 0.085
    pts = [(x - wd / 2, y_base), (x - wd / 2, y_base - H * 0.80), (x - wd * 0.42, y_base - H * 0.88),
           (x, y_base - H), (x + wd * 0.42, y_base - H * 0.88), (x + wd / 2, y_base - H * 0.80), (x + wd / 2, y_base)]
    soft_poly(img, pts, col, alpha)
    for sgn in (-1, 1):  # fins
        soft_poly(img, [(x + sgn * wd / 2, y_base), (x + sgn * wd * 1.15, y_base + H * 0.02), (x + sgn * wd / 2, y_base - H * 0.10)], col, alpha)
    for sgn in (-1, 1):  # side boosters
        bx = x + sgn * wd * 0.78
        soft_poly(img, [(bx - wd * 0.22, y_base), (bx - wd * 0.22, y_base - H * 0.42), (bx, y_base - H * 0.47),
                        (bx + wd * 0.22, y_base - H * 0.42), (bx + wd * 0.22, y_base)], col, alpha)
    if shade is not None:
        x0, x1 = int(x - wd * 1.5), int(x + wd * 1.5) + 1
        y0, y1 = int(y_base - H) - 1, int(y_base + H * 0.03) + 1
        sub = alpha[y0:y1, x0:x1]
        xs = (np.arange(x0, x1, dtype=np.float32) - x) / (wd * 1.2)
        f = np.clip(0.5 - 0.6 * xs, 0, 1) ** 1.3
        light, darkc = np.array(shade[0], np.float32), np.array(shade[1], np.float32)
        colr = darkc + (light - darkc) * f[None, :, None]
        img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - sub[..., None]) + colr * sub[..., None]
        for band_y in (0.30, 0.55, 0.80):   # stage separation lines
            yb_ = int(y_base - H * band_y)
            cv2.line(img, (int(x - wd / 2), yb_), (int(x + wd / 2), yb_), tuple(float(v) * 0.5 for v in shade[1]), 1, cv2.LINE_AA)
    if rim is not None:
        cv2.line(img, (int((x + wd / 2) * 4), int((y_base - H * 0.05) * 4)), (int((x + wd / 2) * 4), int((y_base - H * 0.8) * 4)),
                 rim, 2, cv2.LINE_AA, shift=2)


@register("launch")
class Launch(Scene):
    """Pre-dawn launch: 'pad' (floodlit, venting), 'liftoff' (ignition), 'ascent' (contrail into sunrise)."""
    grade = dict(lift=(0.01, 0.015, 0.035), gamma=(1.0, 1.0, 1.02), gain=(1.02, 0.99, 1.0), saturation=1.08,
                 contrast=1.07, exposure=1.0)

    def __init__(self, w, h, params=None, seed=5):
        super().__init__(w, h, params, seed)
        self.phase = self.p.get("phase", "pad")
        self.CW, self.CH = int(w * 1.25), int(h * 1.5)
        CW, CH = self.CW, self.CH
        self.hy = 0.80 * CH
        sky = vgrad(CH, CW, [(0, (0.010, 0.016, 0.050)), (0.45, (0.05, 0.07, 0.18)), (0.70, (0.30, 0.25, 0.38)),
                             (0.80, (1.10, 0.55, 0.32)), (1, (1.1, 0.55, 0.32))])
        st = stars(CH, CW, 3500, seed + 1)
        yy = np.arange(CH, dtype=np.float32)[:, None]
        sky += st * (1 - smoothstep(0.1 * CH, 0.6 * CH, yy))[..., None] * 0.8
        dawn = radial(CH, CW, 0.72 * CW, self.hy + 30, 1.0, 0.35)
        sky += np.array((1.2, 0.55, 0.25), np.float32) * np.exp(-dawn / 260)[..., None] * 0.8
        n = fbm(CH, CW, 220, 5, seed + 2, aspect=5)
        band = smoothstep(0.45 * CH, 0.62 * CH, yy) * (1 - smoothstep(0.70 * CH, 0.79 * CH, yy))
        dens = smoothstep(0.55, 0.8, n) * band
        sky = over(sky, np.array((0.55, 0.30, 0.30), np.float32), dens * 0.7)
        self.sky = sky.astype(np.float32)
        self.cloud_dens = dens.astype(np.float32)
        land = np.zeros((CH, CW, 3), np.float32)
        la = np.zeros((CH, CW), np.float32)
        m = (yy >= self.hy) & np.ones((1, CW), bool)
        land[m] = (0.012, 0.012, 0.02)
        la[m] = 1
        self.pad_x = 0.55 * CW
        if self.phase == "pad":
            H = 0.28 * CH
            # tower (lattice)
            tx = self.pad_x - H * 0.16
            soft_poly(land, [(tx - 10, self.hy), (tx - 7, self.hy - H * 1.08), (tx + 7, self.hy - H * 1.08), (tx + 10, self.hy)], (0.02, 0.02, 0.03), la)
            for k in range(26):
                yk = self.hy - k * H * 1.08 / 26
                cv2.line(land, (int(tx - 9), int(yk)), (int(tx + 9), int(yk - H * 0.04)), (0.06, 0.06, 0.08), 1, cv2.LINE_AA)
            for k in range(3):
                yk = int(self.hy - H * (0.3 + 0.25 * k))
                cv2.line(land, (int(tx), yk), (int(self.pad_x - 5), yk), (0.03, 0.03, 0.04), 3, cv2.LINE_AA)
            rocket_shape(land, la, self.pad_x, self.hy - 6, H, (0.3, 0.3, 0.33), shade=((1.1, 1.05, 1.0), (0.10, 0.11, 0.15)))
            soft_poly(land, [(self.pad_x - 70, self.hy), (self.pad_x - 50, self.hy - 8), (self.pad_x + 50, self.hy - 8), (self.pad_x + 70, self.hy)],
                      (0.03, 0.03, 0.04), la)
            self.rocket_h = H
            self.lights = [(tx - 80, self.hy - 30), (tx + 160, self.hy - 26), (tx - 200, self.hy - 12), (tx + 300, self.hy - 14)]
        # foreground cornfield silhouette
        r = rng(seed + 21)
        fy = self.hy + 0.10 * CH
        cv2.rectangle(land, (0, int(fy)), (CW, CH), (0.006, 0.006, 0.01), -1)
        cv2.rectangle(la, (0, int(fy)), (CW, CH), 1.0, -1)
        for k in range(120):
            x = r.random() * CW
            if abs(x - self.pad_x) < 0.16 * CW:
                continue
            corn_plant(land, la, x, fy + 8 + r.random() * 20, 0.42 + 0.12 * r.random(),
                       np.array((0.006, 0.006, 0.01)), r, leaves=5, tassel=(0.006, 0.006, 0.01))
        self.land, self.land_a = land, la
        self.puffs = {}

    def _plume_layer(self, t, x, y, scale, intensity, spread):
        """Billowing exhaust clouds at (x, y) (output px); returns (rgb, alpha)."""
        w, h = self.w, self.h
        col = np.zeros((h // 2, w // 2, 3), np.float32)
        a = np.zeros((h // 2, w // 2), np.float32)
        n = 70
        for k in range(n):
            hr = hash01(np.array([k * 5, k * 5 + 1, k * 5 + 2, k * 5 + 3, k * 5 + 4]), 404)
            age = (t * 0.9 + hr[0] * 3.0) % 3.0
            sgn = -1 if hr[1] < 0.5 else 1
            dist = (age ** 0.6) * spread * (0.4 + 0.8 * hr[2])
            px = x + sgn * dist
            py = y - age * 14 * scale * hr[3] - 6
            rad = (18 + 45 * age ** 0.7) * scale * (0.6 + 0.6 * hr[4])
            light = math.exp(-dist / (spread * 0.35)) * intensity
            c = (0.5 + 2.6 * light, 0.42 + 1.4 * light, 0.38 + 0.5 * light)
            cv2.circle(col, (int(px / 2), int(py / 2)), max(1, int(rad / 2)), c, -1, cv2.LINE_AA)
            cv2.circle(a, (int(px / 2), int(py / 2)), max(1, int(rad / 2)), 0.85, -1, cv2.LINE_AA)
        cov = (a > 0).astype(np.float32)
        col = blur(col, 4) / np.maximum(blur(cov, 4), 1e-3)[..., None]
        a = blur(a, 4)
        return cv2.resize(col, (w, h)), cv2.resize(a, (w, h))

    def _flame(self, img, x, y, length, width, t, power=1.0):
        h, w = self.h, self.w
        xx, yy = self._grid()
        flick = 1 + 0.12 * math.sin(t * 47) + 0.08 * math.sin(t * 31 + 1)
        L = length * flick
        u = (yy - y) / L
        core = np.exp(-((xx - x) / (width * (0.35 + 0.65 * np.clip(u, 0, 1)))) ** 2) * (u > 0) * np.exp(-np.clip(u, 0, None) * 2.2)
        img += np.array((6.0, 4.2, 2.2), np.float32) * (core * power)[..., None]
        d = np.sqrt((xx - x) ** 2 + ((yy - y) * 0.6) ** 2)
        img += np.array((3.0, 1.6, 0.6), np.float32) * (np.exp(-d / (width * 2.5)) * power)[..., None]
        img += np.array((0.8, 0.35, 0.12), np.float32) * (np.exp(-d / (width * 12)) * power)[..., None]
        return img

    def render(self, t, dur):
        w, h, CW, CH = self.w, self.h, self.CW, self.CH
        u = min(max(t / max(dur, 1e-3), 0.0), 1.0)    # shots are also sampled inside transition overhangs
        xx, yy = self._grid()
        if self.phase == "pad":
            M = camera(CW, CH, w, h, lerp(1.0, 1.12, ease(u)), 0.52, lerp(0.70, 0.66, ease(u)))
            img = self.view(self.sky, M)
            img = over(img, self.view(self.land, M), self.view(self.land_a, M))
            # floodlights + beams
            for (lx, ly) in self.lights:
                ox, oy = _apply_affine(M, lx, ly)
                d = radial(h, w, ox, oy)
                img += np.array((2.5, 2.2, 1.8), np.float32) * (np.exp(-d / 3) * 2 + np.exp(-d / 25) * 0.5)[..., None]
                ang = np.arctan2(yy - oy, xx - ox)
                beam = np.exp(-((ang + math.pi / 2 + (ox - w / 2) / w * 0.6) / 0.08) ** 2) * np.exp(-d / 400) * (yy < oy)
                img += np.array((0.35, 0.33, 0.30), np.float32) * beam[..., None] * 0.6
            # venting steam near the base
            rx, ry = _apply_affine(M, self.pad_x, self.hy - 4)
            col, a = self._plume_layer(t * 0.5, rx, ry, 0.30, 0.0, 70)
            img = over(img, col * 0.5 + 0.12, a * 0.55)
            return img
        if self.phase == "liftoff":
            # close: rocket rising out of a sea of fire and smoke
            img = vgrad(h, w, [(0, (0.01, 0.02, 0.07)), (0.55, (0.08, 0.08, 0.18)), (1, (0.55, 0.28, 0.16))])
            img += stars(h, w, 900, 55) * 0.5 * (1 - smoothstep(0, h * 0.6, self._grid()[1]))[..., None]
            rise = (u ** 1.5) * h * 0.42
            rx = w * 0.5 + math.sin(t * 40) * 1.5
            base = h * 0.80 - rise
            H = h * 0.62
            lay = np.zeros((h, w, 3), np.float32)
            la = np.zeros((h, w), np.float32)
            rocket_shape(lay, la, rx, base, H, (0.08, 0.08, 0.1), shade=((0.45, 0.45, 0.5), (0.03, 0.03, 0.04)))
            # underside lit by flame
            img = self._flame(img, rx, base + 6, h * 0.5, w * 0.03, t, power=1.0)
            lit = np.exp(-np.abs(yy - base) / 60) * la
            lay += np.array((2.2, 1.2, 0.5), np.float32) * lit[..., None] * 0.6
            img = over(img, lay, la)
            col, a = self._plume_layer(t, rx, h * 1.02, 1.5, 1.0 - u * 0.3, w * 0.55)
            img = over(img, col, a)
            img = self._flame(img, rx, base + 4, h * 0.30, w * 0.022, t, power=0.6)
            if 0.0 <= t < 0.35:   # ignition flash
                img += (1 - t / 0.35) * 2.0
            return img
        # ascent
        def path(s):
            return (0.30 * CW + (0.45 * CW) * s ** 1.7, self.hy - (0.80 * CH) * (1 - (1 - s) ** 1.6))
        sNow = 0.22 + 0.40 * u
        hx_c, hy_c = path(sNow)
        cy = np.clip((hy_c + 0.22 * h) / CH, 0.5 * h / CH + 0.001, 1 - 0.5 * h / CH - 0.001)
        cx = np.clip(hx_c / CW, 0.5 * w / CW + 0.001, 1 - 0.5 * w / CW - 0.001)
        M = camera(CW, CH, w, h, 1.0, float(lerp(0.5, cx, 0.5)), float(cy))
        img = self.view(self.sky, M)
        img = over(img, self.view(self.land, M), self.view(self.land_a, M))
        pts = np.array([path(sv) for sv in np.linspace(0, sNow, 80)], np.float32)
        out = np.stack(_apply_affine(M, pts[:, 0], pts[:, 1]), -1)
        trail = np.zeros((h, w), np.float32)
        for i in range(len(out) - 1):
            f = i / len(out)
            cv2.line(trail, tuple(np.int32(out[i] * 4)), tuple(np.int32(out[i + 1] * 4)), 0.6 + 0.4 * f,
                     max(1, int(10 * (1 - f) + 2)), cv2.LINE_AA, shift=2)
        trail = blur(trail, 5)
        # sunlit upper plume (twilight glow) vs shadowed lower part
        sunlit = smoothstep(0.55 * h, 0.15 * h, yy)
        tc = np.array((0.35, 0.35, 0.45), np.float32) * (1 - sunlit[..., None]) + np.array((1.6, 1.0, 0.75), np.float32) * sunlit[..., None]
        img = over(img, tc, np.clip(trail, 0, 0.85))
        hx, hy_ = out[-1]
        img = self._flame(img, hx, hy_, 40, 4, t, power=0.8)
        d = radial(h, w, hx, hy_)
        img += np.array((1.2, 0.8, 0.5), np.float32) * (np.exp(-d / 90) * 0.35)[..., None]
        return img


# =============================================================================== orbit
@register("orbit")
class Orbit(Scene):
    """Low orbit: Earth's limb at sunrise, a spacecraft drifting past."""
    grade = dict(lift=(0.0, 0.005, 0.02), gamma=(1.0, 1.0, 1.0), gain=(1.0, 1.0, 1.03), saturation=1.1,
                 contrast=1.08, exposure=1.0)

    def __init__(self, w, h, params=None, seed=6):
        super().__init__(w, h, params, seed)
        r = rng(seed)
        self.R = 2.6 * w
        self.c = (0.38 * w, 0.62 * h + self.R)
        self.sun = (0.70 * w, 0.0)
        xx, yy = grid(h, w)
        d = np.sqrt((xx - self.c[0]) ** 2 + (yy - self.c[1]) ** 2)
        self.limb_y = lambda x: self.c[1] - np.sqrt(np.maximum(self.R ** 2 - (x - self.c[0]) ** 2, 0))
        sy = float(self.limb_y(np.array(self.sun[0])) - 6)
        self.sun = (self.sun[0], sy)
        self.d = d
        bg = stars(h, w, 2200, seed + 1) * 0.6
        # atmosphere glow above the limb
        alt = d - self.R
        sd = np.sqrt((xx - self.sun[0]) ** 2 + (yy - self.sun[1]) ** 2)
        near_sun = np.exp(-sd / 380)
        atm = np.exp(-np.clip(alt, 0, None) / 9) * (alt > -2)
        atm2 = np.exp(-np.clip(alt, 0, None) / 40) * (alt > -2)
        blue = np.array((0.15, 0.35, 0.9), np.float32)
        orange = np.array((1.6, 0.65, 0.25), np.float32)
        bg += (blue * (atm * 0.9 + atm2 * 0.3)[..., None]) * (0.3 + 0.7 * np.exp(-sd / 900))[..., None]
        bg += orange * (atm * near_sun * 1.6 + atm2 * near_sun * 0.5)[..., None]
        self.bg = bg.astype(np.float32)
        # planet surface texture in a wide tile (scrolls slowly)
        self.TW = 2 * w
        self.clouds = fbm(h, self.TW, 90, 6, seed + 2, aspect=2.5, wrap_x=True)
        self.land = fbm(h, self.TW, 160, 5, seed + 3, aspect=1.5, wrap_x=True)
        self.inside = (d < self.R).astype(np.float32)
        self.sd = sd
        self.alt = alt

    def _craft(self, img, x, y, s, t):
        dark = (0.015, 0.016, 0.02)
        lay = np.zeros_like(img)
        a = np.zeros(img.shape[:2], np.float32)
        ang = math.radians(-8)
        def R(px, py):
            return (x + (px * math.cos(ang) - py * math.sin(ang)) * s, y + (px * math.sin(ang) + py * math.cos(ang)) * s)
        soft_poly(lay, [R(-70, -9), R(60, -9), R(78, 0), R(60, 9), R(-70, 9)], dark, a)          # hull
        soft_poly(lay, [R(-20, -14), R(10, -14), R(10, 14), R(-20, 14)], dark, a)                 # module
        for sgn in (-1, 1):                                                                        # solar wings
            soft_poly(lay, [R(-40, sgn * 9), R(-10, sgn * 9), R(-6, sgn * 70), R(-44, sgn * 70)], (0.03, 0.05, 0.10), a)
            for k in range(1, 6):
                p0, p1 = R(-42 + 0.3 * k, sgn * (9 + k * 11)), R(-8, sgn * (9 + k * 11))
                cv2.line(lay, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (0.08, 0.1, 0.16), 1, cv2.LINE_AA)
        p0, p1 = R(-70, -9), R(60, -9)
        cv2.line(lay, (int(p0[0] * 4), int(p0[1] * 4)), (int(p1[0] * 4), int(p1[1] * 4)), (2.2, 1.4, 0.8), 2, cv2.LINE_AA, shift=2)
        bl = 0.5 + 0.5 * math.sin(t * 6)
        e = R(-74, 0)
        cv2.circle(lay, (int(e[0]), int(e[1])), 2, (0.6 * bl, 1.5 * bl, 3 * bl), -1, cv2.LINE_AA)
        return over(img, lay, a)

    def render(self, t, dur):
        w, h = self.w, self.h
        u = t / max(dur, 1e-3)
        off = int(t * 14)
        cl = np.roll(self.clouds, -off, axis=1)[:, :w]
        ln = np.roll(self.land, -off // 2, axis=1)[:, :w]
        ocean = np.array((0.010, 0.035, 0.09), np.float32)
        landc = np.array((0.05, 0.06, 0.035), np.float32)
        surf = ocean + (landc - ocean) * smoothstep(0.6, 0.66, ln)[..., None]
        cloud = smoothstep(0.45, 0.8, cl)
        surf = surf + (np.array((0.9, 0.9, 0.95), np.float32) - surf) * cloud[..., None]
        # sunrise lighting near the limb under the sun, night side elsewhere
        light = np.exp(-self.sd / 520) * np.exp(-np.clip(-self.alt, 0, None) / 160)
        warm = np.array((1.6, 0.9, 0.55), np.float32)
        surf = surf * (0.04 + 1.6 * light[..., None]) * (0.6 + 0.4 * warm / 1.6)
        # city lights on the night side
        city = smoothstep(0.62, 0.7, ln) * smoothstep(0.75, 0.95, fbm(h, w, 6, 2, 31)) * (1 - light) * (1 - cloud)
        surf += np.array((1.0, 0.6, 0.25), np.float32) * city[..., None] * 0.25
        img = self.bg.copy()
        img = img * (1 - self.inside[..., None]) + surf * self.inside[..., None]
        # sun just over the limb with a starburst
        sx, sy = self.sun
        sd = self.sd
        rise = smoothstep(0.0, 0.6, u)
        sun_core = np.exp(-sd / 7) * 6 + np.exp(-sd / 45) * 1.2 + np.exp(-sd / 200) * 0.35
        xx, yy = self._grid()
        ang = np.arctan2(yy - sy, xx - sx)
        spikes = (np.abs(np.cos(ang * 3 + 0.4)) ** 400) * np.exp(-sd / 150) * 0.5
        vis = smoothstep(-6, 4, self.limb_y(xx) - yy)
        img += np.array((2.2, 1.9, 1.5), np.float32) * ((sun_core + spikes) * vis * (0.4 + 0.6 * rise))[..., None]
        cx = lerp(-0.05, 0.85, u) * w
        cy = 0.36 * h + math.sin(u * 2) * 8
        img = self._craft(img, cx, cy, 1.3, t)
        return img


# =============================================================================== planet
@register("planet")
class Planet(Scene):
    """A ringed giant world, lit from the side; a tiny craft for scale."""
    grade = dict(lift=(0.0, 0.0, 0.015), gamma=(1.0, 1.0, 1.0), gain=(1.03, 1.0, 0.97), saturation=1.08,
                 contrast=1.1, exposure=1.05)

    def __init__(self, w, h, params=None, seed=7):
        super().__init__(w, h, params, seed)
        self.CW, self.CH = int(w * 1.25), int(h * 1.25)
        CW, CH = self.CW, self.CH
        xx, yy = grid(CH, CW)
        cx, cy, R = 0.60 * CW, 0.55 * CH, 0.36 * CH
        bg = stars(CH, CW, 3000, seed + 1)
        neb = fbm(CH, CW, 260, 6, seed + 2)
        bg += np.array((0.10, 0.04, 0.12), np.float32) * smoothstep(0.5, 0.9, neb)[..., None] * 0.8
        bg += np.array((0.02, 0.05, 0.12), np.float32) * smoothstep(0.4, 0.8, fbm(CH, CW, 180, 5, seed + 3))[..., None] * 0.6
        tilt = math.radians(-14)
        # planet coordinates
        X, Y = (xx - cx) / R, (yy - cy) / R
        Xr = X * math.cos(tilt) - Y * math.sin(tilt)
        Yr = X * math.sin(tilt) + Y * math.cos(tilt)
        rr = np.sqrt(X * X + Y * Y)
        inside = smoothstep(1.003, 0.997, rr)
        Z = np.sqrt(np.clip(1 - rr * rr, 0, 1))
        lat = np.arcsin(np.clip(Yr, -1, 1))
        bands = fbm(CH, CW, 40, 4, seed + 4, aspect=12)
        bandv = np.sin(lat * 14 + bands * 3.0) * 0.5 + 0.5
        c1, c2, c3 = np.array((0.85, 0.66, 0.45)), np.array((0.62, 0.42, 0.28)), np.array((0.95, 0.86, 0.70))
        col = c1 + (c2 - c1) * bandv[..., None] + (c3 - c1) * smoothstep(0.6, 0.9, bands)[..., None] * 0.5
        L = np.array((-0.75, -0.35, 0.56))
        L = L / np.linalg.norm(L)
        ndl = np.clip(X * L[0] + Y * L[1] + Z * L[2], 0, 1)
        shade = ndl ** 0.8 + 0.02
        planet = col * shade[..., None] * 1.3
        limb = np.exp(-(1 - rr) * 30) * inside
        planet += np.array((0.4, 0.3, 0.25), np.float32) * (limb * ndl)[..., None]
        # rings: ellipse in tilted frame
        ring_r = np.sqrt(Xr ** 2 + (Yr / 0.22) ** 2)
        ring_tex = fbm(1, 600, 8, 4, seed + 5)[0]
        idx = np.clip(((ring_r - 1.35) / (2.35 - 1.35) * 599).astype(int), 0, 599)
        ring_a = ((ring_r > 1.35) & (ring_r < 2.35)).astype(np.float32) * (0.25 + 0.75 * ring_tex[idx])
        ring_a *= smoothstep(1.35, 1.42, ring_r) * smoothstep(2.35, 2.25, ring_r)
        front = (Yr > 0)  # lower half of the ring passes in front of the planet
        ring_col = np.array((1.0, 0.9, 0.75), np.float32) * (0.5 + 0.7 * ring_tex[idx])[..., None]
        # planet shadow on the rings (behind, away from light)
        in_shadow = smoothstep(0.1, 0.35, Xr) * smoothstep(1.05, 0.9, np.abs(Yr / 0.22)) * (Yr < 0)
        ring_col = ring_col * (1 - 0.75 * in_shadow[..., None])
        img = bg.copy()
        back = ring_a * (~front) * (1 - inside)
        img = over(img, ring_col, back)
        img = over(img, planet, inside)
        # ring shadow on the planet
        img = img * (1 - 0.6 * (smoothstep(0.02, 0.0, np.abs(Yr - 0.05) - 0.03) * inside)[..., None])
        img = over(img, ring_col, ring_a * front)
        atm = np.exp(-np.clip(rr - 1, 0, None) * 40) * (rr >= 1)
        img += np.array((0.35, 0.25, 0.2), np.float32) * (atm * 0.5)[..., None]
        self.img = img.astype(np.float32)

    def render(self, t, dur):
        u = ease(t / max(dur, 1e-3))
        M = camera(self.CW, self.CH, self.w, self.h, lerp(1.0, 1.10, u), lerp(0.52, 0.56, u), 0.52, rot_deg=lerp(0, 2.0, u))
        img = self.view(self.img, M)
        # tiny craft crossing with engine glow
        w, h = self.w, self.h
        x = lerp(0.15, 0.55, u) * w
        y = lerp(0.78, 0.66, u) * h
        cv2.line(img, (int(x - 8), int(y + 2)), (int(x + 8), int(y - 2)), (0.02, 0.02, 0.03), 3, cv2.LINE_AA)
        d = radial(h, w, x - 10, y + 2.5)
        img += np.array((0.6, 1.2, 2.5), np.float32) * (np.exp(-d / 2.5) * 2 + np.exp(-d / 14) * 0.3)[..., None]
        return img


# =============================================================================== black hole
@register("blackhole")
class BlackHole(Scene):
    """A black hole with a lensed, rotating accretion disk (stylised)."""
    grade = dict(lift=(0.0, 0.0, 0.0), gamma=(1.0, 1.0, 1.0), gain=(1.05, 1.0, 0.95), saturation=1.05,
                 contrast=1.08, exposure=0.95)

    def __init__(self, w, h, params=None, seed=8):
        super().__init__(w, h, params, seed)
        self.cx, self.cy = 0.5 * w, 0.50 * h
        self.R0 = 0.11 * h * self.p.get("size", 1.0)
        xx, yy = grid(h, w)
        dx, dy = xx - self.cx, yy - self.cy
        r = np.sqrt(dx * dx + dy * dy) + 1e-3
        # gravitationally lensed star field: sample source plane at r' = r - Re^2 / r
        Re = self.R0 * 1.9
        bg = stars(h, w, 4500, seed + 1) + np.array((0.03, 0.02, 0.05), np.float32) * fbm(h, w, 200, 5, seed + 2)[..., None]
        k = np.clip(1 - Re * Re / (r * r), -3, 1)
        mx = (self.cx + dx * k).astype(np.float32)
        my = (self.cy + dy * k).astype(np.float32)
        self.bg = cv2.remap(bg, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        self.r = r
        self.theta = np.arctan2(dy, dx)
        # disk plane coordinates for the flat (front) disk: squash vertically
        self.flat = 0.16
        rd = np.sqrt(dx * dx + (dy / self.flat) ** 2)
        self.rd = rd
        self.thd = np.arctan2(dy / self.flat, dx)
        self.NT, self.NR = 1024, 256
        self.tex = fbm(self.NR, self.NT, 18, 5, seed + 3, aspect=6, wrap_x=True)
        self.tex2 = fbm(self.NR, self.NT, 6, 3, seed + 4, aspect=8, wrap_x=True)

    def _disk(self, rr, th, t, r_in, r_out):
        u = (rr - r_in) / (r_out - r_in)
        ri = np.clip(u * (self.NR - 1), 0, self.NR - 1).astype(np.int32)
        omega = 0.9 / (0.25 + np.clip(u, 0, 1)) ** 1.5          # inner parts rotate faster
        ti = ((th / (2 * np.pi) + 0.5 + t * 0.05 * omega) * self.NT).astype(np.int32) % self.NT
        v = self.tex[ri, ti] * 0.7 + self.tex2[ri, ti] * 0.3
        prof = smoothstep(0.0, 0.08, u) * (1 - smoothstep(0.55, 1.0, u)) * (u >= 0) * (u <= 1)
        temp = np.clip(1.15 - u, 0, 1)
        col = (np.array((1.0, 0.40, 0.12), np.float32)[None, None] * (1 - temp[..., None]) +
               np.array((1.5, 1.15, 0.80), np.float32)[None, None] * temp[..., None])
        inten = prof * (0.30 + 1.0 * v ** 1.5) * (0.9 + 2.2 * temp ** 2)
        return col * inten[..., None]

    def render(self, t, dur):
        u = ease(t / max(dur, 1e-3))
        R0 = self.R0
        img = self.bg.copy()
        # lensed far side of the disk: a ring hugging the shadow (above and below)
        halo_r = self.r / R0
        halo = self._disk(halo_r, self.theta * 1.0 + np.pi, t, 1.25, 3.0)
        beam = 1.0 + 0.55 * np.cos(self.theta + 0.0)            # Doppler beaming: one side brighter
        img += halo * beam[..., None] * 0.55
        # photon ring
        img += np.array((2.2, 1.7, 1.2), np.float32) * np.exp(-((self.r - R0 * 1.05) / (R0 * 0.035)) ** 2)[..., None]
        # event-horizon shadow
        sh = smoothstep(R0 * 1.0, R0 * 0.94, self.r)
        img = img * (1 - sh[..., None])
        # flat disk in front (lower half occludes the shadow, upper half behind it)
        disk = self._disk(self.rd / R0, self.thd, t, 1.5, 5.2)
        beam2 = 1.0 + 0.6 * np.cos(self.thd)
        front = (self.theta > 0) | (self.r > R0 * 1.0)
        img += disk * (beam2 * front)[..., None]
        img = bloom(img, threshold=1.2, strength=0.45)
        # slow push-in handled by a mild zoom
        z = lerp(1.0, 1.08, u)
        M = cv2.getRotationMatrix2D((self.cx, self.cy), lerp(-2.0, 2.0, u), z)
        return cv2.warpAffine(img, M, (self.w, self.h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
