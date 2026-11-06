"""Small numpy/OpenCV toolkit for the procedural cinematic scenes.

Everything works in float32 linear-ish RGB (values may exceed 1.0 for
highlights); `tonemap` maps to display range at the end.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def fbm(h: int, w: int, scale: float, octaves: int = 5, seed: int = 0, persistence: float = 0.5,
        aspect: float = 1.0, wrap_x: bool = False) -> np.ndarray:
    """Fractal value noise in [0, 1]. `scale` = feature size of the first octave in px.
    `aspect` > 1 stretches features horizontally. `wrap_x` makes it tile horizontally."""
    r = rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        cell = max(2.0, scale / (2 ** o))
        gw = max(2, int(math.ceil(w / (cell * aspect))))
        gh = max(2, int(math.ceil(h / cell)))
        g = r.random((gh + 3, gw + 3)).astype(np.float32)
        if wrap_x:
            g[:, -3:] = g[:, :3]
        big = cv2.resize(g, (int((gw + 3) * cell * aspect), int((gh + 3) * cell)), interpolation=cv2.INTER_CUBIC)
        if wrap_x:
            per = int(gw * cell * aspect)
            reps = int(math.ceil(w / per)) + 1
            row = big[: h + int(cell), int(cell * aspect): int(cell * aspect) + per]
            big = np.tile(row, (1, reps))
            layer = big[:h, :w]
        else:
            oy, ox = int(cell), int(cell * aspect)
            layer = big[oy:oy + h, ox:ox + w]
        out += amp * layer
        tot += amp
        amp *= persistence
    out /= tot
    lo, hi = np.percentile(out, 0.5), np.percentile(out, 99.5)
    return np.clip((out - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def fbm1d(n: int, scale: float, octaves: int = 4, seed: int = 0) -> np.ndarray:
    return fbm(1, n, scale, octaves, seed)[0]


def vgrad(h: int, w: int, stops) -> np.ndarray:
    """Vertical gradient. stops = [(pos 0..1, (r,g,b)), ...]"""
    ys = np.linspace(0, 1, h, dtype=np.float32)
    pos = np.array([s[0] for s in stops], np.float32)
    cols = np.array([s[1] for s in stops], np.float32)
    col = np.stack([np.interp(ys, pos, cols[:, c]) for c in range(3)], -1)
    return np.repeat(col[:, None, :], w, axis=1).astype(np.float32)


def grid(h: int, w: int):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return xx, yy


def radial(h: int, w: int, cx: float, cy: float, sx: float = 1.0, sy: float = 1.0) -> np.ndarray:
    xx, yy = grid(h, w)
    return np.sqrt(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    return a + (b - a) * t


def over(dst: np.ndarray, src: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = alpha[..., None] if alpha.ndim == 2 else alpha
    return dst * (1 - a) + src * a


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)


def bloom(img: np.ndarray, threshold: float = 0.9, strength: float = 0.35, sigmas=(6, 18, 48)) -> np.ndarray:
    hi = np.maximum(img - threshold, 0)
    small = cv2.resize(hi, (img.shape[1] // 4, img.shape[0] // 4), interpolation=cv2.INTER_AREA)
    acc = np.zeros_like(small)
    for s in sigmas:
        acc += blur(small, s / 4)
    acc = cv2.resize(acc / len(sigmas), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    return img + acc * strength


def tonemap(img: np.ndarray, exposure: float = 1.0) -> np.ndarray:
    """ACES-like filmic curve (Narkowicz fit), output in [0, 1]."""
    x = np.maximum(img * exposure, 0)
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)


def warp_affine(img: np.ndarray, M: np.ndarray, out_w: int, out_h: int, border=cv2.BORDER_REFLECT) -> np.ndarray:
    return cv2.warpAffine(img, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=border)


def camera(src_w: int, src_h: int, out_w: int, out_h: int, zoom: float = 1.0, cx: float = 0.5, cy: float = 0.5,
           rot_deg: float = 0.0) -> np.ndarray:
    """Affine matrix mapping a src canvas to the output: the view is centred on
    (cx, cy) (fractions of the src canvas); zoom=1 is native 1:1 pixel scale, so a
    canvas larger than the output leaves headroom for pans and push-ins."""
    M = cv2.getRotationMatrix2D((src_w * cx, src_h * cy), rot_deg, zoom)
    M[0, 2] += out_w / 2 - src_w * cx
    M[1, 2] += out_h / 2 - src_h * cy
    return M


def ease(u):
    u = np.clip(u, 0, 1)
    return u * u * (3 - 2 * u)


def hash01(k, salt: int = 0):
    """Deterministic pseudo-random in [0,1) for integer arrays."""
    k = (np.asarray(k, np.uint64) * np.uint64(2654435761) + np.uint64(salt * 97 + 13)) & np.uint64(0xFFFFFFFF)
    k ^= k >> np.uint64(15)
    k = (k * np.uint64(2246822519)) & np.uint64(0xFFFFFFFF)
    k ^= k >> np.uint64(13)
    return (k.astype(np.float64) / 4294967296.0).astype(np.float32)


def stars(h: int, w: int, n: int, seed: int, size_pow: float = 3.0, color_var: float = 0.25) -> np.ndarray:
    """Star field image (float RGB)."""
    r = rng(seed)
    img = np.zeros((h, w, 3), np.float32)
    xs = r.random(n) * w
    ys = r.random(n) * h
    mag = r.random(n) ** size_pow
    temp = r.random(n)
    cols = np.stack([1.0 - color_var * (1 - temp) * 0.2, 0.95 + 0.05 * temp, 0.85 + color_var * temp], -1)
    xi, yi = xs.astype(int), ys.astype(int)
    np.add.at(img, (yi, xi), cols * (0.15 + 2.2 * mag[:, None]))
    img = blur(img, 0.6) * 2.2
    big = mag > 0.82
    for x, y, m in zip(xs[big], ys[big], mag[big]):
        cv2.circle(img, (int(x), int(y)), 1, (1.2 * m, 1.2 * m, 1.3 * m), -1, lineType=cv2.LINE_AA)
    return img


def vignette(h: int, w: int, strength: float = 0.35, power: float = 2.2) -> np.ndarray:
    xx, yy = grid(h, w)
    d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 * 0.8 + ((yy - h / 2) / (h / 2)) ** 2 * 0.6)
    return (1 - strength * np.clip(d, 0, 1.6) ** power / 1.6 ** power * 1.6).clip(0.3, 1).astype(np.float32)


def grade(img: np.ndarray, lift=(0, 0, 0), gamma=(1, 1, 1), gain=(1, 1, 1), saturation: float = 1.0,
          contrast: float = 1.0) -> np.ndarray:
    """img in [0,1] display space."""
    x = img * np.array(gain, np.float32) + np.array(lift, np.float32) * (1 - img)
    x = np.clip(x, 0, 1) ** (1 / np.array(gamma, np.float32))
    if contrast != 1.0:
        x = np.clip((x - 0.5) * contrast + 0.5, 0, 1)
    if saturation != 1.0:
        lum = (x * np.array([0.2126, 0.7152, 0.0722], np.float32)).sum(-1, keepdims=True)
        x = np.clip(lum + (x - lum) * saturation, 0, 1)
    return x.astype(np.float32)


def soft_poly(img: np.ndarray, pts, color, alpha_img: np.ndarray | None = None, feather: float = 0.0):
    """Fill a polygon (anti-aliased) with colour; optionally record coverage in alpha_img."""
    p = np.round(np.asarray(pts, np.float32) * 4).astype(np.int32)
    cv2.fillPoly(img, [p], color, lineType=cv2.LINE_AA, shift=2)
    if alpha_img is not None:
        cv2.fillPoly(alpha_img, [p], 1.0, lineType=cv2.LINE_AA, shift=2)


def puff_sprite(r: int, seed: int, light_dir=(-0.4, -0.9)) -> tuple[np.ndarray, np.ndarray]:
    """Soft billowing puff: returns (shade [0..1] lit factor, alpha) of size (2r, 2r)."""
    n = 2 * r
    xx, yy = grid(n, n)
    d = np.sqrt((xx - r) ** 2 + (yy - r) ** 2) / r
    nz = fbm(n, n, max(4, r / 2), 4, seed)
    edge = 1 - d + (nz - 0.5) * 0.5
    alpha = smoothstep(0.0, 0.35, edge)
    lx, ly = light_dir
    shade = np.clip(0.55 + 0.6 * (-(xx - r) / r * lx - (yy - r) / r * ly) * 0.5 + (nz - 0.5) * 0.4, 0, 1.2)
    return shade.astype(np.float32), alpha.astype(np.float32)
