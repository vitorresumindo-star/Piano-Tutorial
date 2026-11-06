"""Handwritten title overlay: large song title, smaller film/subtitle line below,
white with a soft shadow, revealed with a left-to-right 'write-on' wipe."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
FONT_DIR = ROOT / "assets" / "fonts"


def _font(path: str | None, default: str, size: int) -> ImageFont.FreeTypeFont:
    p = Path(path) if path else FONT_DIR / default
    if not p.is_absolute() and not p.exists():
        p = ROOT / p
    return ImageFont.truetype(str(p), size)


class TitleOverlay:
    def __init__(self, w: int, h: int, title: str, subtitle: str = "", cfg: dict | None = None):
        cfg = cfg or {}
        self.w, self.h = w, h
        self.start = float(cfg.get("revealAt", 0.25))
        self.dur = float(cfg.get("revealSeconds", 1.5))
        tf = _font(cfg.get("font"), "dancing-script-700-normal.ttf", int(cfg.get("size", 104)))
        sf = _font(cfg.get("subFont"), "dancing-script-600-normal.ttf", int(cfg.get("subSize", 56)))
        # shrink to fit the width
        while title and tf.getlength(title) > w * 0.9 and tf.size > 40:
            tf = _font(cfg.get("font"), "dancing-script-700-normal.ttf", tf.size - 4)
        ty = int(h * float(cfg.get("y", 0.25)))
        sy = ty + int(tf.size * 0.82)
        self.layers = []
        for text, font, y, delay in ((title, tf, ty, 0.0), (subtitle, sf, sy, 0.55)):
            if not text:
                continue
            img = Image.new("L", (w, h), 0)
            ImageDraw.Draw(img).text((w / 2, y), text, font=font, fill=255, anchor="mm")
            a = np.asarray(img, np.float32) / 255.0
            xs = np.where(a.max(0) > 0)[0]
            x0, x1 = (int(xs.min()), int(xs.max())) if len(xs) else (0, w)
            self.layers.append((a, x0, x1, delay))
        tot = np.zeros((h, w), np.float32)
        for a, *_ in self.layers:
            tot = np.maximum(tot, a)
        self.shadow_wide = cv2.GaussianBlur(cv2.dilate(tot, np.ones((5, 5), np.uint8)), (0, 0), 13)
        self.shadow_tight = cv2.GaussianBlur(np.roll(tot, 3, axis=0), (0, 0), 2.5)
        self.xs = np.arange(w, dtype=np.float32)[None, :]

    def apply(self, img: np.ndarray, t: float) -> np.ndarray:
        """img: display-space float RGB [0,1]; returns the composited image."""
        text = np.zeros((self.h, self.w), np.float32)
        reveal_all = 0.0
        for a, x0, x1, delay in self.layers:
            u = (t - self.start - delay) / self.dur
            if u <= 0:
                continue
            u = min(u, 1.0)
            edge = x0 + (x1 - x0 + 60) * (1 - (1 - u) ** 2.2)
            m = np.clip((edge - self.xs) / 60.0, 0, 1)
            text = np.maximum(text, a * m)
            reveal_all = max(reveal_all, min(1.0, u * 1.5))
        if reveal_all <= 0:
            return img
        sh = np.clip(self.shadow_wide * 0.75 + self.shadow_tight * 0.6, 0, 0.82) * reveal_all
        out = img * (1 - sh[..., None])
        out = out * (1 - text[..., None]) + np.array((1.0, 0.99, 0.97), np.float32) * text[..., None]
        return out
