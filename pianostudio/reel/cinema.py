"""Footage panel: a timeline of shots (procedural scenes or real video clips),
per-shot grading, dissolves, vignette and film grain.

Shot spec (from the project file, times already resolved to seconds):
    {"start": 0.0, "end": 5.8, "scene": "field", "params": {...}}            # procedural
    {"start": 5.8, "end": 11.2, "clip": "footage/shot.mp4", "in": 12.5,      # your own footage
     "focus": [0.5, 0.4], "zoom": [1.0, 1.06], "speed": 1.0}
    optional: "transition": "dissolve" | "white" | "black" | "cut" (into this shot),
              "grade": {"exposure": 1.1, "saturation": 1.05, ...}
"""
from __future__ import annotations

import json
import math
import subprocess

import cv2
import numpy as np

from .scenekit import ease, grade, tonemap, vignette
from .scenes import SCENES


class ProceduralShot:
    def __init__(self, spec: dict, w: int, h: int, fps: int):
        self.spec = spec
        self.w, self.h = w, h
        cls = SCENES[spec["scene"]]
        self.scene = cls(w, h, spec.get("params", {}), seed=int(spec.get("seed", hash(spec["scene"]) % 1000)))
        self.g = dict(cls.grade)
        self.g.update(spec.get("grade", {}))

    def frame(self, t_local: float, dur: float) -> np.ndarray:
        hdr = self.scene.render(t_local, dur)
        g = self.g
        img = tonemap(hdr, g.get("exposure", 1.0))
        return grade(img, g["lift"], g["gamma"], g["gain"], g.get("saturation", 1.0), g.get("contrast", 1.0))


class ClipShot:
    """Streams frames from a video file with ffmpeg: cover-crop around a focus point,
    optional slow push-in (zoom from -> to) and the same grading controls."""

    def __init__(self, spec: dict, w: int, h: int, fps: int):
        self.spec, self.w, self.h, self.fps = spec, w, h, fps
        self.file = spec["clip"]
        self.src_in = float(spec.get("in", 0.0))
        self.speed = float(spec.get("speed", 1.0))
        fx, fy = spec.get("focus", [0.5, 0.5])
        info = json.loads(subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
             "-of", "json", self.file], capture_output=True, text=True, check=True).stdout)
        sw, sh = info["streams"][0]["width"], info["streams"][0]["height"]
        z1 = max(spec.get("zoom", [1.0, 1.0]))
        self.ow, self.oh = int(math.ceil(w * z1 / 2) * 2), int(math.ceil(h * z1 / 2) * 2)  # decode size (with zoom headroom)
        scale = max(self.ow / sw, self.oh / sh)
        self.sw, self.sh = int(math.ceil(sw * scale / 2) * 2), int(math.ceil(sh * scale / 2) * 2)
        self.cx = int(min(max(fx * self.sw - self.ow / 2, 0), self.sw - self.ow))
        self.cy = int(min(max(fy * self.sh - self.oh / 2, 0), self.sh - self.oh))
        self.proc = None
        self.next_idx = None
        self.last = np.zeros((h, w, 3), np.float32)
        self.g = dict(lift=(0, 0, 0), gamma=(1, 1, 1), gain=(1, 1, 1), saturation=1.0, contrast=1.0, exposure=1.0)
        self.g.update(spec.get("grade", {}))

    def _open(self, idx: int):
        if self.proc:
            self.proc.kill()
        t0 = self.src_in + max(idx, 0) / self.fps * self.speed
        vf = (f"setpts=PTS/{self.speed},fps={self.fps},scale={self.sw}:{self.sh}:flags=lanczos,"
              f"crop={self.ow}:{self.oh}:{self.cx}:{self.cy}")
        self.proc = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{t0:.4f}", "-i", self.file, "-an", "-vf", vf,
                                      "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
        self.next_idx = idx

    def frame(self, t_local: float, dur: float) -> np.ndarray:
        idx = int(round(t_local * self.fps))
        if self.proc is None or idx != self.next_idx:
            self._open(idx)
        n = self.ow * self.oh * 3
        buf = self.proc.stdout.read(n)
        self.next_idx += 1
        if len(buf) == n:
            img = np.frombuffer(buf, np.uint8).reshape(self.oh, self.ow, 3).astype(np.float32) / 255.0
            z0, z1 = self.spec.get("zoom", [1.0, 1.0])
            z = z0 + (z1 - z0) * ease(t_local / max(dur, 1e-3))
            zr = z / max(z0, z1)
            M = cv2.getRotationMatrix2D((self.ow / 2, self.oh / 2), 0, zr / (self.ow / self.w) * (self.ow / self.w))
            M[0, 2] += self.w / 2 - self.ow / 2
            M[1, 2] += self.h / 2 - self.oh / 2
            img = cv2.warpAffine(img, M, (self.w, self.h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            g = self.g
            img = np.clip(img * g.get("exposure", 1.0), 0, 1)
            self.last = grade(img, g["lift"], g["gamma"], g["gain"], g.get("saturation", 1.0), g.get("contrast", 1.0))
        return self.last

    def close(self):
        if self.proc:
            self.proc.kill()
            self.proc = None


class Footage:
    def __init__(self, shots: list[dict], w: int, h: int, fps: int, transition: float = 0.6,
                 grain: float = 0.012, vignette_strength: float = 0.32, seed: int = 9):
        self.shots = sorted(shots, key=lambda s: s["start"])
        self.w, self.h, self.fps = w, h, fps
        self.td = transition
        self._inst: dict[int, object] = {}
        self.vig = vignette(h, w, vignette_strength)[..., None]
        r = np.random.default_rng(seed)
        self.grain = [cv2.resize(r.normal(0, 1, (h // 2, w // 2)).astype(np.float32), (w, h)) for _ in range(6)]
        self.grain_amp = grain

    def _shot(self, i: int):
        if i not in self._inst:
            sp = self.shots[i]
            self._inst[i] = (ClipShot if "clip" in sp else ProceduralShot)(sp, self.w, self.h, self.fps)
        return self._inst[i]

    def _render_shot(self, i: int, t: float) -> np.ndarray:
        sp = self.shots[i]
        return self._shot(i).frame(t - sp["start"], sp["end"] - sp["start"])

    def release_before(self, t: float):
        """Free shots that ended before t (keeps worker memory bounded)."""
        for i in list(self._inst):
            if self.shots[i]["end"] + self.td < t:
                inst = self._inst.pop(i)
                if hasattr(inst, "close"):
                    inst.close()

    def render(self, t: float, frame_index: int = 0) -> np.ndarray:
        """Display-space float32 RGB in [0, 1]."""
        idx = 0
        for i, sp in enumerate(self.shots):
            if sp["start"] <= t:
                idx = i
        img = self._render_shot(idx, t)
        sp = self.shots[idx]
        half = self.td / 2
        # blend with the neighbour inside a transition window centred on each cut
        if idx + 1 < len(self.shots) and t > sp["end"] - half:
            nxt = self.shots[idx + 1]
            kind = nxt.get("transition", "dissolve")
            if kind != "cut":
                a = ease((t - (sp["end"] - half)) / self.td)
                img = self._blend(img, self._render_shot(idx + 1, t), a, kind)
        elif idx > 0 and t < sp["start"] + half:
            kind = sp.get("transition", "dissolve")
            if kind != "cut":
                a = ease((t - (sp["start"] - half)) / self.td)
                img = self._blend(self._render_shot(idx - 1, t), img, a, kind)
        img = img * self.vig
        if self.grain_amp > 0:
            g = self.grain[frame_index % len(self.grain)]
            lum = img.mean(-1, keepdims=True)
            img = img + g[..., None] * self.grain_amp * (0.35 + lum * (1 - lum) * 2.6)
        return np.clip(img, 0, 1)

    @staticmethod
    def _blend(a: np.ndarray, b: np.ndarray, x: float, kind: str) -> np.ndarray:
        if kind in ("white", "black"):
            c = 1.0 if kind == "white" else 0.0
            if x < 0.5:
                return a + (c - a) * (x * 2) ** 1.5
            return c + (b - c) * ((x - 0.5) * 2) ** 0.7
        return a * (1 - x) + b * x
