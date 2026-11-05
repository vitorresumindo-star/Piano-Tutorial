"""Sample-based piano rendering (SFZ subset), reverb, metronome and loudness
normalisation -- all offline and deterministic (numpy/scipy only).

Default instrument: Salamander Grand Piano V3 (Yamaha C5, 16 velocity layers),
by Alexander Holm, licensed CC BY 3.0 -- credit is added to video metadata.
If the samples are not installed, a small additive "physical-ish" synth is
used as a fallback so the pipeline still works (lower quality).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.ndimage import minimum_filter1d

SR = 48_000
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SFZ = ROOT / "assets" / "downloads" / "SalamanderGrandPianoV3_44.1khz16bit" / "SalamanderGrandPianoV3.sfz"
CREDIT = "Piano samples: Salamander Grand Piano V3 by Alexander Holm (CC BY 3.0)"


@dataclass
class Region:
    sample: Path
    lokey: int = 0
    hikey: int = 127
    lovel: int = 1
    hivel: int = 127
    keycenter: int | None = None
    amp_veltrack: float = 100.0
    ampeg_release: float = 0.5
    volume: float = 0.0
    trigger: str = "attack"
    rt_decay: float = 0.0
    tune: float = 0.0


def parse_sfz(path: Path) -> list[Region]:
    text = path.read_text(errors="ignore")
    text = re.sub(r"//[^\n]*", "", text)
    regions: list[Region] = []
    group: dict = {}
    current: dict | None = None
    tokens = re.split(r"(<\w+>)", text)
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if tok in ("<group>", "<global>", "<master>"):
            if current is not None:
                regions.append(_mk_region(path, {**group, **current}))
                current = None
            group = {}
            mode = "group"
            continue
        if tok == "<region>":
            if current is not None:
                regions.append(_mk_region(path, {**group, **current}))
            current = {}
            continue
        if tok.startswith("<"):
            continue
        ops = dict(re.findall(r"(\w+)=(\S+)", tok.replace("\\", "/")))
        # sample paths may contain spaces in some sfz files; Salamander does not.
        if current is None:
            group.update(ops)
        else:
            current.update(ops)
    if current is not None:
        regions.append(_mk_region(path, {**group, **current}))
    return [r for r in regions if r.sample is not None]


def _mk_region(sfz: Path, op: dict) -> Region:
    def note(v):
        try:
            return int(v)
        except ValueError:
            m = re.match(r"([a-gA-G])(#|b)?(-?\d)", v)
            base = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}[m.group(1).lower()]
            return 12 * (int(m.group(3)) + 1) + base + (1 if m.group(2) == "#" else -1 if m.group(2) == "b" else 0)
    r = Region(sample=(sfz.parent / op["sample"]) if "sample" in op else None)
    if "key" in op:
        r.lokey = r.hikey = r.keycenter = note(op["key"])
    r.lokey = note(op.get("lokey", r.lokey))
    r.hikey = note(op.get("hikey", r.hikey))
    r.lovel = int(op.get("lovel", 1))
    r.hivel = int(op.get("hivel", 127))
    r.keycenter = note(op["pitch_keycenter"]) if "pitch_keycenter" in op else (r.keycenter if r.keycenter is not None else None)
    r.amp_veltrack = float(op.get("amp_veltrack", 100))
    r.ampeg_release = float(op.get("ampeg_release", 0.5))
    r.volume = float(op.get("volume", 0))
    r.trigger = op.get("trigger", "attack")
    r.rt_decay = float(op.get("rt_decay", 0))
    r.tune = float(op.get("tune", 0))
    if r.keycenter is None:
        # Salamander: a few regions omit pitch_keycenter -> infer from file name
        m = re.search(r"([A-G]#?)(\d)", r.sample.stem.replace("harmL", "").replace("harmS", ""))
        if m:
            names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
            r.keycenter = 12 * (int(m.group(2)) + 1) + names.index(m.group(1))
        else:
            r.keycenter = (r.lokey + r.hikey) // 2
    return r


@lru_cache(maxsize=512)
def load_sample(path: str) -> np.ndarray:
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        x = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        x = data.astype(np.float32) / 2147483648.0
    else:
        x = data.astype(np.float32)
    if x.ndim == 1:
        x = np.stack([x, x], -1)
    if sr != SR:
        g = math.gcd(SR, sr)
        x = signal.resample_poly(x, SR // g, sr // g, axis=0).astype(np.float32)
    return x


@dataclass
class NoteEvent:
    time: float      # seconds in output timeline
    off: float       # key release time (before pedal)
    pitch: int
    velocity: int


@dataclass
class PedalSpan:
    down: float
    up: float


class Piano:
    def __init__(self, sfz: Path | None = None):
        sfz = Path(sfz) if sfz else DEFAULT_SFZ
        self.available = sfz.exists()
        self.regions = parse_sfz(sfz) if self.available else []
        self.attack = [r for r in self.regions if r.trigger == "attack"]
        self.release = [r for r in self.regions if r.trigger == "release"]
        self.name = "Salamander Grand Piano V3" if self.available else "Fallback additive synth"

    def pick(self, regions, pitch, vel):
        for r in regions:
            if r.lokey <= pitch <= r.hikey and r.lovel <= vel <= r.hivel:
                return r
        return None

    def render(self, events: list[NoteEvent], pedal: list[PedalSpan], length: float,
               release_resonance: bool = True) -> np.ndarray:
        n = int(math.ceil(length * SR)) + SR
        out = np.zeros((n, 2), np.float32)
        evs = sorted(events, key=lambda e: e.time)
        last_by_pitch: dict[int, int] = {}
        offs = [self._effective_off(e, pedal) for e in evs]
        # re-strikes damp the previous voice of the same key
        cut = [None] * len(evs)
        for i, e in enumerate(evs):
            j = last_by_pitch.get(e.pitch)
            if j is not None and offs[j] > e.time:
                cut[j] = e.time
            last_by_pitch[e.pitch] = i
        for i, e in enumerate(evs):
            off = offs[i] if cut[i] is None else min(offs[i], cut[i])
            if self.available:
                self._voice_sample(out, e, off, release_resonance, damp_fast=cut[i] is not None)
            else:
                self._voice_synth(out, e, off)
        return out

    @staticmethod
    def _effective_off(e: NoteEvent, pedal: list[PedalSpan]) -> float:
        off = e.off
        for p in pedal:
            if p.down <= off < p.up:
                return max(off, p.up)
        return off

    def _voice_sample(self, out, e: NoteEvent, off: float, rel_res: bool, damp_fast=False):
        r = self.pick(self.attack, e.pitch, e.velocity)
        if r is None:
            return
        x = load_sample(str(r.sample))
        ratio = 2.0 ** ((e.pitch - r.keycenter + r.tune / 100.0) / 12.0)
        release = 0.09 if damp_fast else max(0.12, min(r.ampeg_release, 1.0) * 0.45 if r.ampeg_release <= 1 else r.ampeg_release)
        hold = max(0.0, off - e.time)
        dur = min(hold + release * 1.2, (len(x) - 2) / (SR * ratio))
        ns = int(dur * SR)
        if ns <= 8:
            return
        pos = np.arange(ns) * ratio
        y = np.empty((ns, 2), np.float32)
        idx = np.arange(len(x))
        for c in range(2):
            y[:, c] = np.interp(pos, idx, x[:, c])
        vt = r.amp_veltrack / 100.0
        gain = (1 - vt) + vt * (e.velocity / 127.0) ** 2
        gain *= 10 ** (r.volume / 20)
        t = np.arange(ns) / SR
        env = np.ones(ns, np.float32)
        ri = int(hold * SR)
        if ri < ns:
            tt = t[ri:] - hold
            env[ri:] = np.exp(-6.9 * tt / release)     # -60 dB at `release`
        fade = min(64, ns)
        env[:fade] *= np.linspace(0.6, 1.0, fade)
        y *= (env * gain)[:, None]
        i0 = int(round(e.time * SR))
        i1 = min(len(out), i0 + ns)
        out[i0:i1] += y[: i1 - i0]
        if rel_res and self.release and not damp_fast:
            rr = self.pick(self.release, e.pitch, e.velocity)
            if rr is not None:
                xs = load_sample(str(rr.sample))
                ratio2 = 2.0 ** ((e.pitch - rr.keycenter) / 12.0)
                n2 = int(min(len(xs) / ratio2 - 2, 1.5 * SR))
                if n2 > 8:
                    pos2 = np.arange(n2) * ratio2
                    y2 = np.stack([np.interp(pos2, np.arange(len(xs)), xs[:, c]) for c in range(2)], -1)
                    g2 = ((1 - rr.amp_veltrack / 100) + rr.amp_veltrack / 100 * (e.velocity / 127) ** 2)
                    g2 *= 10 ** (rr.volume / 20) * 10 ** (-rr.rt_decay * min(hold, 6.0) / 20) * 0.6
                    j0 = int(round(off * SR))
                    j1 = min(len(out), j0 + n2)
                    if j0 < len(out):
                        out[j0:j1] += (y2[: j1 - j0] * g2).astype(np.float32)

    def _voice_synth(self, out, e: NoteEvent, off: float):
        f0 = 440.0 * 2 ** ((e.pitch - 69) / 12)
        hold = max(0.05, off - e.time)
        dur = min(hold + 0.6, 6.0)
        ns = int(dur * SR)
        t = np.arange(ns) / SR
        B = 0.0004 * (1 + (e.pitch - 21) / 40)
        y = np.zeros(ns)
        for k in range(1, 14):
            fk = k * f0 * math.sqrt(1 + B * k * k)
            if fk > SR / 2.2:
                break
            amp = (1 / k ** 1.25) * (1 if k > 1 else 1.2)
            decay = (0.6 + 0.15 * k) * (1 + (e.pitch - 40) / 50)
            y += amp * np.sin(2 * math.pi * fk * t + k) * np.exp(-t * decay)
        env = np.ones(ns)
        ri = int(hold * SR)
        env[ri:] = np.exp(-(t[ri:] - hold) * 12)
        env[:48] *= np.linspace(0, 1, 48)
        y *= env * (0.15 * (e.velocity / 127) ** 1.6)
        i0 = int(round(e.time * SR))
        i1 = min(len(out), i0 + ns)
        out[i0:i1, 0] += y[: i1 - i0]
        out[i0:i1, 1] += y[: i1 - i0]


# ---------------------------------------------------------------- effects
def reverb_ir(seconds: float = 2.0, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * 1.4 * SR)
    t = np.arange(n) / SR
    noise = rng.standard_normal((n, 2)).astype(np.float32)
    b_lo, a_lo = signal.butter(2, 3500 / (SR / 2), "low")
    lo = signal.lfilter(b_lo, a_lo, noise, axis=0)
    hi = noise - lo
    ir = lo * np.exp(-6.9 * t / seconds)[:, None] + hi * np.exp(-6.9 * t / (seconds * 0.35))[:, None]
    pre = int(0.014 * SR)
    ir = np.concatenate([np.zeros((pre, 2), np.float32), ir.astype(np.float32)])
    # early reflections
    for d, g in ((0.011, 0.5), (0.019, 0.35), (0.027, 0.3), (0.041, 0.22), (0.053, 0.16)):
        i = int(d * SR)
        ir[i, 0] += g
        ir[int(i * 1.07), 1] += g
    ir /= np.sqrt((ir ** 2).sum(axis=0, keepdims=True))
    return ir


def apply_reverb(x: np.ndarray, amount: float, seconds: float = 2.0) -> np.ndarray:
    if amount <= 0:
        return x
    ir = reverb_ir(seconds)
    wet = np.stack([signal.fftconvolve(x[:, c], ir[:, c])[: len(x)] for c in range(2)], -1)
    b, a = signal.butter(1, 180 / (SR / 2), "high")   # keep the low end dry and clean
    wet = signal.lfilter(b, a, wet, axis=0)
    return (x * (1 - 0.35 * amount) + wet * amount * 0.9).astype(np.float32)


def _biquad_highshelf(G, fc, Q):
    A = 10 ** (G / 40)
    w0 = 2 * math.pi * fc / SR
    alpha = math.sin(w0) / (2 * Q)
    cw = math.cos(w0)
    b = [A * ((A + 1) + (A - 1) * cw + 2 * math.sqrt(A) * alpha), -2 * A * ((A - 1) + (A + 1) * cw),
         A * ((A + 1) + (A - 1) * cw - 2 * math.sqrt(A) * alpha)]
    a = [(A + 1) - (A - 1) * cw + 2 * math.sqrt(A) * alpha, 2 * ((A - 1) - (A + 1) * cw),
         (A + 1) - (A - 1) * cw - 2 * math.sqrt(A) * alpha]
    return np.array(b) / a[0], np.array(a) / a[0]


def _biquad_highpass(fc, Q):
    w0 = 2 * math.pi * fc / SR
    alpha = math.sin(w0) / (2 * Q)
    cw = math.cos(w0)
    b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
    a = [1 + alpha, -2 * cw, 1 - alpha]
    return np.array(b) / a[0], np.array(a) / a[0]


def integrated_loudness(x: np.ndarray) -> float:
    """ITU-R BS.1770-4 integrated loudness (LUFS) with absolute/relative gating."""
    b1, a1 = _biquad_highshelf(4.0, 1681.974450955533, 0.7071752369554196)
    b2, a2 = _biquad_highpass(38.13547087602444, 0.5003270373238773)
    y = signal.lfilter(b2, a2, signal.lfilter(b1, a1, x, axis=0), axis=0)
    blk, hop = int(0.4 * SR), int(0.1 * SR)
    if len(y) < blk:
        return -70.0
    ms = []
    for i in range(0, len(y) - blk + 1, hop):
        ms.append((y[i:i + blk] ** 2).mean(axis=0).sum())
    ms = np.array(ms)
    lk = -0.691 + 10 * np.log10(ms + 1e-12)
    g = ms[lk > -70]
    if not len(g):
        return -70.0
    rel = -0.691 + 10 * np.log10(g.mean()) - 10
    g2 = ms[(lk > -70) & (lk > rel)]
    return float(-0.691 + 10 * np.log10(g2.mean() + 1e-12))


def true_peak_db(x: np.ndarray) -> float:
    up = signal.resample_poly(x, 4, 1, axis=0)
    return 20 * math.log10(np.abs(up).max() + 1e-12)


def limit(x: np.ndarray, ceiling_db: float = -1.0) -> np.ndarray:
    thr = 10 ** (ceiling_db / 20) * 0.97
    peak = np.abs(x).max(axis=1)
    g = np.minimum(1.0, thr / np.maximum(peak, 1e-9))
    look = int(0.004 * SR)
    g = minimum_filter1d(g, size=2 * look + 1)
    # smooth release (one-pole on gain reduction)
    rel = math.exp(-1 / (0.08 * SR))
    gr = 1 - g
    gr = signal.lfilter([1 - rel], [1, -rel], gr)
    gr = np.maximum(gr, 1 - g)
    return (x * (1 - gr)[:, None]).astype(np.float32)


def normalize(x: np.ndarray, target_lufs: float = -14.0, ceiling_db: float = -1.0) -> tuple[np.ndarray, dict]:
    lufs = integrated_loudness(x)
    gain_db = target_lufs - lufs if lufs > -69 else 0.0
    gain_db = min(gain_db, 30.0)
    y = x * 10 ** (gain_db / 20)
    y = limit(y, ceiling_db)
    if true_peak_db(y) > ceiling_db:
        y = limit(y, ceiling_db - 0.5)
    return y, {"input_lufs": round(lufs, 2), "gain_db": round(gain_db, 2),
               "output_lufs": round(integrated_loudness(y), 2), "true_peak_db": round(true_peak_db(y), 2)}


def click(freq: float, accent: bool) -> np.ndarray:
    n = int(0.045 * SR)
    t = np.arange(n) / SR
    y = np.sin(2 * math.pi * freq * t) * np.exp(-t * 90) * (0.5 if accent else 0.33)
    return np.stack([y, y], -1).astype(np.float32)


def write_wav(path: Path, x: np.ndarray):
    x = np.clip(x, -1, 1)
    wavfile.write(str(path), SR, (x * 32767).astype(np.int16))
