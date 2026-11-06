"""Render a reel project to MP4.

    python3 -m pianostudio reel projects/dust_and_starlight            # full render
    python3 -m pianostudio reel projects/dust_and_starlight --frames 0:120 --workers 2
    python3 -m pianostudio reel projects/dust_and_starlight --stills 3,20,45

Pipeline: MIDI -> fingering -> hand choreography + piano roll (cairo) and
footage timeline (procedural scenes or clips) -> raw RGB frames piped to x264
in parallel chunks -> concat + AAC audio -> integrity checks.
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ..export import slugify
from .cinema import Footage
from .hands2d import HandStyle
from .pianoroll import PianoRoll, RollStyle
from .project import PreparedReel, load_project, prepare, render_audio
from .titles import TitleOverlay

ROOT = Path(__file__).resolve().parents[2]


class Compositor:
    def __init__(self, p: PreparedReel):
        cfg = p.cfg
        lay = p.layout
        nc = cfg["notes"]
        style = RollStyle(fall_seconds=float(nc["fallSeconds"]), palette=nc.get("palette") or {},
                          color_mode=nc["colorMode"], hand_colors=tuple(nc["handColors"]),
                          note_labels=bool(nc["labels"]), key_labels=nc["keyLabels"], spelling=nc["spelling"],
                          bar_lines=bool(nc["barLines"]))
        hs = HandStyle.from_hex(cfg["hands"].get("skin"), cfg["hands"].get("outline"))
        self.p = p
        self.roll = PianoRoll(lay, p.notes, style, hs, p.duration, p.bar_times)
        fc = cfg["footage"]
        self.footage = Footage(p.shots, lay.width, lay.footage_h, int(cfg["fps"]), float(fc["transition"]),
                               float(fc["grain"]), float(fc["vignette"]))
        self.title = TitleOverlay(lay.width, lay.footage_h, cfg["title"], cfg.get("subtitle", ""), cfg["titleStyle"])

    def frame(self, i: int) -> np.ndarray:
        fps = int(self.p.cfg["fps"])
        t = i / fps
        top = self.footage.render(t, i)
        top = self.title.apply(top, t)
        top8 = (np.clip(top, 0, 1) * 255 + 0.5).astype(np.uint8)
        bottom = self.roll.render(t)
        if i % 120 == 0:
            self.footage.release_before(t)
        return np.concatenate([top8, bottom], axis=0)


def _x264_cmd(w, h, fps, out, crf, preset, threads=2):
    return ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
            "-i", "-", "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-profile:v", "high", "-g", str(fps * 2),
            "-threads", str(threads), "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
            "-color_range", "tv", str(out)]


def _render_chunk(args):
    project_file, i0, i1, out, crf, preset, fps_override, q = args
    cfg = load_project(project_file)
    if fps_override:
        cfg["fps"] = fps_override
    p = prepare(cfg)
    comp = Compositor(p)
    lay = p.layout
    proc = subprocess.Popen(_x264_cmd(lay.width, lay.height, int(cfg["fps"]), out, crf, preset), stdin=subprocess.PIPE)
    for i in range(i0, i1):
        proc.stdin.write(comp.frame(i).tobytes())
        if q is not None:
            q.put(1)
    proc.stdin.close()
    rc = proc.wait()
    if rc:
        raise RuntimeError(f"x264 failed for chunk {i0}:{i1}")
    return out


def probe(path: Path) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-show_entries",
                        "stream=codec_name,codec_type,width,height,r_frame_rate,nb_read_packets,pix_fmt,duration,"
                        "sample_rate,channels,color_space:format=duration,size,bit_rate",
                        "-of", "json", str(path)], capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


def check(path: Path, w: int, h: int, fps: int, frames: int, duration: float) -> dict:
    info = probe(path)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    num, den = map(int, v["r_frame_rate"].split("/"))
    res = {
        "video_codec_h264": v["codec_name"] == "h264",
        "resolution": (v["width"], v["height"]) == (w, h),
        "fps": abs(num / den - fps) < 1e-6,
        "frame_count": int(v.get("nb_read_packets", 0)) == frames,
        "pixel_format_yuv420p": v.get("pix_fmt") == "yuv420p",
        "audio_aac": bool(a) and a["codec_name"] == "aac",
        "audio_length_matches": bool(a) and abs(float(a["duration"]) - duration) < 0.05,
    }
    res["ok"] = all(res.values())
    res["info"] = {"frames": v.get("nb_read_packets"), "size_mb": round(int(info["format"]["size"]) / 1e6, 2),
                   "bit_rate_kbps": round(int(info["format"]["bit_rate"]) / 1000), "duration": info["format"]["duration"]}
    return res


def sync_report(p: PreparedReel) -> dict:
    """Machine check: for every note the key is fully down in the first video frame at/after
    the onset and the audio event starts at the same instant (both use note.start)."""
    from .hands2d import PRE
    fps = int(p.cfg["fps"])
    errs = []
    roll = PianoRoll(p.layout, p.notes, RollStyle(), HandStyle(), p.duration)
    worst_tip = 0.0
    for hand, track in roll.tracks.items():
        for n in [n for n in p.notes if n.hand == hand]:
            fi = math.ceil(n.start * fps - 1e-9)
            t = fi / fps
            if t >= n.end:
                continue
            st = track.state(t)
            fs = st.fingers[n.finger]
            dx = abs(fs.x - p.layout.x_center(n.pitch))
            worst_tip = max(worst_tip, dx)
            if fs.press < 0.999 or dx > 0.5:
                errs.append({"note": n.id, "time": round(n.start, 3), "finger": n.finger, "press": round(fs.press, 3),
                             "dx_px": round(dx, 2)})
    return {"notes": len(p.notes), "frame_quantisation_ms": round(1000 / fps, 2),
            "key_down_at_onset_failures": errs[:20], "failures": len(errs),
            "worst_fingertip_offset_px": round(worst_tip, 3), "press_lead_ms": round(PRE * 1000, 1)}


def produce_reel(project: str | Path, out_dir: str | Path | None = None, workers: int = 4,
                 frames: tuple[int, int] | None = None, fps: int | None = None, stills: list[float] | None = None,
                 share: bool = True, log=print) -> dict:
    cfg = load_project(project)
    if fps:
        cfg["fps"] = fps
    p = prepare(cfg)
    name = cfg["output"].get("name") or slugify(cfg["title"])
    out = Path(out_dir or cfg["output"].get("dir") or (ROOT / "output" / name))
    out.mkdir(parents=True, exist_ok=True)
    lay = p.layout
    fps_ = int(cfg["fps"])
    log(f"[reel] {cfg['title']} / {cfg.get('subtitle', '')}: {len(p.notes)} notes, {p.duration:.2f}s, "
        f"{p.frames} frames @ {fps_} fps, keys {lay.lo}-{lay.hi} ({lay.n_white} white, {lay.white_w:.1f}px)")
    if p.issues:
        for iss in p.issues:
            log(f"[reel] fingering {iss.severity}: {iss.code} @ {iss.time:.2f}s {iss.message}")
    if stills:
        comp = Compositor(p)
        paths = []
        for t in stills:
            img = comp.frame(int(round(t * fps_)))
            path = out / f"still_{t:06.2f}.png"
            Image.fromarray(img).save(path)
            paths.append(str(path))
        return {"stills": paths}
    t0 = time.time()
    wav = out / "audio.wav"
    stats = render_audio(p, wav)
    log(f"[reel] audio: {stats}")
    i0, i1 = frames if frames else (0, p.frames)
    n = i1 - i0
    workers = max(1, min(workers, n))
    bounds = [i0 + round(n * k / workers) for k in range(workers + 1)]
    tmp = out / "chunks"
    tmp.mkdir(exist_ok=True)
    crf, preset = int(cfg["output"]["crf"]), cfg["output"]["preset"]
    ctx = mp.get_context("fork")
    with ctx.Manager() as mgr:
        q = mgr.Queue()
        jobs = [(cfg["_file"], bounds[k], bounds[k + 1], str(tmp / f"chunk_{k:02d}.mp4"), crf, preset, fps, q)
                for k in range(workers)]
        with ctx.Pool(workers) as pool:
            res = pool.map_async(_render_chunk, jobs)
            done, last = 0, time.time()
            while not res.ready() or not q.empty():
                try:
                    q.get(timeout=1.0)
                    done += 1
                except Exception:
                    pass
                if time.time() - last > 15:
                    el = time.time() - t0
                    eta = el / max(done, 1) * (n - done)
                    log(f"[reel] frames {done}/{n}  ({done / max(el, 1e-6):.1f} fps, ETA {eta / 60:.1f} min)")
                    last = time.time()
            chunks = res.get()
    lst = tmp / "list.txt"
    lst.write_text("".join(f"file '{Path(c).resolve()}'\n" for c in chunks))
    master = out / f"{name}_{lay.width}x{lay.height}_{fps_}fps_HQ.mp4"
    a_off = i0 / fps_
    a_len = n / fps_
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-ss", f"{a_off:.6f}", "-t", f"{a_len:.6f}", "-i", str(wav),
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "320k", "-ar", "48000",
                    "-movflags", "+faststart", str(master)], check=True)
    result = {"master": str(master), "audio": stats, "render_seconds": round(time.time() - t0, 1)}
    result["check_master"] = check(master, lay.width, lay.height, fps_, n, a_len)
    if share:
        sh = out / f"{name}_{lay.width}x{lay.height}_30fps.mp4"
        vf = "fps=30" if fps_ != 30 else "null"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(master), "-vf", vf, "-c:v", "libx264", "-preset", "slow",
                        "-crf", str(cfg["output"]["shareCrf"]), "-profile:v", "high", "-pix_fmt", "yuv420p",
                        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
                        "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", str(sh)], check=True)
        result["share"] = str(sh)
        result["check_share"] = check(sh, lay.width, lay.height, 30, math.ceil(n * 30 / fps_ - 1e-9), a_len)  # fps filter rounds up
    result["sync"] = sync_report(p)
    shutil.rmtree(tmp, ignore_errors=True)
    (out / "render_report.json").write_text(json.dumps(result, indent=2))
    log(f"[reel] done in {result['render_seconds']}s -> {master}")
    return result
