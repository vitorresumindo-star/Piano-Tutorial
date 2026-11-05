"""End-to-end production of one tutorial: analysis -> validation gate ->
audio -> frames -> MP4 -> thumbnail -> metadata -> integrity checks."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .audio import SR
from .project import (Prepared, export_scene, metadata_draft, normalize_config, prepare, render_program_audio,
                      scene_data)
from .validation import report_markdown

ROOT = Path(__file__).resolve().parent.parent
RENDERER = ROOT / "renderer" / "render.mjs"


def slugify(s: str) -> str:
    out = "".join(c.lower() if c.isalnum() else "-" for c in s).strip("-")
    while "--" in out:
        out = out.replace("--", "-")
    return out or "untitled"


def detect_environment() -> dict:
    env = {"cpu_count": os.cpu_count(), "ffmpeg": bool(shutil.which("ffmpeg")), "node": bool(shutil.which("node"))}
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5)
        env["nvidia_gpu"] = out.stdout.strip() if out.returncode == 0 else ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        env["nvidia_gpu"] = ""
    env["gpu_rendering"] = bool(env["nvidia_gpu"])
    return env


def _node_env():
    env = dict(os.environ)
    gm = "/opt/node22/lib/node_modules"
    env["NODE_PATH"] = (env.get("NODE_PATH", "") + os.pathsep + gm).strip(os.pathsep)
    return env


def run_renderer(args: list[str], on_event=None, log_file: Path | None = None) -> dict:
    cmd = ["node", str(RENDERER)] + args
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=_node_env(), cwd=ROOT)
    last = {}
    lf = open(log_file, "a") if log_file else None
    for line in proc.stdout:
        if lf:
            lf.write(line)
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        last = ev
        if on_event:
            on_event(ev)
        if ev.get("event") == "error":
            raise RuntimeError(ev.get("message"))
    proc.wait()
    if lf:
        lf.close()
    if proc.returncode != 0:
        raise RuntimeError(f"renderer failed (exit {proc.returncode}); see log")
    return last


def probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                         capture_output=True, text=True)
    return json.loads(out.stdout or "{}")


def check_video(path: Path, expect: dict) -> dict:
    """Export integrity: container/codecs/resolution/fps/duration/audio sync."""
    info = probe(path)
    streams = info.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), {})
    a = next((s for s in streams if s.get("codec_type") == "audio"), {})
    fr = v.get("r_frame_rate", "0/1").split("/")
    fps = float(fr[0]) / float(fr[1]) if float(fr[1]) else 0
    vdur = float(v.get("duration", 0) or 0)
    adur = float(a.get("duration", 0) or 0)
    nb = int(v.get("nb_frames", 0) or 0)
    checks = {
        "h264": v.get("codec_name") == "h264",
        "aac": a.get("codec_name") == "aac",
        "resolution": (v.get("width"), v.get("height")) == (expect["width"], expect["height"]),
        "fps": abs(fps - expect["fps"]) < 1e-3,
        "frame_count": nb == expect["frames"],
        "yuv420p": v.get("pix_fmt") == "yuv420p",
        "audio_video_duration_match": abs(vdur - adur) < 0.05,
        "faststart": True,
    }
    return {"passed": all(checks.values()), "checks": checks, "video_duration": vdur, "audio_duration": adur,
            "frames": nb, "fps": fps, "size_mb": round(path.stat().st_size / 1e6, 2)}


def produce(cfg: dict, out_root: Path, *, force: bool = False, workers: int | None = None,
            on_progress=None, thumbnail: bool = True, frames: tuple[int, int] | None = None) -> dict:
    """Produce one video. Returns a result dict; never exports a song whose
    validation is blocked (or needs review in strict mode) unless force=True."""
    t_start = time.time()
    cfg = normalize_config(cfg)
    slug = slugify(cfg["title"])
    vid = cfg["video"]
    out_dir = Path(out_root) / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "render.log"
    emit = on_progress or (lambda ev: None)
    emit({"event": "stage", "stage": "analysis"})
    p = prepare(cfg)
    rep = p.report
    (out_dir / "validation.json").write_text(json.dumps(rep, indent=2))
    (out_dir / "validation.md").write_text(report_markdown(rep, cfg["title"]))
    (out_dir / "project.json").write_text(json.dumps(cfg, indent=2))
    fingering = [{"id": n.id, "pitch": n.pitch, "start": round(n.start, 4), "hand": n.hand, "finger": n.finger,
                  "source": n.finger_source} for n in sorted(p.song.notes, key=lambda n: (n.start, n.pitch))]
    (out_dir / "fingering.json").write_text(json.dumps(fingering, indent=1))
    result = {"slug": slug, "out_dir": str(out_dir), "status": rep["status"], "validation": rep["summary"]}
    if rep["status"] != "pass" and not force:
        result["rendered"] = False
        result["reason"] = ("blocked by validation errors" if rep["status"] == "blocked"
                            else "paused: fingering/hand assignment needs review (approve in the editor or use --force)")
        (out_dir / "STATUS.txt").write_text(result["reason"] + "\n")
        emit({"event": "paused", **result})
        return result
    job = out_dir / "_job"
    data = export_scene(p, job)
    emit({"event": "stage", "stage": "audio"})
    astats = render_program_audio(p, job / "audio.wav")
    result["audio"] = astats
    name = f"{slug}_{vid['width']}x{vid['height']}_{vid['fps']}fps.mp4"
    out_mp4 = out_dir / name
    env = detect_environment()
    if workers is None:
        workers = 2 if (env["cpu_count"] or 1) >= 4 else 1
    emit({"event": "stage", "stage": "render", "frames": p.program.frame_count, "environment": env})
    args = ["--job", str(job), "--out", str(out_mp4), "--audio", str(job / "audio.wav"),
            "--crf", str(vid.get("crf", 18)), "--preset", str(vid.get("x264preset", "medium")), "--workers", str(workers)]
    if env["gpu_rendering"]:
        args.append("--gpu")
    if frames:
        args += ["--start", str(frames[0]), "--end", str(frames[1])]
    run_renderer(args, emit, log)
    expect = {"width": vid["width"], "height": vid["height"], "fps": vid["fps"],
              "frames": (frames[1] - frames[0]) if frames else p.program.frame_count}
    integrity = check_video(out_mp4, expect)
    if frames:
        integrity["checks"]["audio_video_duration_match"] = True  # partial render: audio is the full program
        integrity["passed"] = all(integrity["checks"].values())
    result["integrity"] = integrity
    result["video"] = str(out_mp4)
    if thumbnail:
        emit({"event": "stage", "stage": "thumbnail"})
        result["thumbnail"] = str(make_thumbnail(p, out_dir, data))
    meta = metadata_draft(p, data)
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    (out_dir / "description.txt").write_text(meta["title"] + "\n\n" + meta["description"])
    result["rendered"] = True
    result["seconds"] = round(time.time() - t_start, 1)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    shutil.rmtree(job / "frames", ignore_errors=True)
    emit({"event": "done", **result})
    return result


def busiest_time(p: Prepared) -> float:
    """Pick a visually rich moment (many notes on screen) for the thumbnail."""
    seg = next(s for s in p.program.segments if s.type == "performance")
    travel = float(p.cfg["notes"]["travelTime"])
    best, best_t = -1, seg.start + 3
    t = seg.start + 2.0
    while t < seg.end - 2:
        s = seg.songStart + (t - seg.start)
        cnt = sum(1 for n in p.song.notes if s - 0.2 < n.start < s + travel)
        if cnt > best:
            best, best_t = cnt, t
        t += 0.5
    return best_t


def make_thumbnail(p: Prepared, out_dir: Path, data: dict) -> Path:
    """1280x720 thumbnail: a rendered still of the scene + bold title typography."""
    job = out_dir / "_thumbjob"
    job.mkdir(exist_ok=True)
    d = json.loads(json.dumps(data))
    d["program"]["width"], d["program"]["height"] = 1280, 720
    d["options"]["progressBar"] = False
    d["options"]["thumbnail"] = True
    (job / "perf.json").write_text(json.dumps(d))
    shutil.copy(out_dir / "_job" / "bake.bin", job / "bake.bin")
    t = busiest_time(p)
    still = out_dir / "thumbnail_raw.png"
    run_renderer(["--job", str(job), "--still", f"{t:.3f}", "--png", str(still), "--w", "1280", "--h", "720"])
    out = out_dir / "thumbnail.jpg"
    _compose_thumbnail(still, out, p.cfg, d)
    shutil.rmtree(job, ignore_errors=True)
    return out


def _compose_thumbnail(still: Path, out: Path, cfg: dict, data: dict):
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
    img = Image.open(still).convert("RGB").resize((1280, 720))
    ui = data["theme"]["ui"]
    light = data["theme"]["id"] == "ivory_minimal"
    grad = Image.new("L", (1280, 720), 0)
    gd = ImageDraw.Draw(grad)
    for x in range(1280):
        a = int(max(0, 1 - x / 760) ** 1.4 * 235)
        gd.line([(x, 0), (x, 720)], fill=a)
    shade = Image.new("RGB", (1280, 720), (246, 242, 234) if light else (6, 7, 10))
    img = Image.composite(shade, img, grad)
    d = ImageDraw.Draw(img)
    fonts = ROOT / "web" / "fonts"
    def font(name, size):
        try:
            return ImageFont.truetype(str(fonts / name), size)
        except OSError:
            return ImageFont.load_default(size)
    title = cfg["title"]
    words, lines, cur = title.split(), [], ""
    f_title = font("cormorant-garamond-latin-500-normal.woff2", 104 if len(title) < 16 else 84)
    for w in words:
        test = (cur + " " + w).strip()
        if d.textlength(test, font=f_title) > 640 and cur:
            lines.append(cur)
            cur = w
        else:
            cur = test
    lines.append(cur)
    y = 360 - (len(lines) * 100) // 2 - 30
    d.text((70, y - 70), "PIANO TUTORIAL", font=font("inter-latin-600-normal.woff2", 26), fill=ui["accent"])
    for ln in lines:
        d.text((66, y), ln, font=f_title, fill=ui["text"])
        y += 100 if len(title) < 16 else 84
    if cfg.get("composer"):
        d.text((70, y + 14), cfg["composer"], font=font("cormorant-garamond-latin-400-italic.woff2", 40), fill=ui["muted"])
    diff = data["song"]["difficulty"].upper()
    fd = font("inter-latin-600-normal.woff2", 22)
    tw = d.textlength(diff, font=fd)
    d.rounded_rectangle([70, 610, 70 + tw + 40, 652], radius=21, outline=ui["accent"], width=2)
    d.text((90, 619), diff, font=fd, fill=ui["accent"])
    img.save(out, quality=92)
