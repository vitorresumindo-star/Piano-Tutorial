"""Web application backend (Flask).  Start with:  python -m pianostudio serve"""
from __future__ import annotations

import json
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_from_directory

from .project import (DEFAULT_CONFIG, VIDEO_PRESETS, deep_merge, export_scene, list_themes, normalize_config,
                      prepare, render_program_audio)

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
WORK = ROOT / "workspace" / "projects"
OUTPUT = ROOT / "output"
PRESETS = ROOT / "presets"
EXAMPLES = ROOT / "examples" / "midi"

app = Flask(__name__, static_folder=None)
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()


def pdir(pid: str) -> Path:
    d = WORK / pid
    if not d.exists() or "/" in pid or ".." in pid:
        abort(404)
    return d


def load_cfg(pid: str) -> dict:
    return json.loads((pdir(pid) / "project.json").read_text())


def save_cfg(pid: str, cfg: dict):
    (pdir(pid) / "project.json").write_text(json.dumps(cfg, indent=2))


# ---------------------------------------------------------------- static
@app.get("/")
def index():
    return send_from_directory(WEB, "index.html")


@app.get("/web/<path:p>")
def web(p):
    return send_from_directory(WEB, p)


@app.get("/vendor/three/<path:p>")
def three(p):
    return send_from_directory(ROOT / "node_modules" / "three", p)


# ---------------------------------------------------------------- catalog
@app.get("/api/meta")
def meta():
    return jsonify({"themes": list_themes(), "videoPresets": VIDEO_PRESETS, "defaults": DEFAULT_CONFIG,
                    "examples": sorted(p.name for p in EXAMPLES.glob("*.mid")),
                    "presets": sorted(p.stem for p in PRESETS.glob("*.json"))})


@app.get("/api/presets/<name>")
def get_preset(name):
    p = PRESETS / f"{Path(name).name}.json"
    if not p.exists():
        abort(404)
    return jsonify(json.loads(p.read_text()))


@app.post("/api/presets")
def save_preset():
    body = request.get_json(force=True)
    name = "".join(c for c in body.get("name", "preset") if c.isalnum() or c in "-_ ").strip().replace(" ", "_") or "preset"
    keep = ("theme", "colors", "hands", "video", "notes", "structure", "audio", "branding", "range", "strict", "kicker")
    cfg = {k: v for k, v in body.get("config", {}).items() if k in keep}
    PRESETS.mkdir(exist_ok=True)
    (PRESETS / f"{name}.json").write_text(json.dumps(cfg, indent=2))
    return jsonify({"name": name})


# ---------------------------------------------------------------- projects
@app.get("/api/projects")
def projects():
    out = []
    for d in sorted(WORK.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
        f = d / "project.json"
        if f.exists():
            c = json.loads(f.read_text())
            out.append({"id": d.name, "title": c.get("title"), "composer": c.get("composer"), "theme": c.get("theme")})
    return jsonify(out)


@app.post("/api/projects")
def create_project():
    pid = uuid.uuid4().hex[:10]
    d = WORK / pid
    d.mkdir(parents=True)
    preset = {}
    if request.form.get("preset"):
        pp = PRESETS / f"{Path(request.form['preset']).name}.json"
        if pp.exists():
            preset = json.loads(pp.read_text())
    if "file" in request.files:
        f = request.files["file"]
        name = Path(f.filename or "song.mid").name
        dst = d / name
        f.save(dst)
    else:
        ex = Path(request.form.get("example", "minuet_in_g.mid")).name
        src = EXAMPLES / ex
        if not src.exists():
            abort(400)
        dst = d / ex
        shutil.copy(src, dst)
        side = src.with_suffix(".json")
        if side.exists():
            preset = deep_merge(preset, json.loads(side.read_text()))
    title = request.form.get("title") or preset.get("title") or dst.stem.replace("_", " ").title()
    composer = request.form.get("composer") or preset.get("composer", "")
    cfg = deep_merge(DEFAULT_CONFIG, deep_merge(preset, {"midi": str(dst), "title": title, "composer": composer}))
    save_cfg(pid, cfg)
    return jsonify({"id": pid, "config": cfg})


@app.get("/api/projects/<pid>")
def get_project(pid):
    return jsonify({"id": pid, "config": load_cfg(pid)})


@app.put("/api/projects/<pid>")
def put_project(pid):
    body = request.get_json(force=True)
    cfg = load_cfg(pid)
    new = deep_merge(cfg, body)
    if "overrides" in body:            # replace, don't merge
        new["overrides"] = body["overrides"]
    if "colors" in body:
        new["colors"] = body["colors"]
    new["midi"] = cfg["midi"]          # never let the client move the file
    save_cfg(pid, new)
    return jsonify({"id": pid, "config": new})


@app.post("/api/projects/<pid>/analyze")
def analyze(pid):
    """Prepare everything the live preview needs (scene data, baked animation,
    synchronised audio) and return the validation report + notes."""
    d = pdir(pid)
    cfg = load_cfg(pid)
    with_audio = request.args.get("audio", "1") == "1"
    t0 = time.time()
    try:
        p = prepare(cfg)
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 400
    prev = d / "preview"
    data = export_scene(p, prev, preview=True)
    audio = None
    if with_audio:
        audio = render_program_audio(p, prev / "audio.wav")
    issues_by_note = {}
    for i in p.report["issues"]:
        for nid in i.get("note_ids", []):
            issues_by_note.setdefault(str(nid), []).append({"severity": i["severity"], "message": i["message"]})
    return jsonify({"report": p.report, "notes": data["notes"], "program": data["program"], "audio": audio,
                    "issuesByNote": issues_by_note, "difficulty": data["song"]["difficulty"],
                    "range": data["range"], "seconds": round(time.time() - t0, 2), "stamp": time.time()})


@app.get("/api/projects/<pid>/preview/<path:f>")
def preview_file(pid, f):
    return send_from_directory(pdir(pid) / "preview", f, max_age=0)


# ---------------------------------------------------------------- rendering jobs
def _run_job(jid: str, kind: str, fn):
    def emit(ev):
        with LOCK:
            j = JOBS[jid]
            if ev.get("event") == "progress":
                j["progress"] = ev
            elif ev.get("event") == "stage":
                j["stage"] = ev.get("stage")
            elif ev.get("event") in ("song", "song_done"):
                j.setdefault("songs", []).append({k: v for k, v in ev.items() if k != "trace"})
            j["log"].append({k: v for k, v in ev.items() if k not in ("environment", "audio")})
            j["log"] = j["log"][-200:]
    try:
        res = fn(emit)
        with LOCK:
            JOBS[jid].update(state="done", result=res)
    except Exception as e:
        with LOCK:
            JOBS[jid].update(state="failed", error=str(e), trace=traceback.format_exc())


def start_job(kind: str, fn, meta: dict) -> str:
    jid = uuid.uuid4().hex[:10]
    JOBS[jid] = {"id": jid, "kind": kind, "state": "running", "stage": "queued", "log": [], "started": time.time(), **meta}
    threading.Thread(target=_run_job, args=(jid, kind, fn), daemon=True).start()
    return jid


@app.post("/api/projects/<pid>/render")
def render(pid):
    from .export import produce
    body = request.get_json(silent=True) or {}
    cfg = load_cfg(pid)
    out = OUTPUT
    jid = start_job("render", lambda emit: produce(cfg, out, force=bool(body.get("force")), on_progress=emit,
                                                   workers=body.get("workers")), {"project": pid, "title": cfg["title"]})
    return jsonify({"job": jid})


@app.post("/api/batch")
def batch():
    from .batch import run_batch
    body = request.get_json(force=True)
    folder = Path(body.get("folder", ""))
    if not folder.is_absolute():
        folder = ROOT / folder
    if not folder.is_dir():
        return jsonify({"error": f"folder not found: {folder}"}), 400
    preset = {}
    if body.get("preset"):
        pp = PRESETS / f"{Path(body['preset']).name}.json"
        if pp.exists():
            preset = json.loads(pp.read_text())
    out = Path(body.get("out") or OUTPUT / "batch")
    jid = start_job("batch", lambda emit: run_batch(folder, out, preset, force=bool(body.get("force")),
                                                    dry_run=bool(body.get("dryRun")), on_event=emit),
                    {"folder": str(folder)})
    return jsonify({"job": jid})


@app.get("/api/jobs")
def jobs():
    with LOCK:
        return jsonify([{k: v for k, v in j.items() if k not in ("log", "trace")} for j in JOBS.values()])


@app.get("/api/jobs/<jid>")
def job(jid):
    with LOCK:
        j = JOBS.get(jid)
        if not j:
            abort(404)
        return jsonify(j)


@app.get("/output/<path:f>")
def output_file(f):
    return send_from_directory(OUTPUT, f)


def run(host="127.0.0.1", port=8765):
    WORK.mkdir(parents=True, exist_ok=True)
    print(f"Piano Studio running at http://{host}:{port}")
    app.run(host=host, port=port, threaded=True, debug=False)
