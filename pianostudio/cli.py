"""Command line interface.

  python -m pianostudio analyze song.mid               # parse + fingering + validation report
  python -m pianostudio render song.mid --title "..."  # full video
  python -m pianostudio batch midi_folder/ --preset presets/channel.json
  python -m pianostudio reel projects/dust_and_starlight  # vertical cinematic piano reel (9:16)
  python -m pianostudio serve                          # web app on http://localhost:8765
  python -m pianostudio doctor                         # environment check
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .project import VIDEO_PRESETS, deep_merge


def _cfg_from_args(a) -> dict:
    cfg = {}
    if getattr(a, "preset", None):
        cfg = json.loads(Path(a.preset).read_text())
    if getattr(a, "config", None):
        cfg = deep_merge(cfg, json.loads(Path(a.config).read_text()))
    if getattr(a, "midi", None):
        cfg["midi"] = str(Path(a.midi).resolve())
        cfg.setdefault("title", Path(a.midi).stem.replace("_", " ").title())
    for k in ("title", "composer", "theme"):
        if getattr(a, k, None):
            cfg[k] = getattr(a, k)
    if getattr(a, "video", None):
        cfg.setdefault("video", {})["preset"] = a.video
    if getattr(a, "practice", False):
        cfg.setdefault("structure", {})["practice"] = True
    if getattr(a, "no_intro", False):
        cfg.setdefault("structure", {})["intro"] = False
    if getattr(a, "no_outro", False):
        cfg.setdefault("structure", {})["outro"] = False
    if getattr(a, "debug", False):
        cfg["debug"] = True
    if getattr(a, "fingers", False):
        cfg.setdefault("notes", {})["showFingers"] = True
    if getattr(a, "names", False):
        cfg.setdefault("notes", {})["showNoteNames"] = True
    if getattr(a, "cinematic", False):
        cfg["cinematic"] = True
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pianostudio", description="Automated cinematic piano tutorial generator")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--preset", help="channel preset JSON")
        p.add_argument("--config", help="project JSON (overrides preset)")
        p.add_argument("--title")
        p.add_argument("--composer")
        p.add_argument("--theme", help="midnight_studio | ivory_minimal | warm_concert_hall | modern_neon")
        p.add_argument("--video", choices=list(VIDEO_PRESETS), help="export preset")
        p.add_argument("--practice", action="store_true", help="add a slow practice section")
        p.add_argument("--no-intro", action="store_true")
        p.add_argument("--no-outro", action="store_true")
        p.add_argument("--fingers", action="store_true", help="finger numbers on falling notes")
        p.add_argument("--names", action="store_true", help="note names on falling notes")
        p.add_argument("--debug", action="store_true", help="debug render: fingertip contact markers")

    p = sub.add_parser("analyze", help="parse, finger and validate (no rendering)")
    p.add_argument("midi")
    common(p)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("render", help="produce a full tutorial video")
    p.add_argument("midi", nargs="?")
    common(p)
    p.add_argument("--out", default="output")
    p.add_argument("--force", action="store_true", help="render even if validation asks for review")
    p.add_argument("--workers", type=int)
    p.add_argument("--frames", help="render only a frame range a:b (for quick checks)")
    p.add_argument("--cinematic", action="store_true", help="bloom post-processing (slow without a GPU)")

    p = sub.add_parser("batch", help="render every MIDI in a folder")
    p.add_argument("folder")
    p.add_argument("--preset")
    p.add_argument("--out", default="output")
    p.add_argument("--force", action="store_true")
    p.add_argument("--workers", type=int)
    p.add_argument("--dry-run", action="store_true", help="only analyse + validate")

    p = sub.add_parser("reel", help="render a vertical cinematic piano reel from a project folder")
    p.add_argument("project", help="project folder or project.json")
    p.add_argument("--out", help="output folder (default output/<name>)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--fps", type=int, help="override the project frame rate")
    p.add_argument("--frames", help="render only a frame range a:b")
    p.add_argument("--stills", help="comma-separated times (s): save PNG stills instead of a video")
    p.add_argument("--no-share", action="store_true", help="skip the 30 fps share version")

    p = sub.add_parser("serve", help="start the web application")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)

    sub.add_parser("doctor", help="check dependencies and hardware")
    a = ap.parse_args(argv)

    def printer(ev):
        if ev.get("event") == "progress":
            print(f"\r  frames {ev['frames']}/{ev['total']}  {ev['fps']} fps  eta {ev['eta']}s   ", end="", flush=True)
        elif ev.get("event") in ("stage", "paused", "done", "song", "song_done"):
            print(("\n" if ev.get("event") != "song" else "") + json.dumps({k: v for k, v in ev.items() if k != "environment"}))

    if a.cmd == "analyze":
        from .project import prepare
        from .validation import report_markdown
        pr = prepare(_cfg_from_args(a))
        if a.json:
            print(json.dumps(pr.report, indent=2))
        else:
            print(report_markdown(pr.report, pr.cfg["title"]))
        return 0 if pr.report["status"] != "blocked" else 1
    if a.cmd == "render":
        from .export import produce
        cfg = _cfg_from_args(a)
        frames = tuple(int(x) for x in a.frames.split(":")) if a.frames else None
        res = produce(cfg, Path(a.out), force=a.force, workers=a.workers, on_progress=printer, frames=frames)
        print()
        print(json.dumps({k: v for k, v in res.items() if k not in ("audio",)}, indent=2))
        return 0 if res.get("rendered") else 3
    if a.cmd == "batch":
        from .batch import run_batch
        preset = json.loads(Path(a.preset).read_text()) if a.preset else {}
        s = run_batch(Path(a.folder), Path(a.out), preset, force=a.force, workers=a.workers, dry_run=a.dry_run,
                      on_event=printer)
        print(json.dumps(s["counts"], indent=2))
        return 0
    if a.cmd == "reel":
        from .reel.project import MissingMidi
        from .reel.render import produce_reel
        try:
            res = produce_reel(a.project, a.out, workers=a.workers,
                               frames=tuple(int(x) for x in a.frames.split(":")) if a.frames else None,
                               fps=a.fps, stills=[float(x) for x in a.stills.split(",")] if a.stills else None,
                               share=not a.no_share)
        except MissingMidi as e:
            print(str(e))
            return 2
        print(json.dumps(res, indent=2))
        ok = res.get("stills") or res.get("check_master", {}).get("ok")
        return 0 if ok else 3
    if a.cmd == "serve":
        from .server import run
        run(a.host, a.port)
        return 0
    if a.cmd == "doctor":
        from .audio import DEFAULT_SFZ
        from .export import detect_environment
        env = detect_environment()
        root = Path(__file__).resolve().parent.parent
        env["three_js"] = (root / "node_modules" / "three").exists()
        env["piano_samples"] = DEFAULT_SFZ.exists()
        try:
            import subprocess
            r = subprocess.run(["node", "-e", "require('playwright');console.log('ok')"], capture_output=True, text=True,
                               env={**__import__('os').environ, "NODE_PATH": "/opt/node22/lib/node_modules"})
            env["playwright"] = r.stdout.strip() == "ok"
        except FileNotFoundError:
            env["playwright"] = False
        print(json.dumps(env, indent=2))
        missing = [k for k in ("ffmpeg", "node", "three_js", "playwright") if not env.get(k)]
        if not env["piano_samples"]:
            print("note: piano samples missing -> run scripts/setup.sh (fallback synth will be used)")
        if missing:
            print("missing:", ", ".join(missing))
            return 1
        return 0


if __name__ == "__main__":
    sys.exit(main())
