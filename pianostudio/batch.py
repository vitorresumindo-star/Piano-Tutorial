"""Batch production: a folder of MIDI files -> one output folder per song.

For each song: analyse, generate validation report, render ONLY if checks pass
(or the song was approved in a per-song override file), apply the channel
preset, and produce video + thumbnail + metadata.  Songs that need fingering
corrections are paused (status "needs_review") instead of exported.

Per-song settings can be placed next to the MIDI as ``<name>.json`` (title,
composer, overrides, approved, ...). They are merged over the channel preset.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

from .export import produce
from .project import deep_merge


def title_from_filename(p: Path) -> str:
    return p.stem.replace("_", " ").replace("-", " ").strip().title()


def run_batch(midi_dir: Path, out_root: Path, preset: dict | None = None, *, force: bool = False,
              workers: int | None = None, dry_run: bool = False, on_event=None, frames=None) -> dict:
    midi_dir, out_root = Path(midi_dir), Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    files = sorted([*midi_dir.glob("*.mid"), *midi_dir.glob("*.midi")])
    summary = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "input": str(midi_dir), "songs": []}
    emit = on_event or (lambda e: None)
    for f in files:
        song_cfg = {"midi": str(f), "title": title_from_filename(f)}
        side = f.with_suffix(".json")
        if side.exists():
            song_cfg = deep_merge(song_cfg, json.loads(side.read_text()))
        cfg = deep_merge(preset or {}, song_cfg)
        emit({"event": "song", "file": f.name, "title": cfg["title"]})
        try:
            if dry_run:
                from .project import prepare
                p = prepare(cfg)
                res = {"slug": cfg["title"], "status": p.report["status"], "rendered": False,
                       "validation": p.report["summary"]}
            else:
                res = produce(cfg, out_root, force=force, workers=workers, on_progress=emit, frames=frames)
        except Exception as e:  # keep going with the next song
            res = {"status": "failed", "error": str(e), "trace": traceback.format_exc(), "rendered": False}
        res["file"] = f.name
        summary["songs"].append(res)
        emit({"event": "song_done", **{k: v for k, v in res.items() if k != "trace"}})
    summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    summary["counts"] = {
        "total": len(files),
        "rendered": sum(1 for s in summary["songs"] if s.get("rendered")),
        "paused": sum(1 for s in summary["songs"] if s.get("status") == "needs_review" and not s.get("rendered")),
        "blocked": sum(1 for s in summary["songs"] if s.get("status") == "blocked"),
        "failed": sum(1 for s in summary["songs"] if s.get("status") == "failed"),
    }
    (out_root / "batch_summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["# Batch summary", "", f"Input: `{midi_dir}`", "", "| File | Status | Rendered | Notes |", "|---|---|---|---|"]
    for s in summary["songs"]:
        lines.append(f"| {s['file']} | {s.get('status')} | {'yes' if s.get('rendered') else 'no'} | "
                     f"{s.get('reason') or s.get('error') or ''} |")
    (out_root / "batch_summary.md").write_text("\n".join(lines) + "\n")
    return summary
