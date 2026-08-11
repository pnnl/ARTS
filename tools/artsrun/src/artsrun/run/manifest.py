"""What a campaign was asked to run, written before any of it runs.

The track file records what happened; the manifest records what was asked —
every cell with the exact command it will run under, and enough of the
application metadata to vote a consensus.  Together the two files let a reader
rebuild the whole campaign from the run directory alone, without re-resolving
the selection against profiles or benchsets that may have changed since.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from artsrun.model.benchset import ResolvedApp
from artsrun.model.plane import SelectionEntry
from artsrun.model.profile import Launcher, Profile
from artsrun.paths import scratch_dir
from artsrun.run.command import build_command, build_env, render, with_timeout
from artsrun.run.types import Cell, Skipped

VERSION = 1

FILE_NAME = "manifest.json"


def describe_command(cell: Cell, profile: Profile, log_path: Path) -> dict:
    """The exact invocation one cell runs under.

    A Slurm cell is two layers — the submission command and the batch script
    it carries — and both are part of what "the command" means there.
    """
    if profile.launcher is Launcher.SLURM:
        from artsrun.run.slurm import job_script, sbatch_argv

        return {
            "command": render(sbatch_argv(cell, profile, log_path)),
            "script": job_script(cell, profile),
        }
    return {
        "command": render(with_timeout(build_command(cell, profile),
                                       cell.timeout_s)),
        "script": None,
    }


def write_manifest(
    run_dir: Path,
    cells: list[Cell],
    skipped: list[Skipped],
    profile: Profile,
    build_dir: Path,
    *,
    counters_cfg: Path | None = None,
) -> Path:
    entries: dict[str, dict] = {}
    apps: dict[str, dict] = {}
    rows = []
    for cell in cells:
        entries.setdefault(cell.entry.key, cell.entry.model_dump(mode="json"))
        apps.setdefault(cell.app.key, cell.app.model_dump(mode="json"))
        log_path = run_dir / "cells" / cell.log_name
        rows.append({
            "key": cell.key,
            "slug": cell.slug,
            "entry": cell.entry.key,
            "app": cell.app.key,
            "nodes": cell.nodes,
            "repeat": cell.repeat,
            "binary": str(cell.binary),
            "args": list(cell.args),
            "timeout_s": cell.timeout_s,
            "cfg": str(cell.cfg) if cell.cfg else None,
            "env": build_env(cell, profile),
            "log": str(log_path),
            **describe_command(cell, profile, log_path),
        })

    payload = {
        "version": VERSION,
        "written": time.time(),
        "launcher": profile.launcher.value,
        "profile": profile.name,
        "build_dir": str(build_dir),
        "cwd": str(scratch_dir()),
        "stderr": "merged into stdout",
        "stdin": "/dev/null",
        "counters_cfg": str(counters_cfg) if counters_cfg else None,
        "entries": entries,
        "apps": apps,
        "cells": rows,
        "skipped": [
            {"entry": s.entry_key, "app": s.app_key, "nodes": s.nodes,
             "reason": s.reason}
            for s in skipped
        ],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / FILE_NAME
    path.write_text(json.dumps(payload, indent=1))
    return path


class Manifest:
    """A written manifest, read back with its cells as the real thing.

    The entries and applications revalidate through their own models, so a
    reader holds the same `Cell` objects the campaign ran — anything that
    works on a live campaign's cells (extraction, voting) works here too.
    """

    def __init__(self, payload: dict):
        self.payload = payload
        self.launcher = payload.get("launcher", "")
        self.cwd = payload.get("cwd", "")
        raw = payload.get("counters_cfg")
        self.counters_cfg = Path(raw) if raw else None
        entries = {
            k: SelectionEntry.model_validate(v)
            for k, v in payload.get("entries", {}).items()
        }
        apps = {
            k: ResolvedApp.model_validate(v)
            for k, v in payload.get("apps", {}).items()
        }
        self.cells: list[Cell] = []
        self.commands: dict[str, dict] = {}
        for row in payload.get("cells", []):
            cell = Cell(
                entry=entries[row["entry"]],
                app=apps[row["app"]],
                nodes=int(row["nodes"]),
                repeat=int(row["repeat"]),
                binary=Path(row["binary"]),
                args=list(row["args"]),
                timeout_s=int(row["timeout_s"]),
                cfg=Path(row["cfg"]) if row.get("cfg") else None,
                env=dict(row.get("env", {})),
            )
            self.cells.append(cell)
            self.commands[cell.key] = {
                "command": row.get("command", ""),
                "script": row.get("script"),
                "log": row.get("log", ""),
            }
        self.skipped = [
            Skipped(s["entry"], s["app"], int(s["nodes"]), s["reason"])
            for s in payload.get("skipped", [])
        ]

    @classmethod
    def load(cls, run_dir: Path) -> "Manifest | None":
        path = run_dir / FILE_NAME
        if not path.is_file():
            return None
        try:
            return cls(json.loads(path.read_text(errors="replace")))
        except (ValueError, KeyError):
            return None
