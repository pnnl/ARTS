"""Campaign output: a replayable selection, machine-readable results, and the
tables a person reads."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

from artsrun.check import Group, Verdict, minority_report
from artsrun.model.plane import Plane
from artsrun.model.selection import Selection
from artsrun.run.types import CellResult, Skipped, Status

RESULT_COLUMNS = [
    "app", "version", "nodes", "entry", "kind", "repeat",
    "status", "rc", "wall_s", "e2e_s", "scalar", "note", "log",
]

# Display metadata for runtime series, carried into the report so a later
# figure tool inherits one naming and colour convention instead of inventing
# its own.
RT_LABEL = {
    "ocr_excl_purge": "EXCL/PURGE",
    "ocr_excl_retain": "EXCL/RETAIN",
    "ocr_inv_wt": "INV/WT",
    "ocr_inv_wb": "INV/WB",
    "ocr_val_wt": "VAL/WT",
    "ocr_val_wb": "VAL/WB",
    "ocr_val_wt_comb": "VAL+comb/WT",
    "ocr_val_wb_comb": "VAL+comb/WB",
    "xsocr": "XSOCR",
    # OCR-vx ships three runtime families; the one built here is the
    # distributed-memory one, which is what the label should name.
    "ocrvx": "OCR-Vdm",
}
RT_COLOR = {
    "ocr_excl_purge": "#4C72B0",
    "ocr_excl_retain": "#7BA3D8",
    "ocr_inv_wt": "#DD8452",
    "ocr_inv_wb": "#F0B08A",
    "ocr_val_wt": "#55A868",
    "ocr_val_wb": "#8CCB9B",
    "ocr_val_wt_comb": "#C44E52",
    "ocr_val_wb_comb": "#E08A8D",
    "xsocr": "#8172B3",
    "ocrvx": "#937860",
}


def _row(r: CellResult) -> dict:
    return {
        "app": r.cell.app.name,
        "version": r.cell.app.version.value,
        "nodes": r.cell.nodes,
        "entry": r.cell.entry.key,
        "kind": r.cell.entry.kind.value,
        "repeat": r.cell.repeat,
        "status": r.status.value,
        "rc": r.rc,
        "wall_s": round(r.wall_s, 3),
        "e2e_s": round(r.e2e_s, 3) if r.e2e_s is not None else "",
        "scalar": r.scalar or "",
        "note": r.note,
        "log": str(r.log_path) if r.log_path else "",
    }


def write_results_csv(results: list[CellResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for r in results:
            writer.writerow(_row(r))


def write_report_json(
    results: list[CellResult],
    groups: list[Group],
    skipped: list[Skipped],
    selection: Selection,
    path: Path,
) -> None:
    payload = {
        "selection": selection.model_dump(mode="json"),
        "runtimes": {
            key: {"label": RT_LABEL.get(key, key), "color": RT_COLOR.get(key)}
            for key in selection.entries
        },
        "cells": [_row(r) for r in results],
        "consensus": [
            {
                "app": g.app_key,
                "nodes": g.nodes,
                "value": g.consensus,
                "verdicts": {k: v.value for k, v in g.verdicts.items()},
                "unanimous": g.unanimous,
            }
            for g in groups
        ],
        "skipped": [asdict(s) for s in skipped],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def consensus_table(groups: list[Group], plane: Plane, entries: list[str]) -> Table:
    table = Table(title="Consensus by application and node count", expand=False)
    table.add_column("app", style="bold")
    table.add_column("n", justify="right")
    table.add_column("value")
    for key in entries:
        table.add_column(RT_LABEL.get(key, key), justify="center")

    style = {
        Verdict.OK: "[green]OK[/green]",
        Verdict.DISAGREE: "[red]DIFF[/red]",
        Verdict.FAIL: "[red]FAIL[/red]",
        Verdict.NA: "[dim]--[/dim]",
        Verdict.EXPECT_FAIL: "[yellow]EXP![/yellow]",
    }
    for g in groups:
        cells = [style.get(g.verdicts.get(k, Verdict.NA), "?") for k in entries]
        table.add_row(g.app_key, str(g.nodes), g.consensus or "-", *cells)
    return table


def scaling_table(results: list[CellResult], entries: list[str]) -> Table:
    """Measured time per node count, one row per (application, configuration).

    The runtime's own end-to-end stamp (init and teardown excluded) is the
    measurement; process wall is the fallback for a run that predates the
    marker or never reached shutdown recognition.
    """
    node_counts = sorted({r.cell.nodes for r in results})
    table = Table(title="Strong scaling (e2e seconds)", expand=False)
    table.add_column("app", style="bold")
    table.add_column("configuration")
    for n in node_counts:
        table.add_column(f"{n}n", justify="right")

    by_key: dict[tuple[str, str], dict[int, float]] = {}
    for r in results:
        if r.status is not Status.OK:
            continue
        measured = r.e2e_s if r.e2e_s is not None else r.wall_s
        key = (r.cell.app.key, r.cell.entry.key)
        walls = by_key.setdefault(key, {})
        # Repeats collapse to their best observation.
        walls[r.cell.nodes] = min(walls.get(r.cell.nodes, measured), measured)

    for (app_key, entry_key) in sorted(by_key):
        walls = by_key[(app_key, entry_key)]
        row = [f"{walls[n]:.2f}" if n in walls else "-" for n in node_counts]
        table.add_row(app_key, RT_LABEL.get(entry_key, entry_key), *row)
    return table


def write_summary(
    results: list[CellResult],
    groups: list[Group],
    skipped: list[Skipped],
    selection: Selection,
    plane: Plane,
    path: Path,
) -> str:
    console = Console(record=True, width=200, file=open("/dev/null", "w"))
    console.print(consensus_table(groups, plane, selection.entries))
    console.print()
    console.print(scaling_table(results, selection.entries))

    minority = minority_report(groups)
    console.print()
    if minority:
        console.print("[bold]MINORITY REPORT[/bold]")
        for g in minority:
            dissent = ", ".join(
                f"{k}={v.value}" for k, v in g.verdicts.items()
                if v in (Verdict.DISAGREE, Verdict.EXPECT_FAIL, Verdict.FAIL)
            )
            console.print(f"  {g.app_key} @ {g.nodes}n consensus={g.consensus} -> {dissent}")
    else:
        console.print("[bold]No disagreement.[/bold]")

    if skipped:
        console.print()
        console.print(f"[bold]Not run ({len(skipped)} cells)[/bold]")
        seen = set()
        for s in skipped:
            if s.reason in seen:
                continue
            seen.add(s.reason)
            console.print(f"  {s.app_key}: {s.reason}")

    text = console.export_text()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text
