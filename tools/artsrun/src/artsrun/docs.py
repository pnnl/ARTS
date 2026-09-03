"""Structural documents: what one application is, as a page.

The document itself is a per-entry markdown file shipped with the tool
(`data/appdocs/<name>.md`); an entry without one still renders as the
catalog's own facts, so every entry can be opened.
"""

from __future__ import annotations

from artsrun.data import load_doc
from artsrun.model.catalog import AppEntry, Kind


def facts_markdown(entry: AppEntry) -> str:
    """The catalog's own facts, as the document's fallback body."""
    lines = [f"# {entry.name}", ""]
    if entry.provenance:
        lines += [f"*{entry.provenance}*", ""]
    versions = ", ".join(v.value for v in entry.own_versions)
    lines += [
        f"- **binary**: `{entry.binary}`",
        f"- **class**: {entry.cls.value}",
        f"- **versions**: {versions}",
        f"- **completion marker**: `{entry.marker}`",
        f"- **result scalar**: `{entry.result_re}`"
        + (f" (tolerance {entry.tolerance})" if entry.tolerance else ""),
        f"- **calibrated args**: `{' '.join(entry.args) or '(none)'}`",
    ]
    if entry.hpx:
        lines.append(
            f"- **HPX port**: {', '.join(v.value for v in entry.hpx)}")
    if entry.multinode_skip:
        lines.append(f"- **multinode**: skipped — {entry.multinode_skip}")
    if entry.ocrvx_skip:
        lines.append("- **ocr-vx**: skipped — uses extensions it lacks")
    lines += ["", "_No structural document for this entry._"]
    if entry.kind is Kind.TOY:
        lines[-1] = ("_No structural document: a toy exercises one runtime "
                     "mechanism; the mechanism is the story._")
    elif entry.kind is Kind.ATTACK:
        lines[-1] = ("_No structural document: an attack is an adversarial "
                     "characterization probe; its knobs and output line are "
                     "the story._")
    return "\n".join(lines)


def document(entry: AppEntry) -> str:
    """The entry's document, falling back to its catalog facts."""
    return load_doc(entry.name) or facts_markdown(entry)
