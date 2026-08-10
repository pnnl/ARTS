"""Resolve a selection to build targets and compile them in one pass.

The whole selection is compiled before any cell runs, so a campaign never
interleaves building with measuring.  Ninja is the cache: a target that is
already current costs nothing to ask for again.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from artsrun.model.benchset import Benchset
from artsrun.model.catalog import Catalog
from artsrun.model.plane import Plane
from artsrun.model.selection import Selection


class BuildError(RuntimeError):
    pass


@dataclass
class BuildPlan:
    build_dir: Path
    targets: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing


def _cache_value(build_dir: Path, key: str) -> str | None:
    cache = build_dir / "CMakeCache.txt"
    if not cache.is_file():
        return None
    pattern = re.compile(rf"^{re.escape(key)}:[^=]*=(.*)$")
    for line in cache.read_text(errors="ignore").splitlines():
        m = pattern.match(line)
        if m:
            return m.group(1).strip()
    return None


def counter_config_of(build_dir: Path) -> str | None:
    """Which counter file this tree was configured against."""
    return _cache_value(build_dir, "ARTS_COUNTER_CONFIG")


def check_counter_config(build_dir: Path, wanted: Path) -> None:
    """Refuse a mismatch instead of measuring with the wrong counters.

    The selection is compiled in — Preamble.h holds the indices — so a tree
    configured against another file cannot be corrected by rebuilding a
    target.  Reconfiguring recompiles everything, which is the user's call to
    make, not something to do behind a campaign.

    What is compiled in is the file's CONTENT, so that is what decides.  Each
    campaign renders its counters into its own run directory, and comparing
    paths would call every repeat of one set a mismatch.
    """
    have = counter_config_of(build_dir)
    if have and Path(have).resolve() != wanted.resolve():
        try:
            same = (
                Path(have).read_text(errors="ignore")
                == wanted.read_text(errors="ignore")
            )
        except OSError:
            same = False  # the configured file is gone; its content is unknown
        if same:
            return
        raise BuildError(
            f"{build_dir} was configured with counters from\n"
            f"  {have}\n"
            f"but this campaign asks for\n"
            f"  {wanted}\n"
            f"Counter selection is compiled in, so switching means a full "
            f"rebuild:\n"
            f"  cmake -GNinja -B{build_dir} -DCMAKE_BUILD_TYPE=Release "
            f"-DARTS_COUNTER_CONFIG={wanted}"
        )


def check_build_dir(build_dir: Path) -> None:
    """Fail early, and say what to run, rather than reconfiguring silently."""
    if not (build_dir / "build.ninja").is_file():
        raise BuildError(
            f"{build_dir} is not a configured Ninja build tree. Configure it with:\n"
            f"  cmake -GNinja -B{build_dir} -DCMAKE_BUILD_TYPE=Release"
        )
    if (_cache_value(build_dir, "ARTS_BUILD_BENCHMARKS") or "ON") == "OFF":
        raise BuildError(
            f"{build_dir} was configured with ARTS_BUILD_BENCHMARKS=OFF; the "
            f"application targets do not exist there"
        )
    build_type = _cache_value(build_dir, "CMAKE_BUILD_TYPE") or ""
    if build_type and build_type.lower() != "release":
        raise BuildError(
            f"{build_dir} is a {build_type} tree; measurements must come from a "
            f"Release build (reconfigure with -DCMAKE_BUILD_TYPE=Release)"
        )


def available_targets(build_dir: Path) -> set[str]:
    out = subprocess.run(
        ["ninja", "-C", str(build_dir), "-t", "targets", "all"],
        capture_output=True, text=True, check=False,
    ).stdout
    names = set()
    for line in out.splitlines():
        target = line.split(":", 1)[0].strip()
        if target:
            names.add(Path(target).name)
    return names


def plan_targets(
    selection: Selection,
    plane: Plane,
    catalog: Catalog,
    benchset: Benchset,
    build_dir: Path,
) -> BuildPlan:
    """Every executable this campaign will run, deduplicated."""
    entries = [plane.entry(k) for k in selection.entries]
    resolved = {a.key: a for a in benchset.resolve(catalog)}

    wanted: list[str] = []
    for name, versions in selection.apps.items():
        for version in versions:
            app = resolved.get(f"{name}:{version.value}")
            if app is None:
                _, stem = catalog.resolve(name, version)
            else:
                stem = app.binary
            for entry in entries:
                if entry.kind.value == "ocrvx" and app and app.ocrvx_skip:
                    continue
                # The hint layer is already folded into the resolved stem.
                wanted.append(entry.binary(stem, hinted=False))

    targets = sorted(set(wanted))
    have = available_targets(build_dir)
    missing = [t for t in targets if t not in have] if have else []
    return BuildPlan(build_dir=build_dir, targets=targets, missing=missing)


def build(plan: BuildPlan, *, jobs: int | None = None, on_line=None) -> None:
    """Compile the whole plan in a single ninja invocation."""
    if not plan.ok:
        raise BuildError(
            "the build tree has no target for: " + ", ".join(plan.missing[:10])
            + ("…" if len(plan.missing) > 10 else "")
            + "\n(an application in the catalog is not registered in CMake)"
        )
    if shutil.which("ninja") is None:
        raise BuildError("ninja not found on PATH")
    cmd = ["ninja", "-C", str(plan.build_dir)]
    if jobs:
        cmd += ["-j", str(jobs)]
    cmd += plan.targets
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    tail: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        tail.append(line)
        del tail[:-40]
        if on_line:
            on_line(line)
    if proc.wait() != 0:
        raise BuildError("build failed:\n" + "\n".join(tail))
