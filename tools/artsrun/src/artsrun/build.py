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


# How the configure step encodes a counter's settings into the header it
# generates.  Reading them back is what lets a tree be checked against a set
# rather than against the file it happened to be configured from.
_MODES = {0: "OFF", 1: "ONCE", 2: "PERIODIC"}
_LEVELS = {0: "THREAD", 1: "NODE", 2: "CLUSTER"}
_REDUCERS = {0: "SUM", 1: "MAX", 2: "MIN", 3: "MASTER"}

_PREAMBLE = "libs/include/internal/arts/counter/Preamble.h"


def compiled_counters(build_dir: Path) -> dict[str, tuple[str, str, str]] | None:
    """What this tree actually compiled, counter by counter.

    The generated header is the only authority: the selection is turned into
    macros at configure time, so a build carries its counters no matter which
    file they arrived in or whether that file still exists.
    """
    header = build_dir / _PREAMBLE
    if not header.is_file():
        return None
    text = header.read_text(errors="ignore")

    def read(prefix: str) -> dict[str, int]:
        return {
            m.group(1): int(m.group(2))
            for m in re.finditer(rf"^#define {prefix}([A-Z0-9_]+) (\d+)$", text, re.M)
        }

    enabled = read("ENABLE_")
    modes, levels = read("COUNTER_MODE_"), read("COUNTER_LEVEL_")
    reducers = read("COUNTER_REDUCE_") or read("REDUCE_METHOD_")
    if not enabled:
        return None
    return {
        name: (
            _MODES.get(modes.get(name, 0), "OFF") if on else "OFF",
            _LEVELS.get(levels.get(name, 1), "NODE"),
            _REDUCERS.get(reducers.get(name, 0), "SUM"),
        )
        for name, on in enabled.items()
    }


def counter_mismatch(build_dir: Path, counterset) -> list[str]:
    """Counters whose compiled settings differ from what a set asks for.

    What the tree compiled is read back out of its generated header, not out
    of the file it was configured from.  A counter set is values that get
    rendered afresh for every campaign, so the file they land in is
    incidental: a tree configured from a hand-written file may hold exactly
    the wanted selection, and two renderings of one set differ in path only.
    """
    have = compiled_counters(build_dir)
    if have is None or counterset is None:
        return []
    default = ("OFF", "NODE", "SUM")
    want = {
        name: (s.mode.value, s.level.value,
               (s.reduce.value if s.reduce else "SUM"))
        for name, s in counterset.counters.items()
    }
    return sorted(
        name for name in set(have) | set(want)
        if have.get(name, default) != want.get(name, default)
    )


def configure_counters(build_dir: Path, wanted: Path, *, on_line=None) -> None:
    """Point an existing tree at a counter configuration and reconfigure it.

    Counter selection is compiled in, so making a tree match is a build step
    rather than something to hand back to the caller: asking to build with a
    counter set IS asking for the tree that carries it.  Only the counter
    option is passed — everything else stays in the cache, so this cannot
    quietly change the configuration in any other respect.  The rebuild that
    follows is a full one, and says so.
    """
    say = on_line or (lambda _msg: None)
    if shutil.which("cmake") is None:
        raise BuildError("cmake not found on PATH")
    say(f"counters changed — reconfiguring {build_dir} (this rebuilds everything)")
    proc = subprocess.run(
        ["cmake", "-B", str(build_dir), f"-DARTS_COUNTER_CONFIG={wanted}"],
        capture_output=True, text=True,
    )
    for line in proc.stdout.splitlines():
        if "Counter configuration:" in line or "error" in line.lower():
            say(f"  {line.strip()}")
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout).splitlines()[-15:])
        raise BuildError(
            f"could not reconfigure {build_dir} for the counter set:\n{tail}"
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
