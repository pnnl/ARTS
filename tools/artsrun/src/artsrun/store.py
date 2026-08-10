"""Discovery and persistence of the two saved surfaces.

Profiles and benchsets are values, not code: they live under experiments/ and
are not tracked, so a campaign can be reshaped without touching the tool.
"""

from __future__ import annotations

import io
from pathlib import Path

from ruamel.yaml import YAML

from artsrun.model.benchset import Benchset
from artsrun.model.counters import Counterset
from artsrun.model.profile import Profile
from artsrun.paths import benchsets_dir, countersets_dir, profiles_dir

_yaml = YAML()
_yaml.preserve_quotes = True
_yaml.indent(mapping=2, sequence=4, offset=2)


class NotFound(FileNotFoundError):
    pass


def _list(directory: Path) -> list[str]:
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob("*.yaml"))


def _read(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return _yaml.load(fh)


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    _yaml.dump(data, buf)
    path.write_text(buf.getvalue(), encoding="utf-8")


# --- profiles -------------------------------------------------------------
def list_profiles() -> list[str]:
    return _list(profiles_dir())


def profile_path(name: str) -> Path:
    return profiles_dir() / f"{name}.yaml"


def load_profile(name: str) -> Profile:
    path = profile_path(name)
    if not path.is_file():
        known = ", ".join(list_profiles()) or "none"
        raise NotFound(f"no profile '{name}' in {profiles_dir()} (have: {known})")
    data = _read(path)
    data.setdefault("name", name)
    return Profile.model_validate(data)


def save_profile(profile: Profile) -> Path:
    path = profile_path(profile.name)
    _write(path, profile.model_dump(mode="json", exclude_none=True))
    return path


# --- benchsets ------------------------------------------------------------
def list_benchsets() -> list[str]:
    return _list(benchsets_dir())


def benchset_path(name: str) -> Path:
    return benchsets_dir() / f"{name}.yaml"


def load_benchset(name: str) -> Benchset:
    path = benchset_path(name)
    if not path.is_file():
        known = ", ".join(list_benchsets()) or "none"
        raise NotFound(f"no benchset '{name}' in {benchsets_dir()} (have: {known})")
    data = _read(path)
    data.setdefault("name", name)
    return Benchset.model_validate(data)


def save_benchset(benchset: Benchset) -> Path:
    path = benchset_path(benchset.name)
    _write(path, benchset.model_dump(mode="json", exclude_none=True))
    return path


def default_benchset() -> Benchset:
    """The catalog's own defaults, for a run that names no benchset."""
    return Benchset(name="catalog-default", description="catalog defaults")


# --- counter sets ---------------------------------------------------------
def list_countersets() -> list[str]:
    return _list(countersets_dir())


def counterset_path(name: str) -> Path:
    return countersets_dir() / f"{name}.yaml"


def load_counterset(name: str) -> Counterset:
    path = counterset_path(name)
    if not path.is_file():
        known = ", ".join(list_countersets()) or "none"
        raise NotFound(f"no counter set '{name}' in {countersets_dir()} "
                       f"(have: {known})")
    data = _read(path)
    data.setdefault("name", name)
    return Counterset.model_validate(data)


def save_counterset(counterset: Counterset) -> Path:
    path = counterset_path(counterset.name)
    _write(path, counterset.model_dump(mode="json", exclude_none=True))
    return path


def default_counterset() -> Counterset:
    """No counters at all — what a run costs nothing to collect."""
    return Counterset(name="none", description="no counters compiled in")
