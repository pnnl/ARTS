"""Where the repository and its untracked working roots live."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


class RepoNotFound(RuntimeError):
    pass


def _looks_like_repo(p: Path) -> bool:
    return (p / "CMakeLists.txt").is_file() and (p / "libs" / "src").is_dir()


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """The ARTS checkout this tool drives.

    Resolution order: an explicit ARTS_REPO, then the working directory and
    its parents, then the parents of this package.  The tool is normally
    installed editable from inside the checkout, so the last one is the
    common case; the first two let it drive a checkout elsewhere.
    """
    env = os.environ.get("ARTS_REPO")
    if env:
        p = Path(env).expanduser().resolve()
        if not _looks_like_repo(p):
            raise RepoNotFound(f"ARTS_REPO={p} is not an ARTS checkout")
        return p
    for start in (Path.cwd(), Path(__file__).resolve()):
        for cand in (start, *start.parents):
            if _looks_like_repo(cand):
                return cand
    raise RepoNotFound(
        "no ARTS checkout found from the working directory or the installed "
        "package; set ARTS_REPO"
    )


def profiles_dir() -> Path:
    return repo_root() / "experiments" / "profiles"


def benchsets_dir() -> Path:
    return repo_root() / "experiments" / "benchsets"


def countersets_dir() -> Path:
    return repo_root() / "experiments" / "countersets"


def templates_dir() -> Path:
    return repo_root() / "configs" / "templates"


def scratch_dir() -> Path:
    """Where application byproducts land; runs chdir here."""
    return repo_root() / "scratch"


def logs_root() -> Path:
    return repo_root() / "logs" / "exp"


def wall_cache_path() -> Path:
    """Observed cell wall times, used to order the scheduler's queue."""
    return logs_root() / "wall_cache.json"


def default_build_dir() -> Path:
    return repo_root() / "build_release"
