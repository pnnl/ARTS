"""Which counters a campaign compiles in, and how each is reduced.

Selecting counters is a build-time decision: the set is parsed at configure
time into `Preamble.h`, whose indices are compiled into every translation unit
that touches a counter.  Changing the set therefore forces a reconfigure and a
full rebuild, which is why a counter set is a saved artifact rather than a
per-run flag — a campaign picks one and stays on it.

The sampling interval and the output directory are the exception: the runtime
reads those from its own configuration, so they move without rebuilding.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import BaseModel, Field, model_validator

from artsrun.data import load_data


class Mode(StrEnum):
    OFF = "OFF"
    ONCE = "ONCE"
    PERIODIC = "PERIODIC"


class Level(StrEnum):
    """Where a counter's value is reported.

    THREAD keeps each worker's own value; NODE reduces across the threads of a
    rank; CLUSTER reduces again across ranks, and still writes the per-rank
    value so a distribution can be read off the same run.
    """

    THREAD = "THREAD"
    NODE = "NODE"
    CLUSTER = "CLUSTER"


class Reduce(StrEnum):
    SUM = "SUM"
    MAX = "MAX"
    MIN = "MIN"
    MASTER = "MASTER"


class Unit(StrEnum):
    COUNT = "count"
    NS = "ns"
    BYTES = "bytes"


class CounterInfo(BaseModel):
    """What the runtime says a counter is."""

    name: str
    group: str
    unit: Unit = Unit.COUNT
    help: str = ""


class CounterCatalog(BaseModel):
    counters: dict[str, CounterInfo]

    @property
    def names(self) -> list[str]:
        return list(self.counters)

    def groups(self) -> list[tuple[str, list[CounterInfo]]]:
        """In declaration order, which is the runtime's own grouping."""
        out: list[tuple[str, list[CounterInfo]]] = []
        for info in self.counters.values():
            if not out or out[-1][0] != info.group:
                out.append((info.group, []))
            out[-1][1].append(info)
        return out


@lru_cache(maxsize=1)
def load_counter_catalog() -> CounterCatalog:
    raw = load_data("counters.yaml")
    counters = {}
    for name, spec in raw["counters"].items():
        spec = dict(spec or {})
        spec["name"] = name
        counters[name] = CounterInfo.model_validate(spec)
    return CounterCatalog(counters=counters)


class CounterSetting(BaseModel):
    mode: Mode = Mode.OFF
    level: Level = Level.NODE
    reduce: Reduce = Reduce.SUM

    @property
    def enabled(self) -> bool:
        return self.mode is not Mode.OFF

    def render(self) -> str:
        """The one line the build's parser reads."""
        if not self.enabled:
            return "OFF"
        return f"{self.mode.value},{self.level.value},{self.reduce.value}"


class Counterset(BaseModel):
    name: str
    description: str | None = None

    # Runtime-side, so these move without a rebuild.
    capture_interval: int = Field(default=100, ge=1)
    folder: str | None = None

    counters: dict[str, CounterSetting] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _known(self) -> "Counterset":
        catalog = load_counter_catalog()
        unknown = [n for n in self.counters if n not in catalog.counters]
        if unknown:
            raise ValueError(
                f"not counters this runtime defines: {', '.join(sorted(unknown))}"
            )
        return self

    def setting(self, name: str) -> CounterSetting:
        return self.counters.get(name, CounterSetting())

    @property
    def enabled(self) -> list[str]:
        return sorted(n for n, s in self.counters.items() if s.enabled)

    @property
    def periodic(self) -> list[str]:
        return sorted(n for n, s in self.counters.items()
                      if s.mode is Mode.PERIODIC)

    def render_lines(self) -> list[str]:
        """Every counter the runtime knows, in declaration order.

        Absent entries are written as OFF rather than omitted: the build's
        parser leaves an unmentioned counter at its own default, and a set
        should say what it turns off as plainly as what it turns on.
        """
        catalog = load_counter_catalog()
        return [f"{name}={self.setting(name).render()}"
                for name in catalog.names]
