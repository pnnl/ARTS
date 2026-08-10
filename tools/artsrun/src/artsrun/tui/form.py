"""Field descriptions shared by the profile editor and the command line.

One list of what a profile has, what type it is, and what it means; the screen
builds widgets from it and the CLI builds `key=value` parsing from it, so the
two cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FieldSpec:
    key: str                       # dotted for nested values ("slurm.budget")
    label: str
    kind: str                      # text|int|float|bool|choice|int_list|str_list
    help: str = ""
    choices: tuple[str, ...] = ()
    optional: bool = False
    section: str = "run"
    example: str = ""              # shown in the empty box
    none_choice: str = ""          # the choice that means "leave it unset"

    @property
    def path(self) -> list[str]:
        return self.key.split(".")


PROFILE_FIELDS: list[FieldSpec] = [
    FieldSpec("launcher", "launcher", "choice",
              "how ranks are started", choices=("local", "ssh", "slurm")),
    FieldSpec("workers", "workers / node", "int",
              "compute threads per rank", example="15"),
    FieldSpec("progress", "progress / node", "int",
              "network-progress threads per rank", example="1"),
    FieldSpec("repeats", "repeats", "int", "runs per cell", example="1"),
    FieldSpec("cell_timeout_s", "cell timeout (s)", "int",
              "wall budget per cell unless the application overrides it",
              example="300"),

    FieldSpec("provider", "provider", "choice",
              "data plane: tcp anywhere, verbs on InfiniBand. auto lets the "
              "runtime pick, which prefers tcp",
              choices=("auto", "tcp", "verbs"), none_choice="auto",
              optional=True, section="transport"),
    FieldSpec("net_interface", "interface", "text",
              "IP providers only (tcp): binds the source address to this "
              "interface, e.g. ib0 to run tcp over IPoIB when the default "
              "route points at the management network. Naming one with no "
              "usable address is fatal rather than a silent fallback. Empty "
              "leaves the source unconstrained; verbs ignores it entirely",
              optional=True, section="transport"),
    FieldSpec("fabric_domain", "fabric domain", "text",
              "verbs only: pins one HCA on a multi-rail host, named after the "
              "device rather than an interface, e.g. mlx5_0. Empty takes the "
              "provider's first domain",
              optional=True, section="transport"),
    FieldSpec("route_table_size", "route table (2^N)", "int",
              "route table size as a power-of-two exponent: 16 = 65536 entries",
              example="16", section="transport"),
    FieldSpec("regpool_slab_mb", "regpool slab (MB)", "int",
              "registered-memory pool floor the fabric draws payload buffers "
              "from (empty = the runtime's own default)",
              example="64", optional=True, section="transport"),
    FieldSpec("port_count", "connections / node", "int",
              "parallel connections each node opens. A local run has the "
              "runtime claim node_count x this many ports itself; ssh and "
              "slurm must be given the ports below, exactly this many of them",
              example="1", section="transport"),
    FieldSpec("ports", "ports", "int_list",
              "one base port per connection — must name exactly "
              "'connections / node' entries. Locked for a local run, whose "
              "ranks share a machine and claim their own block",
              example="25000", optional=True, section="transport"),

    FieldSpec("ssh.budget", "node budget", "int",
              "nodes available over ssh; must be >= the widest node count, "
              "and hosts must name exactly this many",
              example="4", optional=True, section="ssh"),
    FieldSpec("ssh.hosts", "hosts", "str_list",
              "one hostname or address per node, in rank order — exactly "
              "'node budget' of them",
              example="n01,n02,n03,n04", optional=True, section="ssh"),

    FieldSpec("pin", "pin threads", "bool",
              "bind each thread to one core instead of letting the OS move "
              "it; off makes timings noisier but tolerates a shared machine",
              section="flags"),
    FieldSpec("core_dump", "core dumps", "bool",
              "let a crashing rank write a core file for post-mortem "
              "debugging, at the cost of disk on a bad run", section="flags"),

    FieldSpec("slurm.budget", "node budget", "int",
              "nodes kept in flight at once; must be >= the widest node count "
              "or that cell can never be submitted. 1 = strictly serial",
              example="32", optional=True, section="slurm"),
    FieldSpec("slurm.partition", "partition", "text",
              "which set of nodes to submit to — clusters group their nodes "
              "into partitions with their own hardware and time limits "
              "(empty = the cluster's default)",
              example="compute", optional=True, section="slurm"),
    FieldSpec("slurm.account", "account", "text",
              "the project the node-hours are charged to, for users who "
              "belong to more than one (empty = the default account)",
              example="proj123", optional=True, section="slurm"),
    FieldSpec("slurm.qos", "qos", "text",
              "quality-of-service policy: the priority and resource limits "
              "the jobs run under (empty = the default QoS)",
              example="normal", optional=True, section="slurm"),
    FieldSpec("slurm.poll_interval_s", "poll interval (s)", "float",
              "how often a submitted job's state is checked; longer is "
              "gentler on a busy controller",
              example="10", optional=True, section="slurm"),
]

SECTIONS = [
    ("run", "Run shape"),
    ("transport", "Transport"),
    ("flags", "Flags"),
    ("ssh", "SSH"),
    ("slurm", "Slurm"),
]

# Sections that only apply to one launcher, and which one.
LAUNCHER_SECTIONS = {"ssh": "ssh", "slurm": "slurm"}

# Node counts are edited as toggles on the screen rather than as a text box,
# but the command line still needs a name for them.
NODES_SPEC = FieldSpec("nodes", "node counts", "int_list",
                       "node counts this campaign runs, e.g. 1,2,4,8")

BY_KEY = {f.key: f for f in [*PROFILE_FIELDS, NODES_SPEC]}


def _get(data: dict, path: list[str]) -> Any:
    cur: Any = data
    for part in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
        if cur is None:
            return None
    return cur


def _set(data: dict, path: list[str], value: Any) -> None:
    cur = data
    for part in path[:-1]:
        cur = cur.setdefault(part, {})
    cur[path[-1]] = value


def to_text(spec: FieldSpec, value: Any) -> str:
    """Render one stored value for an input box."""
    if value is None:
        return spec.none_choice
    if spec.kind in ("int_list", "str_list"):
        return ",".join(str(v) for v in value)
    return str(value)


def parse(spec: FieldSpec, text: str) -> Any:
    """Read one input box back, raising with the field's own name."""
    text = text.strip()
    if spec.none_choice and text == spec.none_choice:
        return None
    if not text:
        if spec.optional:
            return None
        raise ValueError(f"{spec.label} is required")
    try:
        if spec.kind == "int":
            return int(text)
        if spec.kind == "float":
            return float(text)
        if spec.kind == "int_list":
            return [int(p) for p in text.replace(" ", ",").split(",") if p]
        if spec.kind == "str_list":
            return [p for p in text.replace(" ", ",").split(",") if p]
    except ValueError as exc:
        raise ValueError(f"{spec.label}: {exc}") from None
    return text


def profile_to_values(profile) -> dict[str, Any]:
    """Stored profile -> one value per field key."""
    data = profile.model_dump(mode="json")
    out: dict[str, Any] = {"nodes": to_text(NODES_SPEC, data.get("nodes"))}
    for spec in PROFILE_FIELDS:
        value = _get(data, spec.path)
        out[spec.key] = value if spec.kind == "bool" else to_text(spec, value)
    return out


def values_to_profile_data(name: str, values: dict[str, Any]) -> dict:
    """Field values -> the dict a Profile is validated from.

    Slurm settings are only carried when that launcher is selected, so
    switching a profile to local does not leave a stale budget behind.
    """
    data: dict[str, Any] = {"name": name}
    launcher = values.get("launcher") or "local"
    if "nodes" in values:
        raw = values["nodes"]
        data["nodes"] = raw if isinstance(raw, list) else parse(NODES_SPEC, str(raw))
    for spec in PROFILE_FIELDS:
        if LAUNCHER_SECTIONS.get(spec.section, launcher) != launcher:
            continue
        # A local run's ranks share a machine and claim their own port block.
        if spec.key == "ports" and launcher == "local":
            continue
        raw = values.get(spec.key)
        value = raw if spec.kind == "bool" else parse(spec, str(raw or ""))
        if value is None:
            continue
        _set(data, spec.path, value)
    if launcher != "slurm":
        data["slurm"] = None
    return data


def blank_values() -> dict[str, Any]:
    """A new profile's starting point: a single-node local run."""
    return {
        "launcher": "local",
        "workers": "4", "progress": "1",
        "repeats": "1", "cell_timeout_s": "300",
        "provider": "", "net_interface": "", "route_table_size": "16",
        "regpool_slab_mb": "", "ports": "", "hosts": "",
        "pin": True, "core_dump": False,
        "slurm.budget": "", "slurm.partition": "",
        "slurm.account": "", "slurm.qos": "", "slurm.poll_interval_s": "",
    }


def first_problem(exc: Exception) -> str:
    """The one thing to fix, rather than pydantic's whole report."""
    from pydantic import ValidationError

    if isinstance(exc, ValidationError):
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"]) or "profile"
        return f"{where}: {first['msg'].removeprefix('Value error, ')}"
    return str(exc)
