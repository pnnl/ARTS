"""Render runtime configurations from the committed templates.

Configurations are written into the run's own directory and named through
ARTS_CONFIG / OCR_CONFIG.  Nothing is ever copied into a build tree or the
repository root: a stray config there answers for a run that forgot to name
one.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from artsrun.model.plane import RuntimeKind
from artsrun.model.profile import Launcher, Profile
from artsrun.paths import templates_dir


def _env(root: Path | None = None) -> Environment:
    return Environment(
        loader=FileSystemLoader(root or templates_dir()),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        trim_blocks=False,
        lstrip_blocks=False,
    )


def render_arts(profile: Profile, nodes: int, *, counter_folder: str | None = None,
                capture_interval: int | None = None) -> str:
    hosts = profile.hosts[:nodes] if profile.launcher is Launcher.SSH else []
    return _env().get_template("arts.cfg.j2").render(
        workers=profile.workers,
        progress=profile.progress,
        launcher=profile.launcher.value,
        nodes=nodes,
        pin=profile.pin,
        route_table_size=profile.route_table_size,
        core_dump=profile.core_dump,
        provider=profile.provider,
        net_interface=profile.net_interface,
        fabric_domain=profile.fabric_domain,
        regpool_slab_mb=profile.regpool_slab_mb,
        # Both runtimes take the stack in BYTES and both read zero as "leave
        # the platform default"; the profile states the size once, in MiB, so
        # the two cannot drift apart.
        stack_size_bytes=profile.stack_size_mb * 1024 * 1024,
        ports=profile.ports,
        port_count=profile.port_count,
        hosts=hosts,
        counter_folder=counter_folder,
        counter_capture_interval=capture_interval,
    )


def render_ocr(profile: Profile, nodes: int) -> str:
    """Reference-runtime configuration.

    The per-node width is the same thread budget the runtime under test gets,
    and so is the worker stack. This key is in BYTES, the same as the ARTS one:
    it reaches pthread_attr_setstacksize unconverted, so a value below
    PTHREAD_STACK_MIN makes thread creation fail outright rather than clamp.

    The binding line is emitted whenever the profile pins: on every launcher a
    rank's block starts at absolute CPU 0 — remote ranks own their host, and
    colocated local ranks are shifted per rank by the `numa` BLOCK policy
    (cpu = offset + width * (mpi_rank % ranks)), which the runtime computes
    itself from its own rank at bring-up.  The key renders atomically with its
    complete BLOCK:<ranks>:<width> value or not at all: the runtime SEGVs on a
    malformed or empty value, and its binding never accepts anything but a
    single lo-hi range plus this policy.
    """
    numa_block = None
    if profile.pin and profile.launcher is Launcher.LOCAL and nodes > 1:
        numa_block = f"BLOCK:{nodes}:{profile.threads_per_node}"
    return _env().get_template("ocr.cfg.j2").render(
        last_thread=profile.threads_per_node - 1,
        stack_size=profile.stack_size_mb * 1024 * 1024,
        binding=profile.pin,
        numa_block=numa_block,
    )


def render_counters(counterset) -> str:
    """The counter file the build parses.

    Only the selection is here; the sampling interval and the output folder
    are runtime keys and go into the runtime configuration instead.
    """
    return _env().get_template("counters.cfg.j2").render(
        description=counterset.description,
        lines=counterset.render_lines(),
    )


def _write_once(path: Path, text: str) -> Path:
    """Write a rendered configuration, refusing to change one already written.

    Only a resumed campaign revisits an existing run directory, and there an
    already-queued job reads the configuration PATH when it starts — silently
    overwriting the file would swap rules under a job submitted before the
    rules changed.  Fresh runs and replays always render into a new stamp
    directory, so they can never trip this.
    """
    if path.exists():
        if path.read_text() != text:
            raise RuntimeError(
                f"rendered configuration changed since this campaign started: "
                f"{path}; start a new campaign instead of resuming this one"
            )
        return path
    path.write_text(text)
    return path


def write_counter_config(counterset, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"counters_{counterset.name}.cfg"
    return _write_once(path, render_counters(counterset))


def write_configs(profile: Profile, nodes: int, out_dir: Path, *,
                  counter_folder: str | None = None,
                  capture_interval: int | None = None) -> dict[str, Path]:
    """Write every configuration this node count needs; return the paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    arts = out_dir / f"arts_{nodes}n.cfg"
    _write_once(arts, render_arts(profile, nodes, counter_folder=counter_folder,
                                  capture_interval=capture_interval))
    ocr = out_dir / f"ocr_{nodes}n.cfg"
    _write_once(ocr, render_ocr(profile, nodes))
    return {"arts": arts, "ocr": ocr}


def write_arts_cfg(profile: Profile, nodes: int, path: Path, *,
                   counter_folder: str | None = None,
                   capture_interval: int | None = None) -> Path:
    """Write one ARTS configuration to an exact path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return _write_once(path, render_arts(profile, nodes,
                                         counter_folder=counter_folder,
                                         capture_interval=capture_interval))


def config_for(kind: RuntimeKind, configs: dict[str, Path]) -> Path | None:
    """Which rendered configuration a runtime reads (ocr-vx is env-driven)."""
    if kind is RuntimeKind.ARTS:
        return configs["arts"]
    if kind is RuntimeKind.XSOCR:
        return configs["ocr"]
    return None
