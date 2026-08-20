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
        # ARTS states the stack in bytes, the reference in MiB; the profile
        # states it once so the two cannot drift apart.
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
    and so is the worker stack — this key is in MiB where the ARTS one is in
    bytes, and both come from the profile's single value. The binding line is
    emitted single-node only, where absolute core numbers stay inside the
    process's own block.
    """
    return _env().get_template("ocr.cfg.j2").render(
        last_thread=profile.threads_per_node - 1,
        stack_size=profile.stack_size_mb,
        binding=(nodes == 1),
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


def write_counter_config(counterset, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"counters_{counterset.name}.cfg"
    path.write_text(render_counters(counterset))
    return path


def write_configs(profile: Profile, nodes: int, out_dir: Path, *,
                  counter_folder: str | None = None,
                  capture_interval: int | None = None) -> dict[str, Path]:
    """Write every configuration this node count needs; return the paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    arts = out_dir / f"arts_{nodes}n.cfg"
    arts.write_text(render_arts(profile, nodes, counter_folder=counter_folder,
                                capture_interval=capture_interval))
    ocr = out_dir / f"ocr_{nodes}n.cfg"
    ocr.write_text(render_ocr(profile, nodes))
    return {"arts": arts, "ocr": ocr}


def write_arts_cfg(profile: Profile, nodes: int, path: Path, *,
                   counter_folder: str | None = None,
                   capture_interval: int | None = None) -> Path:
    """Write one ARTS configuration to an exact path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_arts(profile, nodes, counter_folder=counter_folder,
                                capture_interval=capture_interval))
    return path


def config_for(kind: RuntimeKind, configs: dict[str, Path]) -> Path | None:
    """Which rendered configuration a runtime reads (ocr-vx is env-driven)."""
    if kind is RuntimeKind.ARTS:
        return configs["arts"]
    if kind is RuntimeKind.XSOCR:
        return configs["ocr"]
    return None
