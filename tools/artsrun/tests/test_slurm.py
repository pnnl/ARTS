"""How a Slurm cell script launches its ranks, and what the local launcher
builds -- after the envelope change.

Under Slurm every kind launches through srun inside the cell's own sbatch
script: srun spawns arts's one process per node, and it is how a
site-PMI-integrated MPI launches the reference ranks (mpirun inside the
allocation would bury them in launcher-managed step cgroups of its own).
--cpus-per-task is explicit on the step because srun stopped inheriting it
from the allocation (Slurm >= 22.05) and arts reads its width from exactly
that value; --cpu-bind=none keeps the envelope wrapper the only affinity
actor for the references.  A declared post-verify hook wraps OUTSIDE the
srun line so it runs once per cell, not once per rank.

Locally, both references run inside the envelope wrapper; mpirun appears
wherever there are ranks to distribute and always carries -bind-to none.
"""

from __future__ import annotations

import pytest

from pathlib import Path

from artsrun.model.benchset import ResolvedApp
from artsrun.model.catalog import AppClass, Version
from artsrun.model.plane import RuntimeKind, SelectionEntry
from artsrun.model.profile import Profile
from artsrun.paths import envelope_script
from artsrun.run.command import build_command
from artsrun.run.slurm import job_script

ENV = str(envelope_script())


def _entry(kind: RuntimeKind) -> SelectionEntry:
    return SelectionEntry(
        key=f"{kind.value}_entry", label=kind.value, kind=kind, cell=kind.value,
    )


def _cell(kind: RuntimeKind, nodes: int, cfg: Path | None = None,
          post_verify: str | None = None):
    from artsrun.run.types import Cell

    app = ResolvedApp(
        name="app", version=Version.BASE, binary="app", cls=AppClass.TASK,
        marker="X", scalar_re="X", post_verify=post_verify,
    )
    return Cell(entry=_entry(kind), app=app, nodes=nodes, repeat=1,
                binary=Path("/opt/bin/app"), args=["12", "4"], timeout_s=60,
                cfg=cfg)


def _slurm_profile(**slurm) -> Profile:
    return Profile.model_validate({
        "name": "t", "launcher": "slurm", "nodes": [1, 2, 4],
        "workers": 15, "progress": 1, "ports": [25000], "slurm": slurm,
    })


def _local_profile() -> Profile:
    return Profile.model_validate({
        "name": "t", "launcher": "local", "nodes": [1, 2, 4],
        "workers": 15, "progress": 1,
    })


def _mpich(monkeypatch):
    monkeypatch.setattr("artsrun.run.command._mpi_probe", lambda: "mpich")


def _openmpi(monkeypatch):
    monkeypatch.setattr("artsrun.run.command._mpi_probe", lambda: "openmpi")


# -- slurm scripts: srun for every kind, never mpirun -----------------------

def test_an_arts_slurm_script_has_exactly_one_srun_and_no_mpirun():
    profile = _slurm_profile()
    cell = _cell(RuntimeKind.ARTS, nodes=4)
    script = job_script(cell, profile, Path("/log/cell.rc"))
    # The echo line quotes the launch, so srun appears twice in the script
    # text but is executed once.
    assert "mpirun" not in script
    assert f"srun --ntasks-per-node=1 -N 4 --cpus-per-task=16 {cell.binary}" \
        in script


@pytest.mark.parametrize("nodes", [1, 4])
def test_an_xsocr_slurm_script_runs_srun_with_the_envelope(nodes):
    profile = _slurm_profile()
    cfg = Path(f"/opt/cfg/{nodes}n.cfg")
    cell = _cell(RuntimeKind.XSOCR, nodes=nodes, cfg=cfg)
    script = job_script(cell, profile, Path("/log/cell.rc"))
    assert "mpirun" not in script
    assert (f"srun -N {nodes} -n {nodes} --ntasks-per-node=1 "
            f"--cpus-per-task=16 --cpu-bind=none "
            f"bash {ENV} 16 {nodes} fixed -- {cell.binary} "
            f"-ocr:cfg {cfg} 12 4") in script
    assert "--label" not in script


@pytest.mark.parametrize("nodes", [1, 4])
def test_an_ocrvx_slurm_script_runs_srun_with_the_envelope(nodes):
    profile = _slurm_profile()
    cell = _cell(RuntimeKind.OCRVX, nodes=nodes)
    script = job_script(cell, profile, Path("/log/cell.rc"))
    assert "mpirun" not in script
    assert (f"srun -N {nodes} -n {nodes} --ntasks-per-node=1 "
            f"--cpus-per-task=16 --cpu-bind=none "
            f"bash {ENV} 16 {nodes} fixed -- {cell.binary} 12 4") in script
    assert "export OCRVX_NUM_THREADS=15" in script
    assert "export MV2_ENABLE_AFFINITY=0" in script


def test_slurm_mpi_setting_selects_the_pmi_plugin():
    profile = _slurm_profile(mpi="pmi2")
    cell = _cell(RuntimeKind.XSOCR, nodes=2, cfg=Path("/opt/cfg/2n.cfg"))
    script = job_script(cell, profile, Path("/log/cell.rc"))
    assert "--cpu-bind=none --mpi=pmi2 bash" in script
    # arts never consults PMI; the flag is a reference-cell concern only.
    arts = job_script(_cell(RuntimeKind.ARTS, nodes=2), profile,
                      Path("/log/cell.rc"))
    assert "--mpi" not in arts


def test_post_verify_wraps_outside_srun():
    # The hook runs once on the batch node; inside srun it would run once
    # per rank, concurrently, in the same working directory.
    profile = _slurm_profile()
    cell = _cell(RuntimeKind.XSOCR, nodes=2, cfg=Path("/opt/cfg/2n.cfg"),
                 post_verify="cmp out.txt gold.txt")
    script = job_script(cell, profile, Path("/log/cell.rc"))
    line = next(l for l in script.splitlines() if l.startswith("timeout"))
    assert "sh -c" in line
    assert line.index("srun") > line.index("sh -c")
    assert "cmp out.txt gold.txt" in line


def test_job_script_echoes_its_launch_line():
    profile = _slurm_profile()
    cell = _cell(RuntimeKind.ARTS, nodes=2)
    script = job_script(cell, profile, Path("/log/cell.rc"))
    assert "printf '%s\\n' '$ srun" in script


# -- local launcher: envelope everywhere, mpirun where ranks distribute -----

def test_local_arts_command_is_unchanged(monkeypatch):
    _mpich(monkeypatch)
    profile = _local_profile()
    cell = _cell(RuntimeKind.ARTS, nodes=4)
    assert build_command(cell, profile) == ["/opt/bin/app", "12", "4"]


def test_local_xsocr_command_shapes(monkeypatch):
    _mpich(monkeypatch)
    profile = _local_profile()
    cfg = Path("/opt/cfg/1n.cfg")
    single = build_command(_cell(RuntimeKind.XSOCR, nodes=1, cfg=cfg), profile)
    assert single == [
        "mpirun", "-bind-to", "none", "-n", "1",
        "bash", ENV, "16", "1", "fixed", "--",
        "/opt/bin/app", "-ocr:cfg", str(cfg), "12", "4",
    ]

    cfg4 = Path("/opt/cfg/4n.cfg")
    multi = build_command(_cell(RuntimeKind.XSOCR, nodes=4, cfg=cfg4), profile)
    assert multi == [
        "mpirun", "-bind-to", "none", "-n", "4",
        "bash", ENV, "16", "4", "rank", "--",
        "/opt/bin/app", "-ocr:cfg", str(cfg4), "12", "4",
    ]


def test_local_ocrvx_command_shapes(monkeypatch):
    _mpich(monkeypatch)
    profile = _local_profile()
    single = build_command(_cell(RuntimeKind.OCRVX, nodes=1), profile)
    # A single rank needs no launcher: the envelope wrapper does the taskset
    # itself, after verifying the block.
    assert single == [
        "bash", ENV, "16", "1", "fixed", "--", "/opt/bin/app", "12", "4",
    ]

    multi = build_command(_cell(RuntimeKind.OCRVX, nodes=4), profile)
    assert multi == [
        "mpirun", "-bind-to", "none", "-n", "4",
        "bash", ENV, "16", "4", "rank", "--", "/opt/bin/app", "12", "4",
    ]


def test_local_openmpi_gets_oversubscribe_and_bind_to_none(monkeypatch):
    _openmpi(monkeypatch)
    profile = _local_profile()
    multi = build_command(_cell(RuntimeKind.OCRVX, nodes=4), profile)
    assert multi[:5] == ["mpirun", "--oversubscribe", "--bind-to", "none", "-n"]
    # The one-per-node flag belongs to remote launchers: locally ranks are
    # meant to colocate in their taskset blocks.
    assert "--map-by" not in multi
    assert "ppr:1:node" not in multi
