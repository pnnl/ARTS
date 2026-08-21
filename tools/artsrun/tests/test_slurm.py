"""Which cells get an srun wrapper in a Slurm cell script, and which don't --
and, for the ones that don't, that mpirun is told to place one rank per
node rather than packing them onto the first one.

ARTS has no rank launcher of its own -- srun is what starts the one process
per node the job needs.  XSOCR and OCRVX are MPI programs and already carry
their own launcher (mpirun) in build_command's output, reading the Slurm
allocation from its environment natively; wrapping that in srun a second
time would start a copy of mpirun on every node, each spawning its own full
set of ranks.  But mpirun's own default placement is not "one rank per
node" either -- both major implementations fill the first node to its
core/slot count before moving to the next -- so the generated command must
say so explicitly: `-ppn 1` under MPICH's Hydra, `--map-by ppr:1:node`
under Open MPI.  See docs cited in the ROUND note in
research/appdocs/ledger.md for the primary sources.
"""

from __future__ import annotations

import pytest

from pathlib import Path

from artsrun.model.benchset import ResolvedApp
from artsrun.model.catalog import AppClass, Version
from artsrun.model.plane import RuntimeKind, SelectionEntry
from artsrun.model.profile import Profile
from artsrun.run.command import build_command, mpirun
from artsrun.run.slurm import job_script
from artsrun.run.types import Cell


def _entry(kind: RuntimeKind) -> SelectionEntry:
    return SelectionEntry(
        key=f"{kind.value}_entry", label=kind.value, kind=kind, cell=kind.value,
    )


def _cell(kind: RuntimeKind, nodes: int, cfg: Path | None = None) -> Cell:
    app = ResolvedApp(
        name="app", version=Version.BASE, binary="app", cls=AppClass.TASK,
        marker="X", scalar_re="X",
    )
    return Cell(entry=_entry(kind), app=app, nodes=nodes, repeat=1,
                binary=Path("/opt/bin/app"), args=["12", "4"], timeout_s=60,
                cfg=cfg)


def _slurm_profile() -> Profile:
    return Profile.model_validate({
        "name": "t", "launcher": "slurm", "nodes": [1, 2, 4],
        "workers": 15, "progress": 1, "ports": [25000], "slurm": {},
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


# -- the flavor-detection unit itself: exact flag shape, per flavor ---------

@pytest.mark.parametrize("flavor,flag,expected", [
    ("mpich", ["-ppn", "1"], ["mpirun", "-ppn", "1", "-n", "4"]),
    ("openmpi", ["--map-by", "ppr:1:node"],
     ["mpirun", "--oversubscribe", "--map-by", "ppr:1:node", "-n", "4"]),
])
def test_mpirun_one_per_node_follows_the_detected_flavor(
        monkeypatch, flavor, flag, expected):
    monkeypatch.setattr("artsrun.run.command._mpi_probe", lambda: flavor)
    assert mpirun(4, one_per_node=True) == expected
    assert mpirun(4, one_per_node=False)[-2:] == ["-n", "4"]
    assert not any(f in mpirun(4, one_per_node=False) for f in flag)


# -- slurm scripts: one launcher, never two, and one rank per node ----------

def test_an_arts_slurm_script_has_exactly_one_srun_and_no_mpirun():
    profile = _slurm_profile()
    cell = _cell(RuntimeKind.ARTS, nodes=4)
    script = job_script(cell, profile, Path("/log/cell.rc"))
    assert script.count("srun") == 1
    assert "mpirun" not in script
    assert f"srun --ntasks-per-node=1 -N {cell.nodes}" in script


@pytest.mark.parametrize("flavor,placement", [
    ("mpich", "-ppn 1 -n 4"),
    ("openmpi", "--oversubscribe --map-by ppr:1:node -n 4"),
])
def test_an_xsocr_slurm_script_runs_mpirun_once_pinned_one_per_node(
        monkeypatch, flavor, placement):
    monkeypatch.setattr("artsrun.run.command._mpi_probe", lambda: flavor)
    profile = _slurm_profile()
    cfg = Path("/opt/cfg/4n.cfg")
    cell = _cell(RuntimeKind.XSOCR, nodes=4, cfg=cfg)
    script = job_script(cell, profile, Path("/log/cell.rc"))
    assert "srun" not in script
    assert script.count("mpirun") == 1
    assert f"mpirun {placement} {cell.binary}" in script
    assert f"-ocr:cfg {cfg}" in script


@pytest.mark.parametrize("flavor,placement", [
    ("mpich", "-ppn 1 -n 4"),
    ("openmpi", "--oversubscribe --map-by ppr:1:node -n 4"),
])
def test_an_ocrvx_multinode_slurm_script_runs_mpirun_once_pinned_one_per_node(
        monkeypatch, flavor, placement):
    monkeypatch.setattr("artsrun.run.command._mpi_probe", lambda: flavor)
    profile = _slurm_profile()
    cell = _cell(RuntimeKind.OCRVX, nodes=4)
    script = job_script(cell, profile, Path("/log/cell.rc"))
    assert "srun" not in script
    assert script.count("mpirun") == 1
    assert f"mpirun {placement} {cell.binary}" in script
    assert "export OCRVX_NUM_THREADS=15" in script


# -- local launcher: byte-identical to before this change -------------------
# Pinned to the mpich probe result so these lock the expected shape
# regardless of which MPI happens to be on the host running the suite.

def test_local_arts_command_is_unchanged(monkeypatch):
    _mpich(monkeypatch)
    profile = _local_profile()
    cell = _cell(RuntimeKind.ARTS, nodes=4)
    assert build_command(cell, profile) == ["/opt/bin/app", "12", "4"]


def test_local_xsocr_command_is_unchanged(monkeypatch):
    _mpich(monkeypatch)
    profile = _local_profile()
    cfg = Path("/opt/cfg/1n.cfg")
    single = build_command(_cell(RuntimeKind.XSOCR, nodes=1, cfg=cfg), profile)
    assert single == [
        "mpirun", "-n", "1", "/opt/bin/app", "-ocr:cfg", str(cfg), "12", "4",
    ]

    cfg4 = Path("/opt/cfg/4n.cfg")
    multi = build_command(_cell(RuntimeKind.XSOCR, nodes=4, cfg=cfg4), profile)
    assert multi == [
        "mpirun", "-n", "4", "bash", "-c",
        'r=${PMI_RANK:-${OMPI_COMM_WORLD_RANK:-0}}; s=$((r*16)); '
        'e=$((s+16-1)); exec taskset -c $s-$e "$@"',
        "_", "/opt/bin/app", "-ocr:cfg", str(cfg4), "12", "4",
    ]


def test_local_ocrvx_command_is_unchanged(monkeypatch):
    _mpich(monkeypatch)
    profile = _local_profile()
    single = build_command(_cell(RuntimeKind.OCRVX, nodes=1), profile)
    assert single == ["taskset", "-c", "0-15", "/opt/bin/app", "12", "4"]

    multi = build_command(_cell(RuntimeKind.OCRVX, nodes=4), profile)
    assert multi == [
        "mpirun", "-n", "4", "bash", "-c",
        'r=${PMI_RANK:-${OMPI_COMM_WORLD_RANK:-0}}; s=$((r*16)); '
        'e=$((s+16-1)); exec taskset -c $s-$e "$@"',
        "_", "/opt/bin/app", "12", "4",
    ]


def test_local_command_never_gets_the_one_per_node_flag_even_under_openmpi(
        monkeypatch):
    # The flavor probe is orthogonal to the launcher: only the launcher
    # decides whether ranks are meant to be one-per-node in the first place.
    _openmpi(monkeypatch)
    profile = _local_profile()
    multi = build_command(_cell(RuntimeKind.OCRVX, nodes=4), profile)
    assert "--map-by" not in multi
    assert "ppr:1:node" not in multi
