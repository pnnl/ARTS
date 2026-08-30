"""The ssh launch shapes, the mpirun probe's fatality, the reference
selection guards, and the checker's envelope/world-size demotions."""

from __future__ import annotations

import pytest

from pathlib import Path

from artsrun.model.benchset import ResolvedApp
from artsrun.model.catalog import AppClass, Version
from artsrun.model.plane import RuntimeKind, SelectionEntry, load_plane
from artsrun.model.profile import Profile
from artsrun.paths import envelope_script
from artsrun.run.command import MpiProbeError, build_command, build_env

ENV = str(envelope_script())


def _entry(kind: RuntimeKind) -> SelectionEntry:
    return SelectionEntry(
        key=f"{kind.value}_entry", label=kind.value, kind=kind, cell=kind.value,
    )


def _cell(kind: RuntimeKind, nodes: int, cfg: Path | None = None):
    from artsrun.run.types import Cell

    app = ResolvedApp(
        name="app", version=Version.BASE, binary="app", cls=AppClass.TASK,
        marker="DONE", scalar_re=r"sum (\d+)",
    )
    return Cell(entry=_entry(kind), app=app, nodes=nodes, repeat=1,
                binary=Path("/opt/bin/app"), args=["12", "4"], timeout_s=60,
                cfg=cfg)


def _ssh_profile() -> Profile:
    return Profile.model_validate({
        "name": "t", "launcher": "ssh", "nodes": [1, 2, 4],
        "workers": 15, "progress": 1, "ports": [25000],
        "ssh": {"budget": 4, "hosts": ["n01", "n02", "n03", "n04"]},
    })


# -- ssh: mpirun with a forced ssh bootstrap and an explicit host list ------

def test_ssh_xsocr_mpich_forces_the_ssh_bootstrap(monkeypatch):
    monkeypatch.setattr("artsrun.run.command._mpi_probe", lambda: "mpich")
    cfg = Path("/opt/cfg/4n.cfg")
    argv = build_command(_cell(RuntimeKind.XSOCR, nodes=4, cfg=cfg),
                         _ssh_profile())
    assert argv == [
        "mpirun", "-launcher", "ssh", "-hosts", "n01,n02,n03,n04",
        "-bind-to", "none", "-ppn", "1", "-n", "4",
        "bash", ENV, "16", "4", "fixed", "--",
        "/opt/bin/app", "-ocr:cfg", str(cfg), "12", "4",
    ]


def test_ssh_ocrvx_openmpi_forces_the_ssh_bootstrap(monkeypatch):
    # Both flavors silently switch to a slurm bootstrap when SLURM_* leaks
    # into the environment, ignoring the host list — the bootstrap is forced.
    monkeypatch.setattr("artsrun.run.command._mpi_probe", lambda: "openmpi")
    argv = build_command(_cell(RuntimeKind.OCRVX, nodes=2), _ssh_profile())
    assert argv == [
        "mpirun", "--mca", "plm", "rsh", "--host", "n01,n02",
        "--bind-to", "none", "--map-by", "ppr:1:node", "-n", "2",
        "bash", ENV, "16", "2", "fixed", "--", "/opt/bin/app", "12", "4",
    ]
    # No --oversubscribe on a remote launcher: it would only license the
    # rank packing the envelope's local-rank guard forbids.
    assert "--oversubscribe" not in argv


def test_ssh_single_ocrvx_rank_needs_no_launcher(monkeypatch):
    monkeypatch.setattr("artsrun.run.command._mpi_probe", lambda: "mpich")
    argv = build_command(_cell(RuntimeKind.OCRVX, nodes=1), _ssh_profile())
    assert argv == ["bash", ENV, "16", "1", "fixed", "--",
                    "/opt/bin/app", "12", "4"]


def test_a_failed_mpirun_probe_is_fatal_not_a_default(monkeypatch):
    monkeypatch.setattr("artsrun.run.command._mpi_probe", lambda: None)
    with pytest.raises(MpiProbeError):
        build_command(_cell(RuntimeKind.XSOCR, nodes=2,
                            cfg=Path("/c.cfg")), _ssh_profile())
    # arts never launches through mpirun, so it does not consult the probe.
    assert build_command(_cell(RuntimeKind.ARTS, nodes=2), _ssh_profile()) == [
        "/opt/bin/app", "12", "4",
    ]


def test_a_missing_envelope_script_is_fatal(monkeypatch):
    monkeypatch.setattr("artsrun.run.command._mpi_probe", lambda: "mpich")
    monkeypatch.setattr("artsrun.run.command.envelope_script",
                        lambda: Path("/nonexistent/envelope.sh"))
    with pytest.raises(FileNotFoundError):
        build_command(_cell(RuntimeKind.OCRVX, nodes=1), _ssh_profile())


# -- env: the MPI library's own affinity is neutralized for references ------

def test_reference_env_disables_mvapich_affinity():
    profile = _ssh_profile()
    ref = build_env(_cell(RuntimeKind.OCRVX, nodes=2), profile)
    assert ref["MV2_ENABLE_AFFINITY"] == "0"
    assert ref["OCRVX_NUM_THREADS"] == "15"
    arts = build_env(_cell(RuntimeKind.ARTS, nodes=2), profile)
    assert "MV2_ENABLE_AFFINITY" not in arts


# -- selection guards: what a reference cell needs from the profile ---------

def _selection(**kw):
    from artsrun.model.selection import Selection

    base = dict(profile="t", benchset="b", entries=["xsocr"],
                apps={"nqueens": [Version.BASE]}, node_counts=[1])
    base.update(kw)
    return Selection.model_validate(base)


def _profile(**kw) -> Profile:
    base = dict(name="t", launcher="local", nodes=[1, 2, 4],
                workers=15, progress=1)
    base.update(kw)
    return Profile.model_validate(base)


@pytest.fixture(scope="module")
def plane():
    return load_plane()


@pytest.fixture(scope="module")
def catalog():
    from artsrun.model.catalog import load_catalog

    return load_catalog()


def test_reference_selection_requires_pin(plane, catalog):
    with pytest.raises(ValueError, match="pin"):
        _selection().validate_against(plane, catalog, _profile(pin=False))


def test_arts_only_selection_tolerates_any_geometry(plane, catalog):
    # The guards answer for what the REFERENCES can express; an arts-only
    # campaign (the lcsw sweeps: one worker, no progress thread) must pass.
    sel = _selection(entries=["arts_val_wb"])
    sel.validate_against(plane, catalog,
                         _profile(pin=False, workers=1, progress=0))


def test_reference_selection_requires_two_threads(plane, catalog):
    with pytest.raises(ValueError, match="degenerates"):
        _selection().validate_against(
            plane, catalog, _profile(workers=1, progress=0))


def test_colocated_reference_blocks_must_fit_the_host(plane, catalog):
    # 200 ranks x 16 threads reaches past both u8 fields and any host.
    with pytest.raises(ValueError, match="255|first threads|CPU_SETSIZE"):
        _selection(node_counts=[200]).validate_against(
            plane, catalog, _profile(nodes=[200]))


# -- checker: envelope and world-size demotions -----------------------------

def _result(tmp_path, text: str, status):
    from artsrun.run.types import CellResult

    log = tmp_path / "cell.log"
    log.write_text(text)
    return CellResult(cell=_cell(RuntimeKind.XSOCR, nodes=2), status=status,
                      log_path=log)


def test_envelope_failure_demotes_even_a_measured_timeout(tmp_path):
    # Rank 0 printed marker and stamp, then another rank's envelope died and
    # the survivors idled into the budget: the timeout-leniency branch must
    # not promote this to OK.
    from artsrun import check
    from artsrun.run.types import Status

    text = ("DONE sum 42\n[E2E] 1000000000\n"
            "ARTSRUN-ENVELOPE-FAIL: bind: mask readback is '0-7', wanted '0-15'\n")
    r = check.apply_to(_result(tmp_path, text, Status.TIMEOUT))
    assert r.status is Status.FAIL
    assert "envelope" in r.note


def test_multiple_e2e_stamps_fail_the_cell(tmp_path):
    # N stamps means N independent worlds: a PMI-less launch degrades every
    # rank to a singleton that solves the whole problem alone — with the
    # correct scalar, so the consensus vote cannot catch it either.
    from artsrun import check
    from artsrun.run.types import Status

    text = "DONE sum 42\n[E2E] 1000\nDONE sum 42\n[E2E] 2000\n"
    r = check.apply_to(_result(tmp_path, text, Status.OK))
    assert r.status is Status.FAIL
    assert "world-size" in r.note


def test_a_single_stamp_still_passes(tmp_path):
    from artsrun import check
    from artsrun.run.types import Status

    r = check.apply_to(_result(tmp_path, "DONE sum 42\n[E2E] 1000\n", Status.OK))
    assert r.status is Status.OK
    assert r.scalar == "42"
