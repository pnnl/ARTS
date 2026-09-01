"""How a Flux cell is submitted, launched, and judged.

Under Flux every kind launches through `flux run` inside the cell's own
`flux batch` script, in the proven `-N n -n n -c W` shape (per-resource and
per-task options cannot combine, and -N alone distributes one task per
node).  -c rides on EVERY line — the backstop that keeps a rejected or
ignored affinity off-switch from shrinking a rank to one core — while the
shell's affinity plugins (cpu, gpu, and mpibind where the site loads it)
are disabled so the envelope wrapper and the runtimes' own pinning stay the
only affinity actors.  Job ids are F58 strings handled verbatim; the
marker file, not the queue, is the authority on how a cell ended.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from artsrun.model.benchset import ResolvedApp
from artsrun.model.catalog import AppClass, Version
from artsrun.model.plane import RuntimeKind, SelectionEntry
from artsrun.model.profile import Profile
from artsrun.paths import envelope_script
from artsrun.run.command import build_env
from artsrun.run.flux import (
    FluxBackend,
    FluxError,
    alive_jobs,
    batch_argv,
    flux_build_prefix,
    job_script,
    script_path,
)
from artsrun.run.markers import marker_path
from artsrun.run.types import Cell, CellResult, Status

ENV = str(envelope_script())


def _entry(kind: RuntimeKind) -> SelectionEntry:
    return SelectionEntry(
        key=f"{kind.value}_entry", label=kind.value, kind=kind, cell=kind.value,
    )


def _cell(kind: RuntimeKind, nodes: int, cfg: Path | None = None,
          post_verify: str | None = None):
    app = ResolvedApp(
        name="app", version=Version.BASE, binary="app", cls=AppClass.TASK,
        marker="X", scalar_re="X", post_verify=post_verify,
    )
    return Cell(entry=_entry(kind), app=app, nodes=nodes, repeat=1,
                binary=Path("/opt/bin/app"), args=["12", "4"], timeout_s=60,
                cfg=cfg)


def _flux_profile(**flux) -> Profile:
    return Profile.model_validate({
        "name": "t", "launcher": "flux", "nodes": [1, 2, 4],
        "workers": 15, "progress": 1, "ports": [25000], "flux": flux,
    })


class _FakeFlux:
    """Records every flux invocation and answers from a canned table."""

    def __init__(self, answers=None, rc=0):
        self.calls: list[list[str]] = []
        self.envs: list[dict] = []
        self.answers = answers or {}
        self.rc = rc

    def __call__(self, argv, capture_output=True, text=True, check=False,
                 env=None, input=None):
        self.calls.append(list(argv))
        self.envs.append(dict(env or {}))
        out = ""
        rc = self.rc
        key = tuple(argv[:2])
        if key in self.answers:
            answer = self.answers[key]
            out, rc = answer if isinstance(answer, tuple) else (answer, 0)
        return subprocess.CompletedProcess(argv, rc, stdout=out, stderr="")


# -- the launch line ---------------------------------------------------------

def test_an_arts_flux_script_runs_one_flux_run_in_the_proven_shape():
    profile = _flux_profile()
    cell = _cell(RuntimeKind.ARTS, nodes=4)
    script = job_script(cell, profile, Path("/log/cell.rc"))
    assert "mpirun" not in script
    assert ("flux run -N 4 -n 4 -c16 -o cpu-affinity=off "
            "-o gpu-affinity=off -o mpibind=off "
            f"{cell.binary} 12 4") in script
    assert "--tasks-per-node" not in script
    assert "--label" not in script
    # arts never consults PMI; -o pmi is a reference-cell concern only.
    assert "pmi=" not in script


@pytest.mark.parametrize("nodes", [1, 4])
def test_an_xsocr_flux_script_wraps_the_envelope(nodes):
    profile = _flux_profile()
    cfg = Path(f"/opt/cfg/{nodes}n.cfg")
    cell = _cell(RuntimeKind.XSOCR, nodes=nodes, cfg=cfg)
    script = job_script(cell, profile, Path("/log/cell.rc"))
    assert (f"flux run -N {nodes} -n {nodes} -c16 -o cpu-affinity=off "
            f"-o gpu-affinity=off -o mpibind=off "
            f"bash {ENV} 16 {nodes} fixed -- {cell.binary} "
            f"-ocr:cfg {cfg} 12 4") in script
    assert "export OCRVX_NUM_THREADS" not in script


def test_flux_pmi_setting_reaches_references_only():
    profile = _flux_profile(pmi="cray-pals,simple")
    ref = job_script(_cell(RuntimeKind.OCRVX, nodes=2), profile,
                     Path("/log/cell.rc"))
    assert "-o pmi=cray-pals,simple bash" in ref
    arts = job_script(_cell(RuntimeKind.ARTS, nodes=2), profile,
                      Path("/log/cell.rc"))
    assert "pmi=" not in arts


def test_mpibind_off_is_emitted_only_where_the_site_runs_the_plugin():
    # An instance without the plugin may reject the unknown option name, so
    # the flag rides only on a profile that declares the plugin present.
    bare = job_script(_cell(RuntimeKind.ARTS, nodes=2),
                      _flux_profile(mpibind=False), Path("/log/cell.rc"))
    assert "mpibind" not in bare
    assert "-o cpu-affinity=off -o gpu-affinity=off" in bare


def test_extra_run_flags_ride_every_launch_line():
    profile = _flux_profile(extra_run=["-o", "verbose=1"])
    script = job_script(_cell(RuntimeKind.ARTS, nodes=2), profile,
                        Path("/log/cell.rc"))
    assert "-o mpibind=off -o verbose=1 /opt/bin/app" in script


def test_post_verify_wraps_outside_flux_run():
    profile = _flux_profile()
    cell = _cell(RuntimeKind.XSOCR, nodes=2, cfg=Path("/opt/cfg/2n.cfg"),
                 post_verify="cmp out.txt gold.txt")
    script = job_script(cell, profile, Path("/log/cell.rc"))
    line = next(l for l in script.splitlines() if l.startswith("timeout"))
    assert "sh -c" in line
    assert line.index("flux") > line.index("sh -c")


def test_job_script_records_its_own_marker_atomically():
    profile = _flux_profile()
    script = job_script(_cell(RuntimeKind.ARTS, nodes=2), profile,
                        Path("/log/cell.rc"))
    assert 'echo "$rc $s $(date +%s.%N)" > /log/cell.rc.tmp' in script
    assert "mv /log/cell.rc.tmp /log/cell.rc" in script
    assert "timeout -k 1 60 " in script


# -- submission --------------------------------------------------------------

def test_batch_argv_shape():
    profile = _flux_profile(queue="pbatch", bank="proj",
                            extra_batch=["--requires=-host:bad1"])
    cell = _cell(RuntimeKind.ARTS, nodes=4)
    argv = batch_argv(cell, profile, Path("/log/cell.log"),
                      Path("/log/cell.sh"))
    assert argv == [
        "flux", "batch", "--exclusive", "--nodes=4",
        "--job-name=arts-app-arts_entry-4n", "--output=/log/cell.log",
        "-t", "180s",
        "--queue=pbatch", "--bank=proj", "--requires=-host:bad1",
        "/log/cell.sh",
    ]
    # stderr follows stdout when --error is unset (documented and verified);
    # naming the same file twice would be a second unverified behavior.
    assert not any(a.startswith("--error") for a in argv)


def test_build_prefix_is_a_narrow_job():
    profile = _flux_profile(queue="pbatch", build_queue="pdev", bank="proj",
                            build_cpus=16, build_time="30m")
    assert flux_build_prefix(profile) == [
        "flux", "run", "-n1", "-c16", "--job-name=arts-build",
        "-t", "30m", "--queue=pdev", "--bank=proj",
    ]
    # Without a build queue the cell queue takes the build too — never a
    # short-capped debug queue by default.
    assert "--queue=pbatch" in flux_build_prefix(_flux_profile(queue="pbatch"))


def test_submit_stores_the_printed_id_verbatim_and_forces_ascii(
        tmp_path, monkeypatch):
    fake = _FakeFlux(answers={("flux", "batch"): "f2AbCdE\n"})
    monkeypatch.setattr("artsrun.run.flux.subprocess.run", fake)
    backend = FluxBackend(_flux_profile(), tmp_path)
    cell = _cell(RuntimeKind.ARTS, nodes=2)
    result = backend.submit(cell)
    assert result.status is Status.SUBMITTED
    assert result.extra["job_id"] == "f2AbCdE"
    assert fake.envs[0]["FLUX_F58_FORCE_ASCII"] == "1"
    assert script_path(tmp_path, cell).read_text().startswith("#!/bin/bash")


def test_submit_refuses_to_swap_a_script_under_a_resumed_cell(
        tmp_path, monkeypatch):
    # Write-once by contract, but per-cell by consequence: an exception here
    # would escape the scheduler and kill the rest of the campaign.
    fake = _FakeFlux(answers={("flux", "batch"): "f2AbCdE\n"})
    monkeypatch.setattr("artsrun.run.flux.subprocess.run", fake)
    backend = FluxBackend(_flux_profile(), tmp_path)
    cell = _cell(RuntimeKind.ARTS, nodes=2)
    script_path(tmp_path, cell).write_text("#!/bin/bash\nsomething else\n")
    result = backend.submit(cell)
    assert result.status is Status.FAIL
    assert "changed since this campaign started" in result.note
    assert fake.calls == []  # nothing was submitted


# -- polling -----------------------------------------------------------------

def _submitted(backend, tmp_path, monkeypatch, fake):
    monkeypatch.setattr("artsrun.run.flux.subprocess.run", fake)
    cell = _cell(RuntimeKind.ARTS, nodes=2)
    return backend.submit(cell)


def test_poll_reads_the_marker_first(tmp_path, monkeypatch):
    fake = _FakeFlux(answers={("flux", "batch"): "fX\n"})
    backend = FluxBackend(_flux_profile(), tmp_path)
    result = _submitted(backend, tmp_path, monkeypatch, fake)
    marker_path(tmp_path, result.cell).write_text("0 100.0 160.5")
    updated = backend.poll(result)
    assert updated.status is Status.OK
    assert updated.rc == 0
    assert updated.wall_s == 60.5
    # The queue was never asked: the marker is the authority.
    assert not any(c[:2] == ["flux", "jobs"] for c in fake.calls)


@pytest.mark.parametrize("row,status", [
    ("fX|123|NEW||", Status.SUBMITTED),
    ("fX|123|SCHED||", Status.SUBMITTED),
    ("fX|123|RUN||", Status.RUNNING),
    ("fX|123|CLEANUP||", Status.ENDING),
])
def test_poll_maps_active_states(tmp_path, monkeypatch, row, status):
    fake = _FakeFlux(answers={("flux", "batch"): "fX\n",
                              ("flux", "jobs"): row + "\n"})
    backend = FluxBackend(_flux_profile(), tmp_path)
    result = _submitted(backend, tmp_path, monkeypatch, fake)
    assert backend.poll(result).status is status


def test_poll_leaves_an_unrecognized_state_alone(tmp_path, monkeypatch):
    # A future flux may add states; a merely-queued cell must never be
    # recorded as failed on one.
    fake = _FakeFlux(answers={("flux", "batch"): "fX\n",
                              ("flux", "jobs"): "fX|123|SOMENEWSTATE||\n"})
    backend = FluxBackend(_flux_profile(), tmp_path)
    result = _submitted(backend, tmp_path, monkeypatch, fake)
    assert backend.poll(result).status is Status.SUBMITTED


@pytest.mark.parametrize("row,status,rc", [
    ("fX|123|INACTIVE|COMPLETED|0", Status.OK, 0),
    ("fX|123|INACTIVE|FAILED|7", Status.FAIL, 7),
    ("fX|123|INACTIVE|CANCELED|143", Status.FAIL, 143),
    ("fX|123|INACTIVE|TIMEOUT|124", Status.TIMEOUT, 124),
])
def test_poll_maps_final_results_when_no_marker_exists(
        tmp_path, monkeypatch, row, status, rc):
    fake = _FakeFlux(answers={("flux", "batch"): "fX\n",
                              ("flux", "jobs"): row + "\n"})
    backend = FluxBackend(_flux_profile(), tmp_path)
    result = _submitted(backend, tmp_path, monkeypatch, fake)
    updated = backend.poll(result)
    assert updated.status is status
    assert updated.rc == rc
    assert "no marker" in updated.note


def test_poll_fails_a_job_the_manager_no_longer_knows(tmp_path, monkeypatch):
    fake = _FakeFlux(answers={("flux", "batch"): "fX\n",
                              ("flux", "jobs"): ("JobID fX unknown\n", 1)})
    backend = FluxBackend(_flux_profile(), tmp_path)
    result = _submitted(backend, tmp_path, monkeypatch, fake)
    updated = backend.poll(result)
    assert updated.status is Status.FAIL
    assert "no marker and no job record" in updated.note


def test_abort_cancels_every_outstanding_job(tmp_path, monkeypatch):
    fake = _FakeFlux(answers={("flux", "batch"): "fX\n"})
    backend = FluxBackend(_flux_profile(), tmp_path)
    _submitted(backend, tmp_path, monkeypatch, fake)
    backend.abort()
    assert ["flux", "cancel", "fX"] in fake.calls


# -- alive_jobs / resume -----------------------------------------------------

def test_alive_jobs_queries_one_id_at_a_time(monkeypatch):
    calls = []

    def fake(argv, **kwargs):
        calls.append(list(argv))
        if argv[-1] == "fA":
            return subprocess.CompletedProcess(argv, 0,
                                               stdout="fA|1|RUN\n", stderr="")
        if argv[-1] == "fB":
            return subprocess.CompletedProcess(
                argv, 0, stdout="fB|2|INACTIVE\n", stderr="")
        if argv[-1] == "--count=1":
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        # A purged id errors rather than printing an empty row.
        return subprocess.CompletedProcess(argv, 1, stdout="",
                                           stderr="JobID unknown")

    monkeypatch.setattr("artsrun.run.flux.subprocess.run", fake)
    assert alive_jobs(["fA", "fB", "fC"]) == {"fA"}
    per_id = [c for c in calls if c[:2] == ["flux", "jobs"]
              and c[-1] != "--count=1"]
    assert len(per_id) == 3  # one call per id: no mixed-list error semantics


def test_alive_jobs_aborts_when_flux_itself_cannot_answer(monkeypatch):
    # An unreachable broker must abort the resume, not read as an empty
    # queue — an empty answer would resubmit every still-queued job.
    def fake(argv, **kwargs):
        return subprocess.CompletedProcess(
            argv, 1, stdout="", stderr="Unable to connect to Flux")

    monkeypatch.setattr("artsrun.run.flux.subprocess.run", fake)
    with pytest.raises(FluxError):
        alive_jobs(["fA"])


# -- environment -------------------------------------------------------------

def test_flux_cells_carry_the_launcher_marker_env():
    profile = _flux_profile()
    env = build_env(_cell(RuntimeKind.XSOCR, nodes=2), profile)
    assert env["ARTSRUN_LAUNCHER"] == "flux"
    assert env["ARTS_E2E_MARKER"] == "1"
