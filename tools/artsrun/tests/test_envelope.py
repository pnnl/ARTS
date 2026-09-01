"""The envelope wrapper itself, driven against a fake /proc + /sys tree.

envelope.sh carries the two checks nothing else in the stack performs — the
first-sibling topology check and the bind-readback postcondition — and every
failure mode below is one that is otherwise SILENT at the syscall level
(partial cpuset overlap narrows quietly; CPU_SET past the set size is a
no-op; a rank-variable fallback to 0 collapses every rank onto block 0).
"""

from __future__ import annotations

import os
import subprocess

import pytest

from artsrun.paths import envelope_script

PREFIX = "ARTSRUN-ENVELOPE-FAIL:"


def _sysfs(tmp_path, cpus: dict[int, str]):
    root = tmp_path / "sys"
    for cpu, siblings in cpus.items():
        d = root / f"cpu{cpu}" / "topology"
        d.mkdir(parents=True, exist_ok=True)
        (d / "thread_siblings_list").write_text(siblings + "\n")
    return root


def _run(tmp_path, args, *, cpus, allowed, env=None, taskset="true"):
    status = tmp_path / "status"
    status.write_text(f"Name:\tx\nCpus_allowed_list:\t{allowed}\n")
    full_env = {
        "PATH": os.environ["PATH"],
        "ENVELOPE_STATUS_FILE": str(status),
        "ENVELOPE_SYSFS_ROOT": str(_sysfs(tmp_path, cpus)),
        "ENVELOPE_TASKSET": taskset,
    }
    full_env.update(env or {})
    return subprocess.run(
        ["bash", str(envelope_script()), *[str(a) for a in args]],
        capture_output=True, text=True, env=full_env,
    )


# A two-core no-SMT host and a two-core SMT host with first-sibling
# enumeration (0,2 first threads; 1,3 their twins is the ADJACENT layout).
FIRST_SIBLINGS = {0: "0,2", 1: "1,3", 2: "0,2", 3: "1,3"}
ADJACENT = {0: "0-1", 1: "0-1", 2: "2-3", 3: "2-3"}


def test_clean_pass_execs_the_command(tmp_path):
    proc = _run(tmp_path, [2, 1, "fixed", "--", "echo", "ok"],
                cpus=FIRST_SIBLINGS, allowed="0-1")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_partial_overlap_readback_fails(tmp_path):
    # sched_setaffinity intersects with the cpuset and narrows SILENTLY on a
    # partial overlap; only the readback can tell, so it must.
    proc = _run(tmp_path, [2, 1, "fixed", "--", "echo", "ok"],
                cpus=FIRST_SIBLINGS, allowed="0")
    assert proc.returncode == 90
    assert PREFIX in proc.stderr and "readback" in proc.stderr


def test_sibling_adjacent_enumeration_is_refused(tmp_path):
    proc = _run(tmp_path, [2, 1, "fixed", "--", "echo", "ok"],
                cpus=ADJACENT, allowed="0-1")
    assert proc.returncode == 90
    assert "SMT sibling" in proc.stderr


def test_missing_topology_fails_closed(tmp_path):
    proc = _run(tmp_path, [2, 1, "fixed", "--", "echo", "ok"],
                cpus={0: "0,2"}, allowed="0-1")
    assert proc.returncode == 90
    assert "topology" in proc.stderr


def test_rank_mode_requires_a_rank_variable(tmp_path):
    proc = _run(tmp_path, [2, 4, "rank", "--", "echo", "ok"],
                cpus=FIRST_SIBLINGS, allowed="0-1")
    assert proc.returncode == 90
    assert "PMI_RANK" in proc.stderr

    proc = _run(tmp_path, [2, 4, "rank", "--", "echo", "ok"],
                cpus=FIRST_SIBLINGS, allowed="0-1",
                env={"PMI_RANK": "7"})
    assert proc.returncode == 90
    assert "outside" in proc.stderr


def test_rank_mode_shifts_the_block(tmp_path):
    cpus = {2: "2,6", 3: "3,7"}
    proc = _run(tmp_path, [2, 2, "rank", "--", "echo", "ok"],
                cpus=cpus, allowed="2-3", env={"PMI_RANK": "1"})
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_fixed_mode_rejects_a_second_local_rank(tmp_path):
    proc = _run(tmp_path, [2, 2, "fixed", "--", "echo", "ok"],
                cpus=FIRST_SIBLINGS, allowed="0-1",
                env={"SLURM_LOCALID": "1"})
    assert proc.returncode == 90
    assert "second rank" in proc.stderr


def test_fixed_mode_cross_checks_the_world_size(tmp_path):
    # The second line of defense against a PMI-less launch: the launcher
    # says how many tasks it started, and that must be the cell's rank count.
    proc = _run(tmp_path, [2, 4, "fixed", "--", "echo", "ok"],
                cpus=FIRST_SIBLINGS, allowed="0-1",
                env={"SLURM_NTASKS": "1"})
    assert proc.returncode == 90
    assert "world" in proc.stderr


def test_real_taskset_applies_the_mask(tmp_path):
    # No stubs for the bind itself: the wrapper must set its own affinity
    # and the readback must agree with the real /proc.  CPUs 0-1 exist on
    # any host running this suite; skip where they are not first siblings.
    proc = subprocess.run(
        ["bash", str(envelope_script()), "1", "1", "fixed", "--",
         "grep", "Cpus_allowed_list", "/proc/self/status"],
        capture_output=True, text=True,
        env={"PATH": os.environ["PATH"]},
    )
    if proc.returncode == 90 and "SMT sibling" in proc.stderr:
        pytest.skip("cpu0 is not a first sibling on this host")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split()[-1] == "0"


# --- launcher-keyed scheduler guards ---------------------------------------
# Scheduler variables leak across nestings: a flux cell inherits its
# broker's SLURM_* inside a Slurm allocation, and a local/ssh cell inside a
# flux allocation inherits FLUX_*.  The guards are therefore keyed on the
# launcher the cell was built for (ARTSRUN_LAUNCHER, exported by build_env).

def test_leaked_flux_vars_do_not_judge_a_non_flux_cell(tmp_path):
    proc = _run(tmp_path, [2, 2, "fixed", "--", "echo", "ok"],
                cpus=FIRST_SIBLINGS, allowed="0-1",
                env={"FLUX_JOB_SIZE": "1", "FLUX_TASK_LOCAL_ID": "1"})
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_a_flux_cell_fails_on_a_world_size_mismatch(tmp_path):
    proc = _run(tmp_path, [2, 2, "fixed", "--", "echo", "ok"],
                cpus=FIRST_SIBLINGS, allowed="0-1",
                env={"ARTSRUN_LAUNCHER": "flux", "FLUX_JOB_SIZE": "1"})
    assert proc.returncode == 90
    assert "world: FLUX_JOB_SIZE=1" in proc.stderr


def test_a_flux_cell_refuses_a_colocated_second_rank(tmp_path):
    proc = _run(tmp_path, [2, 2, "fixed", "--", "echo", "ok"],
                cpus=FIRST_SIBLINGS, allowed="0-1",
                env={"ARTSRUN_LAUNCHER": "flux", "FLUX_TASK_LOCAL_ID": "1"})
    assert proc.returncode == 90
    assert "colocation: FLUX_TASK_LOCAL_ID=1" in proc.stderr


def test_a_flux_cell_ignores_leaked_slurm_vars(tmp_path):
    # The mirror image: a flux-on-Slurm rehearsal leaks the broker's
    # SLURM_NTASKS/SLURM_LOCALID into every task.
    proc = _run(tmp_path, [2, 2, "fixed", "--", "echo", "ok"],
                cpus=FIRST_SIBLINGS, allowed="0-1",
                env={"ARTSRUN_LAUNCHER": "flux", "FLUX_JOB_SIZE": "2",
                     "SLURM_NTASKS": "1", "SLURM_LOCALID": "1"})
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_a_non_flux_cell_keeps_the_slurm_world_guard(tmp_path):
    proc = _run(tmp_path, [2, 2, "fixed", "--", "echo", "ok"],
                cpus=FIRST_SIBLINGS, allowed="0-1",
                env={"SLURM_NTASKS": "1"})
    assert proc.returncode == 90
    assert "world: SLURM_NTASKS=1" in proc.stderr
