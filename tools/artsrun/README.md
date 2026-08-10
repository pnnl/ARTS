# artsrun

Selection, build and run driver for ARTS experiments.

One tool holds the three things an experiment is made of — which coherence
configurations to run, on which machine geometry, over which applications —
compiles everything selected in a single pass, runs the resulting cells, and
votes a consensus on their result scalars.

## Install

The repository's `.venv` is uv-managed:

```bash
uv pip install -e tools/artsrun
```

Without uv:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e tools/artsrun
```

`requirements.txt` at the repository root is generated from this package's
`pyproject.toml` (`uv pip compile tools/artsrun/pyproject.toml -o
requirements.txt`); regenerate it when a dependency changes.

## Use

```bash
artsrun                                  # the selection screens
artsrun plane                            # the configuration plane
artsrun apps -b paper-main               # the catalog, with each app's versions
artsrun profile list | show X | validate | fields
artsrun profile new junction2 --from junction   # copy, then $EDITOR
artsrun profile set junction2 workers=31 slurm.budget=16
artsrun benchset list | show X | new Y --from X | edit Y
artsrun benchset set paper-main nqueens --args "12 4" --versions asborn,hinted
artsrun config render -p bentley -n 4    # inspect a rendered configuration
artsrun run -p bentley -b paper-main --dry-run
artsrun run -p junction -b paper-main --detach
artsrun report                            # reprint the latest summary

artsrun counters --set perf --enabled     # what a counter set turns on
artsrun counterset list | show X | render X
artsrun run -p bentley -b paper-main -c perf
```

On the screens: `1`–`4` switch surfaces, `space` toggles, `a` is the shared
all/none control for whichever surface is showing, `r` runs, `d` dry-runs.

The **Profile** screen edits the settings themselves — launcher, workers and
progress threads, provider, ports, hosts, Slurm budget — and saves under the
same name or a new one; an invalid combination reports the one thing to fix
and writes nothing. Node counts sit in *Run shape* as the list itself: `✕` removes one, the box
at the end adds one. Removing changes the machine's shape (Save writes the
list); unchecking changes only this campaign, so narrowing one run costs
nothing and needs no save.

Only the sections the launcher decides are shown — the SSH roster for ssh,
the Slurm settings for slurm. Every launcher states `connections / node`;
only the remote ones name the ports, and then the list must hold exactly that
many. A local run's ranks share a machine, so the runtime claims its own block
of `node_count × connections` instead.

Both remote launchers declare how many nodes they may use, and both refuse a
budget below the widest node count — over ssh the hosts must number exactly
the budget, under Slurm the budget caps what is in flight at once.

Only what a campaign actually decides is a field. The width a run occupies
follows from workers + progress, so nothing else states a core count: a local
run's ranks are placed by the runtime, and a Slurm job owns its nodes
outright and derives `--cpus-per-task` from that same width.

The **Applications** screen lists the suite in two groups. *Applications* come
first — benchmarks with a provenance, the ones a result is claimed about — and
*Microbenchmarks and fixtures* after, each exercising one runtime mechanism or
carrying no workload at all. Only the first group is enabled by default.

Arguments are edited inline. An empty box means the catalog's own calibration,
shown as the placeholder; type into it and the benchmark set records an
override.

## Where things live

| | |
|---|---|
| Application structure and calibration | `src/artsrun/data/apps.yaml` (committed) |
| Configuration plane | `src/artsrun/data/protocols.yaml` (committed) |
| Configuration templates | `configs/templates/*.j2` (committed) |
| Machine and node settings | `experiments/profiles/*.yaml` (untracked) |
| Application rosters and overrides | `experiments/benchsets/*.yaml` (untracked) |
| Counter selections | `experiments/countersets/*.yaml` (untracked) |
| Counters the runtime defines | `src/artsrun/data/counters.yaml` (committed) |
| Campaign output | `logs/exp/<timestamp>/` |

Each application offers up to three versions: **as-born** (as
published), **hinted** (its placement hint layer compiled in), and
**restructured** (its decomposition redesigned as a separate target,
for the cases hints could not fix).

A benchset carries only its deltas: anything it omits falls through to the
catalog, so a campaign that changes one application's arguments is a two-line
file.

The **Counters** screen selects what the build measures. Each counter takes a
mode (`OFF` / `ONCE` / `PERIODIC`), a level, and a reduction:

| Level | What is written |
|---|---|
| `THREAD` | each worker's own value |
| `NODE` | reduced across a rank's threads |
| `CLUSTER` | reduced across ranks, and the per-rank value is still written |

**Counter selection is a build-time decision.** The set is parsed at configure
time into `Preamble.h`, whose indices are compiled into every file that
touches a counter, so switching sets means a reconfigure and a full rebuild.
The driver refuses to run against a tree configured with a different counter
file rather than measure with the wrong ones, and prints the `cmake` line to
fix it. The sampling interval and the output folder are the exception — the
runtime reads those from its own configuration, so they move freely.

## Tests

```bash
.venv/bin/python -m pytest tools/artsrun/tests -q
```

The configuration-rendering tests compare against the committed `configs/`
trees, so they fail if a template drifts from what the runtimes have been
running.
