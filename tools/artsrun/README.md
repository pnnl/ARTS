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

Profiles are a machine's own untracked settings, so a fresh checkout has
none. The screens still open — on an unsaved single-node local profile;
adjust it on the Profile tab and Save writes the machine's first one. The
same bootstrap exists on the command line:

```bash
artsrun profile new <machine>     # single-node local defaults, opens $EDITOR
```

The build tree bootstraps the same way: a first real run configures the
experiment build (Release, benchmarks on — its first build also compiles
the vendored dependencies, so it is long) and every later run reuses it.
Dry runs configure nothing, and an existing tree is only verified — a
Debug or no-benchmark tree is somebody's deliberate configuration and is
reported, not replaced. Under a Slurm profile, every piece of build work —
the first configure, a counter reconfigure, the ninja pass — runs inside a
small job of its own rather than on the login node: one task, a few cpus,
nothing exclusive, so it slots into whatever gap the queue has.

## Use

```bash
artsrun                                  # the selection screens
artsrun plane                            # the configuration plane
artsrun apps -b paper-main               # the catalog, with each app's versions
artsrun profile list | show X | validate | fields
artsrun profile new junction2 --from junction   # copy, then $EDITOR
artsrun profile set junction2 workers=31 slurm.partition=pbatch
artsrun benchset list | show X | new Y --from X | edit Y
artsrun benchset set paper-main nqueens --args "12 4" --versions asborn,optimized
artsrun config render -p ferrari -n 4    # inspect a rendered configuration
artsrun run -p ferrari -b paper-main --dry-run
artsrun run -p junction -b paper-main --detach
artsrun watch                             # live view of the latest campaign
artsrun watch 20260811-220547             # …or of a named one
artsrun report                            # reprint the latest summary

artsrun counters --set perf --enabled     # what a counter set turns on
artsrun counterset list | show X | render X
artsrun run -p ferrari -b paper-main -c perf
```

On the screens: `1`–`4` switch surfaces, `5` is the run tab, `space`
toggles, `a` is the shared all/none control for whichever surface is
showing, `r` runs, `d` dry-runs. The run tab is the live table with a
single control row above it; build output appears below only while there is
nothing else to watch and folds away when the table takes over (`l` brings
it back).

Running on a terminal opens the **live view** — one table row per cell (app,
version, runtime, coherence axes, node count, repeat), recolored as cells
queue, run, finish, and vote; `--plain` keeps the old line output. A click
(or `enter`) opens the cell: the exact command it runs under (for Slurm, the
sbatch call and the batch script), its environment, return code, a following
tail of its log (stderr is merged into stdout; stdin is `/dev/null`), and
buttons for the configurations behind the run — the rendered runtime
configuration the cell was handed and, when counters are on, the counter
file the build was configured against. While the pane is open it follows the
cursor; `esc` closes it.
Consensus is re-voted as results land, so a configuration that strays turns
its row `DIFF` the moment it disagrees — not at the end of the campaign.
The `scaling` button on the summary row (or `g`) swaps the table for the
**strong-scaling reading** of this run — scaling belongs to the campaign it
was measured in, so it lives on the same surface: best wall per node
count for every (application, version, configuration), each wider cell
carrying its speedup against the row's smallest measured node count and
coloured by parallel efficiency — an anti-scaling row turns red the moment
its wider run comes back slower. Because the screen already says all of
this, the end of a campaign prints only a one-line verdict and the path to
`summary.txt`; the full tables still land there. `artsrun watch` attaches
the same view to a running (or finished) campaign from any terminal; a
`--detach`ed campaign is watched the same way, and `d` detaches the view
again without touching the run.

The **Profile** screen edits the settings themselves — launcher, workers and
progress threads, provider, ports, hosts, Slurm partitions — and saves under
the same name or a new one; an invalid combination reports the one thing to
fix and writes nothing. Node counts sit in *Run shape* as the list itself: `✕` removes one, the box
at the end adds one. Removing changes the machine's shape (Save writes the
list); unchecking changes only this campaign, so narrowing one run costs
nothing and needs no save.

Only the sections the launcher decides are shown — the SSH roster for ssh,
the Slurm settings for slurm. Every launcher states `connections / node`;
only the remote ones name the ports, and then the list must hold exactly that
many. A local run's ranks share a machine, so the runtime claims its own block
of `node_count × connections` instead.

An ssh profile declares how many nodes it may use and must name exactly that
many hosts. A Slurm profile declares no budget at all: **every cell is
submitted up front** — scheduling the queue is Slurm's whole purpose — and
each job writes its own outcome marker on the shared filesystem as it ends.
The submitting login node is thereby optional: if it dies with the queue
full, nothing is lost — `artsrun watch` and `artsrun report` reconstruct
finished cells from the markers, and `--resume` resubmits only what neither
finished nor still sits in the queue. Only an explicit stop cancels jobs;
a dead process leaves the queue alone. Each job's `--time` comes from the
cell's own timeout, so the backfill scheduler can slot short jobs early;
the in-job `timeout -k` fires first, which is what lets even a timed-out
cell write its marker. `slurm.build_partition` (e.g. a debug partition)
and `slurm.build_cpus` (default 8) shape where and how wide build work
queues.

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

A campaign directory holds `manifest.json` (every cell with the exact command
it runs under — written before anything runs), `track.jsonl` (events as they
happen: submitted / started / running / finished, with the final status and
scalar), `selection.yaml` (replayable), per-cell logs under `cells/`, and the
end-of-run `results.csv` / `report.json` / `summary.txt`. The live view is a
pure reader of the first four, which is why it can attach to any campaign
from anywhere.

Each application offers up to three versions: **as-born** (as published,
including whatever hints its authors already gave it), **optimized** (the
same code structure with EDT/DB placement hints added or changed — as
statically optimized as hints alone can make it), and **restructured**
(the decomposition itself redesigned as a separate target, for the cases
hints could not fix).

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
