# Legacy harness material

The performance and correctness harnesses that used to live in
`benchmarks/scripts/` were replaced by `tools/artsrun` (`artsrun`), which
carries the same knowledge as data rather than as accreted code: the
application catalog is `tools/artsrun/src/artsrun/data/apps.yaml`, machine
settings are profiles under `experiments/profiles/`, and application rosters
are benchsets under `experiments/benchsets/`.

What is kept here, and why:

- `correctness_harness.py` — the consensus harness. Its case table was the
  source of the scalar regexes, tolerances, pinned answers and structural
  skip reasons now in the catalog. Kept because it carried uncommitted local
  changes at the time of the cutover.
- `microbench/` — the read/write-mix probe drivers (`rwmix_*`, `rwrounds_*`)
  and the exclusion-contention probe. These drive the bespoke probe
  applications (`benchmarks/apps/rwmix.c`, `rwrounds.c`), which are still
  built; the drivers were never part of the application matrix and are not
  wired into `artsrun`. `harness_common.py` sits alongside them because they
  import it.
- `figures/` — the figure generators. Figure production is outside the
  driver's scope; it consumes a campaign's `results.csv` / `report.json`.
  The runtime series naming and colours they established are carried forward
  in `artsrun/report.py` (`RT_LABEL` / `RT_COLOR`).

Nothing here is on a supported path. The drivers expect the old layout
(`benchmarks/scripts/`) and will need their imports adjusted if revived.
