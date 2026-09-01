# HPX reference apps

The normal benchmark build source-builds the pinned HPX submodule in an
isolated build tree, installs it under the ARTS build directory, and uses that
package to build these apps. HPX uses its distributed runtime with the MPI
parcelport; the TCP parcelport is disabled. Like every other vendored runtime
in this repo, HPX is built as static libraries: an app binary embeds HPX,
Boost, and HPX's own pinned hwloc (the release this HPX version validates
against, built static from a checksummed dist tarball), and resolves nothing
through the loader beyond the host toolchain and MPI. (An MPI library may still load the host's
hwloc for itself; the embedded copy is kept out of the dynamic symbol table
so the two never mix.)

```bash
cmake -GNinja -Bbuild -DCMAKE_BUILD_TYPE=Release
cmake --build build --target nqueens_hpx
```

## Threading and pinning

One locality is one process (one MPI rank). Every app here bakes in two
runtime defaults:

- `hpx.use_process_mask=1` — an externally installed CPU affinity mask is
  authoritative: HPX sizes and pins its worker threads inside it instead of
  rebinding from the machine topology (which is HPX's stock behavior).
- `hpx.os_threads=cores` — the default worker count is one per *physical
  core* in the mask, so SMT siblings never carry a second worker. An
  explicit `--hpx:threads=N` overrides.

So a bare run uses every physical core of the machine, a run under a
narrowed mask uses exactly the cores of that mask, and one locality per
host — the remote launcher shape — needs no flags at all.
`--hpx:print-bind` prints the realized per-worker binding for verification.

Colocated localities (several per host) additionally depend on one
backported upstream fix: stock v1.11.0 shifts each newly registering
locality's binding by the core footprints its peers reported (AGAS
`first_used_core`), computed against the machine topology with no regard
for the process mask, so colocated localities land outside the disjoint
per-rank masks the launcher set up. Upstream fixed this on master
(`6df14adf09`, "Don't apply PU offset for localities that explicitly use
core bindings"); the configure step applies that commit verbatim onto the
pristine submodule tree from `third_party/patches/hpx/` — idempotently, so
the submodule working tree stays at v1.11.0 plus exactly this patch. With it, a
colocated run places every locality inside its own mask:

```bash
mpiexec -bind-to none -n 2 \
    bash tools/artsrun/envelope.sh 16 2 rank -- \
    build/benchmarks/hpx/nqueens_hpx 12 8
```

## N-Queens

`nqueens_hpx` implements the same bitmask search, cutoff, rounds, and static
early-level placement as the hinted OCR N-Queens app:

```bash
build/benchmarks/hpx/nqueens_hpx 8 3
mpiexec -bind-to none -n 2 \
    bash tools/artsrun/envelope.sh 16 2 rank -- \
    build/benchmarks/hpx/nqueens_hpx 8 3 1 3
```

The application arguments are `size cutoff [rounds [scatter-levels]]`.
Search nodes below the cutoff run sequentially. Before `scatter-levels`, each
subtree is mapped to a locality by the same mixed column-mask key as the OCR
hinted version; deeper descendants remain on their current locality. Both
commands above print `8-queens; 8x8; sols: 92`, and with `ARTS_E2E_MARKER`
set the console locality prints the same `[E2E] <ns>` stamp as the other
runtimes.

`tests/nqueens_core_test.cpp` checks the shared solver core (known solution
counts, cutoff boundary) and runs automatically as the app project's test
step on every build.

## Experiment driver

`artsrun` offers HPX as the off-plane selection entry `hpx`: it runs the
applications whose catalog row carries `hpx: true`, in the version row the
port mirrors (nqueens: hinted), launched like the other references (mpirun
or srun + the CPU envelope) with the same `[E2E]` measurement and consensus
vote. `artsrun run --entries hpx --apps nqueens ...` runs it alone.
