# LCS_distributed_ST

*Same recursive quad-tree wavefront DP as `LCS_shared`, but with `S` and
`T` chopped into per-block labeled datablocks — the score matrix stays
one shared DB, only the read-only inputs are distributed.*
Source: `third_party/ocr-apps/apps/LCS/refactored/ocr/intel-jesmin-lcs_distributed_ST_datablocks/lcs_ST_distributed.c`
(~695 lines; author Jesmin Jahan Tithi, Intel 2016).

## Overview

Structurally identical to `LCS_shared`: the same `recLCSEdt`
quad-tree recursion (`x11` unblocked, `x12`/`x21` gated on `x11`,
`x22` gated on both), the same base-case `seqLCSEdt` antidiagonal
kernel, the same compact `O(N)` antidiagonal-offset score
representation, the same `S[i] != T[j]` match term with
`GAP_PENALTY=0`, and the same rank-0 native `serial_lcs` self-
consistency check before any EDT is created. The one structural
difference — what "ST" (S/T-distributed) names — is that `S` and `T` are
no longer single datablocks: each is chopped into one fixed-size chunk per
base case (`num_labels = 2^d`), allocated through `ocrGuidRangeCreate`/
`ocrGuidFromIndex` (labeled GUIDs) instead of one `ocrDbCreate` each. The
`score` matrix, by contrast, is **still one shared DB**, exactly as in
`LCS_shared` — this variant distributes only the read-only inputs, not
the read/write state, which is the deliberate midpoint between
`LCS_shared` (nothing distributed) and `LCS_all_db_distributed`
(everything distributed, including `score`). `shutDownEdt` prints `LCS
length: N` and asserts it against the native recomputation. DAG shape
depends only on `N`/`base`, never on string content.

## Parameters

| arg | meaning | default | CLI reachability |
|-----|---------|---------|-------------------|
| `argv[1]` = `N` | string length | 1024 | ✓ `atol` in `mainEdt`, reaches the recursion via `LCS_task_params.N` (`paramv`) — multinode-safe |
| `argv[2]` = `base` | recursion cut-off: the quad-tree halves until `n ≤ base` | 256 | ✓ same, via `LCS_task_params.base`; indirectly sizes each `S`/`T` label block, which is cut to the resulting base-case width `N>>d` (equal to `base` only when `N/base` is a power of two) |
| `argv[3]` = `num_workers` | intended worker count | 16 | ⚠ parsed on rank 0, used only in one `ocrPrintf` — no functional effect, does not set ARTS's thread count |

`GAP_PENALTY` is compile-time only, no argv path.

## Structure

Same recursion-depth definition as `LCS_shared`: `d` = number of `n←n>>1`
shifts of `N` until `≤ base`; the quad-tree is a perfect 4-ary tree of
depth `d`. Let `L = 2^d` be the number of base-case columns, which is also
the number of `S` (and of `T`) tiles: each is `N>>d` characters wide, one
per base case.

| object | count | size |
|--------|-------|------|
| `recLCSEdt` (incl. root) | `(4^(d+1)-1)/3` | — |
| `seqLCSEdt` (leaf) | `4^d` | — |
| `mainEdt` / `shutDownEdt` | 1 each | — |
| DBs | `3 + 2L` — 1 shared `score` (ordinary, RW) + 1 `S`-pointer-array + 1 `T`-pointer-array (ordinary, local scratch, **never wired as a dependency to any EDT**) + `L` labeled `S` tiles + `L` labeled `T` tiles | `score`=`16·(N+1)` B; pointer arrays=`8·L` B each; with `w = N>>d`, `S`/`T` tile 0 = `4·(w+1)` B, tiles `1..L-1` = `4·w` B each |
| Events | `6·4^d − 1`, identical closed form to `LCS_shared` (the recursive `recLCSEdt`/`seqLCSEdt` wiring — and its `EDT_PROP_FINISH`/output-event usage — is unchanged; only the `S`/`T` access path differs, and that never touches event creation): every `recLCSEdt` create (root + `x11`/`x12`/`x21`/`x22`, count `(4^(d+1)-1)/3`) costs 3 events (1 app STICKY + 1 runtime finish + 1 runtime output); every `seqLCSEdt` create (count `4^d`) costs 2 (1 app STICKY + 1 runtime output, no finish); `shutDownEdt` costs 0 | — |
| EDT templates | `2·4^d + 1`, none destroyed | — |

Worked numbers at the calibrated `args = [57344, 1024, 48]` (`N=57344,
base=1024`): 6 shifts (`57344→…→896≤1024`) give `d=6`, so `recLCSEdt` =
5,461, `seqLCSEdt` = 4,096, total EDTs = 9,559, Events = `6·4⁶−1` =
24,575 — all identical to `LCS_shared`'s counts at its own (different)
`d=6`. `L = 2⁶ = 64`; DBs = `3 + 128 = 131` (`score`≈896 KiB; 64 `S` tiles
+ 64 `T` tiles, ≈3.5 KiB each). Note that this `N`/`base` pair is **not** a
power-of-two ratio (`57344/1024 = 56`): the base cases are `896` wide, and
it is that width — not `base` — the tiles are cut to.

Counter cross-check: verified (1 node, `8 2 1` (`d=2`) vs `16 2 1`
(`d=3`)): ΔNUM_EDT_CREATE = 112, ΔNUM_DB_CREATE = 8, ΔNUM_EVENT_CREATE =
288 — exactly the formulas' deltas (EDTs 39→151; DBs `3+2L` = 11→19 as
`L = 2^d` goes 4→8; Events `6·4²−1=95` → `6·4³−1=383`); the runtime adds a
constant baseline of +1 EDT and +1 DB per run (Events' baseline is +0),
giving measured totals 40/12/95 → 152/20/383.

## Wiring

- Root `recLCSEdt`'s template is `depc=1`, wired only to `score` (RW,
  slot 0) — unlike `LCS_shared`'s root, there is no `S`/`T` dependence at
  all (those lines are present in the source but commented out); `S`/`T`
  travel down the recursion only as the `s_labels`/`t_labels` *range*
  GUIDs inside `paramv`.
- Internal (non-leaf) `recLCSEdt` calls carry no DB dependence, same
  event-chain wiring as `LCS_shared` (`x11` unblocked, `x12`/`x21` on
  `x11`'s event, `x22` on both).
- Each leaf resolves its own `S`/`T` block via `ocrGuidFromIndex(...,
  (p->xi-1)/n)` / `(..., (p->xj-1)/n)` — the base case's own extent `n`,
  which is exactly the tile width — then wires `seqLCSEdt`'s 3 slots:
  `S` (RO, slot 0), `T` (RO, slot 1), `score` (RW, slot 2, the same
  single shared DB as `LCS_shared`).
- **DB concurrency**: `score` is unchanged from `LCS_shared` — one
  RW-exclusive DB, exactly one holder at a time, `4^d` serialized leaf
  turns regardless of node/worker count; this variant does nothing to
  relieve that bottleneck. `S`/`T` are now split into `L` separate RO
  DBs, so the *maximum* concurrent-reader count for any *one* block drops
  from "every ready leaf" (`LCS_shared`) to "every ready leaf whose `xi`
  falls in that one block's index range" (up to `2^d` leaves sharing an
  `xi`, varying `xj`) — spreading read traffic across `L` DBs instead of
  concentrating it on one, without touching the write bottleneck at all.

## Flow

Identical phase shape to `LCS_shared`: rank-0-only preamble (create
`score` + the `S`/`T` pointer arrays + `L` labeled tiles each, native
`O(N)` fill, then the `O(N²)` native `serial_lcs` self-check), then the
quad-tree unfolds to depth `d`, `4^d` leaves each taking a serialized
turn on `score`, `shutDownEdt` on completion. The only Flow difference
from `LCS_shared` is that the preamble now issues `2L` extra
`ocrDbCreate` calls (one per labeled tile) instead of 2 (`S_guid`,
`T_guid`) — still entirely serial, still entirely on rank 0.

## Placement (as-born)

No `OCR_APP_OPTIMIZED_PLACEMENT` guard in this file — every
`ocrEdtCreate` passes `NULL_HINT`. But **labeled GUIDs are placed
differently from ordinary ones**: a range created via
`ocrGuidRangeCreate` gets each index's home fixed round-robin
(`home = index % nranks`) at range-creation time, baked into the GUID's
own rank bits — independent of the hint passed to the later
`ocrDbCreate` on that GUID, and independent of which rank actually calls
it. Effective policy:

- **EDTs**: NULL hint → round-robin (`ARTS_HINT_ANY_RANK`) —
  `recLCSEdt`/`seqLCSEdt` scatter with no relation to their tile
  position.
- **`score`, `S_guid`, `T_guid` (pointer array)**: ordinary `ocrDbCreate`,
  NULL hint → home = creator = rank 0 (all created inside `mainEdt`).
- **`S`/`T` labeled tiles**: home = `label_index % nranks`, fixed
  regardless of the `NULL_HINT` `mainEdt` passes to their `ocrDbCreate` —
  genuinely spread across every rank, even though `mainEdt` (rank 0)
  performs every one of the `2L` creates. When the resolved home differs
  from rank 0, that create is itself a remote operation (a coherent-home
  stub install, not a local allocation).

Consequence: `score` behaves exactly as in `LCS_shared` — pinned at rank
0, a remote round trip for most of its `4^d` serialized turns. `S`/`T`
reads, by contrast, are now spread over `nranks` possible homes instead
of concentrated at rank 0 — but since the *EDT's* rank (round-robin, a
creation-order counter) and its needed tile's home (`index % nranks`, a
position-derived value) come from unrelated schemes, a leaf landing on
the same rank as its `S`/`T` block is coincidence, not design; most
`S`/`T` acquires are still remote. Net effect: this variant relieves
pressure on rank 0 specifically for `S`/`T` traffic without making any
individual acquire more likely to be local, and it leaves the `score`
bottleneck exactly as severe as `LCS_shared`'s.

## Sizing

Same dials and the same anti-scaling shape as `LCS_shared` — `d` (via
`N`, `base`) sets both `S`/`T` tile-payload-per-block and the `4^d`
sequential `score` turns that dominate wall time; node/worker count
changes how much of that traffic is remote, not how much of it is
concurrent. The only new consideration versus `LCS_shared`: `base` is a
cut-off, not a tile size, so the tile count is `2^d` and the tile width
`N>>d` — for a `N/base` that is not a power of two there are more, narrower
tiles than the ratio suggests. The calibrated `args = [57344, 1024, 48]`
gives `d=6` (4,096 leaf turns, `L=64` label blocks of 896 characters); as
with `LCS_shared`, this size is picked for
observability, not to saturate a target worker count, since worker count
does not relieve the `score` bottleneck.
