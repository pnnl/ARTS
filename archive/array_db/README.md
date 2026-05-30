# Archived: Array DB + partial-DB access (`arts_db_get` / `arts_db_put`)

Archived 2026-05-30 during the active-messaging refactor. The active-messaging
spec scopes Array DB as **orthogonal** to the refactor (a thin wrapper over the
core `arts_db_create` / `arts_edt_create` / `arts_add_dependence` APIs), and
ties the partial-DB raw RPC (`arts_db_get` / `arts_db_put`) to the same
deprecation. Removing them lets the wire protocol collapse to the plan's exact
`MSG_*` set with no legacy `DB_MOVE` / `GET_FROM_DB` / `PUT_IN_DB` messages.

## Contents

- `array_db.c`, `array_db.h` — the distributed Array DB library.
- `internal_include_array_db/internal.h` — its internal header.
- `CMakeLists.txt` — the original `arts_array_db` OBJECT-library build file.
- `tests/` — the 18 tests that exercised the archived APIs (Array DB +
  `arts_db_get`/`arts_db_put`).

## What was removed from the live tree (recoverable via git history)

The partial-DB-access plumbing that backed `arts_db_get` / `arts_db_put`:

- `arts_db_get` / `arts_db_put` (public API in `arts.h`) +
  `arts_db_op_hint_t` / `ARTS_DB_OP_HINT_DEFAULTS`.
- `internal_get_from_db` / `internal_put_in_db` (`db.c` / `db.h`).
- Wire messages `DB_MOVE` / `GET_FROM_DB` / `PUT_IN_DB`, their packet
  structs, dispatcher cases, and remote senders/handlers
  (`arts_remote_{get_from_db,put_in_db,handle_*,memory_move_no_free}`).
- OoO replay kinds `OOO_DB_GET_FROM` / `OOO_DB_PUT_IN` and the
  `arts_out_of_order_{get_from_db,put_in_db}` deferrers.

## Downstream consumers parked alongside

- `examples/gpu/bfs` — calls `arts_db_put`; its `add_subdirectory(bfs)` is
  commented out in `examples/gpu/CMakeLists.txt`. Port to core
  acquire/release (or to `arts_put_in_db_from_gpu`) to revive it.

## Reviving

Re-introducing Array DB means re-adding the partial-access substrate (or
reimplementing the wrappers on top of core acquire/release) and restoring the
build wiring. The simplest path is to reimplement `arts_get_from_array_db` /
`arts_put_in_array_db` on the core DB acquire/release path rather than the old
raw RPC.
