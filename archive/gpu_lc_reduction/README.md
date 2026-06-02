# Archived: multi-GPU LC (Location Consistency) tree/ring reduction

Archived 2026-05-30 during the cleanup refactor. These are the pre-cleanup
HEAD (`4dc9e92`) snapshots of the GPU LC-sync translation unit and its header.
They are preserved here because the **multi-GPU reduction subtree** they
contain is a substantial, modular piece of work that is plausibly useful for
future multi-GPU Location-Consistency reduce-on-release support — but it was
**dead in the live tree** (only reachable from a commented-out
`/* RING IS NOT WORKING */` block inside `gpu_lc_reduce`), so the dead-code pass
removed it. Recoverable from git regardless; archived for discoverability.

## What's the reduction code (the part worth keeping)

In `gpu_lc_sync_functions.cu` (snapshot):

- `gpu_depth_first` / `gpu_depth_first_rec` / `check_max` — build a spanning
  structure over the participating GPUs (from a device-presence `mask`).
- `gpu_ring_reduction` — the ring-order reduction driver (the WIP "not working"
  path).
- `gpu_shadow_reduction_launch` / `gpu_copy_launch` — per-edge launch helpers
  (shadow-buffer reduce + peer copy).
- `gpu_lc_return_db` — pick the surviving replica after a reduction.

Its launch helper `do_reduction_now(gpu_id, sink, src, ...)` lived in
`libs/src/core/gpu/gpu_stream_buffer.cu` (still present there at HEAD; only the
now-dead `do_reduction_now` definition was removed from the live file — recover
it from git if reviving this).

## Status / caveats

- The **ring** path was explicitly WIP ("RING IS NOT WORKING"); the
  depth-first/tree-build helpers are the more complete part.
- Reviving needs a **multi-GPU** box: the P2P (`cudaMemcpyPeer`) reduction paths
  cannot run or be validated on a single-GPU machine.

## Live code

The non-reduction LC-sync logic (version locks, shadow-copy make/merge,
`gpu_lc_reduce` dispatch, the working tree-reduction `gpu_tree_reduction*`)
continues in `libs/src/core/gpu/gpu_lc.cu` (renamed from
`gpu_lc_sync_functions.cu` in the same refactor).
