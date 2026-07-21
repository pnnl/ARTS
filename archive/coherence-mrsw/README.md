# MRSW coherence protocol (archived 2026-07-14)

Evicted from the live tree during the memory-model/protocol rebranding
(OCR vs DB-WRF memory-model axis; RCU / RWLOCK protocol axis).

MRSW (Multi-Reader Single-Writer) implemented the OCR memory model with a
single global writer serialized at a time — a strict subset of the RCU
(formerly MRNEW) protocol's node-exclusive parallelism. It contributed no
distinct point to the evaluation matrix and is not used in the paper, so it
was archived rather than maintained.

Contents:
- `src/mrsw/` — protocol TU (was `libs/src/core/coherence/mrsw/`)
- `include/mrsw/` — protocol types (was `libs/include/internal/arts/coherence/mrsw/`)
- `tests/` — the 12 MRSW-gated tests (were `tests/unit/mrsw_*.c`,
  `tests/ocr/mrsw_eq_mrnew_results.c`)

To resurrect: restore the directories, re-add the `ARTS_PROTOCOL_MRSW`
selection branch in the root `CMakeLists.txt` / `libs/src/core/CMakeLists.txt`,
re-add the `|| defined(ARTS_PROTOCOL_MRSW)` guards in
`coherence/coherence.h` / `coherence/handlers.h` / `coherence/types.h`
(see the commit that removed them), and re-register the tests.
