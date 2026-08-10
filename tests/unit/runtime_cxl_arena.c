/******************************************************************************
** Copyright 2019 Battelle Memorial Institute
** Licensed under the Apache License, Version 2.0
******************************************************************************/
/// @file runtime_cxl_arena.c
/// @brief Exercises the CXL arena bring-up arm of arts_runtime_node_init and
///        the resulting CXL DB allocation path.
///
/// Under ARTS_USE_CXL, node_init creates the cxl_deque (round-robin one arena
/// per device, or a single static arena), inits cxl_local_lock, and asserts the
/// arena address range is valid; ARTS_CXL_NATIVE has only rank 0 create the
/// shared deque while other ranks attach via arts_cxl_deque_get().  The
/// observable consequence is that CXL DB creation (which carves from that
/// arena) succeeds and returns a usable shared pointer.  We allocate several
/// CXL DBs to walk the round-robin/static arena selection and verify each
/// returns non-NULL and round-trips bytes through the creator + consumer flush.
///
/// config_specific: requires ARTS_USE_CXL (rapid API + CXL device).  Self-skips
/// (prints SKIP, returns 0) when ARTS_USE_CXL is not defined — there is no CXL
/// CI on this box.  The real body uses only public API + ARTS_DB_CXL.

#if !defined(ARTS_USE_CXL)

#include <stdio.h>
int main(void) {
  (void)printf("SKIP runtime_cxl_arena: requires ARTS_USE_CXL\n");
  return 0;
}

#else /* ARTS_USE_CXL */

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

#define N_CXL_DBS 4u
#define ELEMS 16u

void cxl_arena_consumer(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  uint64_t base = paramv[0];
  uint64_t *d = (uint64_t *)depv[0].ptr;
  bool ok = (d != NULL);
  for (unsigned int i = 0; i < ELEMS && ok; i++) {
    if (d[i] != base + i) {
      ok = false;
    }
  }
  if (!ok) {
    (void)fprintf(stderr, "FAIL runtime_cxl_arena: consumer mismatch\n");
    arts_abort(1);
    return;
  }
  arts_printf("  PASS: CXL arena DB (base %lu) round-tripped\n",
              (unsigned long)base);
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== runtime_cxl_arena ===\n");

  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);

  /* Several CXL DBs to walk the arena selection (round-robin per device, or the
   * single static arena). Each must carve a valid, distinct region. */
  for (unsigned int n = 0; n < N_CXL_DBS; n++) {
    void *ptr = NULL;
    arts_guid_t db = arts_db_create(&ptr, ELEMS * sizeof(uint64_t), ARTS_DB_CXL,
                                    ARTS_DB_PROP_NONE, NULL);
    if (ptr == NULL) {
      (void)fprintf(stderr, "FAIL runtime_cxl_arena: create %u NULL ptr\n", n);
      arts_abort(1);
      return;
    }
    uint64_t base = (uint64_t)(0x100u * (n + 1));
    uint64_t *d = (uint64_t *)ptr;
    for (unsigned int i = 0; i < ELEMS; i++) {
      d[i] = base + i;
    }
    arts_db_release(db, DB_MODE_RW);

    uint64_t pv[1] = {base};
    arts_guid_t c =
        arts_edt_create(cxl_arena_consumer, 1, pv, 1,
                        &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
    arts_add_dependence(db, c, 0, DB_MODE_RO);
  }

  arts_event_wait(fe);
  arts_printf("PASS runtime_cxl_arena: %u CXL arena DBs allocated\n",
              N_CXL_DBS);
  arts_shutdown();
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}

#endif /* ARTS_USE_CXL */
