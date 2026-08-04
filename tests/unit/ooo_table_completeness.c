/* SPDX-License-Identifier: Apache-2.0
 *
 * ooo_table_completeness — per-config g_ooo_table[] handler-wiring audit.
 *
 * Invariants under test (ooo.c g_ooo_table + ooo.h enum arts_ooo_kind):
 *
 *  (1) NO NULL SLOT.  Every index in [0, OOO_KIND_COUNT) must map to a
 *      non-NULL handler.  A config that adds a kind to the enum but forgets to
 *      wire it in the initializer leaves an implicit NULL function pointer →
 *      call-through-NULL on the first defer of that kind, with no diagnostic
 *      (census 16-ooo.md §6 LOW).  This test makes that a hard, deterministic
 *      failure at run time (and the per-slot mapping below makes it a
 * build-time symbol-resolution failure too, since each expected handler is
 * named).
 *
 *  (2) EXACT PER-CONFIG KIND SET.  The model-agnostic kinds are always present;
 *      each build's OOO_DB_* arm wires only that protocol's coherence kinds
 *      (HOME vs OWNER vs WRF_VAL vs EXCL).  The test asserts each named enum maps
 *      to its expected handler (the OOO_<NAME> == arts_handler_<name> naming
 *      invariant) and that OOO_KIND_COUNT equals the count of kinds that
 *      protocol defines — pinning the per-config table shape.
 *
 * The g_ooo_table initializer references handler symbols by name, so this TU
 * provides each as a distinct, identifiable stub; the test then checks that the
 * table entry for a given kind is exactly that stub.  Standalone — #includes
 * ooo.c, no ARTS runtime, no threads.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arts/gas/route_table.h"
#include "arts/ooo.h"
#include "arts/utils/shared.h"

/* Distinct stub bodies — each must be a unique address so the per-slot mapping
 * check is meaningful (the compiler must not merge identical-body functions).
 * We tag each with its own static int read in the body to defeat ICF. */
#define MK_HANDLER(name)                                                       \
  static volatile int g_tag_##name;                                            \
  void name(void *i, void *a) {                                                \
    (void)i;                                                                   \
    (void)a;                                                                   \
    g_tag_##name++;                                                            \
  }

MK_HANDLER(arts_handler_event_satisfy_slot)
MK_HANDLER(arts_handler_edt_satisfy_slot)
MK_HANDLER(arts_handler_event_add_dependence)
MK_HANDLER(arts_handler_edt_destroy)
MK_HANDLER(arts_handler_event_destroy)
MK_HANDLER(arts_handler_db_destroy)
MK_HANDLER(arts_db_acquire_replay_dep)
MK_HANDLER(arts_handler_db_snapshot_request)
MK_HANDLER(arts_handler_db_publish)
#if defined(ARTS_PROTOCOL_EXCL)
MK_HANDLER(arts_handler_db_excl_request)
#ifdef ARTS_RELEASE_PURGE
MK_HANDLER(arts_handler_db_excl_release)
#endif
#elif defined(ARTS_PROTOCOL_INV)
MK_HANDLER(arts_handler_db_grant_request)
MK_HANDLER(arts_handler_db_inv_request)
#ifdef ARTS_WRITE_POLICY_WB
MK_HANDLER(arts_handler_db_inv_redirect)
#endif
#elif defined(ARTS_WRITE_POLICY_WT) || defined(ARTS_WRITE_POLICY_WB)
MK_HANDLER(arts_handler_db_grant_request)
#endif

/* route_table + allocator shims (ooo.c needs them at link time even though the
 * completeness test never reserves a slot or allocates a payload). */
static arts_route_item_t g_slot;
void arts_route_table_reserve_or_lookup(arts_guid_t key,
                                        arts_route_item_t **out) {
  (void)key;
  *out = &g_slot;
}
void *arts_malloc(size_t size) { return malloc(size); }
void *arts_calloc(size_t n, size_t s) { return calloc(n, s); }
void arts_free(void *p) { free(p); }

/* Expose the file-static g_ooo_table by compiling ooo.c into this TU. */
#include "../../libs/src/core/ooo.c"

/* A small reflection table: each ENTRY pins kind -> expected handler symbol. */
struct expect_s {
  ooo_kind_t kind;
  arts_ooo_handler_fn_t fn;
  const char *name;
};

#define E(K, F) {K, F, #K " -> " #F}

static const struct expect_s g_expect[] = {
    E(OOO_EVENT_SATISFY_SLOT, arts_handler_event_satisfy_slot),
    E(OOO_EDT_SATISFY_SLOT, arts_handler_edt_satisfy_slot),
    E(OOO_EVENT_ADD_DEPENDENCE, arts_handler_event_add_dependence),
    E(OOO_EDT_DESTROY, arts_handler_edt_destroy),
    E(OOO_EVENT_DESTROY, arts_handler_event_destroy),
    E(OOO_DB_DESTROY, arts_handler_db_destroy),
#if defined(ARTS_PROTOCOL_EXCL) && defined(ARTS_RELEASE_PURGE)
    E(OOO_DB_ACQUIRE, arts_db_acquire_replay_dep),
    E(OOO_DB_EXCL_REQUEST, arts_handler_db_excl_request),
    E(OOO_DB_EXCL_RELEASE, arts_handler_db_excl_release),
#elif defined(ARTS_PROTOCOL_EXCL) && defined(ARTS_RELEASE_RETAIN)
    E(OOO_DB_ACQUIRE, arts_db_acquire_replay_dep),
    E(OOO_DB_EXCL_REQUEST, arts_handler_db_excl_request),
#elif defined(ARTS_PROTOCOL_INV)
    E(OOO_DB_ACQUIRE, arts_db_acquire_replay_dep),
    E(OOO_DB_GRANT_REQUEST, arts_handler_db_grant_request),
    E(OOO_DB_INV_REQUEST, arts_handler_db_inv_request),
    E(OOO_DB_PUBLISH, arts_handler_db_publish),
#ifdef ARTS_WRITE_POLICY_WB
    E(OOO_DB_INV_REDIRECT, arts_handler_db_inv_redirect),
#endif
#elif defined(ARTS_WRITE_POLICY_WT)
    E(OOO_DB_ACQUIRE, arts_db_acquire_replay_dep),
    E(OOO_DB_SNAPSHOT_REQUEST, arts_handler_db_snapshot_request),
    E(OOO_DB_GRANT_REQUEST, arts_handler_db_grant_request),
    E(OOO_DB_PUBLISH, arts_handler_db_publish),
#elif defined(ARTS_WRITE_POLICY_WB)
    E(OOO_DB_ACQUIRE, arts_db_acquire_replay_dep),
    E(OOO_DB_SNAPSHOT_REQUEST, arts_handler_db_snapshot_request),
    E(OOO_DB_GRANT_REQUEST, arts_handler_db_grant_request),
#elif defined(ARTS_PROTOCOL_WRF_VAL)
    E(OOO_DB_ACQUIRE, arts_db_acquire_replay_dep),
    E(OOO_DB_SNAPSHOT_REQUEST, arts_handler_db_snapshot_request),
    E(OOO_DB_PUBLISH, arts_handler_db_publish),
#endif
};

#define N_EXPECT ((int)(sizeof(g_expect) / sizeof(g_expect[0])))

/* Compile-time pin: the number of kinds this protocol enumerates (excluding
 * the OOO_KIND_COUNT sentinel) MUST equal the explicit expectation list.  If a
 * kind is added/removed in the enum without updating this test (or the table),
 * the build fails here. */
_Static_assert(OOO_KIND_COUNT == N_EXPECT,
               "g_ooo_table per-config kind count drifted from the expected "
               "per-protocol set");

int main(void) {
  const char *cfg =
#if defined(ARTS_PROTOCOL_EXCL) && defined(ARTS_RELEASE_PURGE)
      "EXCL+HOME"
#elif defined(ARTS_PROTOCOL_EXCL) && defined(ARTS_RELEASE_RETAIN)
      "EXCL+OWNER"
#elif defined(ARTS_PROTOCOL_INV) && defined(ARTS_WRITE_POLICY_WB)
      "INV+OWNER"
#elif defined(ARTS_PROTOCOL_INV)
      "INV+HOME"
#elif defined(ARTS_WRITE_POLICY_WT)
      "VAL+HOME"
#elif defined(ARTS_WRITE_POLICY_WB)
      "VAL+OWNER"
#elif defined(ARTS_PROTOCOL_WRF_VAL)
      "WRF_VAL"
#else
      "?"
#endif
      ;

  /* (1) No NULL slot anywhere in [0, OOO_KIND_COUNT). */
  for (int k = 0; k < OOO_KIND_COUNT; k++) {
    if (g_ooo_table[k] == NULL) {
      (void)fprintf(
          stderr,
          "FAIL [%s]: g_ooo_table[%d] is NULL — a kind in the enum is "
          "unwired (call-through-NULL on first defer)\n",
          cfg, k);
      return 1;
    }
  }

  /* (2) Each named kind maps to exactly its expected handler symbol, and the
   * expectation list covers every index [0, OOO_KIND_COUNT) bijectively. */
  int covered[OOO_KIND_COUNT];
  for (int k = 0; k < OOO_KIND_COUNT; k++) {
    covered[k] = 0;
  }
  for (int e = 0; e < N_EXPECT; e++) {
    ooo_kind_t k = g_expect[e].kind;
    if ((int)k < 0 || (int)k >= OOO_KIND_COUNT) {
      (void)fprintf(stderr, "FAIL [%s]: expected kind %d out of range (%s)\n",
                    cfg, (int)k, g_expect[e].name);
      return 1;
    }
    if (g_ooo_table[k] != g_expect[e].fn) {
      (void)fprintf(stderr, "FAIL [%s]: %s — table has %p, expected %p\n", cfg,
                    g_expect[e].name, (void *)g_ooo_table[k],
                    (void *)g_expect[e].fn);
      return 1;
    }
    if (covered[k]) {
      (void)fprintf(stderr, "FAIL [%s]: kind %d expected twice (%s)\n", cfg,
                    (int)k, g_expect[e].name);
      return 1;
    }
    covered[k] = 1;
  }
  for (int k = 0; k < OOO_KIND_COUNT; k++) {
    if (!covered[k]) {
      (void)fprintf(
          stderr,
          "FAIL [%s]: g_ooo_table[%d] populated but not pinned by the "
          "expectation list (untested kind)\n",
          cfg, k);
      return 1;
    }
  }

  printf("PASS ooo_table_completeness [%s]: %d kinds, all wired + pinned\n",
         cfg, OOO_KIND_COUNT);
  return 0;
}
