/* finish_event_stress.c — stress recursive cross-node finish-EDT.
 *
 * Each iteration creates a finish-EDT that spawns a depth-DEPTH tree of
 * children, each fanning out to FANOUT children on cycling rank targets.
 * The finish_event must drain to zero on every iteration; if any iteration
 * stalls or fires early, the test hangs (caught by ctest TIMEOUT) or the
 * iter_term EDT observes iter < ITERATIONS at shutdown.
 *
 * The iteration counter lives in paramv (passed through the EDT chain) —
 * no file-scope mutable state. */
#include "arts.h"
#include "arts/utils/atomics.h"

#define ITERATIONS 100
#define FANOUT 3
#define DEPTH 6

/* Forward declaration: iter_term calls start_iter to chain the next
 * iteration; mutual recursion through EDT spawn requires this. */
void start_iter(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]);

void recursive_spawn(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                     arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  /* Children inherit the enclosing finish scope automatically (ARTS
   * runtime semantics).  No explicit finish-scope wiring needed here. */
  unsigned int depth = (unsigned int)paramv[0];
  if (depth == 0) {
    return; /* leaf */
  }
  unsigned int n = arts_get_total_ranks();
  unsigned int my_rank = arts_get_current_rank();
  for (int i = 0; i < FANOUT; i++) {
    arts_edt_hint_t hint = ARTS_EDT_HINT_DEFAULTS;
    /* Cycle through ranks to force cross-node spawn (and oscillation
     * back to caller's rank for FANOUT >= rank_count). */
    hint.rank = (my_rank + 1 + i) % n;
    uint64_t child_paramv[1] = {depth - 1};
    arts_edt_create(recursive_spawn, 1, child_paramv, 0, &hint);
  }
}

/* iter_term: paramv[0] = completed iteration count (1-based after increment).
 * Called when the finish event for one iteration fires. */
void iter_term(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  /* paramv[0] carries the 1-based iteration number for this term. */
  unsigned int iter = (unsigned int)paramv[0];
  if ((iter % 10) == 0) {
    arts_printf("finish_event_stress: iter %u/%u\n", iter, ITERATIONS);
  }
  if (iter < ITERATIONS) {
    uint64_t npv[1] = {(uint64_t)(iter + 1)};
    arts_edt_create(start_iter, 1, npv, 0, NULL);
  } else {
    arts_printf("PASS: %u iterations of recursive cross-node finish-EDT\n",
                ITERATIONS);
    arts_shutdown();
  }
}

/* start_iter: paramv[0] = 1-based iteration number to pass to iter_term. */
void start_iter(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  uint64_t iter = paramv[0];
  arts_guid_t term = arts_edt_create(iter_term, 1, &iter, 1, NULL);
  arts_event_hint_t fh = ARTS_EVENT_HINT_FINISH;
  arts_guid_t fe = arts_event_create(&fh);
  arts_edt_hint_t hint = ARTS_EDT_HINT_DEFAULTS;
  hint.finish_event = fe;
  uint64_t init_paramv[1] = {DEPTH};
  arts_edt_create(recursive_spawn, 1, init_paramv, 0, &hint);
  arts_add_dependence(fe, term, 0, DB_MODE_NULL);
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  /* Start iteration 1. */
  uint64_t pv[1] = {1};
  arts_edt_create(start_iter, 1, pv, 0, NULL);
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
