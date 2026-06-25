/* SPDX-License-Identifier: Apache-2.0
 *
 * T067 — LOCK lock_compute_next (lock/home.c), the PURE lock-state transition
 *        function over [state_bit | w:31 | r:31].
 *
 * lock_compute_next(cur, op, &grant) is a deterministic pure function: given
 * the current packed lock_state and an op (RW/RO acquire/release), it returns
 * the next packed state and sets *grant to NONE / ONE_RW / ALL_RO.  This test
 * pins its FULL truth table with explicit, hand-derived expected values — no
 * oracle that merely re-implements the function — covering:
 *   - none→rw / none→ro grants
 *   - RW arriving while RO held (w==0 && r>0): parks, bit→RO, no grant
 *   - rw→rw (w>0 acquire): bit UNCHANGED, no grant (held writer serves it)
 *   - D7: new RO grant while RW waiting (RO phase extends)
 *   - RO arriving while RW held: parks, bit→RW
 *   - D6: rw→rw chain on release (w-1>0 grants next writer)
 *   - rw→ro on release (w-1==0 && r>0 drains all RO)
 *   - ro→rw on release (r-1==0 && w>0 grants one writer)
 *   - state_bit normalization to 0 whenever a counter reaches 0.
 *
 * lock_compute_next is non-static (exposed for this test); the file is built
 * standalone by #including lock/home.c.  LOCK-only; self-skips elsewhere.
 *
 * Build: -DARTS_PROTOCOL_LOCK=1 -DARTS_UNIT_STANDALONE_SHIMS.
 */

#include <stdio.h>

#if !defined(ARTS_PROTOCOL_LOCK) || !defined(ARTS_TIMING_EAGER)
int main(void) {
  printf(
      "PASS lock_compute_next: skipped (EAGER LOCK-only; the EAGER lock_state "
      "machine [state_bit|w|r] exists only in the LOCK+EAGER build)\n");
  return 0;
}
#else

#include "arts/coherence/lock/types.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* op + grant codes mirror lock/home.c (kept in sync via the same header
 * defines for the packed-state macros). */
#define T_RW_ACQ 0
#define T_RO_ACQ 1
#define T_RW_REL 2
#define T_RO_REL 3
#define G_NONE 0
#define G_ONE_RW 1
#define G_ALL_RO 2

/* Forward decl of the SUT (defined in lock/home.c, #included below). */
uint64_t lock_compute_next(uint64_t cur, int op, uint32_t *out_grant);

static int g_rc = 0;
static int g_checks = 0;

/* Assert: applying `op` to MAKE_STATE(in_bit,in_w,in_r) yields
 * MAKE_STATE(ex_bit,ex_w,ex_r) and grant==ex_grant. */
static void chk(const char *name, uint32_t in_bit, uint32_t in_w, uint32_t in_r,
                int op, uint32_t ex_bit, uint32_t ex_w, uint32_t ex_r,
                uint32_t ex_grant) {
  uint64_t cur = LOCK_MAKE_STATE(in_bit, in_w, in_r);
  uint32_t grant = 0xffffffff;
  uint64_t next = lock_compute_next(cur, op, &grant);
  uint32_t nw = LOCK_STATE_W(next), nr = LOCK_STATE_R(next),
           nb = LOCK_STATE_BIT(next);
  g_checks++;
  if (nw != ex_w || nr != ex_r || nb != ex_bit || grant != ex_grant) {
    (void)fprintf(
        stderr,
        "FAIL [%s]: in(bit=%u w=%u r=%u) op=%d => out(bit=%u w=%u r=%u "
        "grant=%u) expected(bit=%u w=%u r=%u grant=%u)\n",
        name, in_bit, in_w, in_r, op, nb, nw, nr, grant, ex_bit, ex_w, ex_r,
        ex_grant);
    g_rc = 1;
  }
}

int main(void) {
  const uint32_t RW = LOCK_PHASE_BIT_RW; /* 0 */
  const uint32_t RO = LOCK_PHASE_BIT_RO; /* 1 */

  /* ===== RW_ACQ ===== */
  /* none -> rw : w0 r0 => w1 r0, grant ONE_RW, bit normalized 0 (r==0). */
  chk("rwacq none->rw", 0, 0, 0, T_RW_ACQ, 0, 1, 0, G_ONE_RW);
  /* RW arriving while RO held (w==0 && r>0): parks, bit->RO, no grant. */
  chk("rwacq w0 r1 ->park RO", 0, 0, 1, T_RW_ACQ, RO, 1, 1, G_NONE);
  chk("rwacq w0 r3 ->park RO", RO, 0, 3, T_RW_ACQ, RO, 1, 3, G_NONE);
  /* rw->rw (already a writer participant, w>0): bit UNCHANGED, no grant. */
  chk("rwacq w1 r0 rw->rw", 0, 1, 0, T_RW_ACQ, 0, 2, 0, G_NONE);
  /* w>0 with readers present: bit must stay whatever it was (not forced RO). */
  chk("rwacq w1 r2 bit RW unchanged", RW, 1, 2, T_RW_ACQ, RW, 2, 2, G_NONE);
  chk("rwacq w2 r2 bit RO unchanged", RO, 2, 2, T_RW_ACQ, RO, 3, 2, G_NONE);

  /* ===== RO_ACQ ===== */
  /* none -> ro : w0 r0 => grant ALL_RO; bit set RO but normalized to 0 since
   * w==0 after (one counter zero => bit 0). */
  chk("roacq none->ro", 0, 0, 0, T_RO_ACQ, 0, 0, 1, G_ALL_RO);
  /* ro -> ro : w0 r1 => grant ALL_RO, bit normalized 0 (w==0). */
  chk("roacq ro->ro", RO, 0, 1, T_RO_ACQ, 0, 0, 2, G_ALL_RO);
  /* D7: RW waiting (w>0) but RO phase held (bit RO, r>0): RO extends, grant. */
  chk("roacq D7 extend RO", RO, 1, 1, T_RO_ACQ, RO, 1, 2, G_ALL_RO);
  /* RW held (w>0, bit RW): RO parks, bit->RW, no grant. */
  chk("roacq w1 r0 ->park RW", RW, 1, 0, T_RO_ACQ, RW, 1, 1, G_NONE);
  chk("roacq w2 bit RW park", RW, 2, 0, T_RO_ACQ, RW, 2, 1, G_NONE);

  /* ===== RW_REL ===== */
  /* rw->none : w1 r0 => w0 r0, no grant, bit 0. */
  chk("rwrel rw->none", 0, 1, 0, T_RW_REL, 0, 0, 0, G_NONE);
  /* D6 rw->rw : w2 r0 => w1, grant ONE_RW. */
  chk("rwrel D6 rw->rw", RW, 2, 0, T_RW_REL, 0, 1, 0, G_ONE_RW);
  /* rw->rw with readers waiting still grants the next writer (w-1>0). */
  chk("rwrel D6 rw->rw r>0", RW, 3, 2, T_RW_REL, RW, 2, 2, G_ONE_RW);
  /* rw->ro : w1 r2 => w0, drain ALL_RO, bit->RO normalized 0 (w==0). */
  chk("rwrel rw->ro drain", RW, 1, 2, T_RW_REL, 0, 0, 2, G_ALL_RO);

  /* ===== RO_REL ===== */
  /* ro->none : w0 r1 => w0 r0, no grant. */
  chk("rorel ro->none", 0, 0, 1, T_RO_REL, 0, 0, 0, G_NONE);
  /* ro still held (r-1>0): no grant. */
  chk("rorel ro->ro", RO, 0, 3, T_RO_REL, 0, 0, 2, G_NONE);
  /* ro->rw : last reader leaves (r-1==0) while w>0 => grant ONE_RW, bit->RW
   * normalized 0 (r==0). */
  chk("rorel ro->rw", RO, 2, 1, T_RO_REL, 0, 2, 0, G_ONE_RW);
  /* r-1==0 but w==0 => nothing. */
  chk("rorel r0 w0 nothing", 0, 0, 1, T_RO_REL, 0, 0, 0, G_NONE);

  /* ===== state_bit normalization invariant ===== */
  /* Any transition leaving one counter at 0 must clear bit to 0. */
  chk("norm rwrel leaves w0 bit0", RO, 1, 0, T_RW_REL, 0, 0, 0, G_NONE);
  chk("norm rorel leaves r0 bit0", RO, 0, 1, T_RO_REL, 0, 0, 0, G_NONE);

  if (g_rc != 0) {
    return 1;
  }
  printf("PASS lock_compute_next: %d truth-table transitions "
         "(RW/RO ACQ/REL, D6 rw->rw chain, D7 new-RO-while-RW-waiting, "
         "state_bit normalization) all correct\n",
         g_checks);
  return 0;
}

#ifdef ARTS_UNIT_STANDALONE_SHIMS
void *arts_calloc(size_t nmemb, size_t size) { return calloc(nmemb, size); }
void arts_free(void *ptr) { free(ptr); }
void *arts_malloc(size_t size) { return malloc(size); }

/* lock/home.c is monolithic — its handler bodies (never called by this pure
 * truth-table test) reference the broader runtime.  Satisfy the linker with
 * inert stubs. */
#include "arts/coherence/buffer.h"
#include "arts/utils/shared.h"
unsigned int arts_global_rank_id = 0;
void mark_edt_ready_by_guid(arts_guid_t edt_guid, unsigned int slot) {
  (void)edt_guid;
  (void)slot;
}
void arts_transport_send_async(int rank, char *message, unsigned int length) {
  (void)rank;
  (void)message;
  (void)length;
}
void arts_transport_loopback_post(const void *packet, unsigned int size) {
  (void)packet;
  (void)size;
}
arts_shared_ptr_t arts_db_buf_acquire(struct arts_db_cache_s *cache) {
  (void)cache;
  return NULL;
}
void arts_db_buf_release(arts_shared_ptr_t *h) { (void)h; }
struct arts_db_buffer_s *arts_db_buf_install(struct arts_db_cache_s *cache,
                                             uint64_t new_version,
                                             const void *data_payload,
                                             uint64_t db_size) {
  (void)cache;
  (void)new_version;
  (void)data_payload;
  (void)db_size;
  return NULL;
}
void arts_db_buf_write_inplace(struct arts_db_cache_s *cache, const void *data,
                               uint64_t db_size) {
  (void)cache;
  (void)data;
  (void)db_size;
}
void *arts_shared_get(arts_shared_ptr_t p) {
  (void)p;
  return NULL;
}
void arts_send_db_lock_release_ack(unsigned int releaser_rank,
                                   arts_guid_t db_guid, uint64_t cv) {
  (void)releaser_rank;
  (void)db_guid;
  (void)cv;
}
void arts_send_db_cache_destroy(unsigned int sharer_rank, arts_guid_t db_guid) {
  (void)sharer_rank;
  (void)db_guid;
}
bool arts_route_table_set_destroyed(arts_guid_t key) {
  (void)key;
  return false;
}
#endif

#endif /* ARTS_PROTOCOL_LOCK */

#if defined(ARTS_PROTOCOL_LOCK)
#include "core/coherence/lock/arbiters.c"
#endif
