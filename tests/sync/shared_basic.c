/* SPDX-License-Identifier: Apache-2.0
 *
 * Single-thread invariants for arts_shared_ptr_t: make / copy / release /
 * get and the atomic slot API.  Standalone — compiled against shared.c
 * directly with libc-backed arts_calloc/arts_free shims (see the bottom of
 * this file), so it needs no ARTS runtime.
 */

#include "arts/utils/shared.h"

#include <assert.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>

static _Atomic(int) g_deleted; /* total deleter invocations */
static void *g_last_deleted;   /* object pointer of the most recent delete */

static void counting_deleter(void *obj) {
  atomic_fetch_add_explicit(&g_deleted, 1, memory_order_relaxed);
  g_last_deleted = obj;
}

int main(void) {
  int obj_a = 0xA;
  int obj_b = 0xB;

  /* make: strong=1, get returns the object, deleter not yet run. */
  arts_shared_ptr_t a = arts_shared_make(&obj_a, counting_deleter);
  assert(a != NULL);
  assert(arts_shared_get(a) == &obj_a);
  assert(atomic_load(&g_deleted) == 0);

  /* copy: strong 1->2; releasing one ref does NOT run the deleter. */
  arts_shared_ptr_t a2 = arts_shared_copy(a);
  assert(a2 == a);
  arts_shared_release(&a);
  assert(a == NULL);                    /* release nulls the handle */
  assert(atomic_load(&g_deleted) == 0); /* still one ref alive */

  /* last release runs the deleter exactly once with the right object. */
  arts_shared_release(&a2);
  assert(a2 == NULL);
  assert(atomic_load(&g_deleted) == 1);
  assert(g_last_deleted == &obj_a);

  /* copy/release of NULL are no-ops. */
  assert(arts_shared_copy(NULL) == NULL);
  arts_shared_ptr_t n = NULL;
  arts_shared_release(&n); /* must not crash */

  /* Atomic slot: store transfers the make ref into the slot. */
  arts_atomic_shared_ptr_t slot;
  atomic_store(&slot, (arts_shared_ptr_t)NULL);
  assert(arts_atomic_shared_load(&slot) == NULL); /* empty slot */

  arts_shared_ptr_t b = arts_shared_make(&obj_b, counting_deleter);
  arts_atomic_shared_store(&slot, b); /* slot owns b's ref */

  /* load yields a fresh caller-owned ref; releasing it must NOT destroy
   * the object (the slot still holds a ref). */
  arts_shared_ptr_t local = arts_atomic_shared_load(&slot);
  assert(local == b);
  assert(arts_shared_get(local) == &obj_b);
  arts_shared_release(&local);
  assert(atomic_load(&g_deleted) == 1); /* obj_b still alive in the slot */

  /* exchange to NULL hands the slot's ref back to us; releasing it now
   * destroys obj_b. */
  arts_shared_ptr_t old =
      arts_atomic_shared_exchange(&slot, (arts_shared_ptr_t)NULL);
  assert(old == b);
  arts_shared_release(&old);
  assert(atomic_load(&g_deleted) == 2);
  assert(g_last_deleted == &obj_b);

  /* store over an occupied slot releases the previous occupant. */
  arts_shared_ptr_t c1 = arts_shared_make(&obj_a, counting_deleter);
  arts_shared_ptr_t c2 = arts_shared_make(&obj_b, counting_deleter);
  arts_atomic_shared_store(&slot, c1);
  arts_atomic_shared_store(&slot, c2); /* c1 released here */
  assert(atomic_load(&g_deleted) == 3);
  arts_atomic_shared_store(&slot, (arts_shared_ptr_t)NULL); /* c2 released */
  assert(atomic_load(&g_deleted) == 4);

  /* abandon: cancel an unpublished cb WITHOUT running the deleter — the
   * object stays owned by the caller (install-race loser semantics). */
  int obj_c = 0xC;
  arts_shared_ptr_t aborted = arts_shared_make(&obj_c, counting_deleter);
  arts_shared_abandon(&aborted);
  assert(aborted == NULL);
  assert(atomic_load(&g_deleted) == 4); /* deleter NOT run for obj_c */
  /* The cb must have recycled cleanly: make + fully release another object
   * (would corrupt / miscount on a broken abandon). */
  arts_shared_ptr_t reuse = arts_shared_make(&obj_c, counting_deleter);
  arts_shared_release(&reuse);
  assert(atomic_load(&g_deleted) == 5);
  assert(g_last_deleted == &obj_c);
  arts_shared_ptr_t an = NULL;
  arts_shared_abandon(&an); /* NULL no-op must not crash */

  printf("shared_basic: OK\n");
  return 0;
}

/* ── libc-backed shims so the test links without the ARTS runtime ──────── */
void *arts_calloc(size_t nmemb, size_t size) { return calloc(nmemb, size); }
void arts_free(void *ptr) { free(ptr); }
