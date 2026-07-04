/******************************************************************************
** This material was prepared as an account of work sponsored by an agency   **
** of the United States Government.  Neither the United States Government    **
** nor the United States Department of Energy, nor Battelle, nor any of      **
** their employees, nor any jurisdiction or organization that has cooperated **
** in the development of these materials, makes any warranty, express or     **
** implied, or assumes any legal liability or responsibility for the accuracy,*
** completeness, or usefulness or any information, apparatus, product,       **
** software, or process disclosed, or represents that its use would not      **
** infringe privately owned rights.                                          **
**                                                                           **
** Reference herein to any specific commercial product, process, or service  **
** by trade name, trademark, manufacturer, or otherwise does not necessarily **
** constitute or imply its endorsement, recommendation, or favoring by the   **
** United States Government or any agency thereof, or Battelle Memorial      **
** Institute. The views and opinions of authors expressed herein do not      **
** necessarily state or reflect those of the United States Government or     **
** any agency thereof.                                                       **
**                                                                           **
**                      PACIFIC NORTHWEST NATIONAL LABORATORY                **
**                                  operated by                              **
**                                    BATTELLE                               **
**                                     for the                               **
**                      UNITED STATES DEPARTMENT OF ENERGY                   **
**                         under Contract DE-AC05-76RL01830                  **
**                                                                           **
** Copyright 2019 Battelle Memorial Institute                                **
** Licensed under the Apache License, Version 2.0 (the "License");           **
** you may not use this file except in compliance with the License.          **
** You may obtain a copy of the License at                                   **
**                                                                           **
**    https://www.apache.org/licenses/LICENSE-2.0                            **
**                                                                           **
** Unless required by applicable law or agreed to in writing, software       **
** distributed under the License is distributed on an "AS IS" BASIS, WITHOUT **
** WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the  **
** License for the specific language governing permissions and limitations   **
******************************************************************************/

/* Registered slab pool.  See arts/memory/regpool.h for the contract.
 *
 * Layout of a slab:  a large, NUMA-bound anonymous mapping, left unpopulated
 * at creation (see regpool_map_slab) so the NUMA bind below applies before
 * any page is faulted in.  When a fabric domain is supplied the whole mapping
 * is pinned behind one memory-registration handle so a NIC can target any
 * byte with no per-alloc registration.  The mapping is then handed to an
 * allocator arena that is *exclusive* — the arena's heaps allocate only
 * inside it and, on exhaustion, return NULL instead of falling back to
 * unregistered OS memory.  That is the mechanism that keeps every returned
 * pointer inside a registered range.
 *
 * The slab table is grow-only: entries are never moved or removed, only
 * appended (publishing the new count with a release-store) or, for a direct
 * (oversize single-allocation) slab, reclaimed in place and reused. Lookups
 * take an acquire-load of the count and scan without a lock, so a lookup may
 * run concurrently with an allocation that is appending, freeing, or reusing
 * a slab. Records live in a fixed-capacity array so their addresses never
 * move, which is what lets arts_regpool_lookup hand back a stable pointer
 * into the table.
 *
 * A direct slab's whole mapping backs exactly one allocation, so once the
 * caller frees it the mapping has no remaining user and can be torn down
 * immediately instead of waiting for process exit — the same precondition an
 * arena-slab free already relies on: a caller frees only after its refcount
 * has dropped to zero and any in-flight transport operation against that
 * memory has completed, so no lookup can legitimately still be resolving
 * that address by the time the free runs. Reclaiming it in place, rather
 * than only at cleanup, keeps the fixed-capacity table from being exhausted
 * by a workload whose oversize allocations are short-lived and churn through
 * the same handful of live slots.
 *
 * Reclaiming and reusing a table entry without a reader-side lock needs its
 * own per-entry publication discipline, since the entry already counts
 * toward the table's published length: an `is_free` flag on the entry, set
 * under the pool lock, is the gate. A lookup checks the flag with an
 * acquire-load before touching the entry's range, so it either sees the
 * entry fully torn down (flag true, base/len ignored) or fully valid (flag
 * false, base/len safe to compare) — never a torn mix of old and new
 * fields. The lock-holding side upholds that by writing every other field
 * of a reused entry before it clears the flag with a release-store (the same
 * fully-built-then-published order the initial append already uses for the
 * table's count), and by writing nothing else once it sets the flag on a
 * freed entry until reuse claims it.
 */

#include "arts/memory/regpool.h"

#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h> /* posix_memalign / free — non-mimalloc fallback path */
#include <string.h>

#include "arts/system/print.h" /* ARTS_ERROR — fail loudly on confinement loss */

#ifdef ARTS_TRANSPORT_OFI
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#endif

/* ------------------------------------------------------------------------- */
/* The pool is meaningful only with an arena-capable allocator.  Without one   */
/* it degrades to inert stubs so the library still links in a system-malloc    */
/* configuration; init reports failure rather than pretending to register.     */
/* ------------------------------------------------------------------------- */
#ifdef ARTS_MALLOC_MIMALLOC

#include <ctype.h>
#include <dirent.h>
#include <mimalloc.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

/* MPOL_BIND from the kernel's set_mempolicy ABI, declared locally so the
 * module needs neither libnuma nor <linux/mempolicy.h> (which can clash with
 * other headers). */
#ifndef ARTS_MPOL_BIND
#define ARTS_MPOL_BIND 2
#endif

#define REGPOOL_MAX_SLABS 4096u
#define REGPOOL_MAX_NODES 64u /* single-word NUMA bitmask covers node < 64 */
/* Base alignment for every mapping: >= the allocator's arena slice alignment
 * (so a slab is used in full) and huge-page friendly. */
#define REGPOOL_BASE_ALIGN ((size_t)2 * 1024 * 1024)
/* Allocator arena minimum; smaller slabs are rejected by the arena manager. */
#define REGPOOL_MIN_SLAB ((size_t)32 * 1024 * 1024)
/* Payload alignment floor (matches the DB/CXL 64-byte payload invariant). */
#define REGPOOL_ALIGN_FLOOR ((size_t)64)

/* One slab record.  The embedded public view is what lookups return; the extra
 * fields drive free() and growth. */
typedef struct regpool_slab_s {
  arts_regpool_mr_t mr;   /* public: base, len, mr, rkey, numa_node          */
  bool is_direct;         /* true = oversize single-allocation direct slab    */
  mi_arena_id_t arena;    /* backing arena (arena slabs only; NULL if direct) */
  atomic_bool is_free;    /* tombstone: entry torn down, slot open for reuse.
                           * Only ever set on a direct slab (arena slabs are
                           * never freed live). Readers gate on this
                           * (acquire) before touching mr.base/mr.len; the
                           * lock-holding writer clears it (release) only
                           * after every other field of a reused entry is
                           * fully written. */
} regpool_slab_t;

/* --- module state (guarded by g_lock except where marked atomic) --------- */
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static bool g_inited;
static struct fid_domain *g_domain;
static size_t g_slab_bytes;
static unsigned g_numa_nodes;

static regpool_slab_t g_slabs[REGPOOL_MAX_SLABS];
static atomic_size_t g_slab_count; /* release on append, acquire on read */

/* Monotonic memory-registration key source.  When the provider does NOT
 * negotiate FI_MR_PROV_KEY the application must supply a UNIQUE requested_key
 * per fi_mr_reg — registering a second slab with a duplicate key (e.g. 0)
 * fails with FI_ENOKEY.  A unique key is also harmless when the provider DOES
 * assign keys itself (it then ignores requested_key), so a monotone counter is
 * correct for either negotiation. */
static atomic_uint_least64_t g_mr_key_next;

/* Current arena a node's allocations draw from.  A grow publishes a fresh
 * arena here; per-thread heaps notice the change and re-bind. */
static _Atomic(mi_arena_id_t) g_node_arena[REGPOOL_MAX_NODES];

/* Per-thread allocator heap, bound to one exclusive arena.  ARTS worker threads
 * live for the whole process, so a heap created here is never destroyed — its
 * lifetime is the process lifetime, and the arena it draws from outlives it. */
static __thread mi_heap_t *t_heap;
static __thread mi_arena_id_t t_arena;
static __thread int t_node = -1;

/* ------------------------------------------------------------------------- */
/* helpers                                                                     */
/* ------------------------------------------------------------------------- */

static inline size_t align_up_sz(size_t v, size_t a) {
  return (v + a - 1) & ~(a - 1);
}

/* Count NUMA nodes from sysfs; fall back to 1 (single node / no sysfs). */
static unsigned regpool_detect_nodes(void) {
  DIR *d = opendir("/sys/devices/system/node");
  if (d == NULL)
    return 1;
  unsigned n = 0;
  struct dirent *e;
  while ((e = readdir(d)) != NULL) {
    if (strncmp(e->d_name, "node", 4) == 0 &&
        isdigit((unsigned char)e->d_name[4]))
      n++;
  }
  closedir(d);
  return n ? n : 1;
}

/* NUMA node the calling thread currently runs on; fall back to node 0. */
static int regpool_current_node(void) {
  unsigned cpu = 0, node = 0;
  if (syscall(SYS_getcpu, &cpu, &node, NULL) == 0)
    return (int)node;
  return 0;
}

/* Best-effort NUMA binding.  A memory policy only governs *future* faults on
 * the range — it does not migrate pages already resident, and this call
 * passes no MPOL_MF_MOVE (nothing should be resident yet to move).  Callers
 * must therefore invoke this before the range is populated by any means
 * (prefaulting, pinning/registration, or first touch) or the bind is a no-op
 * that silently leaves the range on whatever node absorbed the earlier
 * fault.  A failure (single-node kernel, no permission, unsupported) is not
 * an error — the range simply stays unbound. */
static void regpool_bind_numa(void *base, size_t len, int node) {
  if (g_numa_nodes <= 1 || node < 0 || node >= (int)REGPOOL_MAX_NODES)
    return;
  unsigned long mask = 1UL << (unsigned)node;
  (void)syscall(SYS_mbind, base, len, ARTS_MPOL_BIND, &mask,
                (unsigned long)(sizeof(mask) * 8), 0UL);
}

/* Map `len` bytes aligned to `align`, NUMA-bind, and (when a domain is set)
 * register.  Over-maps by `align` and trims so the base is aligned
 * regardless of what the kernel hands back.  The mapping is left unpopulated
 * (no prefault) so that no page is resident before regpool_bind_numa()
 * applies the policy; registration (which pins, and therefore faults, every
 * page) or the allocator's first touch is what actually populates the
 * range, and by then the policy is already in place.  On success fills
 * *out_* and returns true; on any failure unwinds and returns false. */
static bool regpool_map_slab(int node, size_t len, size_t align, void **out_base,
                             struct fid_mr **out_mr, uint64_t *out_rkey) {
  size_t over = len + align;
  void *raw = mmap(NULL, over, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (raw == MAP_FAILED)
    return false;

  uintptr_t aligned = ((uintptr_t)raw + (align - 1)) & ~(uintptr_t)(align - 1);
  size_t head = (size_t)(aligned - (uintptr_t)raw);
  size_t tail = over - head - len;
  if (head)
    munmap(raw, head);
  if (tail)
    munmap((void *)(aligned + len), tail);
  void *base = (void *)aligned;

  /* Must run before fi_mr_reg() (which faults every page while pinning) or
   * any allocator first-touch — a bind after pages are already resident
   * affects only future faults and silently fails to relocate this range. */
  regpool_bind_numa(base, len, node);

  struct fid_mr *mr = NULL;
  uint64_t rkey = 0;
#ifdef ARTS_TRANSPORT_OFI
  if (g_domain != NULL) {
    uint64_t requested_key =
        atomic_fetch_add_explicit(&g_mr_key_next, 1, memory_order_relaxed);
    int rc = fi_mr_reg(g_domain, base, len,
                       FI_SEND | FI_RECV | FI_READ | FI_WRITE | FI_REMOTE_WRITE,
                       0, requested_key, 0, &mr, NULL);
    if (rc != 0) {
      munmap(base, len);
      return false;
    }
    rkey = fi_mr_key(mr);
  }
#else
  (void)g_domain;
#endif

  *out_base = base;
  *out_mr = mr;
  *out_rkey = rkey;
  return true;
}

/* Append a fully-formed slab record and publish it.  Caller holds g_lock.
 * Returns the record, or NULL if the table is full. */
static regpool_slab_t *regpool_append(void *base, size_t len, struct fid_mr *mr,
                                      uint64_t rkey, int node, bool is_direct,
                                      mi_arena_id_t arena) {
  size_t idx = atomic_load_explicit(&g_slab_count, memory_order_relaxed);
  if (idx >= REGPOOL_MAX_SLABS)
    return NULL;
  regpool_slab_t *s = &g_slabs[idx];
  s->mr.base = base;
  s->mr.len = len;
  s->mr.mr = mr;
  s->mr.rkey = rkey;
  s->mr.numa_node = node;
  s->is_direct = is_direct;
  s->arena = arena;
  atomic_store_explicit(&s->is_free, false, memory_order_relaxed);
  /* Release: the fully-written record must be visible before the count that
   * exposes it to lock-free readers. */
  atomic_store_explicit(&g_slab_count, idx + 1, memory_order_release);
  return s;
}

/* Publish a fully-formed direct-slab record, preferring an already-published
 * tombstoned slot over growing the table.  Caller holds g_lock.  Only the
 * direct-slab path calls this: arena slabs are never freed live, so every
 * tombstoned entry in the table is, and will again be, a direct slab. */
static regpool_slab_t *regpool_publish_direct_locked(void *base, size_t len,
                                                     struct fid_mr *mr,
                                                     uint64_t rkey, int node,
                                                     mi_arena_id_t arena) {
  size_t n = atomic_load_explicit(&g_slab_count, memory_order_relaxed);
  for (size_t i = 0; i < n; i++) {
    regpool_slab_t *s = &g_slabs[i];
    if (!atomic_load_explicit(&s->is_free, memory_order_relaxed))
      continue;
    /* Reused while still tombstoned, so concurrent lookups skip this entry
     * for the whole duration of the rewrite below. */
    s->mr.base = base;
    s->mr.len = len;
    s->mr.mr = mr;
    s->mr.rkey = rkey;
    s->mr.numa_node = node;
    s->is_direct = true;
    s->arena = arena;
    /* Release: the fully-rewritten record must be visible before the slot
     * is handed back to lock-free readers. */
    atomic_store_explicit(&s->is_free, false, memory_order_release);
    return s;
  }
  return regpool_append(base, len, mr, rkey, node, true, arena);
}

/* Create one arena slab for `node` and publish it as the node's current arena.
 * Caller holds g_lock. */
static bool regpool_grow_locked(int node) {
  void *base;
  struct fid_mr *mr;
  uint64_t rkey;
  if (!regpool_map_slab(node, g_slab_bytes, REGPOOL_BASE_ALIGN, &base, &mr,
                        &rkey))
    return false;

  /* Hand the pinned range to an exclusive arena.  is_committed=true (the
   * mapping is accessible — committed by mmap; individual pages are
   * populated later, by registration or by the allocator's first touch),
   * is_pinned=true (the arena must never decommit/purge/reset a registered
   * range), exclusive=true (only heaps created for this arena draw from it —
   * the confinement mechanism). */
  mi_arena_id_t arena = NULL;
  if (!mi_manage_os_memory_ex(base, g_slab_bytes, /*is_committed=*/true,
                              /*is_pinned=*/true, /*is_zero=*/false, node,
                              /*exclusive=*/true, &arena)) {
#ifdef ARTS_TRANSPORT_OFI
    if (mr != NULL)
      fi_close(&mr->fid);
#endif
    munmap(base, g_slab_bytes);
    return false;
  }

  if (regpool_append(base, g_slab_bytes, mr, rkey, node, false, arena) == NULL) {
#ifdef ARTS_TRANSPORT_OFI
    if (mr != NULL)
      fi_close(&mr->fid);
#endif
    /* Arena metadata now references this range; the OS reclaims it at exit. */
    return false;
  }
  atomic_store_explicit(&g_node_arena[node], arena, memory_order_release);
  return true;
}

/* Oversize path: a dedicated mapping that backs exactly one allocation, kept in
 * the same table so lookups resolve it.  Bypasses the arena, whose max object
 * size and contiguity would otherwise fragment or reject the request. */
static void *regpool_alloc_direct(size_t size, size_t align, int node) {
  size_t a = align > REGPOOL_BASE_ALIGN ? align : REGPOOL_BASE_ALIGN;
  size_t len = align_up_sz(size, a);
  void *base;
  struct fid_mr *mr;
  uint64_t rkey;

  pthread_mutex_lock(&g_lock);
  if (!regpool_map_slab(node, len, a, &base, &mr, &rkey)) {
    pthread_mutex_unlock(&g_lock);
    return NULL;
  }
  regpool_slab_t *s = regpool_publish_direct_locked(base, len, mr, rkey, node, NULL);
  pthread_mutex_unlock(&g_lock);
  if (s == NULL) {
#ifdef ARTS_TRANSPORT_OFI
    if (mr != NULL)
      fi_close(&mr->fid);
#endif
    munmap(base, len);
    return NULL;
  }
  /* base is aligned to `a` >= requested align, so it satisfies the request. */
  return base;
}

/* Arena path: allocate from the calling thread's heap, re-binding it whenever
 * the node's current arena has advanced (a grow happened), and growing once on
 * exhaustion. */
static void *regpool_alloc_arena(size_t size, size_t align, int node) {
  mi_arena_id_t cur = atomic_load_explicit(&g_node_arena[node],
                                           memory_order_acquire);
  if (cur == NULL)
    return NULL; /* node not initialized */
  if (t_heap == NULL || t_arena != cur || t_node != node) {
    t_heap = mi_heap_new_in_arena(cur);
    t_arena = cur;
    t_node = node;
    if (t_heap == NULL)
      return NULL;
  }

  void *p = mi_heap_malloc_aligned(t_heap, size, align);
  if (p != NULL)
    return p;

  /* Exhausted: grow and retry once with a heap bound to the new arena. */
  if (!arts_regpool_grow(node))
    return NULL;
  cur = atomic_load_explicit(&g_node_arena[node], memory_order_acquire);
  t_heap = mi_heap_new_in_arena(cur);
  t_arena = cur;
  t_node = node;
  if (t_heap == NULL)
    return NULL;
  return mi_heap_malloc_aligned(t_heap, size, align);
}

/* ------------------------------------------------------------------------- */
/* public API                                                                  */
/* ------------------------------------------------------------------------- */

bool arts_regpool_init(struct fid_domain *domain_or_null, size_t slab_bytes,
                       unsigned int numa_nodes) {
  pthread_mutex_lock(&g_lock);
  if (g_inited) {
    pthread_mutex_unlock(&g_lock);
    return false;
  }

  /* The pool drives arena growth off NULL returns from exhausted exclusive
   * arenas — an intended control-flow signal, not a failure.  The allocator
   * would otherwise print each exhaustion as an "out of memory" diagnostic.
   * This option gates the allocator's diagnostic messages generally (error
   * and warning prints in debug builds), not just the exhaustion one;
   * disabling it silences all of them, but it is purely a print gate — it
   * does not affect the allocator's safety aborts on genuine metadata
   * corruption, which are unconditional. */
  mi_option_disable(mi_option_show_errors);

  if (numa_nodes == 0)
    numa_nodes = regpool_detect_nodes();
  if (numa_nodes > REGPOOL_MAX_NODES)
    numa_nodes = REGPOOL_MAX_NODES;

  size_t slab = align_up_sz(slab_bytes, REGPOOL_BASE_ALIGN);
  if (slab < REGPOOL_MIN_SLAB)
    slab = REGPOOL_MIN_SLAB;

  g_domain = domain_or_null;
  g_slab_bytes = slab;
  g_numa_nodes = numa_nodes;
  atomic_store_explicit(&g_slab_count, 0, memory_order_relaxed);
  for (unsigned i = 0; i < REGPOOL_MAX_NODES; i++)
    atomic_store_explicit(&g_node_arena[i], NULL, memory_order_relaxed);
  g_inited = true;

  bool ok = true;
  for (unsigned i = 0; i < numa_nodes && ok; i++)
    ok = regpool_grow_locked((int)i);
  pthread_mutex_unlock(&g_lock);

  if (!ok) {
    arts_regpool_cleanup();
    return false;
  }
  return true;
}

void arts_regpool_cleanup(void) {
  pthread_mutex_lock(&g_lock);
  size_t n = atomic_load_explicit(&g_slab_count, memory_order_acquire);
  for (size_t i = 0; i < n; i++) {
    regpool_slab_t *s = &g_slabs[i];
    /* A tombstoned direct slab was already closed/unmapped by the free that
     * reclaimed it live; its mr/base/len are stale and must not be touched
     * again here. */
    if (atomic_load_explicit(&s->is_free, memory_order_relaxed))
      continue;
#ifdef ARTS_TRANSPORT_OFI
    if (s->mr.mr != NULL)
      fi_close(&s->mr.mr->fid);
#endif
    /* Direct slabs are the pool's own mappings and are unmapped here.  Arena
     * slabs are owned by the allocator's arena registry; the vendored allocator
     * exposes no public arena-unload, so unmapping one out from under it would
     * dangle its metadata.  The OS reclaims those ranges at process exit — the
     * pool's lifetime is the process lifetime. */
    if (s->is_direct)
      munmap(s->mr.base, s->mr.len);
  }
  atomic_store_explicit(&g_slab_count, 0, memory_order_release);
  for (unsigned i = 0; i < REGPOOL_MAX_NODES; i++)
    atomic_store_explicit(&g_node_arena[i], NULL, memory_order_release);
  g_domain = NULL;
  g_slab_bytes = 0;
  g_numa_nodes = 0;
  g_inited = false;
  pthread_mutex_unlock(&g_lock);
}

bool arts_regpool_grow(int numa_node) {
  if (numa_node < 0)
    numa_node = 0;
  pthread_mutex_lock(&g_lock);
  if (!g_inited) {
    pthread_mutex_unlock(&g_lock);
    return false;
  }
  if ((unsigned)numa_node >= g_numa_nodes)
    numa_node = 0;
  bool ok = regpool_grow_locked(numa_node);
  pthread_mutex_unlock(&g_lock);
  return ok;
}

void *arts_regpool_alloc_aligned(size_t size, size_t align) {
  if (size == 0)
    return NULL;
  size_t slab = g_slab_bytes;
  if (slab == 0)
    return NULL; /* pool not initialized */
  if (align < REGPOOL_ALIGN_FLOOR)
    align = REGPOOL_ALIGN_FLOOR;

  int node = regpool_current_node();
  if ((unsigned)node >= g_numa_nodes)
    node = 0;

  /* Oversize requests (> half a slab) take the direct path; the rest draw from
   * the node's arena, which grows once on exhaustion. */
  void *p = (size > slab / 2) ? regpool_alloc_direct(size, align, node)
                              : regpool_alloc_arena(size, align, node);

  /* Fail loudly: an allocation that could not be satisfied even after a grow
   * cannot be papered over — the payload it would back has nowhere to live. */
  if (p == NULL)
    ARTS_ERROR("regpool: could not satisfy %zu-byte allocation (align %zu)",
               size, align);

  /* Confinement guard.  An external arena's contiguity is not contractually
   * guaranteed by the allocator, so a returned pointer that resolves to no
   * registered slab would be an address the NIC cannot reach — a correctness
   * failure, not a soft error. */
  if (arts_regpool_lookup(p) == NULL)
    ARTS_ERROR("regpool: allocation %p (size %zu) escaped all registered slabs",
               p, size);
  return p;
}

void arts_regpool_free(void *p) {
  if (p == NULL)
    return;
  const arts_regpool_mr_t *m = arts_regpool_lookup(p);
  if (m == NULL)
    return; /* not ours (or already reclaimed) */
  regpool_slab_t *s =
      (regpool_slab_t *)((char *)m - offsetof(regpool_slab_t, mr));
  if (!s->is_direct) {
    mi_free(p);
    return;
  }

  /* Oversize direct slab: its mapping backs exactly this one allocation, so
   * this free is the mapping's only remaining user (the same
   * refcount-zero-and-transport-quiesced precondition an arena free already
   * relies on) and it can be torn down right now instead of waiting for
   * process exit. */
  pthread_mutex_lock(&g_lock);
  if (atomic_load_explicit(&s->is_free, memory_order_relaxed)) {
    /* Already reclaimed by a prior call — defend against a duplicate free
     * turning into a double close/unmap. */
    pthread_mutex_unlock(&g_lock);
    return;
  }
#ifdef ARTS_TRANSPORT_OFI
  if (s->mr.mr != NULL)
    fi_close(&s->mr.mr->fid);
#endif
  munmap(s->mr.base, s->mr.len);
  /* Release: publish the tombstone only after the mapping is fully torn
   * down, so no lookup that observes it can resolve into memory that is no
   * longer mapped. */
  atomic_store_explicit(&s->is_free, true, memory_order_release);
  pthread_mutex_unlock(&g_lock);
}

const arts_regpool_mr_t *arts_regpool_lookup(const void *p) {
  size_t n = atomic_load_explicit(&g_slab_count, memory_order_acquire);
  const char *cp = (const char *)p;
  for (size_t i = 0; i < n; i++) {
    const regpool_slab_t *s = &g_slabs[i];
    /* Acquire: pairs with the release-store that publishes either a fresh
     * append or a reused entry, so a "not free" observation here is
     * guaranteed to see that entry's fully-written base/len below. */
    if (atomic_load_explicit(&s->is_free, memory_order_acquire))
      continue;
    const arts_regpool_mr_t *m = &s->mr;
    if (cp >= (const char *)m->base &&
        cp < (const char *)m->base + m->len)
      return m;
  }
  return NULL;
}

#else /* !ARTS_MALLOC_MIMALLOC — no arena allocator: fall back to a plain
       * system aligned allocator instead of an inert (always-failing) stub.
       * Every caller in this configuration still routes through this API
       * and expects working memory back, so init must succeed; what it
       * cannot provide without an arena allocator is slabs, NUMA binding, or
       * fabric registration.  arts_regpool_lookup therefore always misses
       * (no registered range exists to resolve to) and a non-NULL domain is
       * silently ignored — callers that need RDMA-targetable memory require
       * the arena-allocator build; this build only guarantees plain,
       * symmetric alloc/free. */

bool arts_regpool_init(struct fid_domain *domain_or_null, size_t slab_bytes,
                       unsigned int numa_nodes) {
  (void)domain_or_null;
  (void)slab_bytes;
  (void)numa_nodes;
  return true;
}
void arts_regpool_cleanup(void) {}
void *arts_regpool_alloc_aligned(size_t size, size_t align) {
  if (size == 0) {
    return NULL;
  }
  size_t a = align < sizeof(void *) ? sizeof(void *) : align;
  void *p = NULL;
  if (posix_memalign(&p, a, size) != 0) {
    return NULL;
  }
  return p;
}
void arts_regpool_free(void *p) { free(p); }
const arts_regpool_mr_t *arts_regpool_lookup(const void *p) {
  (void)p;
  return NULL;
}
bool arts_regpool_grow(int numa_node) {
  /* No slab concept in this build — honestly report "nothing was grown"
   * rather than claim success for a no-op. */
  (void)numa_node;
  return false;
}

#endif /* ARTS_MALLOC_MIMALLOC */
