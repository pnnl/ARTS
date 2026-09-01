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

#include <errno.h> /* mmap failure diagnostics */
#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h> /* posix_memalign / free — non-mimalloc fallback path */
#include <string.h>

#include "arts/system/print.h" /* ARTS_ERROR — fail loudly on confinement loss */

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h> /* struct fid_ep for endpoint-bound MRs */

/* Per-node availability estimate over one node's meminfo stream.
 *
 * MemFree alone misjudges a node whose RAM is file cache: those pages are
 * reclaimed on demand by the very fault path that populates a bound
 * mapping, so a cache-heavy node is healthy, not full.  Count the file LRU
 * lists — which exclude unevictable/mlocked pages by construction — minus
 * the pages whose reclaim must first complete writeback, discounted by
 * half: the same haircut the kernel's own MemAvailable applies to page
 * cache.  An anon-heavy node keeps avail ~= MemFree and is still refused
 * by the callers' clamps, which is the guard against faulting a strictly
 * bound range on a node with nothing left to reclaim.
 *
 * Contract: SIZE_MAX when MemFree cannot be read (unknown must not veto
 * growth); MemFree alone when the LRU fields are absent.  Allocator-
 * independent and pure over the stream, so it is testable in isolation. */
size_t arts_regpool_parse_node_avail(FILE *f) {
  unsigned long memfree_kb = 0;
  unsigned long act_kb = 0;
  unsigned long inact_kb = 0;
  unsigned long dirty_kb = 0;
  unsigned long wb_kb = 0;
  unsigned long nfs_kb = 0;
  unsigned long wbtmp_kb = 0;
  bool have_free = false;
  bool have_act = false;
  bool have_inact = false;
  char line[192];
  while (fgets(line, sizeof(line), f) != NULL) {
    unsigned long v;
    if (sscanf(line, "Node %*d MemFree: %lu", &v) == 1) {
      memfree_kb = v;
      have_free = true;
    } else if (sscanf(line, "Node %*d Active(file): %lu", &v) == 1) {
      act_kb = v;
      have_act = true;
    } else if (sscanf(line, "Node %*d Inactive(file): %lu", &v) == 1) {
      inact_kb = v;
      have_inact = true;
    } else if (sscanf(line, "Node %*d Dirty: %lu", &v) == 1) {
      dirty_kb = v;
    } else if (sscanf(line, "Node %*d Writeback: %lu", &v) == 1) {
      wb_kb = v;
    } else if (sscanf(line, "Node %*d NFS_Unstable: %lu", &v) == 1) {
      nfs_kb = v;
    } else if (sscanf(line, "Node %*d WritebackTmp: %lu", &v) == 1) {
      wbtmp_kb = v;
    }
  }
  if (!have_free) {
    return SIZE_MAX;
  }
  size_t avail_kb = memfree_kb;
  if (have_act && have_inact) {
    unsigned long file_kb = act_kb + inact_kb;
    unsigned long inflight_kb = dirty_kb + wb_kb + nfs_kb + wbtmp_kb;
    if (file_kb > inflight_kb) {
      avail_kb += (size_t)(file_kb - inflight_kb) / 2;
    }
  }
  return avail_kb * 1024;
}

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
/* madvise advice that faults a range in and REPORTS failure (Linux 5.14+),
 * declared locally for older toolchain headers. */
#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23
#endif

#define REGPOOL_MAX_SLABS 4096u
#define REGPOOL_MAX_NODES 64u /* single-word NUMA bitmask covers node < 64 */
/* Base alignment for every mapping: >= the allocator's arena slice alignment
 * (so a slab is used in full) and huge-page friendly. */
#define REGPOOL_BASE_ALIGN ((size_t)2 * 1024 * 1024)
/* Allocator arena minimum; smaller slabs are rejected by the arena manager. */
#define REGPOOL_MIN_SLAB ((size_t)32 * 1024 * 1024)
/* Slab-size granule and ceiling, mirroring the vendored allocator's arena
 * geometry: a managed range is trimmed to 32 MiB slices, and one range
 * consumes one global arena-table slot per 16 GiB (a larger range is split
 * into that many sub-arenas, each holding its own slot).  Sizing slabs on
 * the granule loses nothing to trimming, and capping them at exactly the
 * per-slot maximum makes every grow cost exactly one slot — the table, not
 * the machine, is the scarce resource, and the pool's reach is free slots
 * times the cap on any machine.  Drift against the vendored allocator
 * surfaces as trimming waste or sub-arena splitting, both visible in the
 * pool-state dump on an exhausted allocation. */
#define REGPOOL_SLAB_GRANULE ((size_t)32 * 1024 * 1024)
#define REGPOOL_SLAB_CAP ((size_t)16 * 1024 * 1024 * 1024)
/* Payload alignment floor (matches the DB/CXL 64-byte payload invariant). */
#define REGPOOL_ALIGN_FLOOR ((size_t)64)
/* Held back from a node's availability estimate before any bound mapping is
 * sized against it — room for the kernel and concurrent consumers, so a
 * strict-bind populate never races the node to its last page. */
#define REGPOOL_NODE_HEADROOM ((size_t)2 * 1024 * 1024 * 1024)

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
/* Non-NULL selects the endpoint-bound registration discipline
 * (FI_MR_ENDPOINT): every MR is bound to this endpoint and enabled after
 * registration, and the remote key is read only after the enable.  The
 * endpoint must outlive every registration — arts_regpool_unregister exists
 * so the transport can close all MRs before closing it. */
static struct fid_ep *g_ep;
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

/* Per-node grow count (guarded by g_lock).  Drives exponential slab sizing:
 * successive slabs double up to a cap, so a workload whose live footprint is
 * far above the base slab reaches it in O(log) grows instead of consuming an
 * arena-table slot per base-slab-worth of demand (the allocator caps how many
 * arenas a process may register). */
static unsigned g_node_grow_count[REGPOOL_MAX_NODES];

/* One warning per node exhaustion, not one per refused grow: a full node is
 * re-tried by every allocation that prefers it, and each retry would print.
 * Guarded by g_lock (set and cleared only inside a grow). */
static bool g_node_full_warned[REGPOOL_MAX_NODES];

/* Memoized serving node for a node with no arena of its own (refused at
 * init, not yet recovered): -1 = none chosen yet.  Placement is a
 * preference, never a reason to refuse memory that exists, so such a
 * node's threads are served from the fallback's arena on the fast path;
 * the node's own arena is re-checked on every allocation, so a later
 * successful grow reclaims its threads automatically and the memo goes
 * stale unused. */
static _Atomic int g_node_fallback[REGPOOL_MAX_NODES];

/* Diagnostic override: nodes listed (comma-separated) in
 * ARTS_REGPOOL_FORCE_FULL_NODES report zero available bytes, so the
 * refusal and fallover paths are exercisable deterministically without
 * starving a machine.  Parsed once at init, under g_lock. */
static uint64_t g_forced_full;

/* Kernels predating the populate advice report EINVAL; there demand
 * faulting is the only behavior available and the pre-populate guard
 * degrades to the availability clamp alone — as it always was on such
 * kernels.  Latched on first sight (map_slab always runs under g_lock). */
static bool g_populate_unsupported;

/* Per-thread allocator heap, bound to one exclusive arena at a time.  The
 * binding moves on exhaustion (see regpool_thread_bind); the superseded heap
 * is always deleted so its empty pages return to their arena for reuse. */
static __thread mi_heap_t *t_heap;
static __thread mi_arena_id_t t_arena;
static __thread int t_node = -1;

/* Per-(thread, arena) heap cache.  A heap, once created for an arena, is
 * never deleted while the runtime runs: mi_heap_delete migrates live pages
 * and returns all-free pages to the arena, and that page-retirement path
 * races with lock-free cross-thread frees landing on the same pages (the
 * abandon-vs-free window).  A cached live heap keeps owning its pages, so
 * frees from any thread take mimalloc's ordinary supported path, and a later
 * re-bind to the same arena reuses the heap (no stranded blocks, no
 * footprint ratchet — the concerns that motivated deletion — since the heap
 * remains reachable and allocatable). */
#define REGPOOL_THREAD_HEAP_SLOTS 512
typedef struct {
  mi_arena_id_t arena;
  mi_heap_t *heap;
} regpool_theap_slot_t;
static __thread regpool_theap_slot_t t_heap_cache[REGPOOL_THREAD_HEAP_SLOTS];
static __thread unsigned t_heap_cache_count;

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

/* Bytes one NUMA node can still give a bound mapping (see
 * arts_regpool_parse_node_avail for the estimate), or SIZE_MAX when the
 * kernel does not expose it (no sysfs, single-node) — unknown must not
 * veto growth. */
static size_t regpool_node_avail_bytes(int node) {
  if (node >= 0 && node < (int)REGPOOL_MAX_NODES &&
      (g_forced_full & (1ULL << (unsigned)node)) != 0) {
    return 0;
  }
  char path[64];
  snprintf(path, sizeof path, "/sys/devices/system/node/node%d/meminfo", node);
  FILE *f = fopen(path, "r");
  if (f == NULL)
    return SIZE_MAX;
  size_t r = arts_regpool_parse_node_avail(f);
  fclose(f);
  return r;
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
  if (raw == MAP_FAILED) {
    ARTS_WARN("regpool: mmap(%zu MiB) failed: %s", over >> 20,
              strerror(errno));
    return false;
  }

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

  /* Populate every page NOW.  Under the strict bind above, a first-touch
   * fault on an exhausted node is a mempolicy-constrained OOM-kill the
   * process never observes: the allocator hands out addresses against the
   * not-yet-resident range, the cross-node fallback cascade never sees a
   * failure, and the process dies silently at the touch.  Populating at
   * map time narrows that window to slab creation — and the population's
   * RESULT must be honored for even that to hold: it can stop early on a
   * signal (EINTR/EAGAIN) or fail outright (EFAULT and kin), and a
   * partially populated slab would fault its tail later under the strict
   * bind.  Interruptions are retried over the whole range (populated pages
   * are cheap no-ops); a real failure fails the map so the caller can step
   * down or relocate.  What the return can NOT report is bind exhaustion
   * itself — that dies inside the fault path as a constrained OOM — so the
   * availability clamp and its headroom remain the guard against
   * over-sizing; this check covers the reportable failures.  The cost is
   * that a slab commits in full at creation. */
  if (!g_populate_unsupported) {
    for (int tries = 0;; tries++) {
      if (madvise(base, len, MADV_POPULATE_WRITE) == 0) {
        break;
      }
      if (errno == EINVAL) {
        /* Unsupported advice on this kernel — a fresh anonymous mapping
         * admits no other reading of EINVAL.  Not a failure: fall back to
         * demand faulting for the process lifetime. */
        g_populate_unsupported = true;
        ARTS_WARN("regpool: MADV_POPULATE_WRITE unsupported by this kernel "
                  "— slabs fall back to demand faulting");
        break;
      }
      if ((errno != EINTR && errno != EAGAIN) || tries >= 1000) {
        ARTS_WARN("regpool: populate(%zu MiB) failed: %s", len >> 20,
                  strerror(errno));
        munmap(base, len);
        return false;
      }
    }
  }

  struct fid_mr *mr = NULL;
  uint64_t rkey = 0;
  if (g_domain != NULL) {
    uint64_t requested_key =
        atomic_fetch_add_explicit(&g_mr_key_next, 1, memory_order_relaxed);
    int rc = fi_mr_reg(g_domain, base, len,
                       FI_SEND | FI_RECV | FI_READ | FI_WRITE | FI_REMOTE_WRITE,
                       0, requested_key, 0, &mr, NULL);
    if (rc != 0) {
      /* Distinguishable from the mmap failure above: the mapping existed but
       * the fabric refused to pin it — on providers that lock pages this is
       * typically the locked-memory limit (RLIMIT_MEMLOCK), not RAM. */
      ARTS_WARN("regpool: fi_mr_reg(%zu MiB) failed: %s", len >> 20,
                fi_strerror((int)-rc));
      munmap(base, len);
      return false;
    }
    if (g_ep != NULL) {
      /* Endpoint-bound discipline: the region becomes usable only after it
       * is bound to the endpoint and enabled, and with provider-assigned
       * keys the key exists only after the enable. */
      rc = fi_mr_bind(mr, &g_ep->fid, 0);
      if (rc == 0) {
        rc = fi_mr_enable(mr);
      }
      if (rc != 0) {
        ARTS_WARN("regpool: fi_mr_bind/enable(%zu MiB) failed: %s", len >> 20,
                  fi_strerror((int)-rc));
        fi_close(&mr->fid);
        munmap(base, len);
        return false;
      }
    }
    rkey = fi_mr_key(mr);
    if (g_ep != NULL && rkey == FI_KEY_NOTAVAIL) {
      ARTS_ERROR("regpool: MR key unavailable after enable — provider broke "
                 "the key-after-enable contract");
    }
  }

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

/* What the machine can still give a new slab: MemAvailable, with a fixed
 * fraction held back so the pool never races the rest of the process (and
 * the OS) to the last page.  A registration faults every page in, so sizing
 * past this turns a clean refusal into the OOM killer; an unregistered slab
 * is clamped by the same number because availability is a property of the
 * machine, not of whether the range will be registered.  0 on any parse
 * trouble — the caller treats that as "no clamp beyond the doubling
 * itself". */
static size_t regpool_mem_available(void) {
  FILE *f = fopen("/proc/meminfo", "r");
  if (f == NULL)
    return 0;
  char line[128];
  size_t kb = 0;
  while (fgets(line, sizeof(line), f) != NULL) {
    if (sscanf(line, "MemAvailable: %zu kB", &kb) == 1)
      break;
  }
  fclose(f);
  return (kb >> 4) * 15 * 1024; /* 15/16 of it, in bytes */
}

/* Create one arena slab for `node` and publish it as the node's current arena.
 * Caller holds g_lock. */
static bool regpool_grow_locked(int node) {
  void *base;
  struct fid_mr *mr;
  uint64_t rkey;
  /* Exponential slab sizing with a ceiling.  The size doubles from the
   * configured slab — the first grow is exactly one configured slab — up to
   * REGPOOL_SLAB_CAP, then stays there.  The ceiling exists because the
   * allocator's arena table is a bounded GLOBAL process resource, shared
   * with the default heap that backs the runtime's ordinary allocations:
   * one managed range costs one table slot per REGPOOL_SLAB_CAP of length
   * whatever size is offered, so capped grows cost exactly one slot each,
   * and the pool's reach is the table's free slots times the cap on any
   * machine — the table, not a machine-derived constant, is where growth
   * ends, and it ends loudly.
   *
   * The size is further clamped to what the machine has right now,
   * registered or not: an overcommitting kernel happily grants a mapping
   * far beyond physical memory, and a registration faults every page in.
   * A map or registration failure retries at half the size; failure is
   * reported only when even one base slab cannot be obtained.  Every
   * candidate size stays a granule multiple by construction (base and cap
   * are granule-aligned, sizes move by doubling and halving between them,
   * and every descent is floored at the base slab), which
   * regpool_map_slab's tail trim and the allocator's slice geometry both
   * rely on. */
  unsigned grows = g_node_grow_count[node];
  size_t want = g_slab_bytes;
  while (grows-- > 0 && want < REGPOOL_SLAB_CAP)
    want <<= 1;
  if (want > REGPOOL_SLAB_CAP)
    want = REGPOOL_SLAB_CAP;
  {
    size_t avail = regpool_mem_available();
    while (avail != 0 && want > avail && want > g_slab_bytes)
      want >>= 1;
    if (want < g_slab_bytes)
      want = g_slab_bytes;
  }
  /* Under the strict NUMA bind a slab must also fit the NODE, not just the
   * machine: populating a bound range on a genuinely full node risks the
   * mempolicy-constrained OOM killer rather than a clean ENOMEM.  Clamp to
   * the node's available bytes — free pages plus discounted reclaimable
   * file cache, since the populate's fault path reclaims cache on demand
   * (headroom held back for the kernel and concurrent consumers) — so the
   * node's tail is still used, and fail the grow — warned once per
   * exhaustion, cleared when the node grows again — when not even a base
   * slab fits; the caller's cascade then grows another node. */
  {
    size_t node_avail = regpool_node_avail_bytes(node);
    if (node_avail != SIZE_MAX) {
      size_t usable = node_avail > REGPOOL_NODE_HEADROOM
                          ? node_avail - REGPOOL_NODE_HEADROOM
                          : 0;
      while (want > usable && want > g_slab_bytes)
        want >>= 1;
      if (want > usable) {
        if (!g_node_full_warned[node]) {
          g_node_full_warned[node] = true;
          ARTS_WARN("regpool: node %d has %zu MiB available — no room for "
                    "even a %zu MiB slab; growth falls over to the remaining "
                    "nodes",
                    node, node_avail >> 20, g_slab_bytes >> 20);
        }
        return false;
      }
    }
  }
  while (!regpool_map_slab(node, want, REGPOOL_BASE_ALIGN, &base, &mr, &rkey)) {
    if (want <= g_slab_bytes)
      return false;
    want >>= 1;
    if (want < g_slab_bytes)
      want = g_slab_bytes;
  }

  /* Hand the pinned range to an exclusive arena.  is_committed=true (the
   * mapping is accessible — committed by mmap; individual pages are
   * populated later, by registration or by the allocator's first touch),
   * is_pinned=true (the arena must never decommit/purge/reset a registered
   * range), exclusive=true (only heaps created for this arena draw from it —
   * the confinement mechanism). */
  /* is_zero=true: the slab is a fresh anonymous mapping, which the kernel
   * guarantees zero-filled, and nothing between map and manage writes into
   * it (registration only pins; the NUMA bind only sets policy).  Declaring
   * this lets the allocator's zeroed-allocation path skip the redundant
   * memset on first-touch blocks and clear only recycled ones. */
  /* No retry on refusal: the range is granule-sized and at most one slot's
   * worth, so nothing about it can be "too big" — the only refusal left is
   * an exhausted global arena table, which no smaller size cures.  That is
   * the pool's genuine end of reach, reported loudly here and fatally at
   * the allocation that finds every node unable to grow. */
  mi_arena_id_t arena = NULL;
  if (!mi_manage_os_memory_ex(base, want, /*is_committed=*/true,
                              /*is_pinned=*/true, /*is_zero=*/true, node,
                              /*exclusive=*/true, &arena)) {
    ARTS_WARN("regpool: allocator refused a %zu MiB slab — global arena "
              "table exhausted; no further growth is possible at any size",
              want >> 20);
    if (mr != NULL)
      fi_close(&mr->fid);
    munmap(base, want);
    return false;
  }

  if (regpool_append(base, want, mr, rkey, node, false, arena) == NULL) {
    if (mr != NULL)
      fi_close(&mr->fid);
    /* Arena metadata now references this range; the OS reclaims it at exit. */
    return false;
  }
  g_node_grow_count[node]++;
  g_node_full_warned[node] = false;
  atomic_store_explicit(&g_node_arena[node], arena, memory_order_release);
  return true;
}

/* Grow only if no slab has been appended since the caller last observed
 * `seen_slabs` — collapses a herd of threads that exhausted the same arena
 * concurrently into a single mapping (the losers re-try allocation against
 * the winner's slab instead of each mapping one of their own). */
static bool regpool_grow_if_unchanged(int node, size_t seen_slabs) {
  pthread_mutex_lock(&g_lock);
  if (!g_inited) {
    pthread_mutex_unlock(&g_lock);
    return false;
  }
  if (atomic_load_explicit(&g_slab_count, memory_order_acquire) != seen_slabs) {
    pthread_mutex_unlock(&g_lock);
    return true; /* someone else already grew — retry allocation first */
  }
  bool ok = regpool_grow_locked(node);
  pthread_mutex_unlock(&g_lock);
  return ok;
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
  /* Placement is a preference here as everywhere, and a bound populate of
   * an oversize mapping on a node with nothing to reclaim is the same
   * constrained-OOM hazard a slab grow is clamped against — so candidates
   * are screened by the same availability estimate (unknown never vetoes),
   * the preferred node first, then the rest, then an unbound mapping; a
   * candidate that passes the screen can still fail the map itself
   * (population), which just moves on to the next. */
  int used_node = node;
  bool mapped = false;
  for (unsigned k = 0; k <= g_numa_nodes && !mapped; k++) {
    int cand;
    if (k == 0) {
      cand = node;
    } else {
      cand = (int)(k - 1);
      if (cand == node)
        continue;
    }
    size_t av = regpool_node_avail_bytes(cand);
    if (av != SIZE_MAX &&
        (av <= REGPOOL_NODE_HEADROOM || av - REGPOOL_NODE_HEADROOM < len))
      continue;
    used_node = cand;
    mapped = regpool_map_slab(cand, len, a, &base, &mr, &rkey);
  }
  if (!mapped) {
    used_node = -1;
    mapped = regpool_map_slab(-1, len, a, &base, &mr, &rkey);
  }
  if (!mapped) {
    pthread_mutex_unlock(&g_lock);
    return NULL;
  }
  regpool_slab_t *s =
      regpool_publish_direct_locked(base, len, mr, rkey, used_node, NULL);
  pthread_mutex_unlock(&g_lock);
  if (s == NULL) {
    if (mr != NULL)
      fi_close(&mr->fid);
    munmap(base, len);
    return NULL;
  }
  /* base is aligned to `a` >= requested align, so it satisfies the request. */
  return base;
}

/* Re-bind the calling thread's heap to `arena` via the per-thread heap
 * cache: switch to the arena's cached heap, creating it on first use.  See
 * the cache's comment for why heaps are never deleted mid-run. */
static bool regpool_thread_bind(mi_arena_id_t arena, int node) {
  mi_heap_t *h = NULL;
  for (unsigned i = 0; i < t_heap_cache_count; i++) {
    if (t_heap_cache[i].arena == arena) {
      h = t_heap_cache[i].heap;
      break;
    }
  }
  if (h == NULL) {
    h = mi_heap_new_in_arena(arena);
    if (h == NULL)
      return false;
    if (t_heap_cache_count < REGPOOL_THREAD_HEAP_SLOTS) {
      t_heap_cache[t_heap_cache_count].arena = arena;
      t_heap_cache[t_heap_cache_count].heap = h;
      t_heap_cache_count++;
    }
    /* Cache overflow leaves the heap uncached but live: correctness is
     * unaffected, a re-bind simply creates another heap. */
  }
  t_heap = h;
  t_arena = arena;
  t_node = node;
  return true;
}

/* Sweep the existing arena slabs newest-first (the newest is the least
 * likely to be fully consumed), re-binding the thread's heap to each
 * candidate and attempting the allocation.  `node_filter` < 0 admits every
 * node's arenas — the locality-fallback pass; NUMA placement is a
 * preference, never a reason to refuse memory that exists.  Skips the arena
 * that already refused this request. */
static void *regpool_sweep_arenas(size_t size, size_t align, bool zero,
                                  int node_filter, mi_arena_id_t refused) {
  size_t n = atomic_load_explicit(&g_slab_count, memory_order_acquire);
  for (size_t i = n; i-- > 0;) {
    regpool_slab_t *s = &g_slabs[i];
    if (s->is_direct || atomic_load_explicit(&s->is_free, memory_order_acquire))
      continue;
    if (s->arena == NULL || s->arena == refused)
      continue;
    if (node_filter >= 0 && s->mr.numa_node != node_filter)
      continue;
    if (!regpool_thread_bind(s->arena, (int)s->mr.numa_node))
      return NULL;
    void *p = zero ? mi_heap_zalloc_aligned(t_heap, size, align)
                   : mi_heap_malloc_aligned(t_heap, size, align);
    if (p != NULL)
      return p;
  }
  return NULL;
}

/* Exhaustion slow path.  A heap can only draw from the single arena it is
 * bound to, so recovery is a re-binding cascade: (1) sweep this node's
 * existing arenas (space freed into an earlier arena is reachable only
 * through a heap bound to it), (2) grow this node, (3) drop the locality
 * preference — sweep every node's arenas, then grow any other node.  Loops
 * until the allocation succeeds or every node's grow fails at the
 * map/registration level; only that is genuine exhaustion.  A transient
 * miss (a peer raced away a fresh slab) re-enters the cascade.
 * `refused` names the one arena that already failed this request (NULL
 * when none was tried — entry from a node with no arena of its own), so
 * the sweeps skip exactly the arena known to be exhausted and no other. */
static void *regpool_alloc_arena_slow(size_t size, size_t align, bool zero,
                                      int node, mi_arena_id_t refused) {
  for (;;) {
    size_t seen = atomic_load_explicit(&g_slab_count, memory_order_acquire);
    void *p = regpool_sweep_arenas(size, align, zero, node, refused);
    if (p != NULL)
      return p;

    if (regpool_grow_if_unchanged(node, seen)) {
      mi_arena_id_t cur =
          atomic_load_explicit(&g_node_arena[node], memory_order_acquire);
      if (cur == NULL) {
        /* "Someone else grew" was another node's slab and this node still
         * has no arena.  A NULL arena id must never reach a heap bind: it
         * addresses the allocator's unmanaged default space, outside every
         * registered slab — the confinement the pool exists to provide.
         * Re-enter the cascade; the fresh slab is found by the sweeps. */
        continue;
      }
      if (!regpool_thread_bind(cur, node)) {
        ARTS_WARN("regpool: heap re-bind failed for node %d", node);
        return NULL;
      }
      p = zero ? mi_heap_zalloc_aligned(t_heap, size, align)
               : mi_heap_malloc_aligned(t_heap, size, align);
      if (p != NULL)
        return p;
      continue; /* raced away — re-enter the cascade */
    }

    /* This node cannot grow: fall back across nodes before failing. */
    p = regpool_sweep_arenas(size, align, zero, -1, refused);
    if (p != NULL)
      return p;
    bool grew = false;
    for (unsigned o = 0; o < g_numa_nodes && !grew; o++) {
      if ((int)o == node)
        continue;
      if (atomic_load_explicit(&g_node_arena[o], memory_order_acquire) == NULL)
        continue;
      grew = regpool_grow_if_unchanged(
          (int)o, atomic_load_explicit(&g_slab_count, memory_order_acquire));
    }
    if (!grew) {
      ARTS_WARN("regpool: no node can grow (%zu-byte alloc, node %d)", size,
                node);
      return NULL;
    }
    /* The grown node's fresh slab is found by the next sweep pass. */
  }
}

/* Arena path: allocate from the calling thread's heap; on exhaustion enter
 * the re-binding cascade above.  A node with no arena of its own (refused
 * at init, not yet recovered) is served from a memoized fallback node's
 * arena on this same fast path — the node's own slot is re-checked every
 * call, so a later successful grow reclaims its threads automatically. */
static void *regpool_alloc_arena(size_t size, size_t align, bool zero,
                                 int node) {
  mi_arena_id_t cur = atomic_load_explicit(&g_node_arena[node],
                                           memory_order_acquire);
  int home = node;
  if (cur == NULL) {
    int fb = atomic_load_explicit(&g_node_fallback[node],
                                  memory_order_acquire);
    if (fb >= 0) {
      cur = atomic_load_explicit(&g_node_arena[fb], memory_order_acquire);
      home = fb;
    }
    if (cur == NULL) {
      for (unsigned o = 0; o < g_numa_nodes; o++) {
        mi_arena_id_t a =
            atomic_load_explicit(&g_node_arena[o], memory_order_acquire);
        if (a != NULL) {
          cur = a;
          home = (int)o;
          atomic_store_explicit(&g_node_fallback[node], (int)o,
                                memory_order_release);
          break;
        }
      }
    }
    if (cur == NULL) {
      /* No node has an arena yet: let the cascade try to grow this one
       * (nothing was tried, so nothing is refused). */
      return regpool_alloc_arena_slow(size, align, zero, node, NULL);
    }
  }
  if (t_heap == NULL || t_arena != cur || t_node != home) {
    if (!regpool_thread_bind(cur, home))
      return NULL;
  }

  void *p = zero ? mi_heap_zalloc_aligned(t_heap, size, align)
                 : mi_heap_malloc_aligned(t_heap, size, align);
  if (p != NULL)
    return p;
  /* The preferred node (not the fallback) leads the cascade, so a starved
   * node is re-probed for growth exactly when serving capacity runs out.
   * A memo that just failed to serve is dropped first: the cascade may
   * settle on a different node, and a dead memo would otherwise re-route
   * every later allocation through this slow path. */
  if (home != node) {
    atomic_store_explicit(&g_node_fallback[node], -1, memory_order_release);
  }
  return regpool_alloc_arena_slow(size, align, zero, node, t_arena);
}

/* ------------------------------------------------------------------------- */
/* public API                                                                  */
/* ------------------------------------------------------------------------- */

bool arts_regpool_init(struct fid_domain *domain_or_null,
                       struct fid_ep *ep_or_null, size_t slab_bytes,
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

  /* The configured slab is carried on the allocator's slice granule, and no
   * single slab exceeds one table slot's worth — see REGPOOL_SLAB_GRANULE /
   * REGPOOL_SLAB_CAP.  A machine that must reach the table's full extent
   * with fewer ladder steps raises the configured slab, not the cap. */
  size_t slab = align_up_sz(slab_bytes, REGPOOL_SLAB_GRANULE);
  if (slab < REGPOOL_MIN_SLAB)
    slab = REGPOOL_MIN_SLAB;
  if (slab > REGPOOL_SLAB_CAP)
    slab = REGPOOL_SLAB_CAP;

  g_domain = domain_or_null;
  g_ep = ep_or_null;
  g_slab_bytes = slab;
  g_numa_nodes = numa_nodes;
  atomic_store_explicit(&g_slab_count, 0, memory_order_relaxed);
  for (unsigned i = 0; i < REGPOOL_MAX_NODES; i++) {
    atomic_store_explicit(&g_node_arena[i], NULL, memory_order_relaxed);
    atomic_store_explicit(&g_node_fallback[i], -1, memory_order_relaxed);
  }
  g_forced_full = 0;
  {
    const char *ff = getenv("ARTS_REGPOOL_FORCE_FULL_NODES");
    if (ff != NULL && ff[0] != '\0') {
      const char *p = ff;
      while (*p != '\0') {
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p) {
          /* A diagnostic knob's whole value is determinism: a silently
           * dropped token would make its absence look like a pass. */
          ARTS_WARN("regpool: unparsable node list token ignored: \"%s\"", p);
          break;
        }
        if (v >= 0 && v < (long)REGPOOL_MAX_NODES) {
          g_forced_full |= 1ULL << (unsigned)v;
        } else {
          ARTS_WARN("regpool: node %ld out of range in force-full list", v);
        }
        if (*end != ',')
          break;
        p = end + 1;
      }
      if (g_forced_full != 0)
        ARTS_WARN("regpool: diagnostic override — nodes mask 0x%llx treated "
                  "as full",
                  (unsigned long long)g_forced_full);
    }
  }
  g_inited = true;

  /* Per-node carving is best-effort: every node is tried (no short-circuit
   * — a refused node must not shadow the ones after it), a refused node is
   * left with a NULL arena (its threads are served through the fallback
   * path and it recovers on demand), and only ZERO carved nodes is an init
   * failure. */
  bool any = false;
  for (unsigned i = 0; i < numa_nodes; i++) {
    if (regpool_grow_locked((int)i))
      any = true;
  }
  pthread_mutex_unlock(&g_lock);

  if (!any) {
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
    if (s->mr.mr != NULL)
      fi_close(&s->mr.mr->fid);
    /* Direct slabs are the pool's own mappings and are unmapped here.  Arena
     * slabs are owned by the allocator's arena registry; the vendored allocator
     * exposes no public arena-unload, so unmapping one out from under it would
     * dangle its metadata.  The OS reclaims those ranges at process exit — the
     * pool's lifetime is the process lifetime. */
    if (s->is_direct)
      munmap(s->mr.base, s->mr.len);
  }
  atomic_store_explicit(&g_slab_count, 0, memory_order_release);
  for (unsigned i = 0; i < REGPOOL_MAX_NODES; i++) {
    atomic_store_explicit(&g_node_arena[i], NULL, memory_order_release);
    atomic_store_explicit(&g_node_fallback[i], -1, memory_order_release);
    g_node_grow_count[i] = 0;
    g_node_full_warned[i] = false;
  }
  g_forced_full = 0;
  g_domain = NULL;
  g_ep = NULL;
  g_slab_bytes = 0;
  g_numa_nodes = 0;
  g_inited = false;
  pthread_mutex_unlock(&g_lock);
}

void arts_regpool_unregister(void) {
  pthread_mutex_lock(&g_lock);
  size_t n = atomic_load_explicit(&g_slab_count, memory_order_acquire);
  size_t closed = 0;
  for (size_t i = 0; i < n; i++) {
    regpool_slab_t *s = &g_slabs[i];
    if (atomic_load_explicit(&s->is_free, memory_order_relaxed)) {
      continue;
    }
    if (s->mr.mr != NULL) {
      int rc = fi_close(&s->mr.mr->fid);
      if (rc != 0) {
        ARTS_WARN("regpool: unregister fi_close(mr) failed: %s",
                  fi_strerror(-rc));
      }
      s->mr.mr = NULL;
      s->mr.rkey = 0;
      closed++;
    }
  }
  /* Detach from the fabric: a slab mapped after this point registers
   * nothing (NULL-domain behavior), rather than touching a domain or
   * endpoint the transport is about to close. */
  g_domain = NULL;
  g_ep = NULL;
  pthread_mutex_unlock(&g_lock);
  if (closed != 0) {
    ARTS_INFO("regpool: closed %zu slab registrations ahead of endpoint "
              "teardown",
              closed);
  }
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

static void *regpool_alloc_common(size_t size, size_t align, bool zero) {
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

  /* Oversize requests (> half a slab) take the direct path; the rest draw
   * from the node's arena, which grows on exhaustion.  The direct path is a
   * fresh anonymous mapping and therefore already zero-filled — a zeroed
   * request needs no extra work there. */
  void *p = (size > slab / 2) ? regpool_alloc_direct(size, align, node)
                              : regpool_alloc_arena(size, align, zero, node);

  /* Fail loudly: an allocation that could not be satisfied even after a grow
   * cannot be papered over — the payload it would back has nowhere to live.
   * Dump the pool's shape first so exhaustion is distinguishable from an
   * allocator-path defect in the field. */
  if (p == NULL) {
    size_t n = atomic_load_explicit(&g_slab_count, memory_order_acquire);
    size_t total = 0, node_total = 0;
    unsigned node_slabs = 0;
    for (size_t i = 0; i < n; i++) {
      regpool_slab_t *s = &g_slabs[i];
      if (atomic_load_explicit(&s->is_free, memory_order_acquire))
        continue;
      total += s->mr.len;
      if (!s->is_direct && s->mr.numa_node == node) {
        node_total += s->mr.len;
        node_slabs++;
      }
    }
    ARTS_WARN("regpool state: slabs=%zu total=%zu MiB; node %d: arenas=%u "
              "(%zu MiB, %u grows); heap=%s",
              n, total >> 20, node, node_slabs, node_total >> 20,
              g_node_grow_count[node], t_heap ? "bound" : "NULL");
    ARTS_ERROR("regpool: could not satisfy %zu-byte allocation (align %zu)",
               size, align);
  }

  /* Confinement guard.  An external arena's contiguity is not contractually
   * guaranteed by the allocator, so a returned pointer that resolves to no
   * registered slab would be an address the NIC cannot reach — a correctness
   * failure, not a soft error. */
  if (arts_regpool_lookup(p) == NULL)
    ARTS_ERROR("regpool: allocation %p (size %zu) escaped all registered slabs",
               p, size);
  return p;
}

void *arts_regpool_alloc_aligned(size_t size, size_t align) {
  return regpool_alloc_common(size, align, /*zero=*/false);
}

/* Zeroed variant: the allocator clears only blocks recycled from dirty
 * pages — fresh slab memory is kernel-zeroed and declared so at manage
 * time, so the common create-then-initialize pattern skips a full payload
 * memset (and the page faults it forces) on the caller's critical path. */
void *arts_regpool_zalloc_aligned(size_t size, size_t align) {
  return regpool_alloc_common(size, align, /*zero=*/true);
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
  if (s->mr.mr != NULL)
    fi_close(&s->mr.mr->fid);
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

bool arts_regpool_init(struct fid_domain *domain_or_null,
                       struct fid_ep *ep_or_null, size_t slab_bytes,
                       unsigned int numa_nodes) {
  (void)domain_or_null;
  (void)ep_or_null;
  (void)slab_bytes;
  (void)numa_nodes;
  return true;
}
void arts_regpool_cleanup(void) {}
void arts_regpool_unregister(void) {}
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
void *arts_regpool_zalloc_aligned(size_t size, size_t align) {
  void *p = arts_regpool_alloc_aligned(size, align);
  if (p != NULL)
    memset(p, 0, size);
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
