/* Init fallover + fallback serving for a node the clamp refuses.
 *
 * The diagnostic override (ARTS_REGPOOL_FORCE_FULL_NODES) marks the CALLING
 * thread's node as full before init, so this exercises deterministically —
 * with no real memory pressure — the path a genuinely starved node takes:
 * init must still succeed (best-effort carving falls over to the remaining
 * nodes), and an allocation from the refused node's own thread must be
 * served from another node's arena while still resolving through
 * arts_regpool_lookup (confinement holds across the fallback).
 *
 * Requires >= 2 NUMA nodes and the arena allocator; skipped (pass) below
 * that, and registered only for arena-allocator builds. */
#define _GNU_SOURCE
#include <assert.h>
#include <ctype.h>
#include <dirent.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "arts/memory/regpool.h"

static unsigned count_nodes(void) {
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

int main(void) {
  if (count_nodes() < 2) {
    printf("SKIP regpool_node_fallover: single NUMA node — nothing to fall "
           "over to\n");
    return 0;
  }

  /* Pin to the current CPU so the thread's node cannot drift away from the
   * one the override marks full. */
  unsigned cpu = 0, node = 0;
  if (syscall(SYS_getcpu, &cpu, &node, NULL) != 0) {
    printf("SKIP regpool_node_fallover: getcpu unavailable\n");
    return 0;
  }
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  assert(sched_setaffinity(0, sizeof(set), &set) == 0);

  char buf[16];
  snprintf(buf, sizeof(buf), "%u", node);
  assert(setenv("ARTS_REGPOOL_FORCE_FULL_NODES", buf, 1) == 0);

  /* Init must survive the refused node (best-effort carving). */
  assert(arts_regpool_init(NULL, NULL, (size_t)64 * 1024 * 1024, 0));

  /* An allocation from the refused node's own thread must succeed via the
   * fallback arena AND stay confined to a registered slab — and the slab
   * must belong to a DIFFERENT node, or the override did nothing and this
   * test is passing vacuously. */
  void *p = arts_regpool_alloc_aligned(1 << 20, 64);
  assert(p != NULL);
  const arts_regpool_mr_t *m = arts_regpool_lookup(p);
  assert(m != NULL);
  assert(m->numa_node != (int)node);
  memset(p, 0xA5, 1 << 20);
  arts_regpool_free(p);

  /* A second allocation exercises the memoized fallback fast path. */
  void *q = arts_regpool_alloc_aligned(1 << 16, 64);
  assert(q != NULL);
  assert(arts_regpool_lookup(q) != NULL);
  arts_regpool_free(q);

  arts_regpool_cleanup();
  printf("PASS regpool_node_fallover: init fell over node %u and served its "
         "thread from node %d's arena\n",
         node, m->numa_node);
  return 0;
}
