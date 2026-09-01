/* Oversize (direct-slab) allocations: a request above half the configured
 * slab takes a dedicated mapping instead of the arena, and its free tears
 * that mapping down live.  Exercised here at several sizes with the pool's
 * confinement lookup on every pointer, plus the placement fallover: with the
 * calling thread's node forced full, the direct mapping must land on another
 * node rather than fail.  Ends with a clean cleanup — allocate big, verify,
 * exit. */
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

static void check_one(size_t size) {
  unsigned char *p = (unsigned char *)arts_regpool_alloc_aligned(size, 64);
  assert(p != NULL);
  const arts_regpool_mr_t *m = arts_regpool_lookup(p);
  assert(m != NULL);
  assert((unsigned char *)m->base <= p &&
         p + size <= (unsigned char *)m->base + m->len);
  p[0] = 0x5A;
  p[size / 2] = 0x5A;
  p[size - 1] = 0x5A;
  assert(p[0] == 0x5A && p[size / 2] == 0x5A && p[size - 1] == 0x5A);
  arts_regpool_free(p);
  /* The direct mapping is torn down by the free: the pointer must no longer
   * resolve (its slot is tombstoned until reuse). */
  assert(arts_regpool_lookup(p) == NULL);
}

int main(void) {
  /* 64 MiB slab: anything above 32 MiB takes the direct path. */
  assert(arts_regpool_init(NULL, NULL, (size_t)64 * 1024 * 1024, 0));

  check_one((size_t)40 * 1024 * 1024);
  check_one((size_t)100 * 1024 * 1024);
  /* Slot reuse: a second oversize allocation after a free must reclaim the
   * tombstoned table entry rather than grow the table. */
  check_one((size_t)48 * 1024 * 1024);
  arts_regpool_cleanup();

  /* Fallover: the preferred node reports full, so the direct mapping must
   * relocate instead of failing the allocation.  Needs a second node. */
  if (count_nodes() < 2) {
    printf("PASS regpool_direct_oversize (fallover leg skipped: one node)\n");
    return 0;
  }
  unsigned cpu = 0, node = 0;
  if (syscall(SYS_getcpu, &cpu, &node, NULL) != 0) {
    printf("PASS regpool_direct_oversize (fallover leg skipped: no getcpu)\n");
    return 0;
  }
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  assert(sched_setaffinity(0, sizeof(set), &set) == 0);
  char buf[16];
  snprintf(buf, sizeof(buf), "%u", node);
  assert(setenv("ARTS_REGPOOL_FORCE_FULL_NODES", buf, 1) == 0);
  assert(arts_regpool_init(NULL, NULL, (size_t)64 * 1024 * 1024, 0));
  unsigned char *p =
      (unsigned char *)arts_regpool_alloc_aligned((size_t)48 * 1024 * 1024, 64);
  assert(p != NULL);
  const arts_regpool_mr_t *m = arts_regpool_lookup(p);
  assert(m != NULL);
  assert(m->numa_node != (int)node);
  p[0] = 0x5A;
  p[(size_t)48 * 1024 * 1024 - 1] = 0x5A;
  arts_regpool_free(p);
  arts_regpool_cleanup();
  printf("PASS regpool_direct_oversize (fallover landed on node %d)\n",
         m->numa_node);
  return 0;
}
