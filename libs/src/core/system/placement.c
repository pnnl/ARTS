/* SPDX-License-Identifier: Apache-2.0
 *
 * Pure thread placement (no hwloc, no runtime state) — see placement.h for
 * the contract.  topology.c collects the machine's PUs and calls this; tests
 * drive it with synthetic topologies.
 */
#include "arts/system/placement.h"

#include <stdlib.h>

struct pu_sort_s {
  struct arts_pu_desc_s d;
  unsigned int rank_in_core; /* 0 = first PU of its physical core */
};

/* Physical-core identity: (pkg, core).  core os_index alone is ambiguous —
 * many machines number cores per package, so both sockets carry core 0. */
static int cmp_by_core(const void *a, const void *b) {
  const struct pu_sort_s *pa = (const struct pu_sort_s *)a;
  const struct pu_sort_s *pb = (const struct pu_sort_s *)b;
  if (pa->d.pkg_os_index != pb->d.pkg_os_index) {
    return (pa->d.pkg_os_index < pb->d.pkg_os_index) ? -1 : 1;
  }
  if (pa->d.core_os_index != pb->d.core_os_index) {
    return (pa->d.core_os_index < pb->d.core_os_index) ? -1 : 1;
  }
  if (pa->d.pu_os_index != pb->d.pu_os_index) {
    return (pa->d.pu_os_index < pb->d.pu_os_index) ? -1 : 1;
  }
  return 0;
}

/* Base order: SMT round 0 (one PU per core) before the sibling rounds;
 * NUMA-compact within a round; os_index ascending within a NUMA node. */
static int cmp_base(const void *a, const void *b) {
  const struct pu_sort_s *pa = (const struct pu_sort_s *)a;
  const struct pu_sort_s *pb = (const struct pu_sort_s *)b;
  if (pa->rank_in_core != pb->rank_in_core) {
    return (pa->rank_in_core < pb->rank_in_core) ? -1 : 1;
  }
  if (pa->d.numa_id != pb->d.numa_id) {
    return (pa->d.numa_id < pb->d.numa_id) ? -1 : 1;
  }
  if (pa->d.pu_os_index != pb->d.pu_os_index) {
    return (pa->d.pu_os_index < pb->d.pu_os_index) ? -1 : 1;
  }
  return 0;
}

bool arts_placement_compute(const struct arts_pu_desc_s *pus,
                            unsigned int total, unsigned int offset,
                            unsigned int workers, unsigned int progress,
                            struct arts_placement_s *out) {
  unsigned int nthreads = workers + progress;
  if (nthreads == 0 || offset + nthreads > total) {
    return false;
  }

  struct pu_sort_s *all = malloc(total * sizeof(*all));
  if (all == NULL) {
    return false;
  }
  for (unsigned int i = 0; i < total; i++) {
    all[i].d = pus[i];
    all[i].rank_in_core = 0;
  }

  /* SMT rank per physical core. */
  qsort(all, total, sizeof(*all), cmp_by_core);
  unsigned int r = 0;
  for (unsigned int i = 0; i < total; i++) {
    if (i > 0 && (all[i].d.pkg_os_index != all[i - 1].d.pkg_os_index ||
                  all[i].d.core_os_index != all[i - 1].d.core_os_index)) {
      r = 0;
    }
    all[i].rank_in_core = r++;
  }

  qsort(all, total, sizeof(*all), cmp_base);
  const struct pu_sort_s *slice = all + offset;

  /* Workers pack the compact front; progress threads take the tail, which
   * clusters them together on the slice's last locality domain instead of
   * scattering them among the workers. */
  for (unsigned int t = 0; t < nthreads; t++) {
    out[t].pu_os_index = slice[t].d.pu_os_index;
    out[t].core_os_index = slice[t].d.core_os_index;
    out[t].pkg_os_index = slice[t].d.pkg_os_index;
    out[t].numa_id = slice[t].d.numa_id;
    out[t].role = (t < workers) ? ARTS_ROLE_WORKER : ARTS_ROLE_PROGRESS;
    out[t].group_pos = (t < workers) ? t : t - workers;
  }

  free(all);
  return true;
}
