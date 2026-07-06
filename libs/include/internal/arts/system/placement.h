/* SPDX-License-Identifier: Apache-2.0 */
#ifndef ARTS_SYSTEM_PLACEMENT_H
#define ARTS_SYSTEM_PLACEMENT_H

#include <stdbool.h>

#include "arts/system/topology.h" /* enum arts_thread_role */

#ifdef __cplusplus
extern "C" {
#endif

/* One processing unit as discovered from the machine (hwloc or a test's
 * synthetic topology).  os_index values follow the OS's numbering: core
 * os_index may REPEAT across packages (per-package core ids), so a physical
 * core is identified by the (pkg_os_index, core_os_index) pair, never by
 * core_os_index alone. */
struct arts_pu_desc_s {
  unsigned int pu_os_index;
  unsigned int core_os_index;
  unsigned int pkg_os_index;
  unsigned int numa_id;
};

/* One placed thread: the PU it pins to plus its role identity. */
struct arts_placement_s {
  unsigned int pu_os_index;
  unsigned int core_os_index;
  unsigned int pkg_os_index;
  unsigned int numa_id;
  enum arts_thread_role role;
  unsigned int group_pos; /* sequential per role */
};

/*
 * arts_placement_compute — pure thread placement.
 *
 * Orders the machine's PUs into base order (SMT round 0 first, NUMA-compact
 * within a round, os_index ascending within a NUMA node) and takes this
 * rank's slice [offset, offset + workers + progress): out[0..workers) are
 * the workers on the slice's compact front, out[workers..) the progress
 * threads on its tail.  The tail CLUSTERS the progress threads on the
 * slice's last locality domain — they serve the fabric device, not any
 * particular worker's cache, so packing them together displaces the fewest
 * worker cores.  group_pos is sequential per role.
 *
 * Deterministic: output depends only on the PU set and the scalar
 * arguments, never on input array order.
 *
 * `pus` is read-only; `total` is the machine PU count; `out` must hold
 * workers + progress entries.  Returns false when the slice does not fit
 * ([offset, offset+W+P) exceeds total) or on allocation failure.
 */
bool arts_placement_compute(const struct arts_pu_desc_s *pus,
                            unsigned int total, unsigned int offset,
                            unsigned int workers, unsigned int progress,
                            struct arts_placement_s *out);

#ifdef __cplusplus
}
#endif

#endif /* ARTS_SYSTEM_PLACEMENT_H */
