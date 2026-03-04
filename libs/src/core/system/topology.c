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
#include "arts/system/topology.h"

#include "arts/system/print.h"

#include <hwloc.h>
#include <hwloc/helper.h>
#include <stdlib.h>

unsigned int num_numa_domains = 1;

static const char *role_char(enum arts_thread_role role) {
  switch (role) {
  case ARTS_ROLE_WORKER:
    return "W";
  case ARTS_ROLE_SENDER:
    return "S";
  case ARTS_ROLE_RECEIVER:
    return "R";
  default:
    return "?";
  }
}

void print_mask(struct thread_mask_s *threads, unsigned int num_threads) {
  (void)threads;
  (void)num_threads;
  ARTS_INFO(" Id  Role  GrpPos  NUMA  Pkg  Core    PU  Pin");
  for (unsigned int i = 0; i < num_threads; i++) {
    ARTS_INFO("%3u    %s    %3u     %3u   %3u   %3u   %3u    %1u",
              threads[i].id, role_char(threads[i].role), threads[i].group_pos,
              threads[i].numa_domain_id, threads[i].package_id,
              threads[i].core_id, threads[i].pu_id, threads[i].pin);
  }
}

/* Find the nearest NUMA domain for a PU by checking cpuset membership. */
static unsigned int find_numa_for_pu(hwloc_topology_t topology,
                                     hwloc_obj_t pu_obj) {
  hwloc_obj_t numa = NULL;
  while (
      (numa = hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_NUMANODE, numa))) {
    if (hwloc_bitmap_isset(numa->cpuset, pu_obj->os_index)) {
      return numa->os_index;
    }
  }
  return 0; /* UMA or no NUMA node found */
}

/* Walk up from obj to the nearest ancestor of the given type. */
static hwloc_obj_t ancestor_by_type(hwloc_obj_t obj, hwloc_obj_type_t type) {
  for (hwloc_obj_t cur = obj->parent; cur; cur = cur->parent) {
    if (cur->type == type) {
      return cur;
    }
  }
  return NULL;
}

/*
 * PU entry for the collect-sort-assign algorithm.
 * Stores enough metadata to sort PUs into NUMA-aware, HT-aware order.
 */
struct pu_entry_s {
  unsigned int pu_os_index;
  unsigned int core_os_index;
  unsigned int pkg_os_index;
  unsigned int numa_id;
  unsigned int pu_rank_in_core; /* 0 = first PU in core, 1 = HT sibling, ... */
};

/* Sort by (pu_rank_in_core ASC, numa_id ASC, pu_os_index ASC).
 * This gives: round 0 (one PU per core, NUMA 0 first), round 1 (HT siblings),
 * etc. */
static int pu_entry_compare(const void *a, const void *b) {
  const struct pu_entry_s *pa = (const struct pu_entry_s *)a;
  const struct pu_entry_s *pb = (const struct pu_entry_s *)b;
  if (pa->pu_rank_in_core != pb->pu_rank_in_core) {
    return (pa->pu_rank_in_core < pb->pu_rank_in_core) ? -1 : 1;
  }
  if (pa->numa_id != pb->numa_id) {
    return (pa->numa_id < pb->numa_id) ? -1 : 1;
  }
  if (pa->pu_os_index != pb->pu_os_index) {
    return (pa->pu_os_index < pb->pu_os_index) ? -1 : 1;
  }
  return 0;
}

/* Helper comparator for computing pu_rank_in_core:
 * sort by (core_os_index ASC, pu_os_index ASC). */
static int pu_entry_by_core(const void *a, const void *b) {
  const struct pu_entry_s *pa = (const struct pu_entry_s *)a;
  const struct pu_entry_s *pb = (const struct pu_entry_s *)b;
  if (pa->core_os_index != pb->core_os_index) {
    return (pa->core_os_index < pb->core_os_index) ? -1 : 1;
  }
  if (pa->pu_os_index != pb->pu_os_index) {
    return (pa->pu_os_index < pb->pu_os_index) ? -1 : 1;
  }
  return 0;
}

void get_thread_mask(struct arts_config_s *config, struct thread_mask_s *flat) {
  /* Init hwloc topology */
  hwloc_topology_t topology;
  if (hwloc_topology_init(&topology) < 0) {
    ARTS_ERROR("hwloc_topology_init() failed");
  }
  if (hwloc_topology_load(topology) < 0) {
    ARTS_ERROR("hwloc_topology_load() failed");
  }

  /* Oversubscription check (accounts for PU offset in local multi-node) */
  unsigned int total_pus = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_PU);
  if (total_pus == 0) {
    ARTS_ERROR("hwloc detected 0 PUs — cannot assign thread topology");
    return;
  }
  unsigned int pu_offset = 0;
  if (config->shared_pu_pool) {
    pu_offset = config->my_rank * config->thread_count;
    ARTS_INFO("Local multi-node rank %u: PU offset %u (threads %u)",
              config->my_rank, pu_offset, config->thread_count);
  }
  if (pu_offset + config->thread_count > total_pus) {
    ARTS_ERROR("Rank %u: PU range [%u..%u) exceeds available PUs (%u)",
               config->my_rank, pu_offset, pu_offset + config->thread_count,
               total_pus);
    return;
  }

  /* Phase 1: Collect all PUs with topology metadata */
  struct pu_entry_s *pus = malloc(total_pus * sizeof(*pus));
  for (unsigned int i = 0; i < total_pus; i++) {
    hwloc_obj_t pu = hwloc_get_obj_by_type(topology, HWLOC_OBJ_PU, i);
    hwloc_obj_t core = ancestor_by_type(pu, HWLOC_OBJ_CORE);
    hwloc_obj_t pkg = ancestor_by_type(pu, HWLOC_OBJ_PACKAGE);
    pus[i].pu_os_index = pu->os_index;
    pus[i].core_os_index = core ? core->os_index : 0;
    pus[i].pkg_os_index = pkg ? pkg->os_index : 0;
    pus[i].numa_id = find_numa_for_pu(topology, pu);
    pus[i].pu_rank_in_core = 0;
  }

  /* Phase 2: Compute pu_rank_in_core — sort by core, assign ranks within */
  qsort(pus, total_pus, sizeof(*pus), pu_entry_by_core);
  unsigned int rank = 0;
  for (unsigned int i = 0; i < total_pus; i++) {
    if (i > 0 && pus[i].core_os_index != pus[i - 1].core_os_index) {
      rank = 0;
    }
    pus[i].pu_rank_in_core = rank++;
  }

  /* Phase 3: Sort by (pu_rank_in_core, numa_id, pu_os_index) */
  qsort(pus, total_pus, sizeof(*pus), pu_entry_compare);

  /* Phase 4: Assign threads from sorted PU list (offset for local
     multi-node so each rank gets a disjoint PU slice). */
  unsigned int role_count[ARTS_ROLE_MAX] = {0};
  for (unsigned int t = 0; t < config->thread_count; t++) {
    enum arts_thread_role role;
    if (t < config->worker_thread_count) {
      role = ARTS_ROLE_WORKER;
    } else if (t < config->worker_thread_count + config->sender_thread_count) {
      role = ARTS_ROLE_SENDER;
    } else {
      role = ARTS_ROLE_RECEIVER;
    }

    unsigned int pi = pu_offset + t;
    flat[t].id = t;
    flat[t].pu_id = pus[pi].pu_os_index;
    flat[t].core_id = pus[pi].core_os_index;
    flat[t].package_id = pus[pi].pkg_os_index;
    flat[t].numa_domain_id = pus[pi].numa_id;
    flat[t].role = role;
    flat[t].group_pos = role_count[role]++;
    flat[t].pin = config->pin_threads;
  }

  free(pus);

  /* NUMA domain count for public API */
  unsigned int mem = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_NUMANODE);
  num_numa_domains = mem ? mem : 1;

  hwloc_topology_destroy(topology);
}
