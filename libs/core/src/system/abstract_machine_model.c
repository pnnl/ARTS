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
#include "arts/system/abstract_machine_model.h"

#include "arts/system/arts_print.h"

#include <hwloc.h>
#include <hwloc/helper.h>

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

/* Resolve hwloc cpuset for package p (root cpuset as fallback). */
static hwloc_const_cpuset_t pkg_cpuset(hwloc_topology_t topology,
                                       hwloc_obj_t pkg) {
  return pkg ? pkg->cpuset : hwloc_get_root_obj(topology)->cpuset;
}

/* Count objects of a type inside a package's cpuset. Returns at least 1. */
static unsigned int count_inside(hwloc_topology_t topology,
                                 hwloc_const_cpuset_t set,
                                 hwloc_obj_type_t type) {
  unsigned int n = hwloc_get_nbobjs_inside_cpuset_by_type(topology, set, type);
  return n ? n : 1;
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

  /* Oversubscription check */
  unsigned int total_pus = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_PU);
  if (config->thread_count > total_pus) {
    ARTS_ERROR("Thread count (%u) exceeds available PUs (%u)",
               config->thread_count, total_pus);
  }

  unsigned int np = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_PACKAGE);
  if (!np) {
    np = 1;
  }

  /* Stride-based PU assignment — walks hwloc tree via indexed access */
  unsigned int stride = config->pin_stride;
  unsigned int p = 0;
  unsigned int c = 0;
  unsigned int u = 0;
  unsigned int offset = 0;
  unsigned int stride_loop = 0;
  unsigned int role_count[ARTS_ROLE_MAX] = {0};

  for (unsigned int t = 0; t < config->thread_count; t++) {
    /* Resolve [p][c][u] coordinates to hwloc objects */
    hwloc_obj_t pkg = hwloc_get_obj_by_type(topology, HWLOC_OBJ_PACKAGE, p);
    hwloc_const_cpuset_t pset = pkg_cpuset(topology, pkg);

    hwloc_obj_t core =
        hwloc_get_obj_inside_cpuset_by_type(topology, pset, HWLOC_OBJ_CORE, c);
    hwloc_const_cpuset_t cset = core ? core->cpuset : pset;

    hwloc_obj_t pu =
        hwloc_get_obj_inside_cpuset_by_type(topology, cset, HWLOC_OBJ_PU, u);

    /* Determine role */
    enum arts_thread_role role;
    if (t < config->worker_thread_count) {
      role = ARTS_ROLE_WORKER;
    } else if (t < config->worker_thread_count + config->sender_thread_count) {
      role = ARTS_ROLE_SENDER;
    } else {
      role = ARTS_ROLE_RECEIVER;
    }

    /* Fill thread_mask_s directly */
    flat[t].id = t;
    flat[t].pu_id = pu ? pu->os_index : t;
    flat[t].core_id = core ? core->os_index : c;
    flat[t].package_id = pkg ? pkg->os_index : p;
    flat[t].numa_domain_id = pu ? find_numa_for_pu(topology, pu) : 0;
    flat[t].role = role;
    flat[t].group_pos = role_count[role]++;
    flat[t].pin = config->pin_threads;

    /* Advance: stride across cores, wrap across packages, then PUs */
    unsigned int nc = count_inside(topology, pset, HWLOC_OBJ_CORE);
    unsigned int nu = count_inside(topology, cset, HWLOC_OBJ_PU);

    c += stride;
    if (c >= nc) {
      p++;
      while (p < np) {
        hwloc_obj_t next =
            hwloc_get_obj_by_type(topology, HWLOC_OBJ_PACKAGE, p);
        if (next && hwloc_get_nbobjs_inside_cpuset_by_type(
                        topology, next->cpuset, HWLOC_OBJ_CORE) > 0) {
          break;
        }
        p++;
      }
      if (p >= np) {
        p = 0;
        if (stride > 1) {
          offset++;
          stride_loop++;
          if (stride_loop == stride) {
            offset = 0;
            u++;
            stride_loop = 0;
          }
        } else {
          u++;
        }
        if (u >= nu) {
          u = 0;
        }
      }
      c = offset;
    }
  }

  /* NUMA domain count for public API */
  unsigned int mem = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_NUMANODE);
  num_numa_domains = mem ? mem : 1;

  hwloc_topology_destroy(topology);
}
