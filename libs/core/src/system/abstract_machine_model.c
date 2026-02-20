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
#define GNU_SOURCE
#include "arts/system/abstract_machine_model.h"
#include "arts/utils/malloc.h"

#include "arts/runtime/globals.h"
#include "arts/runtime/runtime.h"
#include "arts/system/arts_print.h"
#include "arts/system/threads.h"

#ifdef USE_HWLOC
#include <pthread.h>
#include <sched.h>
#endif

unsigned int num_numa_domains = 1;

enum abstractGroupId {
  ABSTRACT_WORKER = 0,
  ABSTRACT_INBOUND,
  ABSTRACT_OUTBOUND,
  ABSTRACT_MAX
};

void set_thread_mask(struct thread_mask_s *thread_mask, struct unit_mask_s *unit_mask,
                   struct unit_thread_s *unit_thread) {
  thread_mask->numa_domain_id = unit_mask->numa_domain_id;
  thread_mask->core_id = unit_mask->core_id;
  thread_mask->unit_id = unit_mask->unit_id;
  thread_mask->on = unit_mask->on;
#ifdef USE_HWLOC
  thread_mask->core_info = unit_mask->core_info;
  unit_mask->core_info.cpuset = NULL;
#else
  thread_mask->core_info = unit_mask->core_info;
#endif

  thread_mask->id = unit_thread->id;
  thread_mask->group_id = unit_thread->group_id;
  thread_mask->group_pos = unit_thread->group_pos;
  thread_mask->worker = unit_thread->worker;
  thread_mask->network_send = unit_thread->network_send;
  thread_mask->network_receive = unit_thread->network_receive;
  thread_mask->pin = unit_thread->pin;
}

#ifdef USE_HWLOC
#ifndef __APPLE__
static void fill_linux_cpu_set(hwloc_bitmap_t hwloc_set, cpu_set_t *linux_set) {
  CPU_ZERO(linux_set);
  int cpu = hwloc_bitmap_first(hwloc_set);
  while (cpu != -1) {
    if (cpu < CPU_SETSIZE) {
      CPU_SET(cpu, linux_set);
}
    cpu = hwloc_bitmap_next(hwloc_set, cpu);
  }
}
#endif
#endif

void add_a_thread(struct unit_mask_s *mask, bool work_on, bool network_out_on,
                bool network_in_on, unsigned int group_id, unsigned int group_pos,
                bool pin) {
  struct unit_thread_s *next;
  mask->threads++;
  if (mask->list_head == NULL) {
    mask->list_tail = mask->list_head =
        (struct unit_thread_s *)arts_malloc(sizeof(struct unit_thread_s));
    next = mask->list_head;
  } else {
    next = mask->list_tail;
    next->next = (struct unit_thread_s *)arts_malloc(sizeof(struct unit_thread_s));
    next = next->next;
    mask->list_tail = next;
  }

  next->worker = work_on;
  next->network_send = network_out_on;
  next->network_receive = network_in_on;
  next->group_id = group_id;
  next->group_pos = group_pos;
  next->pin = pin;
  next->next = NULL;
  next->id = mask->core_id;
}

#ifdef USE_HWLOC

hwloc_topology_t topology;

void init_topology() {
  hwloc_topology_init(&topology);
#ifndef USE_HWLOC_V2
  hwloc_topology_set_flags(topology, HWLOC_TOPOLOGY_FLAG_IO_BRIDGES);
#endif
  hwloc_topology_load(topology);
}

unsigned int get_number_of_type(hwloc_topology_t topology, hwloc_obj_t obj,
                             hwloc_obj_type_t type) {
  unsigned int count = 0;
  if (obj->type == type) {
    count = 1;
  } else {
    unsigned int i;
    for (i = 0; i < obj->arity; i++) {
      count += get_number_of_type(topology, obj->children[i], type);
}
  }
  return count;
}

void arts_abstract_machine_model_pin_thread(struct arts_core_info_s *core_info) {
  if (!core_info) {
    return;
}
#ifndef __APPLE__
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t),
                         &core_info->linux_cpu_set);
#endif
}

void init_numa_domain_units(hwloc_topology_t topology, hwloc_obj_t obj,
                      hwloc_obj_t numa_domain, unsigned int numa_domain_id,
                      unsigned int *unit_index, struct unit_mask_s *units) {
  if (obj == NULL) {
    return;
  }
  if (obj->type == HWLOC_OBJ_PU) {
    if (obj->parent->type == HWLOC_OBJ_CORE) {
      units[*unit_index].core_id = obj->parent->os_index;
      //            ARTS_INFO("A CORE");
    } else {
      //            ARTS_INFO("NOT A CORE...");
    }
    units[*unit_index].numa_domain_id = numa_domain_id;
    units[*unit_index].unit_id = obj->os_index;
    units[*unit_index].on = 0;

    //        ARTS_INFO("Cluster: %u Unit: %u", cluster->os_index,
    //        obj->os_index);

    units[*unit_index].list_head = NULL;
    units[*unit_index].threads = 0;
    units[*unit_index].core_info.cpuset = hwloc_bitmap_dup(obj->cpuset);
#ifndef __APPLE__
    fill_linux_cpu_set(units[*unit_index].core_info.cpuset,
                    &units[*unit_index].core_info.linux_cpu_set);
#endif
    *unit_index = (*unit_index) + 1;
  } else {
    //        ARTS_INFO("ARITY: %u", obj->arity);
    int i;
    for (i = 0; i < obj->arity; i++) {
      init_numa_domain_units(topology, obj->children[i], numa_domain, numa_domain_id,
                       unit_index, units);
}
  }
}

struct node_mask_s *get_node_mask() {
  struct node_mask_s *node =
      (struct node_mask_s *)arts_malloc(sizeof(struct node_mask_s));
  num_numa_domains = node->num_numa_domains =
      hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_NODE);
  bool is_uma = (num_numa_domains == 0);
  if (is_uma) {
    num_numa_domains = node->num_numa_domains = 1;
}
  node->numa_domain = (struct numa_domain_mask_s *)arts_malloc(sizeof(struct numa_domain_mask_s) *
                                                   node->num_numa_domains);
  unsigned int numa_domain_index = 0;
  unsigned int core_index = 0;
  hwloc_obj_t numa_domain = is_uma ? hwloc_get_root_obj(topology) : NULL;
  hwloc_obj_t core = NULL;
  for (numa_domain_index = 0; numa_domain_index < node->num_numa_domains; numa_domain_index++) {
    if (!is_uma) {
      numa_domain = hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_NODE, numa_domain);
}
    unsigned int numa_domain_id = numa_domain_index;
    if (numa_domain && numa_domain->os_index != HWLOC_UNKNOWN_INDEX) {
      numa_domain_id = numa_domain->os_index;
}
    bool use_global_core_index = false;
#ifdef USE_HWLOC_V2
    if (numa_domain && numa_domain->cpuset) {
      node->numa_domain[numa_domain_index].num_cores =
          hwloc_get_nbobjs_inside_cpuset_by_type(topology, numa_domain->cpuset,
                                                 HWLOC_OBJ_CORE);
    } else {
      node->numa_domain[numa_domain_index].num_cores = 0;
    }
#else
    node->numa_domain[numa_domain_index].num_cores =
        get_number_of_type(topology, numa_domain, HWLOC_OBJ_CORE);
#endif
    if (!node->numa_domain[numa_domain_index].num_cores) {
      node->numa_domain[numa_domain_index].num_cores =
          hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_CORE);
      use_global_core_index = true;
    }
    if (!node->numa_domain[numa_domain_index].num_cores) {
      node->numa_domain[numa_domain_index].num_cores = 1;
      use_global_core_index = true;
    }
    node->numa_domain[numa_domain_index].core = (struct core_mask_s *)arts_malloc(
        sizeof(struct core_mask_s) * node->numa_domain[numa_domain_index].num_cores);
    for (core_index = 0; core_index < node->numa_domain[numa_domain_index].num_cores;
         core_index++) {
      if (use_global_core_index) {
        core = hwloc_get_obj_by_type(topology, HWLOC_OBJ_CORE, core_index);
      } else {
#ifdef USE_HWLOC_V2
        if (numa_domain && numa_domain->cpuset) {
          core = hwloc_get_next_obj_inside_cpuset_by_type(
              topology, numa_domain->cpuset, HWLOC_OBJ_CORE, core);
        } else {
          core = hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_CORE, core);
        }
#else
        core = hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_CORE, core);
#endif
      }
      hwloc_obj_t unit_parent = core ? core : numa_domain;
      node->numa_domain[numa_domain_index].core[core_index].num_units =
          unit_parent ? get_number_of_type(topology, unit_parent, HWLOC_OBJ_PU) : 0;
      if (!node->numa_domain[numa_domain_index].core[core_index].num_units) {
        node->numa_domain[numa_domain_index].core[core_index].num_units =
            hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_PU);
}
      if (!node->numa_domain[numa_domain_index].core[core_index].num_units) {
        node->numa_domain[numa_domain_index].core[core_index].num_units = 1;
}
      node->numa_domain[numa_domain_index].core[core_index].unit =
          (struct unit_mask_s *)arts_malloc(
              sizeof(struct unit_mask_s) *
              node->numa_domain[numa_domain_index].core[core_index].num_units);
      unsigned int unit_index = 0;
      init_numa_domain_units(topology, unit_parent, numa_domain, numa_domain_id, &unit_index,
                       node->numa_domain[numa_domain_index].core[core_index].unit);
    }
  }
  // hwloc_topology_destroy(topology);
  return node;
}

void default_policy(unsigned int number_of_workers, unsigned int number_of_senders,
                   unsigned int number_of_receivers, struct node_mask_s *node,
                   struct arts_config_s *config) {
  unsigned int num_numa_domains = node->num_numa_domains;
  unsigned int num_cores = node->numa_domain[0].num_cores;
  unsigned int num_units = node->numa_domain[0].core[0].num_units;
  //    ARTS_INFO("%d %d %d", num_numa_domains, num_cores, num_units);
  unsigned int cores_per_numa_domain = num_cores * num_units;
  unsigned int core_count = num_numa_domains * num_cores * num_units;
  unsigned int i = 0;
  unsigned int j = 0;
  unsigned int k = 0;
  unsigned int total_threads = 0;
  unsigned int stride = config->pin_stride;
  unsigned int stride_loop = 0;
  unsigned int offset = 0;
  unsigned int network_threads =
      (arts_global_rank_count > 1) * (number_of_receivers + number_of_senders);
  number_of_senders = (arts_global_rank_count > 1) * number_of_senders;
  number_of_receivers = (arts_global_rank_count > 1) * number_of_receivers;
  unsigned int worker_thread_id = 0;
  unsigned int network_out_thread_id = 0;
  unsigned int network_in_thread_id = 0;
  while (total_threads < number_of_workers + network_threads) {
    node->numa_domain[i].core[j].unit[k].on = 1;

    if (total_threads < number_of_workers) {
      add_a_thread(&node->numa_domain[i].core[j].unit[k], 1, 0, 0, ABSTRACT_WORKER,
                 worker_thread_id++, config->pin_threads);
    } else {
      if (total_threads < number_of_workers + number_of_senders) {
        add_a_thread(&node->numa_domain[i].core[j].unit[k], 0, 1, 0, ABSTRACT_OUTBOUND,
                   network_out_thread_id++, config->pin_threads);
      } else if (total_threads <
                 number_of_workers + number_of_receivers + number_of_senders) {
        add_a_thread(&node->numa_domain[i].core[j].unit[k], 0, 0, 1, ABSTRACT_INBOUND,
                   network_in_thread_id++, config->pin_threads);
      }
    }
    total_threads++;
    num_cores = node->numa_domain[i].num_cores;
    num_units = node->numa_domain[i].core[j].num_units;
    j += stride;
    if (j >= num_cores) {
      i++;
      if (i < num_numa_domains) {
        while (node->numa_domain[i].num_cores == 0) {
          i++;
          if (i == num_numa_domains) {
            break;
}
        }
      }
      if (i == num_numa_domains) {
        i = 0;

        if (stride > 1) {
          offset++;
          stride_loop++;
          if (stride_loop == stride) {
            offset = 0;
            k++;
            stride_loop = 0;
          }
        } else {
          k++;
}
        if (k == num_units) {
          k = 0;
        }
      }
      j = offset;
    }
  }
}

unsigned int flatten_mask(struct arts_config_s *config, struct node_mask_s *node,
                         struct thread_mask_s **flat) {
  (void)config;
  unsigned int i;
  unsigned int j;
  unsigned int k;
  unsigned int total;
  unsigned int count = 0;
  for (i = 0; i < node->num_numa_domains; i++) {
    for (j = 0; j < node->numa_domain[i].num_cores; j++) {
      for (k = 0; k < node->numa_domain[i].core[j].num_units; k++) {
        if (node->numa_domain[i].core[j].unit[k].on) {
          count += node->numa_domain[i].core[j].unit[k].threads;
}
      }
    }
  }
  total = count;
  *flat = (struct thread_mask_s *)arts_malloc(sizeof(struct thread_mask_s) * total);
  unsigned int *group_count =
      (unsigned int *)arts_calloc(ABSTRACT_MAX, sizeof(unsigned int));
  count = 0;
  struct unit_thread_s *next;
  for (i = 0; i < node->num_numa_domains; i++) {
    for (j = 0; j < node->numa_domain[i].num_cores; j++) {
      for (k = 0; k < node->numa_domain[i].core[j].num_units; k++) {
        if (node->numa_domain[i].core[j].unit[k].on) {
          next = node->numa_domain[i].core[j].unit[k].list_head;

          while (next != NULL) {
            set_thread_mask(&(*flat)[count], &node->numa_domain[i].core[j].unit[k],
                          next);
            (*flat)[count].group_pos = group_count[next->group_id]++;
            (*flat)[count].id = count;
            ++count;
            struct unit_thread_s *temp = next->next;
            arts_free(next);
            next = temp;
          }
        }
      }
    }
  }
  arts_free(group_count);
  for (i = 0; i < node->num_numa_domains; i++) {
    for (j = 0; j < node->numa_domain[i].num_cores; j++) {
      for (k = 0; k < node->numa_domain[i].core[j].num_units; k++) {
        if (node->numa_domain[i].core[j].unit[k].core_info.cpuset) {
          hwloc_bitmap_free(node->numa_domain[i].core[j].unit[k].core_info.cpuset);
}
      }
      arts_free(node->numa_domain[i].core[j].unit);
    }
    arts_free(node->numa_domain[i].core);
  }
  arts_free(node->numa_domain);
  arts_free(node);
  return total;
}

struct thread_mask_s *get_thread_mask(struct arts_config_s *config) {
  if (config->sender_count > (arts_global_rank_count - 1) * config->ports) {
    config->sender_count = (arts_global_rank_count - 1) * config->ports;
}
  if (config->receiver_count > (arts_global_rank_count - 1) * config->ports) {
    config->receiver_count = (arts_global_rank_count - 1) * config->ports;
}

  unsigned int worker_threads =
      config->thread_count - config->sender_count - config->receiver_count;
  unsigned int total_threads = config->thread_count;

  bool network_on = (arts_global_rank_count > 1);
  struct thread_mask_s *flat;
  init_topology();
  struct node_mask_s *node = get_node_mask();

  default_policy(worker_threads, config->sender_count, config->receiver_count, node,
                config);
  total_threads = flatten_mask(config, node, &flat);

  arts_runtime_node_init(worker_threads, 1, config->sender_count,
                      config->receiver_count, total_threads, 0, config);
  if (config->print_topology) {
    print_mask(flat, total_threads);
}
  return flat;
}

void destroy_thread_mask(struct thread_mask_s *mask) {
  for (unsigned int i = 0; i < arts_node_info.total_thread_count; i++) {
    if (mask[i].core_info.cpuset) {
      hwloc_bitmap_free(mask[i].core_info.cpuset);
}
  }
  if (topology) {
    hwloc_topology_destroy(topology);
    topology = NULL;
  }
  arts_free(mask);
}

void print_topology(struct node_mask_s *node) {
  ARTS_INFO("Node %u", node->num_numa_domains);
  unsigned int i;
  unsigned int j;
  unsigned int k;
  for (i = 0; i < node->num_numa_domains; i++) {
    ARTS_INFO(" Cluster %u", node->numa_domain[i].num_cores);
    for (j = 0; j < node->numa_domain[i].num_cores; j++) {
      ARTS_INFO("  Core %u", node->numa_domain[i].core[j].num_units);
      for (k = 0; k < node->numa_domain[i].core[j].num_units; k++) {
        struct unit_mask_s *unit = &node->numa_domain[i].core[j].unit[k];
        struct unit_thread_s *temp = unit->list_head;
        while (temp != NULL) {
          ARTS_INFO("   Unit %u %u %u %u %u %u %u", temp->id, unit->unit_id,
                    unit->on, temp->worker, temp->network_send,
                    temp->network_receive, unit->core_id);
          temp = temp->next;
        }
      }
    }
  }
}
#else

void arts_abstract_machine_model_pin_thread(struct arts_core_info_s *core_info) {
  arts_pthread_affinity(core_info->cpu_id, true);
}

int artsAffinityFromPthreadValid(unsigned int i, int *validCpus,
                                 unsigned int validCpuCount,
                                 unsigned int num_cores, unsigned int stride) {
  static unsigned int index = 0;
  static unsigned int offset = 0;
  static unsigned int stride_loop = 0;
  static unsigned int count = 0;
  int res = -1;

  if (validCpuCount) {
    if (count == validCpuCount) {
      index = 0;
      offset = 0;
      stride_loop = 0;
      count = 0;
    }

    do {
      res = validCpus[index % num_cores];
      index += stride;
      if (index >= num_cores && stride > 1) {
        stride_loop++;
        offset++;
        if (stride_loop == stride) {
          offset = stride_loop = 0;
        }
        index = offset;
      }
    } while (res == -1);
    count++;
  } else
    ARTS_INFO("Valid set of processors is empty");
  return res;
}

void default_policy(unsigned int number_of_workers, unsigned int number_of_senders,
                   unsigned int number_of_receivers, struct unit_mask_s *flat,
                   unsigned int num_cores, struct arts_config_s *config) {
  unsigned int validCpuCount = 0;
  int *validCpus = arts_valid_pthread_affinity(&validCpuCount);

  unsigned int total_threads = 0;
  unsigned int stride = config->pin_stride;
  unsigned int stride_loop = 0;
  unsigned int i = 0, offset = 0;
  unsigned int network_threads =
      (arts_global_rank_count > 1) * (number_of_receivers + number_of_senders);
  unsigned int networkCores = network_threads * config->cores_per_network_thread;
  unsigned int worker_thread_id = 0;
  unsigned int network_out_thread_id = 0;
  unsigned int network_in_thread_id = 0;

  if (num_cores <= networkCores || validCpuCount <= networkCores)
    ARTS_INFO("Not enough cores. Required cores: %u Total cores: %u "
              "validCpuCount: %u",
              networkCores, num_cores, validCpuCount);
  unsigned int workerCores = num_cores - networkCores;
  int max = -1;
  while (total_threads < number_of_workers) {
    flat[i % workerCores].on = 1;
    int tempAffin =
        artsAffinityFromPthreadValid(i, validCpus, validCpuCount - networkCores,
                                     workerCores, stride); // i % num_cores;
    flat[i % workerCores].core_id = flat[i % workerCores].core_info.cpu_id =
        tempAffin;
    add_a_thread(&flat[i % workerCores], 1, 0, 0, ABSTRACT_WORKER,
               worker_thread_id++, config->pin_threads);
    max = (tempAffin > max) ? tempAffin : max;
    total_threads++;
    //        ARTS_INFO("i: %u -> %u -> %u", i, i%workerCores,
    //        flat[i%workerCores].core_id);
    i += stride;
    if (i >= workerCores && stride > 1) {
      stride_loop++;
      offset++;
      if (stride_loop == stride) {
        offset = stride_loop = 0;
      }
      i = offset;
    }
  }

  unsigned int next = (max - max % stride) + stride;
  for (unsigned int i = 0; i < number_of_senders; i++) {
    for (; next < num_cores; next += config->cores_per_network_thread) {
      if (validCpus[next] > -1) {
        flat[i + workerCores].on = 1;
        flat[i + workerCores].core_id = flat[i + workerCores].core_info.cpu_id =
            validCpus[next];
        add_a_thread(&flat[i + workerCores], 0, 1, 0, ABSTRACT_OUTBOUND,
                   network_out_thread_id++, config->pin_threads);
        next += config->cores_per_network_thread;
        break;
      }
    }
  }
  for (unsigned int i = 0; i < number_of_receivers; i++) {
    for (; next < num_cores; next += config->cores_per_network_thread) {
      if (validCpus[next] > -1) {
        flat[i + workerCores + number_of_senders].on = 1;
        flat[i + workerCores + number_of_senders].core_id =
            flat[i + workerCores + number_of_senders].core_info.cpu_id =
                validCpus[next];
        add_a_thread(&flat[i + workerCores + number_of_senders], 0, 0, 1,
                   ABSTRACT_INBOUND, network_in_thread_id++, config->pin_threads);
        next += config->cores_per_network_thread;
        break;
      }
    }
  }
  if (validCpus)
    arts_free(validCpus);
}

unsigned int flatten_mask(struct arts_config_s *config, unsigned int num_cores,
                         struct unit_mask_s *unit, struct thread_mask_s **flat) {
  unsigned int maskSize = 0;
  unsigned int thread_id = 0;

  for (int i = 0; i < num_cores; i++) {
    if (unit[i].on) {
      maskSize += unit[i].threads;
    }
  }
  *flat = (struct thread_mask_s *)arts_calloc(maskSize, sizeof(struct thread_mask_s));
  unsigned int *group_count = arts_calloc(ABSTRACT_MAX, sizeof(unsigned int));
  struct unit_thread_s *next;
  unsigned int count = 0;
  for (int i = 0; i < num_cores; i++) {
    if (unit[i].on) {
      next = unit[i].list_head;

      while (next != NULL) {
        assert(count < maskSize);
        set_thread_mask(&(*flat)[count], &unit[i], next);
        (*flat)[count].group_pos = group_count[next->group_id]++;
        (*flat)[count].id = count;
        ++count;
        struct unit_thread_s *temp = next->next;
        arts_free(next);
        next = temp;
      }
    }
  }
  arts_free(group_count);
#ifdef USE_HWLOC
  for (int i = 0; i < num_cores; i++) {
    if (unit[i].core_info.cpuset)
      hwloc_bitmap_free(unit[i].core_info.cpuset);
  }
#endif
  arts_free(unit);

  return count;
}

struct thread_mask_s *get_thread_mask(struct arts_config_s *config) {
  if (config->sender_count > (arts_global_rank_count - 1) * config->ports)
    config->sender_count = (arts_global_rank_count - 1) * config->ports;
  if (config->receiver_count > (arts_global_rank_count - 1) * config->ports)
    config->receiver_count = (arts_global_rank_count - 1) * config->ports;

  unsigned int worker_threads =
      config->thread_count - config->sender_count - config->receiver_count;
  unsigned int total_threads = config->thread_count;

  bool network_on = (arts_global_rank_count > 1);
  struct unit_mask_s *unit;
  struct thread_mask_s *flat;

  unsigned int core_count =
      (config->core_count) ? config->core_count : sysconf(_SC_NPROCESSORS_ONLN);

  unit = arts_calloc(core_count, sizeof(struct unit_mask_s));
  default_policy(worker_threads, config->sender_count, config->receiver_count, unit,
                core_count, config);

  total_threads = flatten_mask(config, core_count, unit, &flat);

  if (config->print_topology)
    print_mask(flat, total_threads);
  arts_runtime_node_init(worker_threads, 1, config->sender_count,
                      config->receiver_count, total_threads, 0, config);
  return flat;
}

void destroy_thread_mask(struct thread_mask_s *mask) { arts_free(mask); }

#endif

void print_mask(struct thread_mask_s *units, unsigned int number_of_units) {
  (void)units;
  unsigned int i;
  ARTS_INFO_MASTER(
      " Id   GroupId  GroupPos  Cluster  Core  Unit    On  Worker  "
      "Send  Recv   Pin Status");
  for (i = 0; i < number_of_units; i++) {
    ARTS_INFO_MASTER(
        "%3u    %3u     %3u       %3u     %3u    %3u     %1u     %1u "
        "    %1u     %1u      %1u    %1u",
        units[i].id, units[i].group_id, units[i].group_pos, units[i].numa_domain_id,
        units[i].core_id, units[i].unit_id, units[i].on, units[i].worker,
        units[i].network_send, units[i].network_receive, units[i].pin,
        units[i].status_send);
  }
}
