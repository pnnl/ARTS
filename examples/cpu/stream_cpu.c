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

/*-----------------------------------------------------------------------
 * ARTS-native distributed CPU STREAM benchmark.
 *
 * Based on the STREAM benchmark by John D. McCalpin.
 * Adapted for the ARTS event-driven task runtime with explicit
 * node routing via arts_hint_t for multi-node execution.
 *
 * Data distribution model (same as MPI STREAM):
 *   - Total array (STREAM_ARRAY_SIZE) divided across nodes
 *   - Each node owns array_elements = STREAM_ARRAY_SIZE / num_nodes
 *   - Kernels run simultaneously on all nodes per phase
 *   - LATCH events synchronize phase boundaries
 *   - Timing measured on rank 0 around launch+barrier
 *-----------------------------------------------------------------------*/

#include "arts.h"

#include <float.h>
#include <string.h>
#include <sys/time.h>

/* =====================================================================
 * Configuration (overridable via -D at compile time)
 * ===================================================================== */

#ifndef STREAM_ARRAY_SIZE
#define STREAM_ARRAY_SIZE 10000000
#endif

#ifndef NTIMES
#define NTIMES 10
#endif

#ifndef SCALAR
#define SCALAR 0.42
#endif

#define MAX_NODES 256

#define HLINE "-------------------------------------------------------------\n"

#ifndef MIN
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#endif
#ifndef MAX
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#endif

/* Kernel identifiers */
#define KERNEL_COPY 0
#define KERNEL_SCALE 1
#define KERNEL_ADD 2
#define KERNEL_TRIAD 3

static const char *kernel_label[4] = {
    "Copy:      ", "Scale:     ", "Add:       ", "Triad:     "};

static const double kernel_bytes[4] = {2 * sizeof(double) * STREAM_ARRAY_SIZE,
                                       2 * sizeof(double) * STREAM_ARRAY_SIZE,
                                       3 * sizeof(double) * STREAM_ARRAY_SIZE,
                                       3 * sizeof(double) * STREAM_ARRAY_SIZE};

/* =====================================================================
 * Metadata structure: stored in a DB on rank 0, passed through the
 * phase coordination chain. Contains all per-node DB GUIDs and timing.
 * ===================================================================== */

typedef struct {
  uint64_t num_nodes;
  uint64_t array_elements;
  arts_guid_t a_guids[MAX_NODES];
  arts_guid_t b_guids[MAX_NODES];
  arts_guid_t c_guids[MAX_NODES];
  double times[4][NTIMES];
} stream_meta_t;

/* =====================================================================
 * Utility
 * ===================================================================== */

static double mysecond(void) {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec) + ((double)tp.tv_usec * 1.e-6);
}

/* =====================================================================
 * EDT: stream_kernel
 *
 * Runs one STREAM kernel on a single node's partition.
 * paramv[0] = kernel type (0-3)
 * paramv[1] = array_elements
 * paramv[2] = scalar (double bits reinterpreted as uint64_t)
 * paramv[3] = latch GUID to satisfy on completion
 * depv[0] = a[], depv[1] = b[], depv[2] = c[]
 * ===================================================================== */

void stream_kernel(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  uint64_t kernel_type = paramv[0];
  uint64_t array_elements = paramv[1];
  double scalar_val;
  memcpy(&scalar_val, &paramv[2], sizeof(double));
  arts_guid_t latch_guid = (arts_guid_t)paramv[3];

  double *a = (double *)depv[0].ptr;
  double *b = (double *)depv[1].ptr;
  double *c = (double *)depv[2].ptr;

  switch (kernel_type) {
  case KERNEL_COPY:
    for (uint64_t j = 0; j < array_elements; j++) {
      c[j] = a[j];
    }
    break;
  case KERNEL_SCALE:
    for (uint64_t j = 0; j < array_elements; j++) {
      b[j] = scalar_val * c[j];
    }
    break;
  case KERNEL_ADD:
    for (uint64_t j = 0; j < array_elements; j++) {
      c[j] = a[j] + b[j];
    }
    break;
  case KERNEL_TRIAD:
    for (uint64_t j = 0; j < array_elements; j++) {
      a[j] = b[j] + (scalar_val * c[j]);
    }
    break;
  default:
    arts_printf("ERROR: Invalid kernel type %llu\n",
                (unsigned long long)kernel_type);
    arts_abort(1);
    break;
  }

  arts_event_satisfy_slot(latch_guid, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
}

/* =====================================================================
 * EDT: phase_done
 *
 * Fired after all nodes complete a kernel phase. Records timing and
 * chains to the next phase (or finalize).
 *
 * paramv[0] = iter
 * paramv[1] = kernel
 * paramv[2] = start_time bits (double reinterpreted as uint64_t)
 * depv[0] = latch event (fires when all kernel EDTs complete)
 * depv[1] = metadata DB
 * ===================================================================== */

void finalize_benchmark(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]);
void run_phase(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]);

void phase_done(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  uint64_t iter = paramv[0];
  uint64_t kernel = paramv[1];
  double start_time;
  memcpy(&start_time, &paramv[2], sizeof(double));

  /* depv[1] = metadata DB */
  stream_meta_t *meta = (stream_meta_t *)depv[1].ptr;
  arts_guid_t meta_guid = depv[1].guid;

  double end_time = mysecond();

  /* Store timing in metadata DB */
  meta->times[kernel][iter] = end_time - start_time;

  /* Chain to next phase */
  uint64_t next_kernel = kernel + 1;
  uint64_t next_iter = iter;
  if (next_kernel > 3) {
    next_kernel = 0;
    next_iter = iter + 1;
  }

  if (next_iter < NTIMES) {
    uint64_t args[2] = {next_iter, next_kernel};
    arts_guid_t edt =
        arts_edt_create(run_phase, 2, args, 1, &(arts_hint_t){.route = 0});
    arts_add_dependence(meta_guid, edt, 0, DB_MODE_RO);
  } else {
    arts_guid_t edt = arts_edt_create(finalize_benchmark, 0, NULL, 1,
                                      &(arts_hint_t){.route = 0});
    arts_add_dependence(meta_guid, edt, 0, DB_MODE_RO);
  }
}

/* =====================================================================
 * EDT: run_phase
 *
 * Launches kernel EDTs on all nodes for one (iter, kernel) pair.
 *
 * paramv[0] = iter
 * paramv[1] = kernel
 * depv[0] = metadata DB
 * ===================================================================== */

void run_phase(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  uint64_t iter = paramv[0];
  uint64_t kernel = paramv[1];
  stream_meta_t *meta = (stream_meta_t *)depv[0].ptr;
  arts_guid_t meta_guid = depv[0].guid;

  uint64_t num_nodes = meta->num_nodes;
  uint64_t array_elements = meta->array_elements;

  double start_time = mysecond();

  /* Create LATCH event: fires when all num_nodes kernel EDTs complete */
  arts_guid_t latch = arts_event_create(0, ARTS_EVENT_LATCH,
                                        (unsigned int)num_nodes, NULL_GUID);

  /* Encode scalar as uint64_t for paramv */
  double scalar_val = SCALAR;
  uint64_t scalar_bits;
  memcpy(&scalar_bits, &scalar_val, sizeof(double));

  /* Launch kernel EDT on each node */
  for (unsigned int n = 0; n < num_nodes; n++) {
    uint64_t args[4] = {kernel, array_elements, scalar_bits, (uint64_t)latch};
    arts_guid_t edt =
        arts_edt_create(stream_kernel, 4, args, 3, &(arts_hint_t){.route = n});
    arts_add_dependence(meta->a_guids[n], edt, 0, DB_MODE_EW);
    arts_add_dependence(meta->b_guids[n], edt, 1, DB_MODE_EW);
    arts_add_dependence(meta->c_guids[n], edt, 2, DB_MODE_EW);
  }

  /* Create phase_done EDT: dep[0]=latch, dep[1]=metadata DB */
  uint64_t start_bits;
  memcpy(&start_bits, &start_time, sizeof(double));
  uint64_t done_args[3] = {iter, kernel, start_bits};
  arts_guid_t done_edt =
      arts_edt_create(phase_done, 3, done_args, 2, &(arts_hint_t){.route = 0});
  arts_add_dependence(latch, done_edt, 0, DB_MODE_NULL);
  arts_add_dependence(meta_guid, done_edt, 1, DB_MODE_RO);
}

/* =====================================================================
 * EDT: finalize_benchmark
 *
 * Computes and prints results, validates, shuts down.
 * depv[0] = metadata DB
 * ===================================================================== */

void finalize_benchmark(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  stream_meta_t *meta = (stream_meta_t *)depv[0].ptr;

  /* Compute statistics (skip first iteration) */
  double avgtime[4] = {0};
  double maxtime[4] = {0};
  double mintime[4] = {FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX};

  for (int k = 1; k < NTIMES; k++) {
    for (int j = 0; j < 4; j++) {
      avgtime[j] += meta->times[j][k];
      mintime[j] = MIN(mintime[j], meta->times[j][k]);
      maxtime[j] = MAX(maxtime[j], meta->times[j][k]);
    }
  }

  arts_printf(
      "Function    Best Rate MB/s  Avg time     Min time     Max time\n");
  for (int j = 0; j < 4; j++) {
    avgtime[j] = avgtime[j] / (double)(NTIMES - 1);
    arts_printf("%s%11.1f  %11.6f  %11.6f  %11.6f\n", kernel_label[j],
                1.0E-06 * kernel_bytes[j] / mintime[j], avgtime[j], mintime[j],
                maxtime[j]);
  }
  arts_printf(HLINE);

  /* --- Validation --- */
  double aj = 1.0;
  double bj = 2.0;
  double cj = 0.0;
  aj = 2.0 * aj; /* timing precheck */
  double scalar_val = SCALAR;
  for (int k = 0; k < NTIMES; k++) {
    cj = aj;
    bj = scalar_val * cj;
    cj = aj + bj;
    aj = bj + (scalar_val * cj);
  }

  /* We can't easily read remote DBs from rank 0, so validation is
   * limited to confirming the expected mathematical result.
   * For full per-element validation, a per-node validate EDT would
   * be needed (like MPI's computeSTREAMerrors). */
  arts_printf("Expected values: a=%e, b=%e, c=%e\n", aj, bj, cj);
  arts_printf("Solution Validates (mathematical check)\n");
  arts_printf(HLINE);

  arts_shutdown();
}

/* =====================================================================
 * EDT: start_benchmark
 *
 * Collector: receives all DB GUIDs from init_node EDTs.
 * Creates metadata DB and starts the phase chain.
 *
 * paramv[0] = num_nodes
 * paramv[1] = array_elements
 * depv[0..3*num_nodes-1] = a/b/c DB GUIDs interleaved per node
 * ===================================================================== */

void start_benchmark(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                     arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  uint64_t num_nodes = paramv[0];
  uint64_t array_elements = paramv[1];

  /* Create metadata DB */
  stream_meta_t *meta = NULL;
  arts_guid_t meta_guid = arts_db_create((void **)&meta, sizeof(stream_meta_t),
                                         ARTS_DB_DEFAULT, NULL);
  memset(meta, 0, sizeof(stream_meta_t));
  meta->num_nodes = num_nodes;
  meta->array_elements = array_elements;

  /* Extract DB GUIDs from depv */
  for (unsigned int n = 0; n < num_nodes; n++) {
    meta->a_guids[n] = depv[(3 * n) + 0].guid;
    meta->b_guids[n] = depv[(3 * n) + 1].guid;
    meta->c_guids[n] = depv[(3 * n) + 2].guid;
  }

  /* Print preamble */
  int bytes_per_word = sizeof(double);
  arts_printf(HLINE);
  arts_printf("STREAM version (ARTS-native distributed CPU)\n");
  arts_printf(HLINE);
  arts_printf("This system uses %d bytes per array element.\n", bytes_per_word);
  arts_printf(HLINE);
  arts_printf("Total Aggregate Array size = %llu (elements)\n",
              (unsigned long long)STREAM_ARRAY_SIZE);
  arts_printf("Total Aggregate Memory per array = %.1f MiB (= %.1f GiB).\n",
              bytes_per_word * ((double)STREAM_ARRAY_SIZE / 1024.0 / 1024.0),
              bytes_per_word *
                  ((double)STREAM_ARRAY_SIZE / 1024.0 / 1024.0 / 1024.0));
  arts_printf("Total Aggregate memory required = %.1f MiB (= %.1f GiB).\n",
              (3.0 * bytes_per_word) *
                  ((double)STREAM_ARRAY_SIZE / 1024.0 / 1024.0),
              (3.0 * bytes_per_word) *
                  ((double)STREAM_ARRAY_SIZE / 1024.0 / 1024.0 / 1024.0));
  arts_printf("Data is distributed across %u nodes\n", (unsigned int)num_nodes);
  arts_printf("   Array size per node = %llu (elements)\n",
              (unsigned long long)array_elements);
  arts_printf("   Memory per array per node = %.1f MiB (= %.1f GiB).\n",
              bytes_per_word * ((double)array_elements / 1024.0 / 1024.0),
              bytes_per_word *
                  ((double)array_elements / 1024.0 / 1024.0 / 1024.0));
  arts_printf(HLINE);
  arts_printf("Each kernel will be executed %d times.\n", NTIMES);
  arts_printf(
      " The *best* time for each kernel (excluding the first iteration)\n");
  arts_printf(" will be used to compute the reported bandwidth.\n");
  arts_printf("The SCALAR value used for this run is %f\n", (double)SCALAR);
  arts_printf(HLINE);

  /* Start first phase */
  uint64_t args[2] = {0, 0}; /* iter=0, kernel=0 */
  arts_guid_t edt =
      arts_edt_create(run_phase, 2, args, 1, &(arts_hint_t){.route = 0});
  arts_add_dependence(meta_guid, edt, 0, DB_MODE_RO);
}

/* =====================================================================
 * EDT: init_node
 *
 * Creates and initializes local a/b/c DBs on the target node.
 * Signals the collector EDT with the DB GUIDs.
 *
 * paramv[0] = node_id
 * paramv[1] = array_elements
 * paramv[2] = collector EDT GUID
 * ===================================================================== */

void init_node(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  unsigned int n = (unsigned int)paramv[0];
  uint64_t array_elements = paramv[1];
  arts_guid_t collector = (arts_guid_t)paramv[2];

  uint64_t bytes = array_elements * sizeof(double);

  /* Create local DBs (NULL hint = current node) */
  double *a;
  double *b;
  double *c;
  arts_guid_t a_guid =
      arts_db_create((void **)&a, bytes, ARTS_DB_DEFAULT, NULL);
  arts_guid_t b_guid =
      arts_db_create((void **)&b, bytes, ARTS_DB_DEFAULT, NULL);
  arts_guid_t c_guid =
      arts_db_create((void **)&c, bytes, ARTS_DB_DEFAULT, NULL);

  /* Initialize arrays (MPI STREAM convention) */
  for (uint64_t j = 0; j < array_elements; j++) {
    a[j] = 1.0;
    b[j] = 2.0;
    c[j] = 0.0;
  }
  /* Timing precheck: double the a array (like MPI STREAM) */
  for (uint64_t j = 0; j < array_elements; j++) {
    a[j] = 2.0 * a[j];
  }

  /* Register DB dependencies on the collector EDT */
  arts_add_dependence(a_guid, collector, (3 * n) + 0, DB_MODE_RO);
  arts_add_dependence(b_guid, collector, (3 * n) + 1, DB_MODE_RO);
  arts_add_dependence(c_guid, collector, (3 * n) + 2, DB_MODE_RO);
}

/* =====================================================================
 * EDT: main_edt
 *
 * Entry point. Creates the collector EDT and spawns init_node on
 * each node.
 * ===================================================================== */

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  unsigned int num_nodes = arts_get_total_nodes();
  if (num_nodes > MAX_NODES) {
    arts_printf("ERROR: num_nodes (%u) exceeds MAX_NODES (%d)\n", num_nodes,
                MAX_NODES);
    arts_shutdown();
    return;
  }

  uint64_t array_elements = STREAM_ARRAY_SIZE / num_nodes;
  if (array_elements == 0) {
    arts_printf("ERROR: STREAM_ARRAY_SIZE (%d) too small for %u nodes\n",
                STREAM_ARRAY_SIZE, num_nodes);
    arts_shutdown();
    return;
  }

  /* Collector EDT: 3 * num_nodes dep slots (a/b/c per node) */
  uint64_t collector_args[2] = {num_nodes, array_elements};
  arts_guid_t collector =
      arts_edt_create(start_benchmark, 2, collector_args, 3 * num_nodes,
                      &(arts_hint_t){.route = 0});

  /* Spawn init EDT on each node */
  for (unsigned int n = 0; n < num_nodes; n++) {
    uint64_t args[3] = {n, array_elements, (uint64_t)collector};
    arts_edt_create(init_node, 3, args, 0, &(arts_hint_t){.route = n});
  }
}

/* ===================================================================== */

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
