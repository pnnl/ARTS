#include <bits/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "arts.h"
#include "arts/runtime/rt.h"

#define N_MAX 20

typedef struct {
  int board[N_MAX];
  int row;
  int n;
} nqueen_data_t;

int is_safe(const int board[], int row, int col) {
  for (int i = 0; i < row; i++) {
    if (board[i] == col || board[i] - i == col - row ||
        board[i] + i == col + row) {
      return 0;
    }
  }
  return 1;
}

void join_nqueens(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  arts_guid_t return_guid = paramv[0];
  uint32_t slot = paramv[1];
  int sum = 0;
  for (uint32_t i = 0; i < depc; i++) {
    sum += depv[i].guid;
  }
  arts_signal_edt_value(return_guid, slot, sum);
}

void fork_nqueens(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  arts_guid_t return_guid = paramv[0];
  uint32_t slot = paramv[1];
  nqueen_data_t *current_data = (nqueen_data_t *)depv[0].ptr;

  if (current_data->row == current_data->n) {
    arts_signal_edt_value(return_guid, slot, 1);
    return;
  }

  int safe_positions[N_MAX];
  int count = 0;
  for (int col = 0; col < current_data->n; col++) {
    if (is_safe(current_data->board, current_data->row, col)) {
      safe_positions[count++] = col;
    }
  }

  if (count == 0) {
    arts_signal_edt_value(return_guid, slot, 0);
    return;
  }

  unsigned int num_nodes = arts_get_total_nodes();
  unsigned int current_node = arts_get_current_node();
  arts_guid_t join_guid =
      arts_edt_create(join_nqueens, current_node, 2, paramv, count);
  for (int i = 0; i < count; i++) {
    nqueen_data_t *next_data;
    arts_guid_t db_guid =
        arts_db_create((void **)&next_data, sizeof(nqueen_data_t), ARTS_DB_READ);
    memcpy(next_data->board, current_data->board, sizeof(int) * current_data->row);
    next_data->board[current_data->row] = safe_positions[i];
    next_data->row = current_data->row + 1;
    next_data->n = current_data->n;

    uint64_t new_paramv[2] = {(uint64_t)join_guid, i};
    unsigned int route =
        next_data->row <= 2 ? (current_node + i) % num_nodes : current_node;
    arts_guid_t fork_guid = arts_edt_create(fork_nqueens, route, 2, new_paramv, 1);

    arts_signal_edt(fork_guid, 0, db_guid);
  }
}

void final_nqueens(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  struct timespec end;
  (void)clock_gettime(CLOCK_REALTIME, &end);
  double start_time = paramv[0];
  double end_time = end.tv_sec + (end.tv_nsec / 1e9);

  int n = paramv[1];
  int solutions = depv[0].guid;

  printf("\nResults:\n");
  printf("Execution time: %.4f seconds\n", end_time - start_time);
  printf("Number of solutions: %d\n", solutions);

  // Print expected results for verification
  uint64_t expected[] = {0,       1,        0,        0,        2,
                         10,      4,        40,       92,       352,
                         724,     2680,     14200,    73712,    365596,
                         2279184, 14772512, 95815104, 666090624};
  if (n < 19) {
    printf("Expected solutions for %d-queens: %lu\n", n, expected[n]);
    if (solutions == expected[n]) {
      printf("✓ Result verified!\n");
    } else {
      printf("✗ Result incorrect!\n");
    }
  }

  arts_shutdown();
}

void arts_main(int argc, char **argv) {
  // Parse command line arguments
  if (argc <= 1) {
    printf("Usage: %s <board size>\n", argv[0]);
    arts_shutdown();
  }
  int n = atoi(argv[1]);
  if (n < 1 || n > N_MAX) {
    printf("Board size must be between 1 and 20\n");
    arts_shutdown();
  }

  printf("Solving %d-Queens problem\n", n);
  printf("Using ARTS\n");

  struct timespec start;
  (void)clock_gettime(CLOCK_REALTIME, &start);
  double start_time = start.tv_sec + (start.tv_nsec / 1e9);

  uint64_t final_paramv[2] = {(uint64_t)start_time, (uint64_t)n};
  arts_guid_t final_guid = arts_edt_create(final_nqueens, 0, 2, final_paramv, 1);
  uint64_t fork_paramv[2] = {final_guid, 0};
  arts_guid_t fork_guid = arts_edt_create(fork_nqueens, 0, 2, fork_paramv, 1);
  nqueen_data_t *fork_data;
  arts_guid_t db_guid =
      arts_db_create((void **)&fork_data, sizeof(nqueen_data_t), ARTS_DB_READ);
  for (int i = 0; i < n; i++) {
    fork_data->board[i] = -1;
}
  fork_data->row = 0;
  fork_data->n = n;
  arts_signal_edt(fork_guid, 0, db_guid);
}

int main(int argc, char **argv) { return arts_rt(argc, argv); }
