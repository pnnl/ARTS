#include <bits/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "arts/arts.h"
#include "arts/runtime/RT.h"

#define N_MAX 20

typedef struct {
  int board[N_MAX];
  int row;
  int n;
} nqueen_data_t;

int isSafe(int board[], int row, int col) {
  for (int i = 0; i < row; i++) {
    if (board[i] == col || board[i] - i == col - row ||
        board[i] + i == col + row) {
      return 0;
    }
  }
  return 1;
}

void joinNqueens(uint32_t paramc, uint64_t *paramv, uint32_t depc,
                 artsEdtDep_t depv[]) {
  artsGuid_t returnGuid = paramv[0];
  uint32_t slot = paramv[1];
  int sum = 0;
  for (uint32_t i = 0; i < depc; i++) {
    sum += depv[i].guid;
  }
  artsSignalEdtValue(returnGuid, slot, sum);
}

void forkNqueens(uint32_t paramc, uint64_t *paramv, uint32_t depc,
                 artsEdtDep_t depv[]) {
  artsGuid_t returnGuid = paramv[0];
  uint32_t slot = paramv[1];
  nqueen_data_t *currentData = (nqueen_data_t *)depv[0].ptr;

  if (currentData->row == currentData->n) {
    artsSignalEdtValue(returnGuid, slot, 1);
    return;
  }

  int safePositions[N_MAX];
  int count = 0;
  for (int col = 0; col < currentData->n; col++) {
    if (isSafe(currentData->board, currentData->row, col)) {
      safePositions[count++] = col;
    }
  }

  if (count == 0) {
    artsSignalEdtValue(returnGuid, slot, 0);
    return;
  }

  unsigned int numNodes = artsGetTotalNodes();
  unsigned int currentNode = artsGetCurrentNode();
  artsGuid_t joinGuid =
      artsEdtCreate(joinNqueens, currentNode, 2, paramv, count);
  for (int i = 0; i < count; i++) {
    nqueen_data_t *nextData;
    artsGuid_t dbGuid =
        artsDbCreate((void **)&nextData, sizeof(nqueen_data_t), ARTS_DB_READ);
    memcpy(nextData->board, currentData->board, sizeof(int) * currentData->row);
    nextData->board[currentData->row] = safePositions[i];
    nextData->row = currentData->row + 1;
    nextData->n = currentData->n;

    uint64_t newParamv[2] = {(uint64_t)joinGuid, i};
    unsigned int route =
        nextData->row <= 2 ? (currentNode + i) % numNodes : currentNode;
    artsGuid_t forkGuid = artsEdtCreate(forkNqueens, route, 2, newParamv, 1);

    artsSignalEdt(forkGuid, 0, dbGuid);
  }
}

void finalNqueens(uint32_t paramc, uint64_t *paramv, uint32_t depc,
                  artsEdtDep_t depv[]) {
  struct timespec end;
  clock_gettime(CLOCK_REALTIME, &end);
  double startTime = paramv[0];
  double endTime = end.tv_sec + end.tv_nsec / 1e9;

  int n = paramv[1];
  int solutions = depv[0].guid;

  printf("\nResults:\n");
  printf("Execution time: %.4f seconds\n", endTime - startTime);
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

  artsShutdown();
}

void artsMain(int argc, char **argv) {
  // Parse command line arguments
  if (argc <= 1) {
    printf("Usage: %s <board size>\n", argv[0]);
    artsShutdown();
  }
  int n = atoi(argv[1]);
  if (n < 1 || n > N_MAX) {
    printf("Board size must be between 1 and 20\n");
    artsShutdown();
  }

  printf("Solving %d-Queens problem\n", n);
  printf("Using ARTS\n");

  struct timespec start;
  clock_gettime(CLOCK_REALTIME, &start);
  double startTime = start.tv_sec + start.tv_nsec / 1e9;

  uint64_t finalParamv[2] = {(uint64_t)startTime, (uint64_t)n};
  artsGuid_t finalGuid = artsEdtCreate(finalNqueens, 0, 2, finalParamv, 1);
  uint64_t forkParamv[2] = {finalGuid, 0};
  artsGuid_t forkGuid = artsEdtCreate(forkNqueens, 0, 2, forkParamv, 1);
  nqueen_data_t *forkData;
  artsGuid_t dbGuid =
      artsDbCreate((void **)&forkData, sizeof(nqueen_data_t), ARTS_DB_READ);
  for (int i = 0; i < n; i++)
    forkData->board[i] = -1;
  forkData->row = 0;
  forkData->n = n;
  artsSignalEdt(forkGuid, 0, dbGuid);
}

int main(int argc, char **argv) { return artsRT(argc, argv); }
