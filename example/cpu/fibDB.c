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
#include <assert.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "arts/arts.h"
#include "arts/runtime/memory/DbFunctions.h"

uint64_t start = 0;
#define DPRINTF(...)
// #define DPRINTF(...) \
    do { \
        printf(__VA_ARGS__); \
        fflush(stdout); \
    } while (0)

// #define DPRINTF(...)                                                       \
    do {                                                                   \
        FILE *fp__ = get_dprintf_file();                                   \
        if (fp__) {                                                        \
            fprintf(fp__, "[TID %lu] ", (unsigned long)pthread_self());    \
            fprintf(fp__, __VA_ARGS__);                                    \
            fflush(fp__);                                                  \
        }                                                                  \
    } while (0)

static void signal_handler(int sig) {
  // async-signal-safe printing
  const char *name = strsignal(sig);
  if (!name)
    name = "UNKNOWN";
  write(STDERR_FILENO, "Caught signal: ", 15);
  write(STDERR_FILENO, name, strlen(name));
  write(STDERR_FILENO, "\n", 1);

  if (strcmp(name, "Interrupt") == 0)
    _exit(128 + sig);
}

void catch_all_signals(void) {
  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));

  sa.sa_handler = signal_handler;
  sa.sa_flags = SA_RESTART; // optional but helpful

  sigemptyset(&sa.sa_mask);

  for (int sig = 1; sig < NSIG; sig++) {
    // Skip uncatchable signals
    if (sig == SIGKILL || sig == SIGSTOP)
      continue;

    if (sigaction(sig, &sa, NULL) == -1) {
      // You will get errors for signals your system doesn't support
      // or those blocked/restricted by security.
      // Comment this out if too noisy.
      // perror("sigaction");
    }
  }
}

void fibJoin(uint32_t paramc, uint64_t *paramv, uint32_t depc,
             artsEdtDep_t depv[]) {
  // Validate DB pointers before accessing them
  assert(depv[0].ptr != NULL && "First DB pointer is valid");
  assert(depv[1].ptr != NULL && "Second DB pointer is valid");

  // Extract values from DBs
  int *x_ptr = depv[0].ptr;
  int *y_ptr = depv[1].ptr;
  unsigned int x = *x_ptr;
  unsigned int y = *y_ptr;

  // Create a DB to store the result
  int *result_ptr;
#ifdef USE_CXL
  artsGuid_t resultGuid =
      artsDbCreate((void **)&result_ptr, sizeof(unsigned int), ARTS_DB_CXL);
#else
  artsGuid_t resultGuid =
      artsDbCreate((void **)&result_ptr, sizeof(unsigned int), ARTS_DB_READ);
#endif /* USE_CXL */
  assert(result_ptr && "Result ptr not NULL");
  *result_ptr = x + y;

#ifdef USE_CXL
  artsCXLProducerFlush(resultGuid);
#endif /* USE_CXL */
  // Signal the parent EDT with the result DB
  artsSignalEdt(paramv[0], paramv[1], resultGuid);
}

void fibFork(uint32_t paramc, uint64_t *paramv, uint32_t depc,
             artsEdtDep_t depv[]) {
  unsigned int next = (artsGetCurrentNode() + 1) % artsGetTotalNodes();
  // unsigned int next = artsGetCurrentNode();

  artsGuid_t guid = paramv[0];
  unsigned int slot = paramv[1];

  assert(depv[0].ptr && "depv[0] ptr not null");
  // Get the input number from DB
  unsigned int num = ((unsigned int *)(depv[0].ptr))[0];

  if (num < 2) {
    artsSignalEdt(guid, slot, depv[0].guid);
  } else {
    // Create a join EDT that will combine results
    artsGuid_t joinGuid =
        artsEdtCreate(fibJoin, artsGetCurrentNode(), paramc, paramv, 2);
    // Create DBs for n-1 and n-2
    int *n1_ptr;
#ifdef USE_CXL
    artsGuid_t n1Guid =
        artsDbCreate((void **)&n1_ptr, sizeof(unsigned int), ARTS_DB_CXL);
#else
    artsGuid_t n1Guid =
        artsDbCreate((void **)&n1_ptr, sizeof(unsigned int), ARTS_DB_READ);
#endif /* USE_CXL */
    assert(n1_ptr && "n1_ptr not NULL");
    *n1_ptr = num - 1;
#ifdef USE_CXL
    artsCXLProducerFlush(n1Guid);
#endif /* USE_CXL */

    int *n2_ptr;
#ifdef USE_CXL
    artsGuid_t n2Guid =
        artsDbCreate((void **)&n2_ptr, sizeof(unsigned int), ARTS_DB_CXL);
#else
    artsGuid_t n2Guid =
        artsDbCreate((void **)&n2_ptr, sizeof(unsigned int), ARTS_DB_READ);
#endif /* USE_CXL */
    assert(n2_ptr && "n2_ptr not NULL");
    *n2_ptr = num - 2;
#ifdef USE_CXL
    artsCXLProducerFlush(n2Guid);
#endif /* USE_CXL */

    // Create first child task with n-1
    uint64_t args1[2] = {joinGuid, 0}; // Last param not used since we pass DB
    artsGuid_t fib1 = artsEdtCreate(fibFork, next, 2, args1, 1);
    artsSignalEdt(fib1, 0, n1Guid);

    // Create second child task with n-2
    uint64_t args2[2] = {joinGuid, 1}; // Last param not used since we pass DB
    artsGuid_t fib2 = artsEdtCreate(fibFork, next, 2, args2, 1);
    artsSignalEdt(fib2, 0, n2Guid);
  }
}

void fibDone(uint32_t paramc, uint64_t *paramv, uint32_t depc,
             artsEdtDep_t depv[]) {
  uint64_t time = artsGetTimeStamp() - start;

  // Extract result from DB
  int *result_ptr = depv[0].ptr;
  assert(result_ptr && "fibDone result_ptr not NULL");
  unsigned int result = *result_ptr;

  PRINTF("Fib %u: %u time: %lu nodes: %u workers: %u\n", paramv[0], result,
         time, artsGetTotalNodes(), artsGetTotalWorkers());
  artsShutdown();
}

void initPerNode(unsigned int nodeId, int argc, char **argv) {
  // Nothing to do here
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc,
                   char **argv) {
  if (!nodeId && !workerId) {
    unsigned int num = atoi(argv[1]);

    // Create the done EDT that will receive the final result
    artsGuid_t doneGuid = artsEdtCreate(fibDone, 0, 1, (uint64_t *)&num, 1);

    // Create a DB for the input number
    int *num_ptr;
#if USE_CXL
    artsGuid_t numGuid =
        artsDbCreate((void **)&num_ptr, sizeof(unsigned int), ARTS_DB_CXL);
// artsCXLProducerFlush(numGuid);
#else
    artsGuid_t numGuid =
        artsDbCreate((void **)&num_ptr, sizeof(unsigned int), ARTS_DB_READ);
#endif /* USE_CXL */
    assert(num_ptr && "initPerWorker num_ptr not NULL");
    *num_ptr = num;
#if USE_CXL
    artsCXLProducerFlush(numGuid);
#endif /* USE_CXL */

    // Start the computation
    uint64_t args[2] = {doneGuid, 0}; // Last param not used since we pass DB
    start = artsGetTimeStamp();
    artsGuid_t fibGuid = artsEdtCreate(fibFork, 0, 2, args, 1);
    artsSignalEdt(fibGuid, 0, numGuid);
  }
}

int main(int argc, char **argv) {
  // catch_all_signals();
  PRINTF("Starting ArtsRT with DB-based Fibonacci\n");
  artsRT(argc, argv);
  return 0;
}