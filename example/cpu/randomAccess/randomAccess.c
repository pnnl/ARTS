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

/*
 * This code has been contributed by the DARPA HPCS program.  Contact
 * David Koester <dkoester@mitre.org> or Bob Lucas <rflucas@isi.edu>
 * if you have questions.
 *
 * GUPS (Giga UPdates per Second) is a measurement that profiles the memory
 * architecture of a system and is a measure of performance similar to MFLOPS.
 * The HPCS HPCchallenge RandomAccess benchmark is intended to exercise the
 * GUPS capability of a system, much like the LINPACK benchmark is intended to
 * exercise the MFLOPS capability of a computer.  In each case, we would
 * expect these benchmarks to achieve close to the "peak" capability of the
 * memory system. The extent of the similarities between RandomAccess and
 * LINPACK are limited to both benchmarks attempting to calculate a peak system
 * capability.
 *
 * GUPS is calculated by identifying the number of memory locations that can be
 * randomly updated in one second, divided by 1 billion (1e9). The term
 * "randomly" means that there is little relationship between one address to be
 * updated and the next, except that they occur in the space of one half the
 * total system memory.  An update is a read-modify-write operation on a table
 * of 64-bit words. An address is generated, the value at that address read from
 * memory, modified by an integer operation (add, and, or, xor) with a literal
 * value, and that new value is written back to memory.
 *
 * We are interested in knowing the GUPS performance of both entire systems and
 * system subcomponents --- e.g., the GUPS rating of a distributed memory
 * multiprocessor the GUPS rating of an SMP node, and the GUPS rating of a
 * single processor.  While there is typically a scaling of FLOPS with processor
 * count, a similar phenomenon may not always occur for GUPS.
 *
 * For additional information on the GUPS metric, the HPCchallenge RandomAccess
 * Benchmark,and the rules to run RandomAccess or modify it to optimize
 * performance -- see http://icl.cs.utk.edu/hpcc/
 *
 */

/*
 * This file contains the computational core of the single cpu version
 * of GUPS.  The inner loop should easily be vectorized by compilers
 * with such support.
 *
 * This core is used by both the single_cpu and star_single_cpu tests.
 */

#include "randomAccessDefs.h"

#include "arts/arts.h"
#include "arts/gas/Guid.h"
#include <stdio.h>

unsigned int tileSize = TILESIZE;
unsigned int numTiles = 0;
artsGuid_t *tileGuids = NULL;
uint64_t **tile = NULL;

#define getLocalIndex(v, tableSize) ((v & (tableSize - 1)) % tileSize)
#define getOwnerIndex(v, tableSize) ((v & (tableSize - 1)) / tileSize)

uint64_t start = 0;
artsGuid_t doneGuid = NULL_GUID;

/* Perform updates to main table.  The scalar equivalent is:
 *
 *     u64Int ran;
 *     ran = 1;
 *     for (i=0; i<NUPDATE; i++) {
 *       ran = (ran << 1) ^ (((s64Int) ran < 0) ? POLY : 0);
 *       table[ran & (TableSize-1)] ^= ran;
 *     }
 */

uint64_t hpccStartsCPU(int64_t N) {
  int64_t i, j;
  uint64_t m2[64];
  uint64_t temp;
  volatile uint64_t ran;
  volatile int64_t n = N;

  while (n < 0)
    n += PERIOD2;
  while (n > PERIOD2)
    n -= PERIOD2;
  if (n != 0) {
    temp = 0x1;
    for (i = 0; i < 64; i++) {
      m2[i] = temp;
      temp = (temp << 1) ^ ((int64_t)temp < 0 ? POLY2 : 0);
      temp = (temp << 1) ^ ((int64_t)temp < 0 ? POLY2 : 0);
    }

    for (i = 62; i >= 0; i--)
      if ((n >> i) & 1)
        break;

    ran = 0x2;
    while (i > 0) {
      temp = 0;
      for (j = 0; j < 64; j++)
        if ((ran >> j) & 1)
          temp ^= m2[j];
      ran = temp;
      i -= 1;
      if ((n >> i) & 1)
        ran = (ran << 1) ^ ((int64_t)ran < 0 ? POLY2 : 0);
    }
  } else
    ran = 0x1;

  ran = (ran << 1) ^ ((int64_t)ran < 0 ? POLY2 : 0);
  return ran;
}

/* CPU version of random number generation */
void hpccStarts(int64_t N, uint64_t numUpdates, uint64_t numTiles,
                uint64_t tileSize, uint64_t tableSize, uint64_t *rArray) {
  for (uint64_t localIndex = 0; localIndex < numUpdates; localIndex++) {
    // Add the step offset
    volatile int64_t n = N + localIndex;

    int i, j;
    uint64_t m2[64];
    uint64_t temp;

    uint64_t ran = 0x1;

    while (n < 0)
      n += PERIOD2;
    while (n > PERIOD2)
      n -= PERIOD2;

    if (n) {
      temp = 0x1;
      for (i = 0; i < 64; i++) {
        m2[i] = temp;
        temp = (temp << 1) ^ ((int64_t)temp < 0 ? POLY2 : 0);
        temp = (temp << 1) ^ ((int64_t)temp < 0 ? POLY2 : 0);
      }

      for (i = 62; i >= 0; i--)
        if ((n >> i) & 1)
          break;

      ran = 0x2;
      while (i > 0) {
        temp = 0;
        for (j = 0; j < 64; j++)
          if ((ran >> j) & 1)
            temp ^= m2[j];
        ran = temp;
        i -= 1;
        if ((n >> i) & 1)
          ran = (ran << 1) ^ ((int64_t)ran < 0 ? POLY2 : 0);
      }
    } else
      ran = 0x1;

    ran = (ran << 1) ^ ((int64_t)ran < 0 ? POLY2 : 0);

    rArray[localIndex + numTiles] = ran;
    uint64_t owner = getOwnerIndex(ran, tableSize);
    rArray[owner]++;
  }
}

void updateEdt(uint32_t paramc, uint64_t *paramv, uint32_t depc,
               artsEdtDep_t depv[]) {
  uint64_t tileSize = paramv[0];
  uint64_t numTiles = paramv[1];
  uint64_t tableSize = paramv[2];
  uint64_t numUpdates = paramv[3];
  uint64_t partIndex = paramv[4];

  uint64_t *table = (uint64_t *)depv[0].ptr;
  uint64_t *ran = (uint64_t *)depv[1].ptr;
  ran += numTiles;

  for (uint64_t localIndex = 0; localIndex < numUpdates; localIndex++) {
    uint64_t localRan = ran[localIndex];
    uint64_t globalRanIndex = localRan & (tableSize - 1);
    if (globalRanIndex / tileSize == partIndex) {
      uint64_t localRanIndex = globalRanIndex % tileSize;
      table[localRanIndex] ^= localRan;
    }
  }
  artsCXLProducerFlush(depv[0].guid);
}

void randomEdt(uint32_t paramc, uint64_t *paramv, uint32_t depc,
               artsEdtDep_t depv[]) {

  uint64_t numRemUpdates = paramv[0];
  uint64_t numRandom = (numRemUpdates > MAX_UPDATES_PER_CPU_STEP)
                           ? MAX_UPDATES_PER_CPU_STEP
                           : numRemUpdates;
  uint64_t step = paramv[1];
  uint64_t index = paramv[2];
  int64_t startIndex = (int64_t)(step * MAX_UPDATES_PER_CPU_STEP + index * numRandom);
  uint64_t *rArray = (uint64_t *)depv[0].ptr;
  uint64_t tableSize = TABLESIZE;

  if (numRemUpdates) {
    PRINTF("Get Random: %lu step: %lu index: %lu startIndex: %ld\n", numRandom,
           step, index, startIndex);
    PRINTF("TableSize: %lu rArray: %lu %p numTiles: %u\n", tableSize,
           depv[0].guid, depv[0].ptr, numTiles);

    // Call random function
    hpccStarts(startIndex, numRandom, numTiles, tileSize, tableSize, rArray);

    // Get random counts
    unsigned int elemsToCopy = numTiles;
    uint64_t *count = (uint64_t *)artsCalloc(elemsToCopy, sizeof(uint64_t));
    for (unsigned int i = 0; i < elemsToCopy; i++)
      count[i] = rArray[i];

    // Reserve next randomEdt
    artsGuid_t nextRandomGuid =
        artsReserveGuidRoute(ARTS_EDT, artsGetCurrentNode());
    unsigned int nextRandomDeps = 1;

    // Create update edts
    uint64_t updateArgs[] = {tileSize, numTiles, tableSize, numRandom, 0};
    artsGuid_t epochGuid = artsInitializeAndStartEpoch(NULL_GUID, 0);
    for (uint64_t i = 0; i < numTiles; i++) {
      if (count[i]) {
        updateArgs[4] = i;
        PRINTF("Launching for i: %lu count: %lu tileGuid: %lu\n", i, count[i],
               tileGuids[i]);
        artsGuid_t updateGuid =
            artsEdtCreate(updateEdt, artsGetCurrentNode(), 5, updateArgs, 2);
        artsSignalEdt(updateGuid, 0, tileGuids[i]);
        artsSignalEdt(updateGuid, 1, depv[0].guid);
        nextRandomDeps++;
      }
    }
    artsWaitOnHandle(epochGuid);

    // Free the counts since we are done with them.
    artsFree(count);

    // Create next randomEdt
    uint64_t nextRandom = numRemUpdates - numRandom;
    uint64_t args[] = {nextRandom, step + 1, index};
    artsEdtCreateWithGuid(randomEdt, nextRandomGuid, 3, args, nextRandomDeps);
    artsSignalEdt(nextRandomGuid, 0, depv[0].guid);
  } else
    artsSignalEdt(doneGuid, (unsigned int)-1, NULL_GUID);
}

void syncEdt(uint32_t paramc, uint64_t *paramv, uint32_t depc,
             artsEdtDep_t depv[]) {
  uint64_t time = artsGetTimeStamp() - start;
  PRINTF("Time %lu\n", time);

  uint64_t *Table = (uint64_t *)artsCalloc(TABLESIZE, sizeof(uint64_t));
  for (uint64_t i = 0; i < TABLESIZE; i++)
    Table[i] = i;

#ifdef VALIDATE
  uint64_t temp = 0x1;
  uint64_t tableSize = TABLESIZE;
  for (uint64_t i = 0; i < NUPDATE; i++) {
    temp = (temp << 1) ^ (((int64_t)temp < 0) ? POLY2 : 0);
    Table[temp & (tableSize - 1)] ^= temp;
  }

  bool firstFailure = 1;
  uint64_t totalErrors = 0;
  uint64_t index = 0;
  for (unsigned int i = 0; i < numTiles; i++) {
    uint64_t *tile = (uint64_t *)depv[i].ptr;
    for (unsigned int j = 0; j < tileSize; j++) {
      if (tile[j] != Table[index]) {
        if (firstFailure) {
          firstFailure = 0;
          PRINTF(
              "FAILED on index:%lu Exp: %lu vs Rec: %lu -> %lu\n",
              index, Table[index], tile[j], Table[index] ^ tile[j]);
        }
        totalErrors++;
      }
      index++;
    }
  }
  if (totalErrors)
    PRINTF("%lu errors of %lu!\n", totalErrors, index);
  else
    PRINTF("Verified!\n");
#endif

  double GUPS = (double)NUPDATE / time;
  PRINTF("GUPS: %lf MB: %lu\n", GUPS,
         (TABLESIZE * sizeof(uint64_t)) / (1024 * 1024));
  artsShutdown();
}

void initPerNode(unsigned int nodeId, int argc, char **argv) {
  if (argc > 1)
    tileSize = (unsigned int)atoi(argv[1]);
  numTiles = TABLESIZE / tileSize;
  PRINTF("Random Access Table Size: %u Tile Size: %u Number of Tiles: %u\n",
         TABLESIZE, tileSize, numTiles);

  tileGuids = artsMalloc(sizeof(artsGuid_t) * numTiles);

  if (!nodeId) {
    // Create tiled table
    tile = (uint64_t **)artsCalloc(numTiles, sizeof(uint64_t *));
    uint64_t counter = 0;
    for (unsigned int i = 0; i < numTiles; i++) {
      tileGuids[i] = artsDbCreate((void **)&(tile[i]),
                                  (tileSize + 1) * sizeof(uint64_t),
                                  ARTS_DB_CXL);
      PRINTF("TileGuid[%u]: %lu -> %p\n", i, tileGuids[i], tile[i]);
      for (unsigned int j = 0; j < tileSize; j++)
        tile[i][j] = counter++;
      tile[i][tileSize] = 0;
      artsCXLProducerFlush(tileGuids[i]);
    }
  }

  // Create a sync edt for all partitions
  doneGuid = artsReserveGuidRoute(ARTS_EDT, 0);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId,
                   int argc, char **argv) {
  if (!nodeId && !workerId) {
    // Create update frontier
    unsigned int elemsPerFrontier = numTiles + MAX_UPDATES_PER_CPU_STEP;
    uint64_t *updateFrontier = NULL;
    artsGuid_t updateFrontierGuid = artsDbCreate((void **)&updateFrontier,
                                                 elemsPerFrontier * sizeof(uint64_t),
                                                 ARTS_DB_CXL);
    PRINTF("updateFrontier: %lu %p\n", updateFrontierGuid, updateFrontier);
    for (unsigned int j = 0; j < elemsPerFrontier; j++)
      updateFrontier[j] = 0;
    artsCXLProducerFlush(updateFrontierGuid);

    uint64_t numUpdatesTotal = NUPDATE;
    PRINTF("numUpdatesTotal: %lu\n", numUpdatesTotal);

    // Create randomEdt
    uint64_t args[] = {numUpdatesTotal, 0, 0};
    artsGuid_t updateGuid = artsEdtCreate(randomEdt, 0, 3, args, 1);
    artsSignalEdt(updateGuid, 0, updateFrontierGuid);

    artsEdtCreateWithGuid(syncEdt, doneGuid, 0, NULL, numTiles);
    for (unsigned int i = 0; i < numTiles; i++)
      artsSignalEdt(doneGuid, i, tileGuids[i]);
  }
  start = artsGetTimeStamp();
}

int main(int argc, char **argv) {
  artsRT(argc, argv);
  return 0;
}