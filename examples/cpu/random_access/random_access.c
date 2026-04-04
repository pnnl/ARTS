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
#include "arts/runtime/Globals.h"

artsGuidRange *updateFrontierGuids = NULL;
artsGuid_t updateFrontierGuid;

unsigned int tileSize = TILESIZE;
unsigned int numTiles = 0;
// artsGuidRange *tileGuids = NULL;
artsGuid_t *tileGuids = NULL;
uint64_t **tile = NULL;
#define getLocalIndex(v) ((v & (tableSize - 1)) % tileSize)
#define getOwnerIndex(v) ((v & (tableSize - 1)) / tileSize)
#define getTileGuid(v) artsGetGuid(tileGuids, getOwnerIndex(v))

uint64_t start = 0;
artsGuid_t doneGuid = NULL_GUID;
artsGuid_t updateGuid = NULL_GUID;

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

// void hpccStartsTiled(int64_cu_t N, uint64_cu_t numUpdates,
                          //  uint64_cu_t numTiles, uint64_cu_t tileSize,
                          //  uint64_cu_t tableSize, uint64_cu_t *rArray) {
  
void hpccStartsTiled(uint32_t paramc, uint64_t *paramv, uint32_t depc,
                          artsEdtDep_t depv[]) {
  uint64_cu_t N = paramv[0];
  uint64_cu_t index = paramv[1];
  uint64_cu_t numUpdates = paramv[2];
  uint64_cu_t numTiles = paramv[3];
  uint64_cu_t tableSize = paramv[4];
  uint64_cu_t tileSize = paramv[5];
  uint64_cu_t* rArray = (uint64_cu_t*) paramv[6];
  artsGuid_t updateGuid = paramv[7];
  // unsigned long long int* globalRan = paramv[8];
  artsGuid_t tileGuid = depv[0].guid;
  uint64_cu_t* table = depv[0].ptr;
  unsigned long long int* globalRan = depv[1].ptr;
  globalRan += numTiles;
  
  for (int64_cu_t local = 0; local < MAX_TOTAL_PENDING_UPDATES_CU; local++) {
    int64_cu_t localIndex = index * MAX_TOTAL_PENDING_UPDATES_CU + local;
    if (localIndex < numUpdates) {
      // Add the step offset
      volatile int64_cu_t n = N + localIndex;

      int i, j;
      uint64_cu_t m2[64];
      uint64_cu_t temp;

      uint64_cu_t ran = 0x1;

      while (n < 0)
        n += PERIOD;
      while (n > PERIOD)
        n -= PERIOD;

      if (n) {
        temp = 0x1;
        for (i = 0; i < 64; i++) {
          m2[i] = temp;
          temp = (temp << 1) ^ ((int64_cu_t)temp < 0 ? POLY : 0);
          temp = (temp << 1) ^ ((int64_cu_t)temp < 0 ? POLY : 0);
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
            ran = (ran << 1) ^ ((int64_cu_t)ran < 0 ? POLY : 0);
        }
      } else
        ran = 0x1;

      ran = (ran << 1) ^ ((int64_cu_t)ran < 0 ? POLY : 0);

      rArray[localIndex + numTiles] = ran; //TODO: Check logic here
      uint64_cu_t owner = getOwnerIndex(ran);
      rArray[owner] += 1ULL;
    }
  }
  uint64_cu_t partIndex = index;
  artsCXLProducerFlush(tileGuid);
  artsSignalEdt(updateGuid, (index+1), tileGuid);
}

/* Utility routine to start random number generator at Nth step */
void hpccStarts(int64_cu_t N, uint64_cu_t numUpdates,
                           uint64_cu_t numTiles, uint64_cu_t tileSize,
                           uint64_cu_t tableSize, uint64_cu_t *rArray,
                           artsGuid_t nextGuid) {
  uint64_t next = artsGetCurrentNode();
  for (uint64_cu_t index = 0; index < numTiles; index++) {
    uint64_t args[8] = {N, index, numUpdates, numTiles,
                       tableSize, tileSize, (uint64_t) rArray, nextGuid};
    artsGuid_t hpccTiledGuid = artsEdtCreate(hpccStartsTiled, next, 8, args, 2);
    artsSignalEdt(hpccTiledGuid, 0, tileGuids[index]);
    artsSignalEdt(hpccTiledGuid, 1, updateFrontierGuid);
    next = (next+1)%artsGetTotalNodes();
  }
}

void updateEdt(uint32_t paramc, uint64_t *paramv, uint32_t depc,
                          artsEdtDep_t depv[]) {
  uint64_cu_t tileSize = paramv[0];
  uint64_cu_t numTiles = paramv[1];
  uint64_cu_t tableSize = paramv[2];
  uint64_cu_t numUpdates = paramv[3];
  uint64_cu_t partIndex = paramv[4];
  artsGuid_t nextGuid = paramv[5];
  uint32_t slot = paramv[6];
  
  artsCXLConsumerFlush(depv[0].guid);  // Flush the tile before accessing
  artsCXLConsumerFlush(depv[1].guid);  // Flush the updateFrontier before accessing

  uint64_cu_t *table = (uint64_cu_t *)depv[0].ptr;
  unsigned long long int *ran = (unsigned long long int *)depv[1].ptr;
  ran += numTiles;

  for (uint64_cu_t local = 0; local < numUpdates; local++) {
    uint64_cu_t localRan = ran[local];
    uint64_cu_t globalRanIndex = localRan & (tableSize - 1);
    if (globalRanIndex / tileSize == partIndex) {  // Check if this update belongs to our partition
        uint64_cu_t localRanIndex = globalRanIndex % tileSize;
        __atomic_fetch_xor(&table[localRanIndex], localRan, __ATOMIC_SEQ_CST);
    }
  }
  artsSignalEdt(nextGuid, slot, NULL_GUID);
  artsCXLProducerFlush(depv[0].guid);
}

void updateDriver(uint32_t paramc, uint64_t *paramv, uint32_t depc,
               artsEdtDep_t depv[]) {
  uint64_t tileSize = paramv[0];
  uint64_t numTiles = paramv[1];
  uint64_t tableSize = paramv[2];
  uint64_t numRandom = paramv[3];
  uint64_t nextRandomGuid = paramv[4];

  artsCXLConsumerFlush(depv[0].guid);
  for (unsigned int i = 0; i < numTiles; i++) {
    artsCXLConsumerFlush(depv[i+1].guid);
  }
  
  artsGuid_t readOnly = depv[0].guid;
  uint64_t updateArgs[] = {tileSize, numTiles, tableSize, numRandom, 0, nextRandomGuid, 0};
  uint64_t next = artsGetCurrentNode();
  for (uint64_t i = 0; i < numTiles; i++) {
    updateArgs[4] = i;
    updateArgs[6] = i+1;
    artsGuid_t updateGuid = artsEdtCreate(updateEdt,
                                next,
                                7, updateArgs, 2);
    artsSignalEdt(updateGuid, 0, depv[i+1].guid); // tileGuid
    artsSignalEdt(updateGuid, 1, readOnly);
    next = (next+1)%artsGetTotalNodes();
  }
}

void randomDriver(uint32_t paramc, uint64_t *paramv, uint32_t depc,
               artsEdtDep_t depv[]) {
  uint64_t numRemUpdates = paramv[0];
  uint64_t numRandom = (numRemUpdates > MAX_UPDATES_PER_CPU_STEP)
                          ? MAX_UPDATES_PER_CPU_STEP
                          : numRemUpdates; // Number of updates in the step
  uint64_t step = paramv[1];
  uint64_t index = paramv[2];

  int64_t startIndex =
      (int64_t)(step * MAX_UPDATES_PER_CPU_STEP +
                index * numRandom);
  artsCXLConsumerFlush(depv[0].guid);
  uint64_t *rArray = (uint64_t *)depv[0].ptr;
  uint64_t tableSize = TABLESIZE;

  //TODO: Probably need to separate hpcc and update edts. Need to wait for all hpcc
  // EDTs to finish before running update
  
  if (numRemUpdates) {
    // artsGuid_t nextGuid = artsReserveGuidRoute(ARTS_EDT, 0);
    uint64_t nextRandom = numRemUpdates - numRandom;
    uint64_t args[4] = {nextRandom, step+1, index, numTiles};
    // artsEdtCreateWithGuid(randomDriver, nextGuid, 4, args, numTiles+1);
    artsGuid_t nextGuid = artsEdtCreate(randomDriver, 0, 4, args, numTiles+1);
    artsSignalEdt(nextGuid, 0, updateFrontierGuid);

    uint64_t updateArgs[5] = {tileSize, numTiles, tableSize, numRandom, nextGuid};
    artsGuid_t updateGuid = artsEdtCreate(updateDriver, 0, 5, updateArgs, numTiles+1);
    hpccStarts(startIndex, numRandom, numTiles,
                tileSize, tableSize, (uint64_cu_t*) rArray, updateGuid);
    artsSignalEdt(updateGuid, 0, depv[0].guid);
  }
  else {
    artsSignalEdt(doneGuid, numTiles, NULL_GUID);
  }
}

void syncEdt(uint32_t paramc, uint64_t *paramv, uint32_t depc,
             artsEdtDep_t depv[]) {
  uint64_t time = artsGetTimeStamp() - start;
  PRINTF("Time %lu\n", time);
  
  for (unsigned int i = 0; i < numTiles; i++) {
    artsCXLConsumerFlush(depv[i].guid);
  }

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
              "FAILED on index:%lu Exp: %lu vs Rec: %lu updates: %lu -> %lu\n",
              index, tile[j], Table[index], tile[tileSize],
              Table[index] ^ tile[j]);
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
  if (!nodeId) {
    if (argc > 1)
      tileSize = (unsigned int)atoi(argv[1]);
    numTiles = TABLESIZE / tileSize;
    PRINTF("Random Access Table Size: %u Tile Size: %u Number of Tiles: %u\n",
           TABLESIZE, tileSize, numTiles);

    // Create tiled table
    tileGuids = (artsGuid_t*)calloc(numTiles, sizeof(artsGuid_t)); // TODO: Make sure to free
    tile = (uint64_t **)calloc(numTiles, sizeof(uint64_t *));
    uint64_t counter = 0;
    
    for (unsigned int i = 0; i < numTiles; i++) {
      
      tileGuids[i] = artsDbCreate((void**)&(tile[i]), (tileSize + 1)*sizeof(uint64_t),
                                 ARTS_DB_CXL);

      for (unsigned int j = 0; j < tileSize; j++)
        tile[i][j] = counter++;
      tile[i][tileSize] = 0;
      artsCXLProducerFlush(tileGuids[i]);
    }
    
    // Create update frontiers.  The number of updates a frontier can hold is 1024
    // per thread
    unsigned int elemsPerFrontier = numTiles + MAX_TOTAL_PENDING_UPDATES;
    uint64_t *updateFrontier;
    updateFrontierGuid = artsDbCreate((void**)&updateFrontier, elemsPerFrontier*sizeof(uint64_t), ARTS_DB_CXL);
    for (unsigned int j = 0; j < elemsPerFrontier; j++)
      updateFrontier[j] = 0;
    artsCXLProducerFlush(updateFrontierGuid);

    // Create a LC sync edt for all partitions
    doneGuid = artsReserveGuidRoute(ARTS_EDT, 0);
    updateGuid = artsReserveGuidRoute(ARTS_EDT, 0);
  }
}

void initPerWorker(unsigned int nodeId, unsigned int workerId,
                              int argc, char **argv) {
  if (!nodeId && !workerId) {
    PRINTF("Num updates: %lu\n", NUPDATE);
    uint64_t args[] = {NUPDATE, 0, 0, numTiles};
    artsEdtCreateWithGuid(randomDriver, updateGuid, 4, args, 1); 
    artsSignalEdt(updateGuid, 0, updateFrontierGuid);

    artsEdtCreateWithGuid(syncEdt, doneGuid, 0, NULL, 1 + numTiles);
    for (unsigned int i = 0; i < numTiles; i++) {
      artsSignalEdt(doneGuid, i, tileGuids[i]);
    }
  }
  start = artsGetTimeStamp();
}

int main(int argc, char **argv) {
  artsRT(argc, argv);
  return 0;
}