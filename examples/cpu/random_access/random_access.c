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
 * exercise the MFLOPS capability of a computer.
 *
 * GUPS is calculated by identifying the number of memory locations that can be
 * randomly updated in one second, divided by 1 billion (1e9). The term
 * "randomly" means that there is little relationship between one address to be
 * updated and the next, except that they occur in the space of one half the
 * total system memory.  An update is a read-modify-write operation on a table
 * of 64-bit words. An address is generated, the value at that address read from
 * memory, modified by an XOR with the random value, and that new value is
 * written back to memory.
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "arts.h"
#include "arts/memory/db.h"

#include "random_access_defs.h"

/* =====================================================================
 * Global state (set during init_per_node / init_per_worker)
 * ===================================================================== */

static unsigned int tileSize = TILESIZE;
static unsigned int numTiles = 0;

/* Per-tile datablocks for the random-access table */
static arts_guid_t *tileGuids = NULL;
static uint64_t   **tile      = NULL;

/* The update-frontier datablock holds:
 *   [0 .. numTiles-1]          : per-tile update counts (histogram)
 *   [numTiles .. numTiles+NUPDATE-1] : generated random values
 */
static arts_guid_t updateFrontierGuid = NULL_GUID;

/* Macros to map a random value to a tile index and intra-tile offset */
#define getLocalIndex(v)  (((v) & (TABLESIZE - 1)) % tileSize)
#define getOwnerIndex(v)  (((v) & (TABLESIZE - 1)) / tileSize)

static uint64_t    start    = 0;
static arts_guid_t doneGuid = NULL_GUID;
static arts_guid_t updateGuid = NULL_GUID;

/* =====================================================================
 * hpccStartsTiled
 *
 * EDT that computes a batch of random values for one tile's worth of
 * updates and stores them in the shared update-frontier datablock.
 *
 * paramv[0] = N (starting step index for hpccStarts)
 * paramv[1] = index (which tile / batch this EDT handles)
 * paramv[2] = numUpdates (total updates in this step)
 * paramv[3] = numTiles
 * paramv[4] = tableSize
 * paramv[5] = tileSize
 * paramv[6] = (uint64_t) rArray pointer (update-frontier data)
 * paramv[7] = updateGuid (EDT to signal when done)
 *
 * depv[0]   = tile datablock (read to get tileGuid for flush)
 * depv[1]   = updateFrontier datablock (read/write)
 * ===================================================================== */
void hpccStartsTiled(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                     arts_edt_dep_t depv[]) {
    (void)paramc;
    (void)depc;

    int64_ra_t  N          = (int64_ra_t)paramv[0];
    uint64_ra_t index      = (uint64_ra_t)paramv[1];
    uint64_ra_t numUpdates = (uint64_ra_t)paramv[2];
    uint64_ra_t nTiles     = (uint64_ra_t)paramv[3];
    uint64_ra_t tableSize  = (uint64_ra_t)paramv[4];
    uint64_ra_t tSz        = (uint64_ra_t)paramv[5];
    uint64_ra_t *rArray    = (uint64_ra_t *)paramv[6];
    arts_guid_t nextGuid   = (arts_guid_t)paramv[7];

    arts_guid_t tileGuid   = depv[0].guid;

    /* rArray layout:
     *   [0 .. nTiles-1]           : per-tile update counts (histogram)
     *   [nTiles .. nTiles+numUpdates-1] : generated random values
     */
    uint64_ra_t *globalRan = rArray + nTiles;

    for (uint64_ra_t local = 0; local < MAX_TOTAL_PENDING_UPDATES; local++) {
        int64_ra_t localIndex = (int64_ra_t)(index * MAX_TOTAL_PENDING_UPDATES + local);
        if ((uint64_ra_t)localIndex < numUpdates) {
            /* Compute the random value at step N + localIndex */
            volatile int64_ra_t n = N + localIndex;

            int i, j;
            uint64_ra_t m2[64];
            uint64_ra_t temp;
            uint64_ra_t ran = 0x1ULL;

            while (n < 0)
                n += PERIOD;
            while (n > PERIOD)
                n -= PERIOD;

            if (n) {
                temp = 0x1ULL;
                for (i = 0; i < 64; i++) {
                    m2[i] = temp;
                    temp = (temp << 1) ^ ((int64_ra_t)temp < 0 ? POLY : 0);
                    temp = (temp << 1) ^ ((int64_ra_t)temp < 0 ? POLY : 0);
                }

                for (i = 62; i >= 0; i--)
                    if ((n >> i) & 1)
                        break;

                ran = 0x2ULL;
                while (i > 0) {
                    temp = 0;
                    for (j = 0; j < 64; j++)
                        if ((ran >> j) & 1)
                            temp ^= m2[j];
                    ran = temp;
                    i -= 1;
                    if ((n >> i) & 1)
                        ran = (ran << 1) ^ ((int64_ra_t)ran < 0 ? POLY : 0);
                }
            } else {
                ran = 0x1ULL;
            }

            ran = (ran << 1) ^ ((int64_ra_t)ran < 0 ? POLY : 0);

            /* Store the random value and increment the owner tile's count */
            globalRan[localIndex] = ran;
            uint64_ra_t owner = getOwnerIndex(ran);
            rArray[owner] += 1ULL;
        }
    }

#if CXL_DB
    arts_cxl_producer_flush(tileGuid);
#endif
    /* Signal the updateDriver EDT at slot (index+1) with the tile GUID */
    arts_signal_edt(nextGuid, (uint32_t)(index + 1), tileGuid, DB_MODE_RO);
}

/* =====================================================================
 * hpccStarts
 *
 * Helper that spawns one hpccStartsTiled EDT per tile.
 *
 * N          = starting step index
 * numUpdates = number of updates in this step
 * nTiles     = number of tiles
 * tSz        = tile size
 * tableSize  = total table size
 * rArray     = pointer to the update-frontier data
 * nextGuid   = updateDriver EDT to signal when all tiles are done
 * ===================================================================== */
static void hpccStarts(int64_ra_t N, uint64_ra_t numUpdates,
                       uint64_ra_t nTiles, uint64_ra_t tSz,
                       uint64_ra_t tableSize, uint64_ra_t *rArray,
                       arts_guid_t nextGuid) {
    unsigned int next = arts_get_current_node();
    for (uint64_ra_t index = 0; index < nTiles; index++) {
        uint64_t args[8] = {
            (uint64_t)N,
            (uint64_t)index,
            (uint64_t)numUpdates,
            (uint64_t)nTiles,
            (uint64_t)tableSize,
            (uint64_t)tSz,
            (uint64_t)rArray,
            (uint64_t)nextGuid
        };
        arts_guid_t hpccTiledGuid = arts_edt_create(
            hpccStartsTiled, 8, args, 2,
            &(arts_hint_t){.route = next});
        arts_signal_edt(hpccTiledGuid, 0, tileGuids[index], DB_MODE_RO);
        arts_signal_edt(hpccTiledGuid, 1, updateFrontierGuid, DB_MODE_RO);
        next = (next + 1) % arts_get_total_nodes();
    }
}

/* =====================================================================
 * updateEdt
 *
 * EDT that applies the random updates belonging to one tile partition.
 * For each random value in the frontier, if it maps to this tile, XOR
 * the table entry with the random value.
 *
 * paramv[0] = tileSize
 * paramv[1] = numTiles
 * paramv[2] = tableSize
 * paramv[3] = numUpdates (number of random values in the frontier)
 * paramv[4] = partIndex  (which tile this EDT owns)
 * paramv[5] = nextGuid   (EDT to signal when done)
 * paramv[6] = slot       (slot in nextGuid to signal)
 *
 * depv[0]   = tile datablock (read-write)
 * depv[1]   = updateFrontier datablock (read-only)
 * ===================================================================== */
void updateEdt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
    (void)paramc;
    (void)depc;

    uint64_ra_t tSz        = (uint64_ra_t)paramv[0];
    uint64_ra_t nTiles     = (uint64_ra_t)paramv[1];
    uint64_ra_t tableSize  = (uint64_ra_t)paramv[2];
    uint64_ra_t numUpdates = (uint64_ra_t)paramv[3];
    uint64_ra_t partIndex  = (uint64_ra_t)paramv[4];
    arts_guid_t nextGuid   = (arts_guid_t)paramv[5];
    uint32_t    slot       = (uint32_t)paramv[6];

#if CXL_DB
    arts_cxl_consumer_flush(depv[0].guid);  /* flush tile before accessing */
    arts_cxl_consumer_flush(depv[1].guid);  /* flush frontier before accessing */
#endif

    uint64_ra_t *table = (uint64_ra_t *)depv[0].ptr;
    uint64_ra_t *ran   = (uint64_ra_t *)depv[1].ptr;
    ran += nTiles;  /* skip the per-tile histogram; point at random values */

    for (uint64_ra_t local = 0; local < numUpdates; local++) {
        uint64_ra_t localRan       = ran[local];
        uint64_ra_t globalRanIndex = localRan & (tableSize - 1);
        if (globalRanIndex / tSz == partIndex) {
            uint64_ra_t localRanIndex = globalRanIndex % tSz;
            __atomic_fetch_xor(&table[localRanIndex], localRan, __ATOMIC_SEQ_CST);
        }
    }

    arts_signal_edt(nextGuid, slot, NULL_GUID, DB_MODE_NULL);

#if CXL_DB
    arts_cxl_producer_flush(depv[0].guid);
#endif
}

/* =====================================================================
 * updateDriver
 *
 * EDT that fans out one updateEdt per tile after all hpccStartsTiled
 * EDTs have completed (i.e., the frontier is fully populated).
 *
 * paramv[0] = tileSize
 * paramv[1] = numTiles
 * paramv[2] = tableSize
 * paramv[3] = numRandom (number of updates in this step)
 * paramv[4] = nextRandomGuid (randomDriver EDT for the next step)
 *
 * depv[0]         = updateFrontier datablock (read-only)
 * depv[1..nTiles] = tile datablocks (one per tile, read-write)
 * ===================================================================== */
void updateDriver(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
    (void)paramc;
    (void)depc;

    uint64_t tSz           = paramv[0];
    uint64_t nTiles        = paramv[1];
    uint64_t tableSize     = paramv[2];
    uint64_t numRandom     = paramv[3];
    arts_guid_t nextRandGuid = (arts_guid_t)paramv[4];

#if CXL_DB
    arts_cxl_consumer_flush(depv[0].guid);
    for (unsigned int i = 0; i < nTiles; i++)
        arts_cxl_consumer_flush(depv[i + 1].guid);
#endif

    arts_guid_t readOnly = depv[0].guid;  /* updateFrontier */

    /* updateArgs: [tileSize, numTiles, tableSize, numRandom, partIndex,
     *              nextGuid, slot] */
    uint64_t updateArgs[7] = {tSz, nTiles, tableSize, numRandom, 0,
                               (uint64_t)nextRandGuid, 0};
    unsigned int next = arts_get_current_node();
    for (uint64_t i = 0; i < nTiles; i++) {
        updateArgs[4] = i;
        updateArgs[6] = i + 1;  /* slot in randomDriver: 0 is frontier, 1..nTiles are tiles */
        arts_guid_t updGuid = arts_edt_create(
            updateEdt, 7, updateArgs, 2,
            &(arts_hint_t){.route = next});
        arts_signal_edt(updGuid, 0, depv[i + 1].guid, DB_MODE_RO); /* tile */
        arts_signal_edt(updGuid, 1, readOnly, DB_MODE_RO);          /* frontier */
        next = (next + 1) % arts_get_total_nodes();
    }
}

/* =====================================================================
 * randomDriver
 *
 * Recursive EDT that drives one step of the random-access benchmark.
 * Each step processes up to MAX_UPDATES_PER_CPU_STEP updates.
 *
 * paramv[0] = numRemUpdates (remaining updates after this step)
 * paramv[1] = step          (current step index, for computing start offset)
 * paramv[2] = index         (sub-index within step, currently always 0)
 * paramv[3] = numTiles
 *
 * depv[0]         = updateFrontier datablock
 * ===================================================================== */
void randomDriver(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
    (void)paramc;
    (void)depc;

    uint64_t numRemUpdates = paramv[0];
    uint64_t numRandom     = (numRemUpdates > MAX_UPDATES_PER_CPU_STEP)
                                 ? MAX_UPDATES_PER_CPU_STEP
                                 : numRemUpdates;
    uint64_t step          = paramv[1];
    uint64_t index         = paramv[2];
    uint64_t nTiles        = paramv[3];

    int64_ra_t startIndex = (int64_ra_t)(step * MAX_UPDATES_PER_CPU_STEP
                                         + index * numRandom);

#if CXL_DB
    arts_cxl_consumer_flush(depv[0].guid);
#endif
    uint64_t *rArray   = (uint64_t *)depv[0].ptr;
    uint64_t tableSize = TABLESIZE;

    if (numRemUpdates) {
        uint64_t nextRem  = numRemUpdates - numRandom;
        uint64_t nextArgs[4] = {nextRem, step + 1, index, nTiles};
        arts_guid_t nextGuid = arts_edt_create(
            randomDriver, 4, nextArgs, (uint32_t)(nTiles + 1),
            &(arts_hint_t){.route = 0});
        /* Slot 0 of randomDriver is the frontier; slots 1..nTiles are tiles */
        arts_signal_edt(nextGuid, 0, updateFrontierGuid, DB_MODE_RO);

        uint64_t updateArgs[5] = {tileSize, nTiles, tableSize, numRandom,
                                   (uint64_t)nextGuid};
        arts_guid_t updDriverGuid = arts_edt_create(
            updateDriver, 5, updateArgs, (uint32_t)(nTiles + 1),
            &(arts_hint_t){.route = 0});

        /* Slot 0 of updateDriver is the frontier */
        arts_signal_edt(updDriverGuid, 0, depv[0].guid, DB_MODE_RO);

        /* Spawn hpccStartsTiled EDTs; each signals updateDriver at slot (i+1) */
        hpccStarts(startIndex, numRandom, nTiles, tileSize, tableSize,
                   (uint64_ra_t *)rArray, updDriverGuid);
    } else {
        /* All updates done — signal the sync/done EDT */
        arts_signal_edt(doneGuid, (uint32_t)nTiles, NULL_GUID, DB_MODE_NULL);
    }
}

/* =====================================================================
 * syncEdt
 *
 * Final EDT: prints timing and optionally validates the result.
 *
 * depc = 1 + numTiles
 *   depv[0 .. numTiles-1] = tile datablocks
 *   depv[numTiles]        = NULL_GUID sentinel (from randomDriver)
 * ===================================================================== */
void syncEdt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
             arts_edt_dep_t depv[]) {
    (void)paramc;
    (void)paramv;
    (void)depc;

    uint64_t time = arts_get_time_stamp() - start;
    arts_printf("Time %lu ns\n", time);

#if CXL_DB
    for (unsigned int i = 0; i < numTiles; i++)
        arts_cxl_consumer_flush(depv[i].guid);
#endif

#ifdef VALIDATE
    /* Build the expected table by replaying the sequential LFSR */
    uint64_t *Table = (uint64_t *)calloc(TABLESIZE, sizeof(uint64_t));
    for (uint64_t i = 0; i < TABLESIZE; i++)
        Table[i] = i;

    uint64_t temp      = 0x1ULL;
    uint64_t tableSize = TABLESIZE;
    for (uint64_t i = 0; i < NUPDATE; i++) {
        temp = (temp << 1) ^ (((int64_t)temp < 0) ? POLY : 0);
        Table[temp & (tableSize - 1)] ^= temp;
    }

    bool     firstFailure = true;
    uint64_t totalErrors  = 0;
    uint64_t idx          = 0;
    for (unsigned int i = 0; i < numTiles; i++) {
        uint64_t *tilePart = (uint64_t *)depv[i].ptr;
        for (unsigned int j = 0; j < tileSize; j++) {
            if (tilePart[j] != Table[idx]) {
                if (firstFailure) {
                    firstFailure = false;
                    arts_printf(
                        "FAILED on index:%lu Expected: %lu Received: %lu\n",
                        idx, Table[idx], tilePart[j]);
                }
                totalErrors++;
            }
            idx++;
        }
    }
    free(Table);

    if (totalErrors)
        arts_printf("%lu errors of %lu!\n", totalErrors, idx);
    else
        arts_printf("Verified!\n");
#endif /* VALIDATE */

    double GUPS = (double)NUPDATE / (double)time;
    arts_printf("GUPS: %lf  Table MB: %lu\n", GUPS,
                (TABLESIZE * sizeof(uint64_t)) / (1024UL * 1024UL));

    arts_shutdown();
}

/* =====================================================================
 * init_per_node
 *
 * Called once per node before workers start.  Node 0 allocates all
 * tile datablocks and the update-frontier datablock, then reserves
 * the done and update GUIDs.
 * ===================================================================== */
void init_per_node(unsigned int nodeId, int argc, char **argv) {
    if (!nodeId) {
        if (argc > 1)
            tileSize = (unsigned int)atoi(argv[1]);
        numTiles = (unsigned int)(TABLESIZE / tileSize);

        arts_printf(
            "Random Access  TableSize: %lu  TileSize: %u  NumTiles: %u\n",
            (unsigned long)TABLESIZE, tileSize, numTiles);

        /* Allocate tile datablocks */
        tileGuids = (arts_guid_t *)calloc(numTiles, sizeof(arts_guid_t));
        tile      = (uint64_t **)calloc(numTiles, sizeof(uint64_t *));

        uint64_t counter = 0;
        for (unsigned int i = 0; i < numTiles; i++) {
#if CXL_DB
            tileGuids[i] = arts_db_create((void **)&tile[i],
                                          tileSize * sizeof(uint64_t),
                                          ARTS_DB_CXL, NULL);
#else
            tileGuids[i] = arts_db_create((void **)&tile[i],
                                          tileSize * sizeof(uint64_t),
                                          ARTS_DB_DEFAULT, NULL);
#endif
            for (unsigned int j = 0; j < tileSize; j++)
                tile[i][j] = counter++;
#if CXL_DB
            arts_cxl_producer_flush(tileGuids[i]);
#endif
        }

        /* Allocate the update-frontier datablock:
         *   [0 .. numTiles-1]                  : per-tile update counts
         *   [numTiles .. numTiles+NUPDATE-1]   : generated random values
         */
        unsigned int elemsPerFrontier = (unsigned int)(numTiles + MAX_TOTAL_PENDING_UPDATES);
        uint64_t *updateFrontier;
#if CXL_DB
        updateFrontierGuid = arts_db_create((void **)&updateFrontier,
                                            elemsPerFrontier * sizeof(uint64_t),
                                            ARTS_DB_CXL, NULL);
#else
        updateFrontierGuid = arts_db_create((void **)&updateFrontier,
                                            elemsPerFrontier * sizeof(uint64_t),
                                            ARTS_DB_DEFAULT, NULL);
#endif
        for (unsigned int j = 0; j < elemsPerFrontier; j++)
            updateFrontier[j] = 0;
#if CXL_DB
        arts_cxl_producer_flush(updateFrontierGuid);
#endif

        /* Reserve GUIDs for the done and update EDTs on node 0 */
        doneGuid   = arts_guid_reserve(ARTS_EDT, 0);
        updateGuid = arts_guid_reserve(ARTS_EDT, 0);
    }
}

/* =====================================================================
 * init_per_worker
 *
 * Called on each worker thread after the startup barrier.
 * Worker 0 on node 0 creates the done and randomDriver EDTs and
 * kicks off the computation.
 * ===================================================================== */
void init_per_worker(unsigned int nodeId, unsigned int workerId,
                     int argc, char **argv) {
    (void)argc;
    (void)argv;

    if (!nodeId && !workerId) {
        arts_printf("Num updates: %lu\n", (unsigned long)NUPDATE);

        /* syncEdt waits for all tiles (depv[0..numTiles-1]) plus one
         * NULL signal from randomDriver (depv[numTiles]) */
        arts_edt_create_with_guid(syncEdt, doneGuid, 0, NULL,
                                  numTiles + 1);
        for (unsigned int i = 0; i < numTiles; i++)
            arts_signal_edt(doneGuid, (uint32_t)i, tileGuids[i], DB_MODE_RO);

        /* randomDriver: depv[0] = frontier, depv[1..numTiles] = tiles
         * (tiles are signaled by updateEdt as they complete) */
        uint64_t args[4] = {NUPDATE, 0, 0, numTiles};
        arts_edt_create_with_guid(randomDriver, updateGuid, 4, args,
                                  numTiles + 1);
        arts_signal_edt(updateGuid, 0, updateFrontierGuid, DB_MODE_RO);
    }
    start = arts_get_time_stamp();
}

/* ===================================================================== */

int main(int argc, char **argv) {
    arts_rt(argc, argv);
    return 0;
}