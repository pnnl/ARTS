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

/*-----------------------------------------------------------------------*/
/* Program: CXL Stream                                                   */
/* Based on the STREAM benchmark by John D. McCalpin.                    */
/* Adapted for the new ARTS runtime with CXL shared-memory support.      */
/*                                                                       */
/* This program measures memory transfer rates in MB/s for simple        */
/* computational kernels coded in C, using CXL datablocks.               */
/*                                                                       */
/* Data is divided into tiles. Each tile is stored as an ARTS_DB_CXL     */
/* datablock. Kernel EDTs operate on individual tiles and signal the     */
/* next launcher EDT upon completion. The benchmark chains NTIMES        */
/* iterations of Copy -> Scale -> Add -> Triad in reverse order so that  */
/* the first kernel to fire triggers the entire chain.                   */
/*-----------------------------------------------------------------------*/

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "arts.h"
#include "arts/memory/db.h"

/* =====================================================================
 * Configuration (overridable via -D at compile time)
 * ===================================================================== */

#ifndef N
#define N (1 << 20)
#endif

#ifndef TILESIZE
#define TILESIZE 131072
#endif

#ifndef NTIMES
#define NTIMES 20
#endif

#define OFFSET 0
#define M 20

#define HLINE "-------------------------------------------------------------\n"

#ifndef MIN
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#endif
#ifndef MAX
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#endif

/* =====================================================================
 * CXL mode toggle:
 *   CXL_DB 1  — use ARTS_DB_CXL datablocks + producer/consumer flushes
 *   CXL_DB 0  — use ARTS_DB_DEFAULT datablocks (no flushes), useful for
 *               testing without CXL hardware
 * ===================================================================== */
#ifndef CXL_DB
#define CXL_DB 1
#endif

static const char *label[4] = {
    "Copy:      ", "Scale:     ", "Add:       ", "Triad:     "};

static double bytes[4] = {2 * sizeof(double) * N, 2 * sizeof(double) * N,
                           3 * sizeof(double) * N, 3 * sizeof(double) * N};

/* =====================================================================
 * Global state (set during init_per_node / init_per_worker)
 * ===================================================================== */

static unsigned int tileSize = TILESIZE;
static unsigned int numTiles = 0;

static arts_guid_t *aTileGuids = NULL;
static arts_guid_t *bTileGuids = NULL;
static arts_guid_t *cTileGuids = NULL;

/* doneGuid: the final "done" EDT that validates results */
static arts_guid_t doneGuid = NULL_GUID;

/* firstKernel: the first launcher EDT in the chain; signaled by workers */
static arts_guid_t firstKernel = NULL_GUID;

/* Host-side tile pointers (only valid on node 0 for CXL_DB mode) */
static double **aTile = NULL;
static double **bTile = NULL;
static double **cTile = NULL;

/* Timing arrays */
static double times[4][NTIMES];

static int quantum = 0;

/* =====================================================================
 * Utility
 * ===================================================================== */

static double mysecond(void) {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

static int checktick(void) {
    int i, minDelta, Delta;
    double t1, t2, timesfound[M];

    for (i = 0; i < M; i++) {
        t1 = mysecond();
        while (((t2 = mysecond()) - t1) < 1.0E-6)
            ;
        timesfound[i] = t1 = t2;
    }

    minDelta = 1000000;
    for (i = 1; i < M; i++) {
        Delta = (int)(1.0E6 * (timesfound[i] - timesfound[i - 1]));
        minDelta = MIN(minDelta, MAX(Delta, 0));
    }
    return minDelta;
}

static void checkSTREAMresults(unsigned int tSz, unsigned int totalSize,
                                double **aT, double **bT, double **cT) {
    double aj, bj, cj, scalar;
    double asum, bsum, csum;
    double epsilon;
    int k;

    aj = 1.0;
    bj = 2.0;
    cj = 0.0;
    aj = 2.0E0 * aj;
    scalar = 3.0;
    for (k = 0; k < NTIMES; k++) {
        cj = aj;
        bj = scalar * cj;
        cj = aj + bj;
        aj = bj + scalar * cj;
    }
    aj = aj * (double)(N);
    bj = bj * (double)(N);
    cj = cj * (double)(N);

    asum = 0.0;
    bsum = 0.0;
    csum = 0.0;

    unsigned int nTiles = totalSize / tSz;
    if (totalSize % tSz)
        nTiles++;

    for (unsigned int i = 0; i < nTiles; i++) {
        unsigned int end = (i + 1 < nTiles) ? tSz : totalSize - i * tSz;
        for (unsigned int j = 0; j < end; j++) {
            asum += aT[i][j];
            bsum += bT[i][j];
            csum += cT[i][j];
        }
    }

#define abs_val(a) ((a) >= 0 ? (a) : -(a))
    epsilon = 1.e-8;

    if (abs_val(aj - asum) / asum > epsilon) {
        arts_printf("Failed Validation on array a[]\n");
        arts_printf("        Expected  : %f \n", aj);
        arts_printf("        Observed  : %f \n", asum);
    } else if (abs_val(bj - bsum) / bsum > epsilon) {
        arts_printf("Failed Validation on array b[]\n");
        arts_printf("        Expected  : %f \n", bj);
        arts_printf("        Observed  : %f \n", bsum);
    } else if (abs_val(cj - csum) / csum > epsilon) {
        arts_printf("Failed Validation on array c[]\n");
        arts_printf("        Expected  : %f \n", cj);
        arts_printf("        Observed  : %f \n", csum);
    } else {
        arts_printf("Solution Validates\n");
    }
}

/* =====================================================================
 * Forward declarations
 * ===================================================================== */

void launch2KernelEdt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                      arts_edt_dep_t depv[]);
void launch3KernelEdt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                      arts_edt_dep_t depv[]);

/* =====================================================================
 * EDT: copyKernel
 *
 * Performs: b[i] = a[i]
 *
 * paramv[0] = (unused, reserved for event GUID in launch2KernelEdt)
 * paramv[1] = nextEdt GUID
 * paramv[2] = tile length (number of doubles)
 * paramv[3] = slot index in nextEdt to signal
 * depv[0]   = a tile (source)
 * depv[1]   = b tile (destination)
 * ===================================================================== */

void copyKernel(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
    (void)paramc;
    (void)depc;

    unsigned int len    = (unsigned int)paramv[2];
    arts_guid_t nextEdt = (arts_guid_t)paramv[1];
    uint32_t slot       = (uint32_t)paramv[3];

    double *a = (double *)depv[0].ptr;
    double *b = (double *)depv[1].ptr;

    for (unsigned int idx = 0; idx < len; idx++)
        b[idx] = a[idx];

#if CXL_DB
    arts_cxl_producer_flush(depv[0].guid);
    arts_cxl_producer_flush(depv[1].guid);
#endif
    arts_signal_edt(nextEdt, slot, NULL_GUID, DB_MODE_NULL);
}

/* =====================================================================
 * EDT: scaleKernel
 *
 * Performs: b[i] = scalar * a[i]
 *
 * paramv[0] = (unused)
 * paramv[1] = nextEdt GUID
 * paramv[2] = tile length
 * paramv[3] = slot index in nextEdt
 * paramv[4] = scalar (double bits reinterpreted as uint64_t)
 * depv[0]   = a tile (source)
 * depv[1]   = b tile (destination)
 * ===================================================================== */

void scaleKernel(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
    (void)paramc;
    (void)depc;

    unsigned int len    = (unsigned int)paramv[2];
    arts_guid_t nextEdt = (arts_guid_t)paramv[1];
    uint32_t slot       = (uint32_t)paramv[3];
    double scale;
    memcpy(&scale, &paramv[4], sizeof(double));

    double *a = (double *)depv[0].ptr;
    double *b = (double *)depv[1].ptr;

    for (unsigned int idx = 0; idx < len; idx++)
        b[idx] = scale * a[idx];

#if CXL_DB
    arts_cxl_producer_flush(depv[0].guid);
    arts_cxl_producer_flush(depv[1].guid);
#endif
    arts_signal_edt(nextEdt, slot, NULL_GUID, DB_MODE_NULL);
}

/* =====================================================================
 * EDT: addKernel
 *
 * Performs: c[i] = a[i] + b[i]
 *
 * paramv[0] = (unused)
 * paramv[1] = nextEdt GUID
 * paramv[2] = tile length
 * paramv[3] = slot index in nextEdt
 * depv[0]   = a tile
 * depv[1]   = b tile
 * depv[2]   = c tile (destination)
 * ===================================================================== */

void addKernel(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
    (void)paramc;
    (void)depc;

    unsigned int len    = (unsigned int)paramv[2];
    arts_guid_t nextEdt = (arts_guid_t)paramv[1];
    uint32_t slot       = (uint32_t)paramv[3];

    double *a = (double *)depv[0].ptr;
    double *b = (double *)depv[1].ptr;
    double *c = (double *)depv[2].ptr;

    for (unsigned int idx = 0; idx < len; idx++)
        c[idx] = a[idx] + b[idx];

#if CXL_DB
    arts_cxl_producer_flush(depv[0].guid);
    arts_cxl_producer_flush(depv[1].guid);
    arts_cxl_producer_flush(depv[2].guid);
#endif
    arts_signal_edt(nextEdt, slot, NULL_GUID, DB_MODE_NULL);
}

/* =====================================================================
 * EDT: triadKernel
 *
 * Performs: a[i] = b[i] + scalar * c[i]
 *
 * paramv[0] = (unused)
 * paramv[1] = nextEdt GUID
 * paramv[2] = tile length
 * paramv[3] = slot index in nextEdt
 * paramv[4] = scalar (double bits reinterpreted as uint64_t)
 * depv[0]   = b tile
 * depv[1]   = c tile
 * depv[2]   = a tile (destination)
 *
 * CXL mode: after flushing, signals done with the actual tile GUIDs
 *   (3 signals per tile: a at slot, b at numTiles+slot, c at 2*numTiles+slot).
 *
 * Non-CXL mode: if nextEdt != doneGuid, signals with NULL_GUID.
 *   If nextEdt == doneGuid (last iteration), copies tile data into fresh
 *   ARTS_DB_DEFAULT datablocks and signals done with those 3 DBs.
 * ===================================================================== */

void triadKernel(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
    (void)paramc;
    (void)depc;

    unsigned int len    = (unsigned int)paramv[2];
    arts_guid_t nextEdt = (arts_guid_t)paramv[1];
    uint32_t slot       = (uint32_t)paramv[3];
    double scale;
    memcpy(&scale, &paramv[4], sizeof(double));

    double *b = (double *)depv[0].ptr;
    double *c = (double *)depv[1].ptr;
    double *a = (double *)depv[2].ptr;

    for (unsigned int idx = 0; idx < len; idx++)
        a[idx] = b[idx] + scale * c[idx];

#if CXL_DB
    arts_cxl_producer_flush(depv[0].guid);  /* b tile */
    arts_cxl_producer_flush(depv[1].guid);  /* c tile */
    arts_cxl_producer_flush(depv[2].guid);  /* a tile (result) */
    /* Signal done with the actual tile GUIDs so done can validate */
    arts_signal_edt(nextEdt, slot,                  depv[2].guid, DB_MODE_RO); /* a */
    arts_signal_edt(nextEdt, numTiles + slot,        depv[0].guid, DB_MODE_RO); /* b */
    arts_signal_edt(nextEdt, (2 * numTiles) + slot,  depv[1].guid, DB_MODE_RO); /* c */
#else
    if (nextEdt != doneGuid) {
        arts_signal_edt(nextEdt, slot, NULL_GUID, DB_MODE_NULL);
    } else {
        /* Last triad iteration: copy tile data into read-only DBs for done EDT */
        double *aCopy, *bCopy, *cCopy;
        arts_guid_t aSignal = arts_db_create((void **)&aCopy,
                                             tileSize * sizeof(double),
                                             ARTS_DB_DEFAULT, NULL);
        arts_guid_t bSignal = arts_db_create((void **)&bCopy,
                                             tileSize * sizeof(double),
                                             ARTS_DB_DEFAULT, NULL);
        arts_guid_t cSignal = arts_db_create((void **)&cCopy,
                                             tileSize * sizeof(double),
                                             ARTS_DB_DEFAULT, NULL);
        memcpy(aCopy, depv[2].ptr, sizeof(double) * tileSize);
        memcpy(bCopy, depv[0].ptr, sizeof(double) * tileSize);
        memcpy(cCopy, depv[1].ptr, sizeof(double) * tileSize);

        arts_signal_edt(nextEdt, slot,                    aSignal, DB_MODE_RO);
        arts_signal_edt(nextEdt, numTiles + slot,         bSignal, DB_MODE_RO);
        arts_signal_edt(nextEdt, (2 * numTiles) + slot,   cSignal, DB_MODE_RO);
    }
#endif
}

/* =====================================================================
 * EDT: done
 *
 * Prints timing summary and validates results.
 *
 * paramv: none
 *
 * depc = 3 * numTiles (both CXL and non-CXL modes)
 *   depv[0 .. numTiles-1]           = a tiles
 *   depv[numTiles .. 2*numTiles-1]  = b tiles
 *   depv[2*numTiles .. 3*numTiles-1]= c tiles
 *
 * In CXL mode, consumer flushes are performed before reading depv data.
 * ===================================================================== */

void done(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
          arts_edt_dep_t depv[]) {
    (void)paramc;
    (void)paramv;
    (void)depc;

    int j, k;

    /* --- SUMMARY --- */
    double avgtime[4] = {0};
    double maxtime[4] = {0};
    double mintime[4] = {FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX};

    for (k = 1; k < NTIMES; k++) {
        for (j = 0; j < 4; j++) {
            avgtime[j] += times[j][k];
            mintime[j] = MIN(mintime[j], times[j][k]);
            maxtime[j] = MAX(maxtime[j], times[j][k]);
        }
    }

    arts_printf("Function      Rate (MB/s)   Avg time     Min time     Max time\n");
    for (j = 0; j < 4; j++) {
        avgtime[j] = avgtime[j] / (double)(NTIMES - 1);
        arts_printf("%s%11.4f  %11.4f  %11.4f  %11.4f\n", label[j],
                    1.0E-06 * bytes[j] / mintime[j],
                    avgtime[j], mintime[j], maxtime[j]);
    }
    arts_printf(HLINE);

    /* --- Validation --- */
    double **aTileAll = malloc(sizeof(double *) * numTiles);
    double **bTileAll = malloc(sizeof(double *) * numTiles);
    double **cTileAll = malloc(sizeof(double *) * numTiles);

    for (unsigned int i = 0; i < numTiles; i++) {
        aTileAll[i] = malloc(sizeof(double) * tileSize);
        bTileAll[i] = malloc(sizeof(double) * tileSize);
        cTileAll[i] = malloc(sizeof(double) * tileSize);
#if CXL_DB
        /* Flush CXL cache lines before reading tile data from depv */
        arts_cxl_consumer_flush(depv[i].guid);
        arts_cxl_consumer_flush(depv[numTiles + i].guid);
        arts_cxl_consumer_flush(depv[(2 * numTiles) + i].guid);
#endif
        memcpy(aTileAll[i], depv[i].ptr,                    sizeof(double) * tileSize);
        memcpy(bTileAll[i], depv[numTiles + i].ptr,         sizeof(double) * tileSize);
        memcpy(cTileAll[i], depv[(2 * numTiles) + i].ptr,   sizeof(double) * tileSize);
    }

    if (!arts_get_current_node()) {
        checkSTREAMresults(tileSize, N, aTileAll, bTileAll, cTileAll);
        arts_printf(HLINE);
    }

    for (unsigned int i = 0; i < numTiles; i++) {
        free(aTileAll[i]);
        free(bTileAll[i]);
        free(cTileAll[i]);
    }
    free(aTileAll);
    free(bTileAll);
    free(cTileAll);

    free(aTileGuids);
    free(bTileGuids);
    free(cTileGuids);
    free(aTile);
    free(bTile);
    free(cTile);

    arts_shutdown();
}

/* =====================================================================
 * EDT: launch2KernelEdt
 *
 * Intermediate launcher EDT for 2-array kernels (copy, scale).
 * Creates one kernel EDT per tile and signals them with their tile DBs.
 *
 * paramv[0] = kernel function pointer (arts_edt_t)
 * paramv[1] = tileSize
 * paramv[2] = totalSize (N)
 * paramv[3] = scalar bits (double reinterpreted as uint64_t; 0 if unused)
 * paramv[4] = aTileGuids pointer (artsGuid_t*)
 * paramv[5] = bTileGuids pointer (artsGuid_t*)
 * paramv[6] = nextGuid (next launcher EDT to signal when all tiles done)
 *
 * depc = numTiles (one slot per tile, all satisfied by streamDriver)
 * ===================================================================== */

void launch2KernelEdt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                      arts_edt_dep_t depv[]) {
    (void)paramc;
    (void)depc;
    (void)depv;

    arts_edt_t funPtr       = (arts_edt_t)paramv[0];
    unsigned int tSz        = (unsigned int)paramv[1];
    unsigned int totalSize  = (unsigned int)paramv[2];
    uint64_t scalarBits     = paramv[3];
    arts_guid_t *aGuid      = (arts_guid_t *)paramv[4];
    arts_guid_t *bGuid      = (arts_guid_t *)paramv[5];
    arts_guid_t nextGuid    = (arts_guid_t)paramv[6];

    unsigned int tiles = totalSize / tSz;
    if (totalSize % tSz)
        tiles++;

    unsigned int next = 0;
    arts_guid_t *edtGuids = (arts_guid_t *)malloc(sizeof(arts_guid_t) * tiles);

    /* Build per-tile args: [unused, nextGuid, tileLen, slotIdx, scalarBits] */
    uint64_t args[5] = {0, (uint64_t)nextGuid, tSz, 0, scalarBits};
    uint64_t numArgs = (scalarBits != 0) ? 5 : 4;

    for (unsigned int i = 0; i < tiles; ++i) {
        args[2] = (i + 1 < tiles) ? tSz : totalSize - i * tSz;
        args[3] = i;
        edtGuids[i] = arts_edt_create(funPtr, (uint32_t)numArgs, args, 2,
                                      &(arts_hint_t){.route = next});
        next = (next + 1) % arts_get_total_nodes();
        arts_signal_edt(edtGuids[i], 0, aGuid[i], DB_MODE_RO);
    }
    for (unsigned int i = 0; i < tiles; ++i) {
        arts_signal_edt(edtGuids[i], 1, bGuid[i], DB_MODE_RO);
    }
    free(edtGuids);
}

/* =====================================================================
 * EDT: launch3KernelEdt
 *
 * Intermediate launcher EDT for 3-array kernels (add, triad).
 *
 * paramv[0] = kernel function pointer (arts_edt_t)
 * paramv[1] = tileSize
 * paramv[2] = totalSize (N)
 * paramv[3] = scalar bits (double reinterpreted as uint64_t; 0 if unused)
 * paramv[4] = aTileGuids pointer
 * paramv[5] = bTileGuids pointer
 * paramv[6] = cTileGuids pointer
 * paramv[7] = nextGuid
 *
 * depc = numTiles
 * ===================================================================== */

void launch3KernelEdt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                      arts_edt_dep_t depv[]) {
    (void)paramc;
    (void)depc;
    (void)depv;

    arts_edt_t funPtr       = (arts_edt_t)paramv[0];
    unsigned int tSz        = (unsigned int)paramv[1];
    unsigned int totalSize  = (unsigned int)paramv[2];
    uint64_t scalarBits     = paramv[3];
    arts_guid_t *aGuid      = (arts_guid_t *)paramv[4];
    arts_guid_t *bGuid      = (arts_guid_t *)paramv[5];
    arts_guid_t *cGuid      = (arts_guid_t *)paramv[6];
    arts_guid_t nextGuid    = (arts_guid_t)paramv[7];

    unsigned int tiles = totalSize / tSz;
    if (totalSize % tSz)
        tiles++;

    unsigned int next = 0;
    arts_guid_t *edtGuids = (arts_guid_t *)malloc(sizeof(arts_guid_t) * tiles);

    uint64_t args[5] = {0, (uint64_t)nextGuid, tSz, 0, scalarBits};
    uint64_t numArgs = (scalarBits != 0) ? 5 : 4;

    for (unsigned int i = 0; i < tiles; ++i) {
        args[2] = (i + 1 < tiles) ? tSz : totalSize - i * tSz;
        args[3] = i;
        edtGuids[i] = arts_edt_create(funPtr, (uint32_t)numArgs, args, 3,
                                      &(arts_hint_t){.route = next});
        next = (next + 1) % arts_get_total_nodes();
        arts_signal_edt(edtGuids[i], 0, aGuid[i], DB_MODE_RO);
        arts_signal_edt(edtGuids[i], 1, bGuid[i], DB_MODE_RO);
    }
    for (unsigned int i = 0; i < tiles; ++i) {
        arts_signal_edt(edtGuids[i], 2, cGuid[i], DB_MODE_RO);
    }
    free(edtGuids);
}

/* =====================================================================
 * EDT: timerLaunch2KernelEdt
 *
 * Wrapper around launch2KernelEdt that records start/end times.
 * Stores start time, calls launch2KernelEdt logic, then records end time.
 *
 * paramv[0..6] = same as launch2KernelEdt
 * paramv[7]    = kernel index (0=copy, 1=scale)
 * paramv[8]    = iteration index
 * ===================================================================== */

void timerLaunch2KernelEdt(uint32_t paramc, const uint64_t *paramv,
                            uint32_t depc, arts_edt_dep_t depv[]) {
    uint64_t kernelIdx = paramv[7];
    uint64_t iterIdx   = paramv[8];

    times[kernelIdx][iterIdx] = mysecond();
    launch2KernelEdt(paramc, paramv, depc, depv);
    times[kernelIdx][iterIdx] = mysecond() - times[kernelIdx][iterIdx];
}

/* =====================================================================
 * EDT: timerLaunch3KernelEdt
 *
 * Wrapper around launch3KernelEdt that records start/end times.
 *
 * paramv[0..7] = same as launch3KernelEdt
 * paramv[8]    = kernel index (2=add, 3=triad)
 * paramv[9]    = iteration index
 * ===================================================================== */

void timerLaunch3KernelEdt(uint32_t paramc, const uint64_t *paramv,
                            uint32_t depc, arts_edt_dep_t depv[]) {
    uint64_t kernelIdx = paramv[8];
    uint64_t iterIdx   = paramv[9];

    times[kernelIdx][iterIdx] = mysecond();
    launch3KernelEdt(paramc, paramv, depc, depv);
    times[kernelIdx][iterIdx] = mysecond() - times[kernelIdx][iterIdx];
}

/* =====================================================================
 * EDT: streamDriver
 *
 * Builds the entire phase chain in reverse order (so that the first
 * launcher EDT in the chain is the last one created).  The chain is:
 *
 *   done <- triad_N-1 <- add_N-1 <- scale_N-1 <- copy_N-1
 *        <- triad_N-2 <- ...
 *        <- copy_0    (= firstKernel, signaled by initPerWorker)
 *
 * Each launcher EDT has depc = numTiles so that it fires only after all
 * tile kernel EDTs from the previous phase have completed.
 * ===================================================================== */

void streamDriver(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
    (void)paramc;
    (void)paramv;
    (void)depc;
    (void)depv;

    double scalar = 3.0;
    uint64_t scalarBits;
    memcpy(&scalarBits, &scalar, sizeof(double));

    unsigned int tiles = N / tileSize;
    if (N % tileSize)
        tiles++;

    unsigned int currentNode = arts_get_current_node();
    unsigned int numDeps     = tiles;

    arts_guid_t prevEdt = doneGuid;

    for (int k = NTIMES - 1; k >= 0; k--) {
        /* --- Triad: a[i] = b[i] + scalar * c[i] --- */
        /* paramv: [funPtr, tileSize, N, scalarBits, bGuids, cGuids, aGuids,
         *          nextGuid, kernelIdx=3, iterIdx=k] */
        uint64_t argsTriad[10] = {
            (uint64_t)triadKernel, tileSize, N, scalarBits,
            (uint64_t)bTileGuids, (uint64_t)cTileGuids, (uint64_t)aTileGuids,
            (uint64_t)prevEdt,
            3, (uint64_t)k
        };
        prevEdt = arts_edt_create(timerLaunch3KernelEdt, 10, argsTriad,
                                  numDeps,
                                  &(arts_hint_t){.route = currentNode});

        /* --- Add: c[i] = a[i] + b[i] --- */
        uint64_t argsAdd[10] = {
            (uint64_t)addKernel, tileSize, N, 0,
            (uint64_t)aTileGuids, (uint64_t)bTileGuids, (uint64_t)cTileGuids,
            (uint64_t)prevEdt,
            2, (uint64_t)k
        };
        prevEdt = arts_edt_create(timerLaunch3KernelEdt, 10, argsAdd,
                                  numDeps,
                                  &(arts_hint_t){.route = currentNode});

        /* --- Scale: b[i] = scalar * c[i] --- */
        uint64_t argsScale[9] = {
            (uint64_t)scaleKernel, tileSize, N, scalarBits,
            (uint64_t)cTileGuids, (uint64_t)bTileGuids,
            (uint64_t)prevEdt,
            1, (uint64_t)k
        };
        prevEdt = arts_edt_create(timerLaunch2KernelEdt, 9, argsScale,
                                  numDeps,
                                  &(arts_hint_t){.route = currentNode});

        /* --- Copy: c[i] = a[i] --- */
        uint64_t argsCopy[9] = {
            (uint64_t)copyKernel, tileSize, N, 0,
            (uint64_t)aTileGuids, (uint64_t)cTileGuids,
            (uint64_t)prevEdt,
            0, (uint64_t)k
        };
        if (k == 0) {
            /* First copy launcher is the firstKernel; signaled by workers */
            arts_edt_create_with_guid(timerLaunch2KernelEdt, firstKernel,
                                      9, argsCopy,
                                      arts_get_total_workers());
        } else {
            prevEdt = arts_edt_create(timerLaunch2KernelEdt, 9, argsCopy,
                                      numDeps,
                                      &(arts_hint_t){.route = currentNode});
        }
    }
}

/* =====================================================================
 * init_per_node
 *
 * Called once per node before workers start.  Allocates tile GUIDs and
 * creates the CXL (or default) datablocks on node 0.
 * ===================================================================== */

void init_per_node(unsigned int nodeId, int argc, char **argv) {
    if (argc > 1)
        tileSize = (unsigned int)atoi(argv[1]);

    numTiles = N / tileSize;
    if (N % tileSize)
        numTiles++;

    /* Reserve the done EDT GUID on node 0 */
    doneGuid = arts_guid_reserve(ARTS_EDT, 0);

    if (!nodeId)
        arts_printf("N: %u tileSize: %u numTiles: %u\n", N, tileSize, numTiles);

    aTileGuids = malloc(sizeof(arts_guid_t) * numTiles);
    bTileGuids = malloc(sizeof(arts_guid_t) * numTiles);
    cTileGuids = malloc(sizeof(arts_guid_t) * numTiles);

#if !CXL_DB
    /* Reserve GUIDs round-robin across nodes for non-CXL mode */
    unsigned int owner = 0;
    for (unsigned int i = 0; i < numTiles; i++) {
        aTileGuids[i] = arts_guid_reserve(ARTS_DB, owner);
        bTileGuids[i] = arts_guid_reserve(ARTS_DB, owner);
        cTileGuids[i] = arts_guid_reserve(ARTS_DB, owner);
        owner = (owner + 1) % arts_get_total_nodes();
    }
#endif

#if CXL_DB
    /* CXL mode: all tiles created on node 0 */
    if (!nodeId) {
#endif
        aTile = (double **)calloc(numTiles, sizeof(double *));
        bTile = (double **)calloc(numTiles, sizeof(double *));
        cTile = (double **)calloc(numTiles, sizeof(double *));

#if !CXL_DB
        owner = 0;
#endif
        for (unsigned int i = 0; i < numTiles; i++) {
#if CXL_DB
            aTileGuids[i] = arts_db_create((void **)&aTile[i],
                                           tileSize * sizeof(double),
                                           ARTS_DB_CXL, NULL);
            bTileGuids[i] = arts_db_create((void **)&bTile[i],
                                           tileSize * sizeof(double),
                                           ARTS_DB_CXL, NULL);
            cTileGuids[i] = arts_db_create((void **)&cTile[i],
                                           tileSize * sizeof(double),
                                           ARTS_DB_CXL, NULL);
#else
            if (nodeId == owner) {
                aTile[i] = arts_db_create_with_guid(aTileGuids[i],
                                                    tileSize * sizeof(double),
                                                    ARTS_DB_DEFAULT, NULL, NULL);
                bTile[i] = arts_db_create_with_guid(bTileGuids[i],
                                                    tileSize * sizeof(double),
                                                    ARTS_DB_DEFAULT, NULL, NULL);
                cTile[i] = arts_db_create_with_guid(cTileGuids[i],
                                                    tileSize * sizeof(double),
                                                    ARTS_DB_DEFAULT, NULL, NULL);
            }
#endif
            /* Initialize tile data */
#if CXL_DB
            for (unsigned int j = 0; j < tileSize; j++) {
                aTile[i][j] = 1.0;
                bTile[i][j] = 2.0;
                cTile[i][j] = 0.0;
            }
            arts_cxl_producer_flush(aTileGuids[i]);
            arts_cxl_producer_flush(bTileGuids[i]);
            arts_cxl_producer_flush(cTileGuids[i]);
#else
            if (nodeId == owner) {
                for (unsigned int j = 0; j < tileSize; j++) {
                    aTile[i][j] = 1.0;
                    bTile[i][j] = 2.0;
                    cTile[i][j] = 0.0;
                }
            }
            owner = (owner + 1) % arts_get_total_nodes();
#endif
        }

        arts_printf(HLINE);
        int BytesPerWord = sizeof(double);
        arts_printf("This system uses %d bytes per DOUBLE PRECISION word.\n",
                    BytesPerWord);
        arts_printf(HLINE);
        arts_printf("Array size = %d, Offset = %d\n", N, OFFSET);
        arts_printf("Total memory required = %.1f MB.\n",
                    (3.0 * BytesPerWord) * ((double)N / 1048576.0));
        arts_printf("Each test is run %d times, but only\n", NTIMES);
        arts_printf("the *best* time for each is used.\n");
        arts_printf(HLINE);

        if ((quantum = checktick()) >= 1)
            arts_printf(
                "Your clock granularity/precision appears to be %d microseconds.\n",
                quantum);
        else
            arts_printf(
                "Your clock granularity appears to be less than one microsecond.\n");

        /* Reserve firstKernel GUID on this node */
        firstKernel = arts_guid_reserve(ARTS_EDT, nodeId);
#if CXL_DB
    }
#endif
}

/* =====================================================================
 * init_per_worker
 *
 * Called on each worker thread after the startup barrier.
 * Performs the timing precheck (doubles a[]) and signals firstKernel.
 * Worker 0 on node 0 also creates the done EDT and launches streamDriver.
 * ===================================================================== */

void init_per_worker(unsigned int nodeId, unsigned int workerId,
                     int argc, char **argv) {
    (void)argc;
    (void)argv;

#if CXL_DB
    if (!nodeId) {
#endif
        double t = mysecond();
#if !CXL_DB
        unsigned int owner = 0;
#endif
        for (unsigned int i = 0; i < numTiles; i++) {
#if !CXL_DB
            if (nodeId == owner) {
#endif
                if (i % arts_get_total_workers() == workerId) {
                    for (unsigned int j = 0; j < tileSize; j++)
                        aTile[i][j] = 2.0E0 * aTile[i][j];
#if CXL_DB
                    arts_cxl_producer_flush(aTileGuids[i]);
#endif
                }
#if !CXL_DB
            }
            owner = (owner + 1) % arts_get_total_nodes();
#endif
        }

        /* Signal firstKernel with this worker's slot */
        arts_signal_edt(firstKernel, arts_get_current_worker(),
                        NULL_GUID, DB_MODE_NULL);

        t = 1.0E6 * (mysecond() - t);

        if (!workerId) {
#if !CXL_DB
            if (!nodeId) {
#endif
                arts_printf(
                    "Each test below will take on the order of %d microseconds.\n",
                    (int)t);
                arts_printf("   (= %d clock ticks)\n", (int)(t / quantum));
                arts_printf("Increase the size of the arrays if this shows that\n");
                arts_printf("you are not getting at least 20 clock ticks per test.\n");
                arts_printf(HLINE);
                arts_printf("WARNING -- The above is only a rough guideline.\n");
                arts_printf("For best results, please be sure you know the\n");
                arts_printf("precision of your system timer.\n");
                arts_printf(HLINE);
#if !CXL_DB
            }
#endif

            if (!nodeId) {
                /* done EDT has 3*numTiles deps in both CXL and non-CXL modes */
                arts_edt_create_with_guid(done, doneGuid, 0, NULL,
                                          numTiles * 3);
                arts_edt_create(streamDriver, 0, NULL, 0,
                                &(arts_hint_t){.route = 0});
            }
        }
#if CXL_DB
    }
#endif
}

/* ===================================================================== */

int main(int argc, char **argv) {
    arts_rt(argc, argv);
    return 0;
}