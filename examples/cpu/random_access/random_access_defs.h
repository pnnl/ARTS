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
#ifndef RANDOM_ACCESS_DEFS_H
#define RANDOM_ACCESS_DEFS_H

#ifdef __cplusplus
extern "C" {
#endif

/* =====================================================================
 * CXL mode toggle:
 *   CXL_DB 1  — use ARTS_DB_CXL datablocks + producer/consumer flushes
 *   CXL_DB 0  — use ARTS_DB_DEFAULT datablocks (no flushes)
 * ===================================================================== */
#ifndef CXL_DB
#define CXL_DB 1
#endif

/* Table size: number of 64-bit words in the random-access table */
#define TABLESIZE (32UL * 2UL * 80UL * 1024UL)

/* Default tile size (number of uint64_t elements per tile) */
#define TILESIZE 32UL

/* Number of updates: 4x the table size (standard HPCC ratio) */
#define NUPDATE (4UL * TABLESIZE)

/* Enable validation by default */
#define VALIDATE 1

/* LFSR polynomial and period for the random number generator */
#define POLY  0x0000000000000007ULL
#define PERIOD 1317624576693539401LL

/* Maximum number of updates processed per CPU step (batch size) */
#define MAX_TOTAL_PENDING_UPDATES (1024ULL * 16ULL)
#define MAX_UPDATES_PER_CPU_STEP  MAX_TOTAL_PENDING_UPDATES

typedef unsigned long long int uint64_ra_t;
typedef long long int          int64_ra_t;

#ifdef __cplusplus
}
#endif

#endif /* RANDOM_ACCESS_DEFS_H */