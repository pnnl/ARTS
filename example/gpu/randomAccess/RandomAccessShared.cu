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
 * randomly updated in one second, divided by 1 billion (1e9). The term "randomly"
 * means that there is little relationship between one address to be updated and
 * the next, except that they occur in the space of one half the total system
 * memory.  An update is a read-modify-write operation on a table of 64-bit words.
 * An address is generated, the value at that address read from memory, modified
 * by an integer operation (add, and, or, xor) with a literal value, and that
 * new value is written back to memory.
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

#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include "cublas_v2.h"
#include "cublas_api.h"
#include <cuda_runtime.h>
#include "randomAccessDefs.h"

#define NUMGPUS 6

/* Perform updates to main table.  The scalar equivalent is:
    *
    *     u64Int ran;
    *     ran = 1;
    *     for (i=0; i<NUPDATE; i++) {
    *       ran = (ran << 1) ^ (((s64Int) ran < 0) ? POLY : 0);
    *       table[ran & (TableSize-1)] ^= ran;
    *     }
*/

#define NANOSECS 1000000000

uint64_t getTimeStamp()
{
    struct timespec res;
    clock_gettime(CLOCK_REALTIME, &res);
    uint64_t timeRes = res.tv_sec*NANOSECS+res.tv_nsec;
    return timeRes;
}

uint64_t HPCC_starts_CPU(int64_t N)
{
    uint64_t i, j;
    uint64_t m2[64];
    uint64_t temp;
    volatile uint64_t ran;
    volatile int64_t n = N;
    
    while (n < 0) n += PERIOD2;
    while (n > PERIOD2) n -= PERIOD2;
    if (n != 0) 
    {
        temp = 0x1;
        for (i=0; i<64; i++) 
        {
            m2[i] = temp;
            temp = (temp << 1) ^ ((int64_t) temp < 0 ? POLY2 : 0);
            temp = (temp << 1) ^ ((int64_t) temp < 0 ? POLY2 : 0);
        }

        for (i=62; i>=0; i--)
            if ((n >> i) & 1)
                break;

        ran = 0x2;
        while (i > 0) {
            temp = 0;
            for (j=0; j<64; j++)
                if ((ran >> j) & 1)
                    temp ^= m2[j];
            ran = temp;
            i -= 1;
            if ((n >> i) & 1)
                ran = (ran << 1) ^ ((int64_t) ran < 0 ? POLY2 : 0);
        }
    }
    else
        ran = 0x1;
    
    ran = (ran << 1) ^ ((int64_t) ran < 0 ? POLY2 : 0);
    return ran;
}

/* Utility routine to start random number generator at Nth step */
__device__ uint64_cu_t HPCC_starts(uint64_cu_t N)
{
    volatile int64_cu_t n = N;
    
    int i, j;
    uint64_cu_t m2[64];
    uint64_cu_t temp;

    uint64_cu_t ran = 0x1;
        
    while (n < 0) n += PERIOD;
    while (n > PERIOD) n -= PERIOD;

    if(n)
    {
        temp = 0x1;
        for (i=0; i<64; i++) 
        {
            m2[i] = temp;
            temp = (temp << 1) ^ ((int64_cu_t) temp < 0 ? POLY : 0);
            temp = (temp << 1) ^ ((int64_cu_t) temp < 0 ? POLY : 0);
        }

        for (i=62; i>=0; i--)
            if ((n >> i) & 1)
                break;

        ran = 0x2;
        while (i > 0) {
            temp = 0;
            for (j=0; j<64; j++)
                if ((ran >> j) & 1)
                    temp ^= m2[j];
            ran = temp;
            i -= 1;
            if ((n >> i) & 1)
            ran = (ran << 1) ^ ((int64_cu_t) ran < 0 ? POLY : 0);
        }
    }
    else
        ran = 0x1;
    return ran;
}

__global__ void upate(uint64_cu_t offsetStart, uint64_cu_t offsetEnd, uint64_cu_t numUpdatesPerThread, uint64_cu_t tableSize, uint64_cu_t * table)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_cu_t start = index * numUpdatesPerThread + offsetStart;
    for(uint64_cu_t i=0; i<numUpdatesPerThread; i++)
    {
        uint64_cu_t current = start + i;
        if(current < offsetEnd)
        {
            uint64_cu_t ran = HPCC_starts(current);
            ran = (ran << 1) ^ ((int64_cu_t) ran < 0 ? POLY : 0);
            atomicXor(&table[ran & (tableSize-1)], ran);
        }
        else
            break;
    }
}

int main(int argc, char** argv)
{
    printf("TableSize: %lu -> %lu Updates: %lu\n", TABLESIZE, TABLESIZE*sizeof(uint64_t)/(1024*1024), NUPDATE);
    uint64_t * table = NULL;
    cudaMallocManaged(&table, TABLESIZE * sizeof(uint64_t));

    for (uint64_t i = 0; i < TABLESIZE; i++)
        table[i] = i;

    int blockSize = MAXTHREADS;
    int numBlocks = MAXGRID;
    uint64_t tableSize = TABLESIZE;
    uint64_t totalUpdates = NUPDATE;

    uint64_t numUpdatesPerGpu = totalUpdates / NUMGPUS;
    if(!numUpdatesPerGpu)
        numUpdatesPerGpu = 1;

    uint64_t numUpdatesPerThread = numUpdatesPerGpu / (blockSize * numBlocks);
    if(!numUpdatesPerThread)
        numUpdatesPerThread = 1;

    uint64_t time = getTimeStamp();
    for(unsigned int i=0; i<NUMGPUS; i++)
    {
        cudaSetDevice(i);
        // printf("offset %lu totalUpdates %lu numUpdatesPerThread %lu tableSize %lu\n", i * numUpdatesPerGpu, (i+1) * numUpdatesPerGpu, numUpdatesPerThread, tableSize);
        upate<<<numBlocks, blockSize>>>(i * numUpdatesPerGpu, (i+1) * numUpdatesPerGpu, numUpdatesPerThread, tableSize, (uint64_cu_t*)table);
    }
    
    for(unsigned int i=0; i<NUMGPUS; i++)
    {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }
    time = getTimeStamp() - time;

    #ifdef VALIDATE
    uint64_t * Table = (uint64_t*) malloc(sizeof(uint64_t)*TABLESIZE);
    for(uint64_t i=0; i<TABLESIZE; i++)
        Table[i] = i;

    uint64_t ran = 0x1;
    for (uint64_t i=0; i<NUPDATE; i++) {
        ran = (ran << 1) ^ (((int64_t) ran < 0) ? POLY2 : 0);
        Table[ran & (tableSize-1)] ^= ran;
    }

    uint64_t errors = 0;
    for (uint64_t i = 0; i < TABLESIZE; i++)
    {
        if(table[i] != Table[i])
        {
            if(!errors)
                printf("Error %lu vs %lu\n", table[i], Table[i]);
            errors++;
        }
    }
    printf("Time: %lu Total Errors: %lu\n", time, errors);
    free(Table);
    #endif
    printf("Time: %lu\n", time);
    double GUPS = (double) NUPDATE / time;
    printf("GUPS: %lf %lu\n", GUPS, (TABLESIZE * sizeof(uint64_t)) / (1024*1024));
    // Free memory
    cudaFree(table);
    
 
    return 0;
}