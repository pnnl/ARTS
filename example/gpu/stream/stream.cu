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
/* Program: Stream                                                       */
/* Revision: $Id: stream_omp.c,v 5.4 2009/02/19 13:57:12 mccalpin Exp mccalpin $ */
/* Original code developed by John D. McCalpin                           */
/* Programmers: John D. McCalpin                                         */
/*              Joe R. Zagar                                             */
/*                                                                       */
/* This program measures memory transfer rates in MB/s for simple        */
/* computational kernels coded in C.                                     */
/*-----------------------------------------------------------------------*/
/* Copyright 1991-2003: John D. McCalpin                                 */
/*-----------------------------------------------------------------------*/
/* License:                                                              */
/*  1. You are free to use this program and/or to redistribute           */
/*     this program.                                                     */
/*  2. You are free to modify this program for your own use,             */
/*     including commercial use, subject to the publication              */
/*     restrictions in item 3.                                           */
/*  3. You are free to publish results obtained from running this        */
/*     program, or from works that you derive from this program,         */
/*     with the following limitations:                                   */
/*     3a. In order to be referred to as "STREAM benchmark results",     */
/*         published results must be in conformance to the STREAM        */
/*         Run Rules, (briefly reviewed below) published at              */
/*         http://www.cs.virginia.edu/stream/ref.html                    */
/*         and incorporated herein by reference.                         */
/*         As the copyright holder, John McCalpin retains the            */
/*         right to determine conformity with the Run Rules.             */
/*     3b. Results based on modified source code or on runs not in       */
/*         accordance with the STREAM Run Rules must be clearly          */
/*         labelled whenever they are published.  Examples of            */
/*         proper labelling include:                                     */
/*         "tuned STREAM benchmark results"                              */
/*         "based on a variant of the STREAM benchmark code"             */
/*         Other comparable, clear and reasonable labelling is           */
/*         acceptable.                                                   */
/*     3c. Submission of results to the STREAM benchmark web site        */
/*         is encouraged, but not required.                              */
/*  4. Use of this program or creation of derived works based on this    */
/*     program constitutes acceptance of these licensing restrictions.   */
/*  5. Absolutely no warranty is expressed or implied.                   */
/*-----------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include "arts.h"
#include "artsGpuRuntime.h"
#include "cublas_v2.h"
#include <cuda_runtime.h>
#include "streamUtil.h"

static double avgtime[4] = {0};
static double maxtime[4] = {0};
static double mintime[4] = {FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX};

static const char	* label[4] = {
    "Copy:      ", 
    "Scale:     ", 
    "Add:       ", 
    "Triad:     "};

static double bytes[4] = {
    2 * sizeof(double) * N,
    2 * sizeof(double) * N,
    3 * sizeof(double) * N,
    3 * sizeof(double) * N
    };

int quantum;

unsigned int tileSize = 1024;
unsigned int numTiles;

artsGuidRange * aTileGuids = NULL;
artsGuidRange * bTileGuids = NULL;
artsGuidRange * cTileGuids = NULL;

double **aTile;
double **bTile;
double **cTile;

__global__ void copyKernel(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    unsigned int len = (unsigned int) paramv[0];
    double * a = (double *) depv[0].ptr;
    double * b = (double *) depv[1].ptr;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < len) 
        b[idx] = a[idx];
}

__global__ void scaleKernel(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    unsigned int len = (unsigned int) paramv[0];
    double scale = (double) paramv[1];
    double * a = (double *) depv[0].ptr;
    double * b = (double *) depv[1].ptr;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < len) 
        b[idx] = scale * a[idx];
}

__global__ void addKernel(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    unsigned int len = (unsigned int) paramv[0];
    double * a = (double *) depv[0].ptr;
    double * b = (double *) depv[1].ptr;
    double * c = (double *) depv[2].ptr;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < len) 
        c[idx] = a[idx] + b[idx];
}

__global__ void triadKernal(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    unsigned int len = (unsigned int) paramv[0];
    double scale = (double) paramv[1];
    double * a = (double *) depv[0].ptr;
    double * b = (double *) depv[1].ptr;
    double * c = (double *) depv[2].ptr;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < len) 
        c[idx] = a[idx] + scale * b[idx];
}

void streamDriver(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    register int j, k;
    
    double times[4][NTIMES];

    double scalar = 3.0;
    for (k=0; k<NTIMES; k++)
	{
	    times[0][k] = mysecond();
        launch2KernelEdt(copyKernel, tileSize, N, 0, aTileGuids, cTileGuids);
	    times[0][k] = mysecond() - times[0][k];
        
        times[1][k] = mysecond();
        launch2KernelEdt(scaleKernel, tileSize, N, scalar, cTileGuids, bTileGuids);
	    times[1][k] = mysecond() - times[1][k];
        
        times[2][k] = mysecond();
        launch3KernelEdt(addKernel, tileSize, N, 0, aTileGuids, bTileGuids, cTileGuids);
	    times[2][k] = mysecond() - times[2][k];
        
        times[3][k] = mysecond();
        launch3KernelEdt(triadKernal, tileSize, N, scalar, bTileGuids, cTileGuids, aTileGuids);
        times[3][k] = mysecond() - times[3][k];
    }

    /*	--- SUMMARY --- */
    for (k=1; k<NTIMES; k++) /* note -- skip first iteration */
	{
        for (j=0; j<4; j++)
        {
            avgtime[j] = avgtime[j] + times[j][k];
            mintime[j] = MIN(mintime[j], times[j][k]);
            maxtime[j] = MAX(maxtime[j], times[j][k]);
        }
	}
    
    printf("Function      Rate (MB/s)   Avg time     Min time     Max time\n");
    for (j=0; j<4; j++) {
	    avgtime[j] = avgtime[j]/(double)(NTIMES-1);

        printf("%s%11.4f  %11.4f  %11.4f  %11.4f\n", label[j],
            1.0E-06 * bytes[j]/mintime[j],
            avgtime[j],
            mintime[j],
            maxtime[j]);
    }
    printf(HLINE);

    /* --- Check Results --- */
    checkSTREAMresults(tileSize, N, aTile, bTile, cTile);
    printf(HLINE);
    artsShutdown();
}

extern "C"
void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    if (argc > 1)
        tileSize = (unsigned int) atoi(argv[1]);
    
    numTiles = N / tileSize;
    if(N % tileSize)
        numTiles++;
    
    PRINTF("N: %u tileSize: %u numTiles: %u\n", N, tileSize, numTiles);

    aTileGuids = artsNewGuidRangeNode(ARTS_DB_GPU_WRITE, numTiles, 0);
    bTileGuids = artsNewGuidRangeNode(ARTS_DB_GPU_WRITE, numTiles, 0);
    cTileGuids = artsNewGuidRangeNode(ARTS_DB_GPU_WRITE, numTiles, 0);
    
    if(!nodeId)
    {
        aTile = (double**)artsCalloc(sizeof(double*)*numTiles);
        bTile = (double**)artsCalloc(sizeof(double*)*numTiles);
        cTile = (double**)artsCalloc(sizeof(double*)*numTiles);

        for(unsigned int i=0; i<numTiles; i++)
        {
            aTile[i] = (double*) artsDbCreateWithGuid(artsGetGuid(aTileGuids, i), tileSize * sizeof(double));
            bTile[i] = (double*) artsDbCreateWithGuid(artsGetGuid(bTileGuids, i), tileSize * sizeof(double));
            cTile[i] = (double*) artsDbCreateWithGuid(artsGetGuid(cTileGuids, i), tileSize * sizeof(double));
            for(unsigned int j=0; j<tileSize; j++)
            {
                aTile[i][j] = 1.0;
                bTile[i][j] = 2.0;
                cTile[i][j] = 0.0;
            }
        }

        printf(HLINE);
        int BytesPerWord = sizeof(double);
        printf("This system uses %d bytes per DOUBLE PRECISION word.\n", BytesPerWord);
        printf(HLINE);

        printf("Array size = %d, Offset = %d\n" , N, OFFSET);
        printf("Total memory required = %.1f MB.\n", (3.0 * BytesPerWord) * ( (double) N / 1048576.0));
        printf("Each test is run %d times, but only\n", NTIMES);
        printf("the *best* time for each is used.\n");
        printf(HLINE);

        if((quantum = checktick()) >= 1) 
            printf("Your clock granularity/precision appears to be %d microseconds.\n", quantum);
        else
            printf("Your clock granularity appears to be less than one microsecond.\n");
    }
}

extern "C"
void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{
    if(!nodeId)
    {
        double t = mysecond();
        for(unsigned int i=0; i<numTiles; i++)
        {
            if(i % artsGetTotalWorkers() == workerId)
            {
                for (unsigned int j = 0; j < tileSize; j++)
                    aTile[i][j] = 2.0E0 * aTile[i][j];
            }
        }
        t = 1.0E6 * (mysecond() - t);
        
        if(!workerId)
        {
            printf("Each test below will take on the order of %d microseconds.\n", (int) t);
            printf("   (= %d clock ticks)\n", (int) (t/quantum) );
            printf("Increase the size of the arrays if this shows that\n");
            printf("you are not getting at least 20 clock ticks per test.\n");

            printf(HLINE);

            printf("WARNING -- The above is only a rough guideline.\n");
            printf("For best results, please be sure you know the\n");
            printf("precision of your system timer.\n");
            printf(HLINE);

            artsEdtCreate(streamDriver, 0, 0, NULL, 0);
        }
    }
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}
