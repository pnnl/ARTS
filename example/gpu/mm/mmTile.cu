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
#include <stdio.h>
#include <stdlib.h>
#include "arts.h"
#include "artsGpuRuntime.h"
#include "mmUtil.h"

#define GPUMM 1
#define MATSIZE 1024
#define TILESIZE 16
#define VERIFY 1
#define SMTILE 32 // Hardcoded for Volta

uint64_t start = 0;

int matSize;
int tileSize;
unsigned int numBlocks = 1;

artsGuid_t aMatGuid = NULL_GUID;
artsGuid_t bMatGuid = NULL_GUID;
artsGuid_t cMatGuid = NULL_GUID;
artsGuid_t doneGuid = NULL_GUID;

double * aMatrix = NULL;
double * bMatrix = NULL;
double * cMatrix = NULL;

artsGuidRange * aTileGuids = NULL;
artsGuidRange * bTileGuids = NULL;

__global__ void mmKernel(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    const int blk = (int) paramv[0];
    double *A = (double *) depv[0].ptr;
    double *B = (double *) depv[1].ptr;
    double *C = (double *) depv[2].ptr;

    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    register double sum = 0;

    for(unsigned int k=0; k<blk; k++)
        sum+=A[row * blk + k] * B[k * blk + col];
    C[row * blk + col] = sum;
}

void mmKernelCPU(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    artsGuid_t toSignal = (artsGuid_t) paramv[1];
    unsigned int k = (unsigned int) paramv[2];
    artsGuid_t cTileGuid = (artsGuid_t) paramv[3];
    const int blk = (int) paramv[0];
    double *A = (double *) depv[0].ptr;
    double *B = (double *) depv[1].ptr;
    double *C = (double *) depv[2].ptr;
    
    for(unsigned int i=0; i<blk; i++)
    {
        //rows of B
        for(unsigned int j=0; j<blk; j++)
        {
            //rows of A and columns of B
            for(unsigned int k=0; k<blk; k++)
            {
                C[i * blk + j] += A[i * blk + k] * B[k * blk + j];
            }
        }
    }
    artsSignalEdt(toSignal, k, cTileGuid);
}

void multiplyMM(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    artsGuid_t toSignal = paramv[0];
    
    unsigned int rowSize = tileSize;
    
    // unsigned int i = paramv[1];
    // unsigned int j = paramv[2];
    unsigned int k = paramv[3];
    
    // double * aTile = (double*) depv[0].ptr;
    // double * bTile = (double*) depv[1].ptr;
    double * cTile = NULL;

    artsGuid_t aTileGuid = depv[0].guid;
    artsGuid_t bTileGuid = depv[1].guid;
    artsGuid_t cTileGuid = artsDbCreate((void**) &cTile, sizeof(double) * tileSize * tileSize, ARTS_DB_GPU_WRITE);

    initMatrix(rowSize, cTile, false, true);

#if GPUMM
    dim3 threads(SMTILE, SMTILE);
    dim3 grid((tileSize+SMTILE-1)/SMTILE, (tileSize+SMTILE-1)/SMTILE);
    
    uint64_t args[] = {tileSize};
    artsGuid_t    mulGpuGuid = artsEdtCreateGpu(mmKernel, artsGetCurrentNode(), 1, args, 3, grid, threads, toSignal, k, cTileGuid);
    artsSignalEdt(mulGpuGuid, 0, aTileGuid);
    artsSignalEdt(mulGpuGuid, 1, bTileGuid);
    artsSignalEdt(mulGpuGuid, 2, cTileGuid);
#else
    uint64_t args[] = {tileSize, toSignal, k, cTileGuid};
    artsGuid_t    mulGpuGuid = artsEdtCreate(mmKernelCPU, artsGetCurrentNode(), 4, args, 3);
    artsSignalEdt(mulGpuGuid, 0, aTileGuid);
    artsSignalEdt(mulGpuGuid, 1, bTileGuid);
    artsSignalEdt(mulGpuGuid, 2, cTileGuid);
#endif
}

__global__ void sumMMKernel(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    const unsigned int columnSize = (unsigned int) paramv[0];

    double * cTile = (double *) depv[0].ptr;

    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    for (unsigned int k=1; k<depc; ++k)
    {
        double* toAdd = (double*) depv[k].ptr;
        cTile[row * columnSize + col] += toAdd[row*columnSize+col];
    }
}

void sumMM(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    artsGuid_t doneGuid = paramv[0];

    unsigned int columnSize = tileSize;

    unsigned int row = paramv[1];
    unsigned int col = paramv[2];

    double * cTile;
    unsigned int rowSize    = tileSize;
    artsGuid_t cTileGuid = artsDbCreate((void**) &cTile, sizeof(double) * tileSize * tileSize, ARTS_DB_GPU_WRITE);
    initMatrix(rowSize, cTile, false, true);

    for(unsigned int i=0; i<depc; i++)
    {
        double * toAdd = (double*) depv[i].ptr;
        for(unsigned int j=0; j<columnSize; j++)
        {
            for(unsigned int k=0; k<rowSize; k++)
            {
               cTile[j * rowSize + k] += toAdd[j * rowSize + k];
            }
        }
    }
    artsSignalEdt(doneGuid, 3 + (row * numBlocks + col), cTileGuid);
}

void finishBlockMM(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    double * cMat  = (double*) depv[0].ptr;

    for(unsigned int i=0; i<numBlocks; i++)
        for(unsigned int j=0; j<numBlocks; j++)
        {
            double * cTile = (double*) depv[3 + (i * numBlocks + j)].ptr;
            copyBlock(i, j, tileSize, cTile, matSize, cMat, false);
        }

    uint64_t time = artsGetTimeStamp() - start;

#if VERIFY
    double * aMat  = (double*) depv[1].ptr;
    double * bMat  = (double*) depv[2].ptr;
    printf("Verifying results...\n");
    double *temp = (double*) artsCalloc(matSize * matSize * sizeof(double));
    for (unsigned int i=0; i< matSize; ++i)
        for (unsigned int j=0; j<matSize; ++j)
            for (unsigned int k=0; k<matSize; ++k)
                temp[i*matSize+j] += aMat[i*matSize+k]*bMat[k*matSize+j];

    for (unsigned int i=0; i< matSize; ++i)
        for (unsigned int j=0; j<matSize; ++j)
            if (temp[i * matSize + j] != cMat[i * matSize + j])
            {
                printf("Failed at cMat[%u][%u]\n", i, j);
                printf("Expected: %lf | Obtained: %lf\n", temp[i * matSize + j], cMat[i * matSize + j]);
                artsFree(temp);
                artsShutdown();
                return;
            }

    artsFree(temp);
    PRINTF("Success %lu\n", time);
#else
    PRINTF("Done %lu\n", time);
#endif

    artsShutdown();
}

extern "C"
void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    if (argc == 1)
    {
        matSize = MATSIZE;
        tileSize = TILESIZE;
    } else if (argc == 2)
    {
        matSize = atoi(argv[1]);
        tileSize = TILESIZE;
    } else
    {
        matSize = atoi(argv[1]);
        tileSize = atoi(argv[2]);
    }
    numBlocks = matSize / tileSize;
    doneGuid = artsReserveGuidRoute(ARTS_EDT,     0);
    aMatGuid = artsReserveGuidRoute(ARTS_DB_READ, 0);
    bMatGuid = artsReserveGuidRoute(ARTS_DB_READ, 0);
    cMatGuid = artsReserveGuidRoute(ARTS_DB_READ, 0);
    
    aTileGuids = artsNewGuidRangeNode(ARTS_DB_GPU_READ, numBlocks*numBlocks, 0);
    bTileGuids = artsNewGuidRangeNode(ARTS_DB_GPU_READ, numBlocks*numBlocks, 0);
    
    if(!nodeId)
    {
        aMatrix = (double*) artsDbCreateWithGuid(aMatGuid, matSize * matSize * sizeof(double));
        bMatrix = (double*) artsDbCreateWithGuid(bMatGuid, matSize * matSize * sizeof(double));
        cMatrix = (double*) artsDbCreateWithGuid(cMatGuid, matSize * matSize * sizeof(double));
        
        initMatrix(matSize, aMatrix, true, false);
        initMatrix(matSize, bMatrix, false, false);
        initMatrix(matSize, cMatrix, false, true);
        
        PRINTF("Starting\n");
    }
}

extern "C"
void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{
    unsigned int totalThreads = artsGetTotalNodes() * artsGetTotalWorkers();
    unsigned int globalThreadId = nodeId * artsGetTotalWorkers() + workerId;   
  
    if(!nodeId && !workerId)
    {
        for(unsigned int i=0; i<numBlocks; i++)
        {
            for(unsigned int j=0; j<numBlocks; j++)
            {
                artsGuid_t aTileGuid = artsGetGuid(aTileGuids, i * numBlocks + j);
                double * aTile = (double*) artsDbCreateWithGuid(aTileGuid, sizeof(double) * tileSize * tileSize);
                copyBlock(i, j, tileSize, aTile, matSize, aMatrix, true);

                artsGuid_t bTileGuid = artsGetGuid(bTileGuids, i * numBlocks + j);
                double * bTile = (double*) artsDbCreateWithGuid(bTileGuid, sizeof(double) * tileSize * tileSize);
                copyBlock(i, j, tileSize, bTile, matSize, bMatrix, true);
            }
        }
    }

    for(unsigned int i=0; i<numBlocks; i++)
    {
        for(unsigned int j=0; j<numBlocks; j++)
        {
            if((i * numBlocks + j) % totalThreads == globalThreadId)
            {
#if GPUMM
                uint64_t sumArgs[] = {tileSize};
                dim3 threads (SMTILE, SMTILE);
                dim3 grid((tileSize+SMTILE-1)/SMTILE, (tileSize+SMTILE-1)/SMTILE);

                artsGuid_t sumGuid = artsEdtCreateGpuPT (sumMMKernel, nodeId, 1, sumArgs, numBlocks, grid, threads, doneGuid, 3 + (i * numBlocks + j), 0);
#else
                uint64_t sumArgs[] = {doneGuid, i, j};
                artsGuid_t sumGuid = artsEdtCreate(sumMM, nodeId, 3, sumArgs, numBlocks);
#endif
                for(unsigned int k=0; k<numBlocks; k++)
                {
                    uint64_t args[] = {sumGuid, i, j, k};
                    artsGuid_t mulGuid = artsEdtCreate(multiplyMM, nodeId, 4, args, 2);
                    artsSignalEdt(mulGuid, 0, artsGetGuid(aTileGuids, i * numBlocks + k));
                    artsSignalEdt(mulGuid, 1, artsGetGuid(bTileGuids, k * numBlocks + j));
                }
            }
        }
    }

    if(!nodeId && !workerId)
    {
        artsEdtCreateWithGuid(finishBlockMM, doneGuid, 0, NULL, 3 + numBlocks * numBlocks);
        artsSignalEdt(doneGuid, 0, cMatGuid);
        artsSignalEdt(doneGuid, 1, aMatGuid);
        artsSignalEdt(doneGuid, 2, bMatGuid);
        start = artsGetTimeStamp();
    }
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}