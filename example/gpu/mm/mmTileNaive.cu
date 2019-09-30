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
#define SMTILE 32

uint64_t start = 0;

int mat_size;
int tile_size;
unsigned int numBlocks = 1;

artsGuid_t aMatGuid = NULL_GUID;
artsGuid_t bMatGuid = NULL_GUID;
artsGuid_t cMatGuid = NULL_GUID;
artsGuid_t doneGuid = NULL_GUID;

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
    
    unsigned int rowSize    = tile_size;
    
    unsigned int i = paramv[1];
    unsigned int j = paramv[2];
    unsigned int k = paramv[3];
    
    double * aMat = (double*) depv[0].ptr;
    double * bMat = (double*) depv[1].ptr;
    
    double * aTile = NULL;
    double * bTile = NULL;
    double * cTile = NULL;
    
    artsGuid_t aTileGuid = artsDbCreate((void**) &aTile, sizeof(double) * tile_size * tile_size, ARTS_DB_GPU_READ);
    artsGuid_t bTileGuid = artsDbCreate((void**) &bTile, sizeof(double) * tile_size * tile_size, ARTS_DB_GPU_READ);
    artsGuid_t cTileGuid = artsDbCreate((void**) &cTile, sizeof(double) * tile_size * tile_size, ARTS_DB_GPU_WRITE);
    
    copyBlock(i, k, tile_size, aTile, mat_size, aMat, true);
    copyBlock(k, j, tile_size, bTile, mat_size, bMat, true);
    initMatrix(rowSize, cTile, false, true);
    
#if GPUMM
    dim3 threads(SMTILE, SMTILE);
    dim3 grid((tile_size+SMTILE-1)/SMTILE, (tile_size+SMTILE-1)/SMTILE);
    
    uint64_t args[] = {tile_size};
    artsGuid_t    mulGpuGuid = artsEdtCreateGpu(mmKernel, artsGetCurrentNode(), 1, args, 3, grid, threads, toSignal, k, cTileGuid);
    artsSignalEdt(mulGpuGuid, 0, aTileGuid);
    artsSignalEdt(mulGpuGuid, 1, bTileGuid);
    artsSignalEdt(mulGpuGuid, 2, cTileGuid);
#else
    uint64_t args[] = {tile_size, toSignal, k, cTileGuid};
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

    unsigned int columnSize = tile_size;

    unsigned int row = paramv[1];
    unsigned int col = paramv[2];

    double * cTile;
    unsigned int rowSize    = tile_size;
    artsGuid_t cTileGuid = artsDbCreate((void**) &cTile, sizeof(double) * tile_size * tile_size, ARTS_DB_GPU_WRITE);
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
            copyBlock(i, j, tile_size, cTile, mat_size, cMat, false);
        }

    uint64_t time = artsGetTimeStamp() - start;

#if VERIFY
    double * aMat  = (double*) depv[1].ptr;
    double * bMat  = (double*) depv[2].ptr;
    printf("Verifying results...\n");
    double *temp = (double*) artsCalloc(mat_size * mat_size * sizeof(double));
    for (unsigned int i=0; i< mat_size; ++i)
        for (unsigned int j=0; j<mat_size; ++j)
            for (unsigned int k=0; k<mat_size; ++k)
                temp[i*mat_size+j] += aMat[i*mat_size+k]*bMat[k*mat_size+j];

    for (unsigned int i=0; i< mat_size; ++i)
        for (unsigned int j=0; j<mat_size; ++j)
            if (temp[i * mat_size + j] != cMat[i * mat_size + j])
            {
                printf("Failed at cMat[%u][%u]\n", i, j);
                printf("Expected: %lf | Obtained: %lf\n", cMat[i * mat_size + j], temp[i * mat_size + j]);
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
        mat_size = MATSIZE;
        tile_size = TILESIZE;
    } else if (argc == 2)
    {
        mat_size = atoi(argv[1]);
        tile_size = TILESIZE;
    } else
    {
        mat_size = atoi(argv[1]);
        tile_size = atoi(argv[2]);
    }
    numBlocks = mat_size / tile_size;
    doneGuid = artsReserveGuidRoute(ARTS_EDT,     0);
    aMatGuid = artsReserveGuidRoute(ARTS_DB_READ, 0);
    bMatGuid = artsReserveGuidRoute(ARTS_DB_READ, 0);
    cMatGuid = artsReserveGuidRoute(ARTS_DB_READ, 0);

    if(!nodeId)
    {
        double * aMat = (double*) artsDbCreateWithGuid(aMatGuid, mat_size * mat_size * sizeof(double));
        double * bMat = (double*) artsDbCreateWithGuid(bMatGuid, mat_size * mat_size * sizeof(double));
        double * cMat = (double*) artsDbCreateWithGuid(cMatGuid, mat_size * mat_size * sizeof(double));

        initMatrix(mat_size, aMat, false, false);
        initMatrix(mat_size, bMat, false, false);
        initMatrix(mat_size, cMat, false, true);
    }
}

extern "C"
void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{   
    unsigned int totalThreads = artsGetTotalNodes() * artsGetTotalWorkers();
    unsigned int globalThreadId = nodeId * artsGetTotalWorkers() + workerId;
    
    for(unsigned int i=0; i<numBlocks; i++)
        for(unsigned int j=0; j<numBlocks; j++)
            if((i * numBlocks + j) % totalThreads == globalThreadId)
            {
#if GPUMM
                uint64_t sumArgs[] = {tile_size};
                dim3 threads (tile_size, tile_size);
                dim3 grid((tile_size+SMTILE-1)/SMTILE, (tile_size+SMTILE-1)/SMTILE);
                artsGuid_t sumGuid = artsEdtCreateGpuPT (sumMMKernel, nodeId, 1, sumArgs, numBlocks, grid, threads, doneGuid, 3 + (i * numBlocks + j), 0);
#else
                uint64_t sumArgs[] = {doneGuid, i, j};
                artsGuid_t sumGuid = artsEdtCreate(sumMM, nodeId, 3, sumArgs, numBlocks);
#endif
                for(unsigned int k=0; k<numBlocks; k++)
                {
                    uint64_t args[] = {sumGuid, i, j, k};
                    artsGuid_t mulGuid = artsEdtCreate(multiplyMM, nodeId, 4, args, 2);
                    artsSignalEdt(mulGuid, 0, aMatGuid);
                    artsSignalEdt(mulGuid, 1, bMatGuid);
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
