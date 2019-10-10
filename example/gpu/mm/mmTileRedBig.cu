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

#include "cublas_v2.h"
#include <cuda_runtime.h>

#define MATSIZE 1024
#define TILESIZE 1
#define VERIFY 0
#define SMTILE 32

#define DPRINTF(...)
// #define DPRINTF(...) PRINTF(__VA_ARGS__)

uint64_t start = 0;

unsigned int matSize;
unsigned int tileSize;
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
artsGuid_t    * cTileGuids = NULL;

cublasHandle_t * handle;

typedef struct {
    unsigned int numLeaves;
    unsigned int totalNodes;
    unsigned int interiorNodes;
    artsGuid_t * redDbGuids;
    artsGuid_t * redEdtGuids;
} binaryReductionTree_t;

binaryReductionTree_t **  redTree = NULL;

unsigned int left(unsigned int i) { return 2*i + 1; }
unsigned int right(unsigned int i) { return 2*i + 2; }
unsigned int parent(unsigned int i) { return (i-1)/2; }

unsigned int reserveEdtGuids(artsGuid_t * allGuids, unsigned int index, artsType_t edtType)
{
    if(allGuids[index])
        return artsGuidGetRank(allGuids[index]);
    //left Rank    
    unsigned int rank = reserveEdtGuids(allGuids, left(index), edtType);
    //always reserve left rank
    allGuids[index] = artsReserveGuidRoute(edtType, rank);
    DPRINTF("edt: %d -> %lu\n", index, allGuids[index]);
    //visit right rank
    reserveEdtGuids(allGuids, right(index), edtType);
    return rank;
}

binaryReductionTree_t * initBinaryReductionTree(unsigned int numLeaves, artsEdt_t funPtr, artsType_t dbType, artsType_t edtType, uint32_t paramc, uint64_t * paramv, dim3 grid, dim3 block, artsGuid_t endGuid, uint32_t slot)
{
    binaryReductionTree_t * tree = (binaryReductionTree_t*) artsCalloc(sizeof(binaryReductionTree_t));
    tree->numLeaves = numLeaves;
    tree->totalNodes = 2 * numLeaves - 1;
    tree->interiorNodes = tree->totalNodes - tree->numLeaves;

    //Create space for all the guids
    artsGuid_t * allGuids = (artsGuid_t*) artsCalloc(sizeof(artsGuid_t) * tree->totalNodes);
    tree->redDbGuids = &allGuids[tree->interiorNodes];
    tree->redEdtGuids = allGuids;

    //Reserves the db guids
    for(unsigned int i=0; i<tree->numLeaves; i++)
        allGuids[tree->interiorNodes + i] = artsReserveGuidRoute(dbType, i % artsGetTotalNodes());

    //Reserves the edt guids
    reserveEdtGuids(allGuids, 0, edtType);

    //Check all the guids
    for(unsigned int i=0; i<tree->totalNodes; i++)
    {
        DPRINTF("i: %u guid: %lu rank: %u type: %u\n", i, allGuids[i], artsGuidGetRank(allGuids[i]), artsGuidGetType(allGuids[i]));
    }

    //Set up the signals
    for(unsigned int i=0; i<tree->interiorNodes; i++)
    {
        if(artsIsGuidLocal(tree->redEdtGuids[i]))
        {
            if(!i)
            {
                DPRINTF("Last: %lu -> %lu slot: %u\n", tree->redEdtGuids[i], endGuid, slot);
                artsEdtCreateGpuPTWithGuid(funPtr, tree->redEdtGuids[i], paramc, paramv, 2, grid, block, endGuid, slot, 0);
            }
            else
            {
                int parentIndex = parent(i);
                bool isRight = right(parentIndex) == i;
                artsGuid_t toSignal = tree->redEdtGuids[parentIndex];
                int toSignalSlot = (isRight) ? 1 : 0;
                DPRINTF("%lu -> %lu slot: %u parent: %d\n", tree->redEdtGuids[i], toSignal, toSignalSlot, parentIndex);
                artsEdtCreateGpuPTWithGuid(funPtr, tree->redEdtGuids[i], paramc, paramv, 2, grid, block, toSignal, toSignalSlot, 0);
            }
        }
    }

    return tree;
}

void fireBinaryReductionTree(binaryReductionTree_t * tree)
{
    //Signal the top edts
    for(unsigned int i=0; i<tree->numLeaves; i++)
    {
        int index = tree->interiorNodes + i;
        int parentIndex = parent(index);
        bool isRight = right(parentIndex) == index;
        artsGuid_t toSignal = tree->redEdtGuids[parentIndex];
        int toSignalSlot = (isRight) ? 1 : 0;
        DPRINTF("ToSignal: %lu slot: %u\n", toSignal, toSignalSlot);
        artsSignalEdt(toSignal, toSignalSlot, tree->redDbGuids[i]);
    }
}

void fireDbFromReductionTree(binaryReductionTree_t * tree, unsigned int whichDb)
{
    int index = tree->interiorNodes + whichDb;
    int parentIndex = parent(index);
    bool isRight = right(parentIndex) == index;
    artsGuid_t toSignal = tree->redEdtGuids[parentIndex];
    int toSignalSlot = (isRight) ? 1 : 0;
    DPRINTF("ToSignal: %lu slot: %u\n", toSignal, toSignalSlot);
    artsSignalEdt(toSignal, toSignalSlot, tree->redDbGuids[whichDb]);
}

void multiplyMM(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    // artsGuid_t toSignal = paramv[0];
    unsigned int size = sizeof(double) * tileSize * tileSize;
    unsigned int i = paramv[1];
    unsigned int j = paramv[2];
    unsigned int k = paramv[3];

    // artsGuid_t aTileGuid = depv[0].guid;
    // artsGuid_t bTileGuid = depv[1].guid;
    artsGuid_t cTileGuid = paramv[4];
    
    double * aTileDev  = (double*) depv[0].ptr;
    double * bTileDev  = (double*) depv[1].ptr;
    double * cTileDev = (double*) artsCudaMalloc(size);

    double alpha  = 1.0;
    double beta = 0.0;

    cublasDgemm(handle[artsGetGpuId()], CUBLAS_OP_N, CUBLAS_OP_N, 
        tileSize, tileSize, tileSize, 
        &alpha, 
        aTileDev, tileSize, 
        bTileDev, tileSize, 
        &beta, 
        cTileDev, tileSize);

    double * cTileHost = (double*) artsDbCreateWithGuid(cTileGuid, size);
    artsPutInDbFromGpu(cTileDev, cTileGuid, 0, size, true);
    fireDbFromReductionTree(redTree[i*numBlocks + j], k);
    // artsSignalEdt(toSignal, k, cTileGuid);
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

void finishBlockMM(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    // double * cMat  = (double*) depv[0].ptr;
    // for(unsigned int i=0; i<numBlocks; i++)
    //     for(unsigned int j=0; j<numBlocks; j++)
    //     {
    //         double * cTile = (double*) depv[3 + (i * numBlocks + j)].ptr;
    //         copyBlock(i, j, tileSize, cTile, matSize, cMat, false);
    //     }

    uint64_t time = artsGetTimeStamp() - start;
    PRINTF("Done %lu\n", time);
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

    uint64_t sumArgs[] = {tileSize};
    dim3 threads(SMTILE, SMTILE);
    dim3 grid((tileSize+SMTILE-1)/SMTILE, (tileSize+SMTILE-1)/SMTILE);
    redTree = (binaryReductionTree_t**) artsCalloc(sizeof(binaryReductionTree_t*)*numBlocks*numBlocks);
    for(unsigned int i=0; i<numBlocks; i++)
    {
        for(unsigned int j=0; j<numBlocks; j++)
        {
            redTree[i*numBlocks + j] = initBinaryReductionTree(numBlocks, sumMMKernel, ARTS_DB_GPU_WRITE, ARTS_GPU_EDT, 1, sumArgs, grid, threads, doneGuid, 3 + (i * numBlocks + j));
        }
    }
    if(!nodeId)
        PRINTF("MatSize: %u TileSize: %u\n", matSize, tileSize);
}

extern "C"
void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{   
    unsigned int totalThreads = artsGetTotalNodes() * artsGetTotalWorkers();
    unsigned int globalThreadId = nodeId * artsGetTotalWorkers() + workerId;   
  
    // if(!nodeId && !workerId)
    {
        for(unsigned int i=0; i<numBlocks; i++)
        {
            for(unsigned int j=0; j<numBlocks; j++)
            {
                if((i * numBlocks + j) % totalThreads == globalThreadId)
                {
                    artsGuid_t aTileGuid = artsGetGuid(aTileGuids, i * numBlocks + j);
                    double * aTile = (double*) artsDbCreateWithGuid(aTileGuid, sizeof(double) * tileSize * tileSize);
                    // initMatrix(tileSize, aTile, false, false);
                    // copyBlock(i, j, tileSize, aTile, matSize, aMatrix, true);

                    artsGuid_t bTileGuid = artsGetGuid(bTileGuids, i * numBlocks + j);
                    double * bTile = (double*) artsDbCreateWithGuid(bTileGuid, sizeof(double) * tileSize * tileSize);
                    // initMatrix(tileSize, bTile, false, false);
                    // copyBlock(i, j, tileSize, bTile, matSize, bMatrix, true);
                }
            }
        }
    }

    uint64_t sumArgs[] = {tileSize};
    dim3 threads(SMTILE, SMTILE);
    dim3 grid((tileSize+SMTILE-1)/SMTILE, (tileSize+SMTILE-1)/SMTILE);

    for(unsigned int i=0; i<numBlocks; i++)
    {
        for(unsigned int j=0; j<numBlocks; j++)
        {
            if((i * numBlocks + j) % totalThreads == globalThreadId)
            {
                // artsGuid_t sumGuid = artsEdtCreateGpuPT (sumMMKernel, nodeId, 1, sumArgs, numBlocks, grid, threads, doneGuid, 3 + (i * numBlocks + j), 0);
                artsGuid_t * cGuid = redTree[i*numBlocks + j]->redDbGuids;
                for(unsigned int k=0; k<numBlocks; k++)
                {
                    uint64_t args[] = {0, i, j, k, cGuid[k]};
                    artsGuid_t mulGuid = artsEdtCreateGpuLib(multiplyMM, nodeId, 5, args, 2, grid, threads);
                    artsSignalEdt(mulGuid, 0, artsGetGuid(aTileGuids, i * numBlocks + k));
                    artsSignalEdt(mulGuid, 1, artsGetGuid(bTileGuids, k * numBlocks + j));
                }
                // fireBinaryReductionTree(redTree[i*numBlocks + j]);
            }
        }
    }

    if(!nodeId && !workerId)
    {
        artsEdtCreateWithGuid(finishBlockMM, doneGuid, 0, NULL, 3 + numBlocks * numBlocks);
        artsSignalEdt(doneGuid, 0, NULL_GUID);
        artsSignalEdt(doneGuid, 1, NULL_GUID);
        artsSignalEdt(doneGuid, 2, NULL_GUID);
        PRINTF("Starting\n");
        start = artsGetTimeStamp();
    }
}

extern "C"
void initPerGpu(int devId, cudaStream_t * stream)
{
    if(!devId)
        handle = (cublasHandle_t*) artsCalloc(sizeof(cublasHandle_t) * artsGetNumGpus());
    cublasStatus_t stat = cublasCreate(&handle[devId]);
}

extern "C"
void cleanPerGpu(int devId, cudaStream_t * stream)
{
    cublasStatus_t stat = cublasDestroy(handle[devId]);
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}
