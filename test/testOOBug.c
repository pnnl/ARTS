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

#define MATSIZE 3
#define TILE 1

uint64_t start = 0;

unsigned int numBlocks = 1;
artsGuid_t aMatGuid = NULL_GUID;
artsGuid_t bMatGuid = NULL_GUID;
artsGuid_t cMatGuid = NULL_GUID;
artsGuidRange * aTileGuids = NULL;
artsGuidRange * bTileGuids = NULL;

void printMatrix(unsigned int rowSize, float * mat)
{
    unsigned int columnSize = rowSize;
    for(unsigned int i=0; i<columnSize; i++)
    {
        for(unsigned int j=0; j<rowSize; j++)
        {
            printf("%5.2f ", mat[i*rowSize + j]);
        }
        printf("\n");
    }
}

void initMatrix(unsigned int rowSize, float * mat, bool identity, bool zero)
{
    unsigned int columnSize = rowSize;
    for(unsigned int i=0; i<columnSize; i++)
    {
        for(unsigned int j=0; j<rowSize; j++)
        {
            if(zero)
                mat[i*rowSize + j] = 0;
            else if(identity)
            {
                if(i==j)
                    mat[i*rowSize + j] = 1;
                else
                    mat[i*rowSize + j] = 0;
            }
            else
                mat[i*rowSize + j] = i * rowSize + j;
        }
    }
}

void copyBlock(unsigned int x, unsigned int y, unsigned int tileRowSize, float * tile, unsigned int rowSize, float * mat, bool toTile)
{
    unsigned int tileColumnSize = tileRowSize;
    unsigned int columnSize      = rowSize;
    
    unsigned int xOffset = tileRowSize    * x;
    unsigned int yOffset = tileColumnSize * y;
    
    if(toTile)
    {
        for(unsigned int i=0; i<tileColumnSize; i++)
            memcpy(&tile[ i * tileRowSize ], &mat[ (i + yOffset) * rowSize + xOffset ], tileRowSize * sizeof(float));
    }
    else
    {
        for(unsigned int i=0; i<tileColumnSize; i++)
            memcpy(&mat[ (i + yOffset) * rowSize + xOffset ], &tile[ i * tileRowSize ], tileRowSize * sizeof(float));
    }

}

void initBlockMM(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    unsigned int i = paramv[0];
    unsigned int j = paramv[1];
    
//    PRINTF("%s %u %u\n", __func__, i, j);
    
    float * aMat = (float*) depv[0].ptr;
    float * bMat = (float*) depv[1].ptr;
    
    artsGuid_t aGuid = artsGetGuid(aTileGuids, i * numBlocks + j);
    artsGuid_t bGuid = artsGetGuid(bTileGuids, i * numBlocks + j);
    
    float * aTile = (float*) artsDbCreateWithGuid(aGuid, sizeof(float) * TILE * TILE);
    float * bTile = (float*) artsDbCreateWithGuid(bGuid, sizeof(float) * TILE * TILE);
//    artsDbCreateWithGuidAndData(artsGuid_t guid, void * data, uint64_t size)
    
    copyBlock(i, j, TILE, aTile, MATSIZE, aMat, true);
    copyBlock(i, j, TILE, bTile, MATSIZE, bMat, true);
}

void mmKernelCPU(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    unsigned int I = paramv[4];
    unsigned int J = paramv[5];
    artsGuid_t toSignal = (artsGuid_t) paramv[1];
    unsigned int K = (unsigned int) paramv[2];
//    PRINTF("%s %u %u %u %u SIG: %u\n", __func__, I,K, K,J);
    artsGuid_t cTileGuid = (artsGuid_t) paramv[3];
    const int blk = (int) paramv[0];
    float *A = (float *) depv[0].ptr; 
    float *B = (float *) depv[1].ptr;
    float *C = (float *) depv[2].ptr;
    
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
    artsSignalEdt(toSignal, K, cTileGuid);
}

void multiplyMM(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    artsGuid_t toSignal = paramv[0];
    
    unsigned int rowSize    = TILE;
    unsigned int columnSize = TILE;
    
    unsigned int i = paramv[1];
    unsigned int j = paramv[2];
    unsigned int k = paramv[3];
    
    PRINTF("%s i: %u k: %u x % k: %u j: %u %lu %lu\n", __func__, i, k, k, j, depv[0].guid, depv[1].guid);
    
    float * aTile = (float*) depv[0].ptr;
    float * bTile = (float*) depv[1].ptr;
    float * cTile = NULL;
    
    artsGuid_t cTileGuid = artsDbCreate((void**) &cTile, sizeof(float) * TILE * TILE, ARTS_DB_GPU_WRITE);
    initMatrix(rowSize, cTile, false, true);
    
    uint64_t args[] = {TILE, toSignal, k, cTileGuid, i, j, k};
    artsGuid_t    mulGpuGuid = artsEdtCreate(mmKernelCPU, 0, 7, args, 3);
    artsSignalEdt(mulGpuGuid, 0, depv[0].guid);
    artsSignalEdt(mulGpuGuid, 1, depv[1].guid);
    artsSignalEdt(mulGpuGuid, 2, cTileGuid);
}

void sumMM(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    artsGuid_t doneGuid = paramv[0];
    
    unsigned int rowSize    = TILE;
    unsigned int columnSize = TILE;
    
    unsigned int I = paramv[1];
    unsigned int J = paramv[2];
    
//    PRINTF("%s: i: %u j: %u %lu\n", __func__, I, J, doneGuid);
    
    float * cTile;
    artsGuid_t cTileGuid = artsDbCreate((void**) &cTile, sizeof(float) * TILE * TILE, ARTS_DB_GPU_WRITE);
    initMatrix(rowSize, cTile, false, true);
    
    for(unsigned int i=0; i<depc; i++)
    {
        float * toAdd = (float*) depv[i].ptr;
        for(unsigned int j=0; j<columnSize; j++)
        {
            for(unsigned int k=0; k<rowSize; k++)
            {
                cTile[j * rowSize + k] += toAdd[j * rowSize + k];
            }
        }
    }    
    artsSignalEdt(doneGuid, 1 + (I * numBlocks + J), cTileGuid);
}

void finishBlockMM(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    PRINTF("%s\n", __func__);
    artsGuid_t toSignal = paramv[0]; 
    float * cMat  = (float*) depv[0].ptr;
    for(unsigned int i=0; i<numBlocks; i++)
    {
        for(unsigned int j=0; j<numBlocks; j++)
        {
            float * cTile = (float*) depv[1 + i * numBlocks + j].ptr;
            copyBlock(i, j, TILE, cTile, MATSIZE, cMat, false);
        }
    }
    uint64_t time = artsGetTimeStamp() - start;
//    printMatrix(MATSIZE, cMat);
    PRINTF("DONE %lu\n", time);
    artsShutdown();
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    numBlocks = MATSIZE / TILE;
    
    aMatGuid = artsReserveGuidRoute(ARTS_DB_READ, 0);
    bMatGuid = artsReserveGuidRoute(ARTS_DB_READ, 0);
    cMatGuid = artsReserveGuidRoute(ARTS_DB_READ, 0);
    
    aTileGuids = artsNewGuidRangeNode(ARTS_DB_PIN, numBlocks*numBlocks, 0);
    bTileGuids = artsNewGuidRangeNode(ARTS_DB_PIN, numBlocks*numBlocks, 0);
    if(!nodeId)
    {
        float * aMat = (float*) artsDbCreateWithGuid(aMatGuid, MATSIZE * MATSIZE * sizeof(float));
        float * bMat = (float*) artsDbCreateWithGuid(bMatGuid, MATSIZE * MATSIZE * sizeof(float));
        float * cMat = (float*) artsDbCreateWithGuid(cMatGuid, MATSIZE * MATSIZE * sizeof(float));
        
        initMatrix(MATSIZE, aMat, false, false);
        initMatrix(MATSIZE, bMat,  true, false);
        initMatrix(MATSIZE, cMat, false, true);
        
//        PRINTF("A MATRIX\n");
//        printMatrix(MATSIZE, aMat);
//        PRINTF("B MATRIX\n");
//        printMatrix(MATSIZE, bMat);
//        PRINTF("C MATRIX\n");
//        printMatrix(MATSIZE, cMat);
        PRINTF("Starting\n");
    }
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{   
    if(!nodeId && !workerId)
    {
        artsGuid_t doneGuid = artsEdtCreate(finishBlockMM, 0, 0, NULL, 1 + numBlocks * numBlocks);
        artsSignalEdt(doneGuid, 0, cMatGuid);
        
        for(unsigned int i=0; i<numBlocks; i++)
        {
            for(unsigned int j=0; j<numBlocks; j++)
            {
                uint64_t initArgs[] = {i, j}; 
                artsGuid_t initGuid = artsEdtCreate(initBlockMM, 0, 2, initArgs, 2);
                artsSignalEdt(initGuid, 0, aMatGuid);
                artsSignalEdt(initGuid, 1, bMatGuid);
                
                uint64_t sumArgs[] = {doneGuid, i, j};
                artsGuid_t sumGuid = artsEdtCreate(sumMM, 0, 3, sumArgs, numBlocks);
                PRINTF("SUMGUID: i: %u j: %u %lu\n", i, j, sumGuid);
                for(unsigned int k=0; k<numBlocks; k++)
                {
                    uint64_t args[] = {sumGuid, i, j, k};
                    artsGuid_t mulGuid = artsEdtCreate(multiplyMM, 0, 4, args, 2);
                    PRINTF("%lu Signaling: i: %u k: %u %lu i: %u k: %u %lu\n", mulGuid, i, k, artsGetGuid(aTileGuids, i * numBlocks + k), k, j, artsGetGuid(bTileGuids, k * numBlocks + j));
                    artsSignalEdt(mulGuid, 0, artsGetGuid(aTileGuids, i * numBlocks + k));
                    artsSignalEdt(mulGuid, 1, artsGetGuid(bTileGuids, k * numBlocks + j));
                }
            }
        }
        start = artsGetTimeStamp();
    }
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}

//#include <stdio.h>
//#include <stdlib.h>
//#include "arts.h"
//
//#define NUMGUIDS 128
//
//artsGuid_t doneGuid = NULL_GUID;
//artsGuidRange * guids = NULL;
//
//void create(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
//{
//    unsigned int i = paramv[0];
//    artsGuid_t guid = artsGetGuid(guids, i);
//    artsDbCreateWithGuid(guid, sizeof(float));
//}
//
//void work(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
//{
//    artsGuid_t toSignal = (artsGuid_t) paramv[0];
//    artsSignalEdt(toSignal, 0, NULL_GUID);
//}
//
//void stage(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
//{
//    artsGuid_t toSignal = paramv[0];
//    unsigned int i      = paramv[1];
//    
//    artsGuid_t guid    = artsGetGuid(guids, i);
//    artsGuid_t edtGuid = artsEdtCreate(work, 0, 1, paramv, 1);
//    artsSignalEdt(edtGuid, 0, guid);
//}
//
//void finish(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
//{
//    PRINTF("DONE\n");
//    artsShutdown();
//}
//
//void initPerNode(unsigned int nodeId, int argc, char** argv)
//{
//    doneGuid = artsReserveGuidRoute(ARTS_EDT, 0);
//    guids = artsNewGuidRangeNode(ARTS_DB_GPU, NUMGUIDS, 0);
//}
//
//void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
//{   
//    if(!nodeId && !workerId)
//    {
//        artsEdtCreateWithGuid(finish, doneGuid, 0, NULL, NUMGUIDS);
//        
//        for(unsigned int i=0; i<NUMGUIDS; i++)
//        {
//            artsEdtCreate(create, 0, 1, (uint64_t*)&i, 0);
//            
//            uint64_t args[] = {doneGuid, i};
//            artsEdtCreate(stage, 0, 2, args, 0);                
//        }
//    }
//}
//
//int main(int argc, char** argv)
//{
//    artsRT(argc, argv);
//    return 0;
//}
