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
#include "buffer.h"

#define NUMBUFFERS 2

volatile unsigned int currentBuffer = 0; 
unsigned int *** gpuBufferPtr = NULL; //NUMBUFFERS per GPU (Many)
unsigned int **  cpuBufferPtr = NULL; //NUMBUFFERS per Node (One)

artsGuid_t * masterBufferGuids = NULL;
artsGuid_t * bufferGuids = NULL;
unsigned int *** bufferPtr = NULL;

void createBuffersOnCpu(unsigned int size)
{
    unsigned int numNodes = artsGetTotalNodes();
    unsigned int numGpus = artsGetTotalGpus();
    unsigned int nodeId = artsGetCurrentNode();

    cpuBufferPtr = (unsigned int**) artsCalloc(sizeof(unsigned int*) * NUMBUFFERS);
    for(unsigned int i=0; i<NUMBUFFERS; i++)
        cpuBufferPtr[i] = (unsigned int*) artsCalloc(sizeof(unsigned int) * size);

    gpuBufferPtr = (unsigned int***) artsCalloc(sizeof(unsigned int**) * NUMBUFFERS);
    for(unsigned int i=0; i<NUMBUFFERS; i++)
        gpuBufferPtr[i] = (unsigned int**) artsCalloc(sizeof(unsigned int*) * numGpus);

    masterBufferGuids = (artsGuid_t*) artsCalloc(sizeof(artsGuid_t) * numNodes);
    for(unsigned int i=0; i<numNodes; i++)
        masterBufferGuids[i] = artsReserveGuidRoute(ARTS_DB_READ, i);

    bufferGuids = (artsGuid_t*)artsDbCreateWithGuid(masterBufferGuids[nodeId], sizeof(artsGuid_t) * numNodes * NUMBUFFERS);
    for(unsigned int i=0; i<NUMBUFFERS; i++)
    {
        for(unsigned int j=0; j<numNodes; j++)
            bufferGuids[i*numNodes + j] = artsReserveGuidRoute(ARTS_DB_GPU_READ, j);
    }
}

void createBuffersOnGpu(unsigned int gpu, unsigned int size)
{
    if(!gpuBufferPtr)
        PRINTF("Must run createBuffersOnCpu first!\n");

    for(unsigned int i=0; i<NUMBUFFERS; i++)
    {
        gpuBufferPtr[i][gpu] = (unsigned int*) artsCudaMalloc(sizeof(unsigned int) * size);
    }
}

void createBufferDB()
{
    unsigned int numNodes = artsGetTotalNodes();
    unsigned int numGpus = artsGetTotalGpus();
    unsigned int nodeId = artsGetCurrentNode();

    bufferPtr = (unsigned int***) artsCalloc(sizeof(unsigned int**)*NUMBUFFERS);
    for(unsigned int j=0; j<NUMBUFFERS; j++)
    {
        
        bufferPtr[j] = (unsigned int**)artsDbCreateWithGuid(bufferGuids[j*numNodes + nodeId], sizeof(unsigned int*) * (numGpus+1));
        for(uint64_t i=0; i<numGpus; i++)
            bufferPtr[j][i] = gpuBufferPtr[j][i];
        bufferPtr[j][numGpus] = cpuBufferPtr[j];
    }
}

void freeBuffersOnGpu(unsigned int gpu)
{
    if(!gpuBufferPtr)
        PRINTF("Must run createBuffersOnCpu first!\n");

    for(unsigned int i=0; i<NUMBUFFERS; i++)
    {
        artsCudaFree(gpuBufferPtr[i][gpu]);
    }
}

void printMasterBufferGuids()
{
    unsigned int numNodes = artsGetTotalNodes();
    for(unsigned int i=0; i<numNodes; i++)
        PRINTF("masterBufferGuids[%u]: %lu\n", i, masterBufferGuids[i]);
}

void printLocalBufferGuids()
{
    unsigned int numNodes = artsGetTotalNodes();
    unsigned int nodeId = artsGetCurrentNode();
    for(unsigned int i=0; i<NUMBUFFERS; i++)
    {
        PRINTF("bufferGuids[%u][%u]: %lu\n", i, nodeId, bufferGuids[i*numNodes + nodeId]);
    }
}

void printBufferPtr()
{
    unsigned int numGpus = artsGetTotalGpus();
    for(unsigned int i=0; i<NUMBUFFERS; i++)
        for(unsigned int j=0; j<numGpus+1; j++)
        {
            PRINTF("buffer: %u bufferPtr[%u]: %p\n", i, j, bufferPtr[i][j]);
        }
}

void printRawPtr()
{
    unsigned int numGpus = artsGetTotalGpus();
    for(unsigned int i=0; i<NUMBUFFERS; i++)
    {
        for(unsigned int j=0; j<numGpus; j++)
        {
            PRINTF("buffer: %u gpuBufferPtr[%u]: %p\n", i, j, gpuBufferPtr[i][j]);
        }
        PRINTF("buffer: %u cpuBufferPtr   : %p\n", i,  cpuBufferPtr[i]);
    }
}

artsGuid_t getBufferGuid(unsigned int nodeId, uint64_t level)
{
    unsigned int numNodes = artsGetTotalNodes();
    uint64_t index = level % NUMBUFFERS;
    // PRINTF("Get index: %u nodeId: %u %lu\n", index, nodeId, bufferGuids[index*numNodes + nodeId]);
    return bufferGuids[index*numNodes + nodeId];
}

unsigned int * getLocalBuffer(unsigned int index, uint64_t level)
{
    unsigned int numGpus = artsGetTotalGpus();
    unsigned int bufferIndex = level % NUMBUFFERS;
    if(index == numGpus)
        return cpuBufferPtr[bufferIndex];
    return gpuBufferPtr[bufferIndex][index];
}

void resetBuffer(uint64_t level)
{
    unsigned int numNodes = artsGetTotalNodes();
    unsigned int nodeId = artsGetCurrentNode();
    uint64_t index = level % NUMBUFFERS;

    for(unsigned int j=0; j<numNodes; j++)
    {
        bufferGuids[index*numNodes + j] = artsDbRename(bufferGuids[index*numNodes + j]);
        // PRINTF("RENAME index: %u node: %u %lu\n", index, j, bufferGuids[index*numNodes + j]);
    }

    unsigned int offset = &bufferGuids[index*numNodes] - bufferGuids;
    void * src = (void*)&bufferGuids[index*numNodes];
    for(unsigned int i=0; i<numNodes; i++)
    {
        if(i != nodeId)
            artsPutInDbAt(src, NULL_GUID, masterBufferGuids[i], -1, offset, sizeof(artsGuid_t) * numNodes, i);
    }

    
}