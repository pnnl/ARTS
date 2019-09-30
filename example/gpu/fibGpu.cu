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

uint64_t start = 0;

//This is the GPU kernel
__global__ void fibJoin(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    unsigned int * x = (unsigned int *) depv[0].ptr;
    unsigned int * y = (unsigned int *) depv[1].ptr;
    unsigned int * res = (unsigned int *) depv[2].ptr;
    (*res) = (*x) + (*y);
}

void fibFork(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    unsigned int next = 0; //(artsGetCurrentNode() + 1) % artsGetTotalNodes();
//    PRINTF("NODE: %u WORKER: %u NEXT: %u\n", artsGetCurrentNode(), artsGetCurrentWorker(), next);
    
    artsGuid_t     doneGuid = paramv[0];
    unsigned int   slot     = (unsigned int) paramv[1];
    
    artsGuid_t     resGuid  = depv[0].guid;
    unsigned int * resPtr   = (unsigned int*) depv[0].ptr;
    
    if((*resPtr) < 2)
        artsSignalEdt(doneGuid, slot, resGuid);
    else
    {
        //Create two DB of type ARTS_DB_GPU
        unsigned int * x  = NULL;
        artsGuid_t     xGuid = artsDbCreate((void**) &x, sizeof(unsigned int), ARTS_DB_GPU_WRITE);
        (*x) = (*resPtr) - 1;
        
        unsigned int * y  = NULL;
        artsGuid_t     yGuid = artsDbCreate((void**) &y, sizeof(unsigned int), ARTS_DB_GPU_WRITE);
        (*y) = (*resPtr) - 2;
        
        //Create a continuation edt to run on the GPU
        dim3 grid(1);
        dim3 block(1);
        artsGuid_t joinGuid = artsEdtCreateGpu(fibJoin, next, 0, NULL, 3, grid, block, doneGuid, slot, resGuid);
        artsSignalEdt(joinGuid, 2, resGuid);
        
        //Create the forks which will run on the CPU
        uint64_t args[2] = {joinGuid, 0};
        artsGuid_t forkGuidX = artsEdtCreate(fibFork, next, 2, args, 1);
        artsSignalEdt(forkGuidX, 0, xGuid);
        
        args[1] = 1;
        artsGuid_t forkGuidY = artsEdtCreate(fibFork, next, 2, args, 1);
        artsSignalEdt(forkGuidY, 0, yGuid);
    }
}

void fibDone(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    uint64_t time = artsGetTimeStamp() - start;
    unsigned int * resPtr = (unsigned int*) depv[0].ptr;
    PRINTF("Fib %u: %u time: %lu nodes: %u workers: %u\n", paramv[0], *resPtr, time, artsGetTotalNodes(), artsGetTotalWorkers());
    artsShutdown();
}

extern "C"
void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    
}

extern "C"
void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{   
    if(!nodeId && !workerId)
    {
        unsigned int * resPtr  = NULL;
        artsGuid_t     resGuid = artsDbCreate((void**) &resPtr, sizeof(unsigned int), ARTS_DB_GPU_WRITE);
        if (argc < 2) {
          PRINTF("Format: ./fibGpu NUMBER\n");
          artsShutdown();
          return;
        }
        *resPtr = atoi(argv[1]);
        
        artsGuid_t doneGuid = artsEdtCreate(fibDone, 0, 1, (uint64_t*)resPtr, 1);
        
        uint64_t args[] = {doneGuid, 0};
        artsGuid_t fibGuid = artsEdtCreate(fibFork, 0, 2, args, 1);
        artsSignalEdt(fibGuid, 0, resGuid);
        start = artsGetTimeStamp();
    }
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}
