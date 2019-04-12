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

artsGuid_t dbDestGuid = NULL_GUID;
artsGuid_t shutdownGuid = NULL_GUID;
unsigned int numElements = 0;
unsigned int blockSize = 0;

void dummy(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    artsGuid_t resultGuid = paramv[0];
    unsigned int resultSize = paramv[1];
    unsigned int bufferSize = paramv[2]/sizeof(unsigned int);
    unsigned int * buffer = depv[0].ptr;
    PRINTF("%lu %u %u %p\n", resultGuid, resultSize, bufferSize, buffer);
    unsigned int * sum = artsCalloc(resultSize);
    for(unsigned int i=0; i<bufferSize; i++)
    {
        PRINTF("%u\n", buffer[i]);
        *sum+=buffer[i];
    }
    PRINTF("Sum before: %u\n", *sum);
    artsSetBuffer(resultGuid, sum, resultSize);
}

void startEdt(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    uint64_t args[3];
    
    unsigned int result = 0;
    unsigned int * dataPtr = &result;
    args[0] = artsAllocateLocalBuffer((void**)&dataPtr, sizeof(unsigned int), 1, NULL_GUID);
    args[1] = sizeof(unsigned int);
    
    unsigned int bufferSize = sizeof(unsigned int)*5;
    unsigned int * data = artsCalloc(bufferSize);
    for(unsigned int i=0; i<5; i++)
        data[i] = i;
    args[2] = bufferSize;
    
    artsActiveMessageWithBuffer(dummy, (artsGetCurrentNode()+1) % artsGetTotalNodes(), 3, args, 0, data, bufferSize);
    
    while(!result)
    {
        artsYield();
        PRINTF("Did a yield\n");
    }
    
    PRINTF("Sum: %u\n", result);
    artsShutdown();
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    PRINTF("Starting\n");
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{
    if(!nodeId && !workerId)
    {
        artsEdtCreate(startEdt, 0, 0, NULL, 0);
    }
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}
