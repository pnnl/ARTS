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

artsGuid_t dbGuid = NULL_GUID;
artsGuid_t shutdownGuid = NULL_GUID;
artsGuid_t edtGuidFixed = NULL_GUID;
unsigned int numElements = 0;
unsigned int blockSize = 0;
unsigned int stride = 0;

void getter(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    unsigned int sum = 0;
    for(unsigned int i=0; i<depc; i++)
    {
        unsigned int * data = depv[i].ptr;
        for(unsigned int j=0; j<stride; j++)
        {
            sum+=data[j];
        }
    }
    artsSignalEdtValue(shutdownGuid, artsGetCurrentNode(), sum);
}

void creater(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    unsigned int * data = artsMalloc(sizeof(unsigned int)*numElements);
    for(unsigned int i=0; i<numElements; i++)
    {
        data[i] = i;
    }
    artsDbCreateWithGuidAndData(dbGuid, data, sizeof(unsigned int) * numElements);
    artsEdtCreateWithGuid(getter, edtGuidFixed, 0, NULL, blockSize/stride);
}

void shutDownEdt(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    unsigned int sum = 0;
    for(unsigned int i=0; i<artsGetTotalNodes(); i++)
        sum += (unsigned int)depv[i].guid;
    
    unsigned int compare = 0;
    for(unsigned int i=0; i<numElements; i++)
        compare += i;
    
    if(sum == compare)
        PRINTF("CHECK SUM: %u vs %u\n", sum, compare);
    else
        PRINTF("FAIL SUM: %u vs %u\n", sum, compare);
    artsShutdown();
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    dbGuid = artsReserveGuidRoute(ARTS_DB_PIN, 0);
    shutdownGuid = artsReserveGuidRoute(ARTS_EDT, 0);
    edtGuidFixed = artsReserveGuidRoute(ARTS_EDT, 0);
    numElements = atoi(argv[1]);
    blockSize = numElements / artsGetTotalNodes();
    stride = atoi(argv[2]);
    if(!nodeId)
        PRINTF("numElements: %u blockSize: %u stride: %u\n", numElements, blockSize, stride);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{
    if(blockSize % stride)
    {
        if(!nodeId && !workerId)
        {
            artsShutdown();
        }
        return;
    }
    
    if(!workerId)
    {
        if(!nodeId)
        {
            artsEdtCreate(creater, 0, 0, NULL, 0);
            artsEdtCreateWithGuid(shutDownEdt, shutdownGuid, 0, NULL, artsGetTotalNodes());
        }
        
        unsigned int deps = blockSize/stride;
        if(!nodeId)
        {
            for(unsigned int j=0; j<deps; j++)
            {
                artsGetFromDb(edtGuidFixed, dbGuid, j, sizeof(unsigned int) * (nodeId*blockSize + j*stride), sizeof(unsigned int) * stride);
            }
        }
        else
        {
            artsGuid_t edtGuid = artsEdtCreate(getter, nodeId, 0, NULL, deps);
            for(unsigned int j=0; j<deps; j++)
            {
                artsGetFromDb(edtGuid, dbGuid, j, sizeof(unsigned int) * (nodeId*blockSize + j*stride), sizeof(unsigned int) * stride);
            }
        }
    }
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}
