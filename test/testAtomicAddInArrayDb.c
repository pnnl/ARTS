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

unsigned int elementsPerBlock = 0;
unsigned int blocks = 0;
unsigned int numAdd = 0;
artsArrayDb_t * array = NULL;
artsGuid_t arrayGuid = NULL_GUID;

void end(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    for(unsigned int i=0; i<depc-1; i++)
    {
        unsigned int data = depv[i].guid;
        PRINTF("updates: %u\n", data);
    }
    artsShutdown();
}

//Created by the epochEnd via gather will signal end
void check(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    for(unsigned int i=0; i<blocks; i++)
    {
        unsigned int * data = depv[i].ptr;
        for(unsigned int j=0; j<elementsPerBlock; j++)
        {
            PRINTF("i: %u j: %u %u\n", i, j, data[j]);
        }
    }
    artsSignalEdtValue(paramv[0], numAdd*elementsPerBlock*blocks, 0);
}

//This is run at the end of the epoch
void epochEnd(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{    
    unsigned int numInEpoch = depv[0].guid;
    PRINTF("%u in Epoch\n", numInEpoch);
    artsGatherArrayDb(array, check, 0, 1, paramv, 0);
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    elementsPerBlock = atoi(argv[1]);
    blocks = artsGetTotalNodes();
    numAdd = atoi(argv[2]);
    arrayGuid = artsReserveGuidRoute(ARTS_DB_PIN, 0);
    if(!nodeId)
        PRINTF("ElementsPerBlock: %u Blocks: %u\n", elementsPerBlock, blocks);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{   
    
    if(!workerId && !nodeId)
    {
        //The end will get all the updates and a signal from the gather
        artsGuid_t endGuid = artsEdtCreate(end, 0, 0, NULL, numAdd*elementsPerBlock*blocks + 1);

        artsGuid_t endEpochGuid = artsEdtCreate(epochEnd, 0, 1, &endGuid, 1);
        artsInitializeAndStartEpoch(endEpochGuid, 0);

        array = artsNewArrayDbWithGuid(arrayGuid, sizeof(unsigned int), elementsPerBlock * blocks);

        for(unsigned int j=0; j<numAdd; j++)
        {
            for(unsigned int i=0; i<elementsPerBlock*blocks; i++)
            {
                PRINTF("i: %u Slot: %u edt: %lu\n", i, j*elementsPerBlock*blocks + i, endGuid);
                artsAtomicAddInArrayDb(array, i, 1, endGuid, j*elementsPerBlock*blocks + i);
            }
        }
    }
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}
