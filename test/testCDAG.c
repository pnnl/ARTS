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
unsigned int numReads = 0;
unsigned int numWrites = 0;
unsigned int numDynamicReads = 0;
unsigned int numDynamicWrites = 0;
artsGuid_t shutdownGuid;
artsGuid_t dbGuid;
artsGuid_t * readGuids;
artsGuid_t * writeGuids;

void shutdownEdt(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    unsigned int * array = depv[0].ptr;
    for(unsigned int i=0; i<numWrites; i++)
    {
        PRINTF("i: %u %u\n", i, array[i]);
    }
    artsShutdown();
}

void readTest(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
//    PRINTF("READ\n");
    unsigned int * array = depv[0].ptr;
    for(unsigned int i=0; i<numWrites; i++)
    {
        if(array[i] != 0 && array[i] != i)
        {
            PRINTF("BAD VALUE i: %u %u\n", i, array[i]);
        }
    }
    artsSignalEdtValue(shutdownGuid, -1, 0);
}

void writeTest(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    unsigned int index = paramv[0];
    unsigned int * array = depv[0].ptr;
//    PRINTF("WRITE %u\n", index);
    array[index] += index;

    for(unsigned int i=0; i<numDynamicReads; i++)
    {
        artsGuid_t guid = artsEdtCreate(readTest, artsGetCurrentNode(), 0, NULL, 1);
        artsSignalEdt(guid, 0, dbGuid);
    }

    for(unsigned int i=0; i<numDynamicWrites; i++)
    {
        paramv[0] = (paramv[0]+1) % numWrites;
        artsGuid_t guid = artsEdtCreate(readTest, artsGetCurrentNode(), 0, NULL, 1);
        artsSignalEdt(guid, 0, dbGuid);
    }

    if(!index)
        artsSignalEdt(shutdownGuid, 0, artsGuidCast(dbGuid, ARTS_DB_WRITE));
    else
        artsSignalEdtValue(shutdownGuid, -1, 0);
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    numReads = atoi(argv[1]);
    numWrites = atoi(argv[2]);
    numDynamicReads = atoi(argv[3]);
    numDynamicWrites = atoi(argv[4]);
    if(!nodeId)
        PRINTF("Reads: %u Writes: %u Dynamic Reads: %u Dynamic Writes: %u Final Deps: %u\n", numReads, numWrites, numDynamicReads, numDynamicWrites, numDynamicReads*numWrites+numDynamicWrites*numWrites+numReads+numWrites);

    readGuids = artsMalloc(sizeof(artsGuid_t)*numReads);
    writeGuids = artsMalloc(sizeof(artsGuid_t)*numWrites);

    dbGuid = artsReserveGuidRoute(ARTS_DB_READ, 0);

    for(unsigned int i=0; i<numReads; i++)
        readGuids[i] = artsReserveGuidRoute(ARTS_EDT, i % artsGetTotalNodes());
    for(unsigned int i=0; i<numWrites; i++)
        writeGuids[i] = artsReserveGuidRoute(ARTS_EDT, i % artsGetTotalNodes());

    shutdownGuid = artsReserveGuidRoute(ARTS_EDT, 0);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{
    if(!workerId)
    {
        if(!nodeId)
        {
            unsigned int * ptr = artsDbCreateWithGuid(dbGuid, sizeof(unsigned int) * numWrites);
            for(unsigned int i=0; i<numWrites; i++)
                ptr[i] = 0;

            artsEdtCreateWithGuid(shutdownEdt, shutdownGuid, 0, NULL, numDynamicReads*numWrites+numDynamicWrites*numWrites+numReads+numWrites);
        }

        for(uint64_t i=0; i<numReads; i++)
        {
            if(artsIsGuidLocal(readGuids[i]))
            {
                artsEdtCreateWithGuid(readTest, readGuids[i], 0, NULL, 1);
                artsSignalEdt(readGuids[i], 0, dbGuid);
            }
        }

        for(uint64_t i=0; i<numWrites; i++)
        {
            if(artsIsGuidLocal(writeGuids[i]))
            {
                artsEdtCreateWithGuid(writeTest, writeGuids[i], 1, &i, 1);
                artsSignalEdt(writeGuids[i], 0, artsGuidCast(dbGuid, ARTS_DB_WRITE));
            }
        }


    }
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}
