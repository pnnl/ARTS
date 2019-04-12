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
unsigned int numWrites = 0;
artsGuid_t dbGuid;
artsGuid_t * writeGuids;


void writeTest(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    unsigned int index = paramv[0];
    unsigned int * array = depv[0].ptr;
//    if(array)
//    {
        for(unsigned int i=index; i<numWrites; i++)
            array[i] = index;
//    }
    if(paramc > 1)
    {
        PRINTF("-----------------SIGNALLING NEXT %u\n", index);
        artsSignalEdtValue((artsGuid_t) paramv[1], -1, 0);
    }
    else
    {
        for(unsigned int i=0; i<numWrites; i++)
        {
            PRINTF("i: %u %u\n", i, array[i]);
        }
        artsShutdown();
    } 
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    dbGuid = artsReserveGuidRoute(ARTS_DB_READ, 0);
    
    numWrites = atoi(argv[1]);
    writeGuids = artsMalloc(sizeof(artsGuid_t)*numWrites);
    for(unsigned int i=0; i<numWrites; i++)
        writeGuids[i] = artsReserveGuidRoute(ARTS_EDT, i % artsGetTotalNodes());
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
        }
        
        uint64_t args[2];
        for(uint64_t i=0; i<numWrites; i++)
        {
            if(artsIsGuidLocal(writeGuids[i]))
            {
                args[0] = i;
                
                if(i < numWrites-1)
                {
                    args[1] = writeGuids[i+1];
                    artsEdtCreateWithGuid(writeTest, writeGuids[i], 2, args, 2);
                }
                else
                {
                    artsEdtCreateWithGuid(writeTest, writeGuids[i], 1, args, 2);
                }
                artsSignalEdt(writeGuids[i], 0, artsGuidCast(dbGuid, ARTS_DB_WRITE));
            }
        }
        if(!nodeId)
            artsSignalEdtValue(writeGuids[0], -1, 0);
        
    }
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}

