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
artsGuid_t aGuid = NULL_GUID;
artsGuid_t bGuid = NULL_GUID;

void check(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    unsigned int * ptr;
    artsGuid_t guid = artsDbCreate((void**)&ptr, sizeof(unsigned int), ARTS_DB_ONCE);
    *ptr = 2;
    
    PRINTF("Check: %lu %u newGuid: %lu\n", depv[0].guid, *((unsigned int*)depv[0].ptr), guid);
    artsSignalEdt(bGuid, 0, guid);
}

void shutDownEdt(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    PRINTF("ShutdownEdt: %lu %u\n", depv[0].guid, *((unsigned int*)depv[0].ptr));
    artsShutdown();
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    dbGuid = artsReserveGuidRoute(ARTS_DB_ONCE, 0);
    aGuid = artsReserveGuidRoute(ARTS_EDT, 1);
    bGuid = artsReserveGuidRoute(ARTS_EDT, 2);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{   
    if(!workerId)
    {
        if(nodeId == 0)
        {
            unsigned int * aPtr = artsDbCreateWithGuid(dbGuid, sizeof(unsigned int));
            *aPtr = 1;
        }
        
        if(nodeId == 1)
        {
            artsEdtCreateWithGuid(check, aGuid, 0, NULL, 1);
            artsSignalEdt(aGuid, 0, dbGuid);
        }
        
        if(nodeId == 2)
        {
            artsEdtCreateWithGuid(shutDownEdt, bGuid, 0, NULL, 1);
        }
    }
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}
