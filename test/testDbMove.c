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

artsGuid_t guid[4];
artsGuid_t shutdownGuid = NULL_GUID;

void check(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    for(unsigned int i=0; i<depc; i++)
    {
        for(unsigned int j=0; j<4; j++)
        {
            if(guid[j] == depv[i].guid)
            {
                unsigned int * data = depv[i].ptr;
                PRINTF("j: %u %lu: %u from %u\n", j, depv[i].guid, *data, artsGuidGetRank(depv[i].guid));
            }
        }
    }
    artsSignalEdtValue(shutdownGuid, -1, 0);
}

void shutDownEdt(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    artsShutdown();
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    guid[0] = artsReserveGuidRoute(ARTS_DB_ONCE_LOCAL, 0);
    guid[1] = artsReserveGuidRoute(ARTS_DB_ONCE_LOCAL, 0);
    guid[2] = artsReserveGuidRoute(ARTS_DB_ONCE_LOCAL, 1);
    guid[3] = artsReserveGuidRoute(ARTS_DB_ONCE_LOCAL, 1);
    
    shutdownGuid = artsReserveGuidRoute(ARTS_EDT, 0);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{
    if(!workerId)
    {
        if(nodeId == 0)
        {
            //Local to local
            unsigned int * aPtr = artsDbCreateWithGuid(guid[0], sizeof(unsigned int));
            *aPtr = 1;
            artsDbMove(guid[0], 0);
            
            //Local to remote
            unsigned int * aPtr2 = artsDbCreateWithGuid(guid[1], sizeof(unsigned int));
            *aPtr2 = 2;
            artsDbMove(guid[1], 1);
            
            //Remote to local
            artsDbMove(guid[2], 0);
            
            //Remote to remote
            artsDbMove(guid[3], 2);
        }
        
        if(nodeId == 1)
        {
            unsigned int * bPtr = artsDbCreateWithGuid(guid[2], sizeof(unsigned int));
            *bPtr = 3;
            
            unsigned int * cPtr = artsDbCreateWithGuid(guid[3], sizeof(unsigned int));
            *cPtr = 4;
        }
    }
    if(!workerId)
    {
        if(nodeId == 0)
        {
            artsGuid_t edtGuid = artsEdtCreate(check, nodeId, 0, NULL, 2);
            artsSignalEdt(edtGuid, 0, guid[0]);
            artsSignalEdt(edtGuid, 1, guid[2]);
            
            artsEdtCreateWithGuid(shutDownEdt, shutdownGuid, 0, NULL, 3);
        }
        
        if(nodeId == 1)
        {
            artsGuid_t edtGuid = artsEdtCreate(check, nodeId, 0, NULL, 1);
            artsSignalEdt(edtGuid, 0, guid[1]);
        }
        
        if(nodeId == 2)
        {
            artsGuid_t edtGuid = artsEdtCreate(check, nodeId, 0, NULL, 1);
            artsSignalEdt(edtGuid, 0, guid[3]);
        }
    }
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}
