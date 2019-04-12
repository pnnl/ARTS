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
#include <string.h>
#include "arts.h"

unsigned int elemsPerNode = 4;
artsArrayDb_t * array = NULL;

void shutdown(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    PRINTF("Depc: %u\n", depc);
    for(unsigned int i=0; i<depc; i++)
    {
        unsigned int * data = depv[i].ptr;
        for(unsigned int j=0; j<elemsPerNode; j++)
        {
            PRINTF("%u: %u\n", i*elemsPerNode+j, data[j]);
        }
    }
    artsShutdown();
}

void check(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    artsGatherArrayDb(array, shutdown, 0, 0, NULL, 0);
}

void edtFunc(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    unsigned int index = paramv[0];
    artsGuid_t checkGuid = paramv[1];
    unsigned int * value = depv[0].ptr;
    *value = index;
    PRINTF("%u:  %u %p\n", index, *value, value);
    artsSignalEdtValue(checkGuid, 0, 0);
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{   
    if(!nodeId && !workerId)
    {
        artsGuid_t checkGuid = artsEdtCreate(check, 0, 0, NULL, elemsPerNode * artsGetTotalNodes());
        artsGuid_t guid = artsNewArrayDb(&array, sizeof(unsigned int), elemsPerNode * artsGetTotalNodes());
        artsForEachInArrayDbAtData(array, 1, edtFunc, 1, &checkGuid);
//        artsForEachInArrayDb(array, edtFunc, 1, &checkGuid);
    }
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}
