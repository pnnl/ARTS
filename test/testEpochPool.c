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
#include "shadAdapter.h"

uint64_t numDummy = 0;

void dummytask(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]) 
{
    uint64_t index = paramv[0];
    uint64_t dep = paramv[1];
    PRINTF("Dep: %lu ID: %lu Current Node: %u Current Worker: %u\n", dep, index, artsGetCurrentNode(), artsGetCurrentWorker());
}

void rootTask(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[]) 
{    
    uint64_t dep = paramv[0];
    PRINTF("Root: %lu\n", dep);
    if(dep)
    {
        artsGuid_t poolGuid = artsInitializeAndStartEpoch(NULL_GUID, 0);
        
        dep--;
        unsigned int numNodes = artsGetTotalNodes();
//        artsEdtCreateShad(rootTask, (artsGetCurrentNode()+1)%numNodes, 1, &dep);
        artsEdtCreateDep(rootTask, (artsGetCurrentNode()+1)%numNodes, 1, &dep, 0, false);
        
//        uint64_t args[2];
//        args[0] = dep;
//        
//        for(uint64_t i=0; i<numDummy; i++)
//        {
//            args[1] = i;
//            artsEdtCreateDep(dummytask, i%numNodes, 2, args, 0, false);
//        }
        PRINTF("Waiting on %lu\n", poolGuid);
        if(artsWaitOnHandle(poolGuid))
            PRINTF("Done waiting on %lu dep: %lu\n", poolGuid, dep);
    }
    
    if(dep+1 == numDummy)
        artsShutdown();
}

void initPerNode(unsigned int nodeId, int argc, char** argv) 
{
    numDummy = (uint64_t) atoi(argv[1]);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv) 
{
    if(!nodeId && !workerId)
    {
        PRINTF("Starting\n");
        uint64_t arg = numDummy;
        artsEdtCreateShad(rootTask, 0, 1, &arg);
    }
}

int main(int argc, char** argv) 
{
    artsRT(argc, argv);
    return 0;
}
