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

uint64_t start = 0;

void fib(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    artsGuid_t resultGuid = paramv[0];
    int num = paramv[1];
    int sum = num;
    int x = -1;
    int y = -1;
    
    if(num >= 2) 
    {
        artsTicket_t ctx = artsGetContextTicket();
        unsigned int count = 2;
        
        int * xPtr = &x;
        int * yPtr = &y;
        artsGuid_t xGuid = artsAllocateLocalBuffer((void**)&xPtr, sizeof(int), 1, NULL_GUID);
        artsGuid_t yGuid = artsAllocateLocalBuffer((void**)&yPtr, sizeof(int), 1, NULL_GUID);
        
        uint64_t args[3];
        
        args[0] = xGuid;
        args[1] = num-2;
        args[2] = ctx;
        artsEdtCreate(fib, 0, 3, args, 0);
        
        args[0] = yGuid;
        args[1] = num-1;
        artsEdtCreate(fib, 0, 3, args, 0);
        
        artsContextSwitch(2);
        sum = x + y;
    }
    
    if(resultGuid)
    {
        artsSetBuffer(resultGuid, &sum, sizeof(int));
        artsSignalContext(paramv[2]);
    }
    else
    {
        uint64_t time = artsGetTimeStamp() - start;
        PRINTF("Fib %d: %d %lu\n", num, sum, time);
        artsShutdown();
    }
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{

}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{   
    if(!nodeId && !workerId)
    {
        int num = atoi(argv[1]);
        uint64_t args[] = {NULL_GUID, num};
        start = artsGetTimeStamp();
        artsGuid_t guid = artsEdtCreate(fib, 0, 2, args, 0);
    }
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}
