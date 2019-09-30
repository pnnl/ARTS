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
#include "artsGpuStream.h"
#include "artsGpuRuntime.h"

#define SOMEARGS 10

artsGuid_t localDbCreate(void **addr, uint64_t size, artsType_t mode, artsGuid_t guid)
{
    unsigned int dbSize = size + sizeof(struct artsDb);
//    void * ptr = artsMalloc(dbSize);
    void * ptr = artsCudaMallocHost(dbSize);
    if(ptr)
    {
        struct artsHeader *header = (struct artsHeader*)ptr;
        header->type = mode;
        header->size = dbSize;
        struct artsDb * dbRes = (struct artsDb *)header;
        dbRes->guid = guid;
        dbRes->dbList = NULL;
        *addr = (void*)((struct artsDb *) ptr + 1);
    }
    return guid;
}

__global__ void kernel(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    uint64_t * ptr = (uint64_t *)depv[threadIdx.x].ptr;
    *ptr = paramv[threadIdx.x];
}

int main(void)
{
    // dim3 grid(1);
    // dim3 block(SOMEARGS);
    
    // uint64_t paramv[SOMEARGS];
    // artsEdtDep_t depv[SOMEARGS];
    // artsGpu_t * artsGpu;
    
    // PRINTF("INIT STREAM\n");
    // artsInitGpus(1, 1, 1);
    
    // for(unsigned int i=0; i<SOMEARGS; i++)
    // {
    //     paramv[i] = i;
    //     depv[i].guid = localDbCreate(&depv[i].ptr, sizeof(artsGuid_t), ARTS_DB_READ, 999);
    //     depv[i].mode = ARTS_DB_READ;
    // }
    
    // PRINTF("LAUNCHING 1 %u\n", SOMEARGS);
    // artsScheduleToGpuInternal(kernel, SOMEARGS, paramv, SOMEARGS, depv, grid, block, NULL, artsGpu);
    
    // PRINTF("WAITING\n");
    // artsGpuSynchronize(artsGpu);
    
    // for(unsigned int i=0; i<SOMEARGS; i++)
    // {
    //     artsGuid_t * ptr = (artsGuid_t *) depv[i].ptr;
    //     PRINTF("RES: %lu\n", *ptr);
    // }
    
    // PRINTF("DESTROYING\n");
    // artsCleanupGpus();
    return 0;
}

