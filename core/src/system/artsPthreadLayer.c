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
#define _GNU_SOURCE
#include <pthread.h>
#include "arts.h"
#include "artsRuntime.h"
#include "artsConfig.h"
#include "artsRemote.h"
#include "artsGuid.h"
#include "limits.h"
#include "artsGlobals.h"
#include "artsCounter.h"
#include "artsThreads.h"
#include "artsTMT.h"
#include <unistd.h>

unsigned int artsGlobalRankId;
unsigned int artsGlobalRankCount;
unsigned int artsGlobalMasterRankId;
struct artsConfig * gConfig;
struct artsConfig * config;

pthread_t * nodeThreadList;

void * artsThreadLoop(void * data)
{
    struct threadMask * unit = (struct threadMask*) data;
    if(unit->pin)
        artsAbstractMachineModelPinThread(unit->coreInfo);
    artsRuntimePrivateInit(unit, gConfig);
    artsRuntimeLoop();
    artsRuntimePrivateCleanup();
    return NULL;
    //pthread_exit(NULL);
}


void artsThreadMainJoin()
{
    artsRuntimeLoop();
    artsRuntimePrivateCleanup();
    int i;
    for (i = 1; i < artsNodeInfo.totalThreadCount; i++)
        pthread_join(nodeThreadList[i], NULL);
    artsRuntimeGlobalCleanup();
    //artsFree(args);
    artsFree(nodeThreadList);
}

void artsThreadInit( struct artsConfig * config  )
{
    gConfig = config;
    struct threadMask * mask = getThreadMask(config);
    nodeThreadList = artsMalloc(sizeof (pthread_t) * artsNodeInfo.totalThreadCount);
    unsigned int i = 0, threadCount=artsNodeInfo.totalThreadCount;
        
    if(config->stackSize)
    {
        void * stack;
        pthread_attr_t attr;
        long pageSize = sysconf(_SC_PAGESIZE);
        size_t size = ((config->stackSize%pageSize > 0) + (config->stackSize/pageSize))*pageSize;
        for (i=1; i<threadCount; i++)
        {
            pthread_attr_init(&attr);
            pthread_attr_setstacksize(&attr,size);
            pthread_create(&nodeThreadList[i], &attr, &artsThreadLoop, &mask[i]);
        }
    }
    else
    {
        for (i=1; i<threadCount; i++)
            pthread_create(&nodeThreadList[i], NULL, &artsThreadLoop, &mask[i]);
    }
    if(mask->pin)
        artsAbstractMachineModelPinThread(mask->coreInfo);
    artsRuntimePrivateInit(&mask[0], config);
}

void artsShutdown()
{
    if(artsGlobalRankCount>1)
        artsRemoteShutdown();

    if(artsGlobalRankCount==1)
        artsRuntimeStop();
}

void artsThreadSetOsThreadCount(unsigned int threads)
{
    pthread_setconcurrency(threads);
}

void artsPthreadAffinity(unsigned int cpuCoreId) {
    cpu_set_t cpuset;
    pthread_t thread;
    thread = pthread_self();
    CPU_ZERO(&cpuset);
    CPU_SET(cpuCoreId, &cpuset);
    if(pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset))
        PRINTF("Failed to set affinity %u\n", cpuCoreId);  
}

int * artsValidPthreadAffinity(unsigned int * size)
{
    unsigned int count = 0;
    cpu_set_t cpuset;
    pthread_t thread = pthread_self();
    
    int * affin = (int*) artsMalloc(sizeof(int) * CPU_SETSIZE);
    for(int i=0; i<CPU_SETSIZE; i++) {
        CPU_ZERO(&cpuset);
        CPU_SET(i, &cpuset);
        if(pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset))
            affin[i] = -1;
        else {
            affin[i] = i;
            count++;
        }
        
    }
    *size = count;
    return affin;
}
