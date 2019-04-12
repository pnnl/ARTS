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
#include "artsAbstractMachineModel.h"
#include "artsRuntime.h"
#include "artsGlobals.h"
#include "artsThreads.h"

unsigned int numNumaDomains = 1;

enum abstractGroupId
{
    abstractWorker=0,
    abstractInbound,
    abstractOutbound,
    abstractMax
};

void setThreadMask( struct threadMask *threadMask, struct unitMask * unitMask, struct unitThread * unitThread )
{
    threadMask->clusterId = unitMask->clusterId;
    threadMask->coreId = unitMask->coreId;
    threadMask->unitId = unitMask->unitId;
    threadMask->on = unitMask->on;
    threadMask->coreInfo = unitMask->coreInfo;

    threadMask->id = unitThread->id;
    threadMask->groupId = unitThread->groupId;
    threadMask->groupPos = unitThread->groupPos;
    threadMask->worker = unitThread->worker;
    threadMask->networkSend = unitThread->networkSend;
    threadMask->networkReceive = unitThread->networkReceive;
    threadMask->pin = unitThread->pin;
}

void addAThread( struct unitMask * mask, bool workOn, bool networkOutOn, bool networkInOn, unsigned int groupId, unsigned int groupPos, bool pin)
{
    struct unitThread * next;
    mask->threads++;
    if(mask->listHead == NULL)
    {
        mask->listTail = mask->listHead = artsMalloc(sizeof(struct unitThread));
        next = mask->listHead;
    }
    else
    {
        next = mask->listTail;
        next->next = artsMalloc(sizeof(struct unitThread));
        next = next->next;
        mask->listTail = next;
    }

    next->worker = workOn;
    next->networkSend = networkOutOn;
    next->networkReceive = networkInOn;
    next->groupId = groupId;
    next->groupPos = groupPos;
    next->pin = pin;
    next->next = NULL;
    next->id = mask->coreId;
}

#ifdef HWLOC

#include <hwloc.h>

struct artsCoreInfo
{
    hwloc_cpuset_t cpuset;
    hwloc_topology_t topology;
};

hwloc_topology_t getTopology()
{
    hwloc_topology_t topology;
    hwloc_topology_init(&topology);
#ifndef HWLOC_V2
    hwloc_topology_set_flags(topology, HWLOC_TOPOLOGY_FLAG_IO_BRIDGES);
#endif
    hwloc_topology_load(topology);
    return topology;
}

unsigned int getNumberOfType(hwloc_topology_t topology, hwloc_obj_t obj, hwloc_obj_type_t type)
{
    unsigned int count = 0;
    if(obj->type == type)
        count = 1;
    else
    {
        unsigned int i;
        for (i=0; i<obj->arity; i++)
            count+=getNumberOfType(topology, obj->children[i], type);
    }
    return count;
}

void artsAbstractMachineModelPinThread(struct artsCoreInfo * coreInfo )
{
//    PRINTF("BINDING\n");
    hwloc_set_cpubind(coreInfo->topology, coreInfo->cpuset, HWLOC_CPUBIND_THREAD);
}

void initClusterUnits(hwloc_topology_t topology, hwloc_obj_t obj, hwloc_obj_t cluster, unsigned int * unitIndex, struct unitMask * units)
{
    if(obj->type == HWLOC_OBJ_PU)
    {
        if(obj->parent->type == HWLOC_OBJ_CORE)
        {
            units[*unitIndex].coreId = obj->parent->os_index;
//            PRINTF("A CORE\n");
        }
        else
        {
//            PRINTF("NOT A CORE...\n");
        }
        units[*unitIndex].clusterId = cluster->os_index;
        units[*unitIndex].unitId = obj->os_index;
        units[*unitIndex].on = 0;

//        PRINTF("Cluster: %u Unit: %u\n", cluster->os_index, obj->os_index);
        
        units[*unitIndex].listHead = NULL;
        units[*unitIndex].threads = 0;
        units[*unitIndex].coreInfo = artsMalloc(sizeof(struct artsCoreInfo));
        units[*unitIndex].coreInfo->topology = topology;
        units[*unitIndex].coreInfo->cpuset = hwloc_bitmap_dup(obj->cpuset);
        *unitIndex=(*unitIndex)+1;
    }
    else
    {
//        PRINTF("ARITY: %u\n", obj->arity);
        int i;
        for (i=0; i<obj->arity; i++)
            initClusterUnits(topology, obj->children[i], cluster, unitIndex, units);
    }
}

struct nodeMask * initTopology()
{
    hwloc_topology_t topology = getTopology();

    struct nodeMask * node = (struct nodeMask*) artsMalloc(sizeof(struct nodeMask));
    numNumaDomains = node->numClusters = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_NODE);
    node->cluster = (struct clusterMask*) artsMalloc(sizeof(struct clusterMask)*node->numClusters);
    unsigned int clusterIndex = 0;
    unsigned int coreIndex = 0;
    hwloc_obj_t cluster = NULL;
    hwloc_obj_t core = NULL;
    for(clusterIndex = 0; clusterIndex<node->numClusters; clusterIndex++)
    {
        cluster = hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_NODE, cluster);
        node->cluster[clusterIndex].numCores = getNumberOfType(topology, cluster, HWLOC_OBJ_CORE);
        node->cluster[clusterIndex].core = (struct coreMask*) artsMalloc(sizeof(struct coreMask)*node->cluster[clusterIndex].numCores);
        for(coreIndex=0; coreIndex<node->cluster[clusterIndex].numCores; coreIndex++)
        {
            core = hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_CORE, core);
            node->cluster[clusterIndex].core[coreIndex].numUnits = getNumberOfType(topology, core, HWLOC_OBJ_PU);
            node->cluster[clusterIndex].core[coreIndex].unit = (struct unitMask*) artsMalloc(sizeof(struct unitMask)*node->cluster[clusterIndex].core[coreIndex].numUnits);
            unsigned int unitIndex = 0;
            initClusterUnits(topology, core, cluster, &unitIndex, node->cluster[clusterIndex].core[coreIndex].unit);
        }
    }
    //hwloc_topology_destroy(topology);
    return node;
}

void defaultPolicy(unsigned int numberOfWorkers, unsigned int numberOfSenders, unsigned int numberOfReceivers, struct nodeMask * node, struct artsConfig * config)
{
    unsigned int numClusters = node->numClusters;
    unsigned int numCores = node->cluster[0].numCores;
    unsigned int numUnits = node->cluster[0].core[0].numUnits;
//    PRINTF("%d %d %d\n", numClusters, numCores, numUnits);
    unsigned int coresPerCluster = numCores*numUnits;
    unsigned int coreCount = numClusters*numCores*numUnits;
    unsigned int i=0, j=0, k=0, totalThreads=0;
    unsigned int stride = config->pinStride;
    unsigned int strideLoop =0;
    unsigned int offset = 0;
    unsigned int networkThreads = (artsGlobalRankCount>1)*(numberOfReceivers+numberOfSenders);
    numberOfSenders = (artsGlobalRankCount>1) * numberOfSenders;
    numberOfReceivers = (artsGlobalRankCount>1) * numberOfReceivers;
    unsigned int workerThreadId = 0;
    unsigned int networkOutThreadId = 0;
    unsigned int networkInThreadId = 0;
    while(totalThreads < numberOfWorkers+networkThreads)
    {
        node->cluster[i].core[j].unit[k].on = 1;

        if( totalThreads< numberOfWorkers )
        {
            addAThread(&node->cluster[i].core[j].unit[k], 1, 0, 0, abstractWorker, workerThreadId++, config->pinThreads);
        }
        else
        {
            if(totalThreads < numberOfWorkers + numberOfSenders )
            {
                addAThread(&node->cluster[i].core[j].unit[k], 0, 1, 0, abstractOutbound, networkOutThreadId++, config->pinThreads);
            }
            else if(totalThreads < numberOfWorkers+numberOfReceivers+numberOfSenders)
            {
                    addAThread(&node->cluster[i].core[j].unit[k], 0, 0, 1, abstractInbound, networkInThreadId++, config->pinThreads);
            }
        }
        totalThreads++;
        numCores = node->cluster[i].numCores;
        numUnits = node->cluster[i].core[j].numUnits;
        j+=stride;
        if(j >= numCores)
        {
            i++;
            if(i<numClusters)
            {
                while(node->cluster[i].numCores == 0)
                {
                    i++;
                    if(i == numClusters)
                        break;
                }
            }
            if(i == numClusters)
            {
                i=0;

                if(stride > 1)
                {
                    offset++;
                    strideLoop++;
                    if(strideLoop == stride)
                    {
                        offset = 0;
                        k++;
                        strideLoop=0;
                    }
                }
                else
                k++;
                if(k == numUnits)
                {
                    k=0;
                }
            }
            j=offset;
        }
    }
}

unsigned int flattenMask(struct artsConfig * config, struct nodeMask * node, struct threadMask ** flat)
{
    unsigned int i, j, k, total, count=0;
    for(i=0; i<node->numClusters; i++)
    {
        for(j=0; j<node->cluster[i].numCores; j++)
        {
            for(k = 0; k<node->cluster[i].core[j].numUnits; k++)
            {
                if(node->cluster[i].core[j].unit[k].on)
                    count+=node->cluster[i].core[j].unit[k].threads;
            }
        }
    }
    total = count;
    *flat = (struct threadMask*) artsMalloc(sizeof(struct threadMask)*total);
    unsigned int * groupCount = artsCalloc(sizeof(unsigned int)*abstractMax);
    count = 0;
    struct unitThread * next;
    for(i=0; i<node->numClusters; i++)
    {
        for(j=0; j<node->cluster[i].numCores; j++)
        {
            for(k = 0; k<node->cluster[i].core[j].numUnits; k++)
            {
                if(node->cluster[i].core[j].unit[k].on)
                {
                    next = node->cluster[i].core[j].unit[k].listHead;

                    while(next != NULL)
                    {
                        setThreadMask(&(*flat)[count], &node->cluster[i].core[j].unit[k], next);
                        (*flat)[count].groupPos = groupCount[next->groupId]++;
                        (*flat)[count].id = count;
                        ++count;
                        next = next->next;
                    }
                }
            }
        }
    }
    for(i=0; i<node->numClusters; i++)
    {
        for(j=0; j<node->cluster[i].numCores; j++)
        {
            artsFree(node->cluster[i].core[j].unit);
        }
        artsFree(node->cluster[i].core);
    }
    artsFree(node->cluster);
    artsFree(node);
    return total;
}

struct threadMask * getThreadMask(struct artsConfig * config)
{
    if(config->senderCount > (artsGlobalRankCount-1)*config->ports)
        config->senderCount = (artsGlobalRankCount-1)*config->ports;
    if(config->recieverCount > (artsGlobalRankCount-1)*config->ports)
        config->recieverCount = (artsGlobalRankCount-1)*config->ports;

    unsigned int workerThreads = config->threadCount;
    unsigned int totalThreads = config->threadCount;

    bool networkOn = (artsGlobalRankCount>1);
    struct threadMask * flat;
    struct nodeMask * topology = initTopology();

    defaultPolicy(workerThreads, config->senderCount, config->recieverCount, topology, config);
    totalThreads = flattenMask(config, topology, &flat);

    artsRuntimeNodeInit(workerThreads, 1, config->senderCount, config->recieverCount, totalThreads, 0, config);
    if(config->printTopology)
        printMask(flat,totalThreads);
    return flat;
}

void printTopology(struct nodeMask * node)
{
    PRINTF("Node %u\n", node->numClusters);
    unsigned int i, j, k;
    for(i=0; i<node->numClusters; i++)
    {
        PRINTF(" Cluster %u\n", node->cluster[i].numCores);
        for(j=0; j<node->cluster[i].numCores; j++)
        {
            PRINTF("  Core %u\n", node->cluster[i].core[j].numUnits);
            for(k=0; k<node->cluster[i].core[j].numUnits; k++)
            {
                struct unitMask * unit = &node->cluster[i].core[j].unit[k];
                struct unitThread * temp = unit->listHead;
                while(temp != NULL)
                {
                    PRINTF("   Unit %u %u %u %u %u %u %u\n",
                        temp->id, unit->unitId, unit->on, temp->worker,
                        temp->networkSend, temp->networkReceive, unit->coreId);
                    temp = temp->next;
                }
            }
        }
    }
}
#else

#include <unistd.h>

struct artsCoreInfo
{
    unsigned int cpuId;
};

void artsAbstractMachineModelPinThread(struct artsCoreInfo * coreInfo )
{
    artsPthreadAffinity(coreInfo->cpuId);
}

int artsAffinityFromPthreadValid(unsigned int i, int * validCpus, unsigned int validCpuCount, unsigned int numCores, unsigned int stride)
{
    static unsigned int index = 0;
    static unsigned int offset = 0;
    static unsigned int strideLoop = 0;
    static unsigned int count = 0;
    int res = -1;
    
    if(validCpuCount) {
        if(count == validCpuCount) {
            index = 0;
            offset = 0;
            strideLoop = 0;
            count = 0;
        }
        
        do {
            res = validCpus[index%numCores];
            index+=stride;
            if(index >= numCores && stride > 1)
            {
                strideLoop++;
                offset++;
                if(strideLoop == stride)
                {
                    offset=strideLoop=0;
                }
                index=offset;
            }
        } while(res == -1);
        count++;
    }
    else
        PRINTF("Valid set of processors is empty\n");
    return res;
}

void defaultPolicy(unsigned int numberOfWorkers, unsigned int numberOfSenders, unsigned int numberOfReceivers, struct unitMask * flat, unsigned int numCores,  struct artsConfig * config)
{
    unsigned int validCpuCount = 0;
    int * validCpus = artsValidPthreadAffinity(&validCpuCount);
    
    unsigned int totalThreads=0;
    unsigned int stride = config->pinStride;
    unsigned int strideLoop =0;
    unsigned int i=0, offset = 0;
    unsigned int networkThreads = (artsGlobalRankCount>1)*(numberOfReceivers+numberOfSenders);
    unsigned int networkCores = networkThreads * config->coresPerNetworkThread;
    unsigned int workerThreadId = 0;
    unsigned int networkOutThreadId = 0;
    unsigned int networkInThreadId = 0;
    
    if(numCores <= networkCores || validCpuCount <= networkCores)
        PRINTF("Not enough cores. Required cores: %u Total cores: %u validCpuCount: %u\n", networkCores, numCores, validCpuCount);
    unsigned int workerCores = numCores - networkCores;
    int max = -1;
    while(totalThreads < numberOfWorkers)
    {
        flat[i%workerCores].on = 1;
        flat[i%workerCores].coreInfo = artsMalloc(sizeof(struct artsCoreInfo));
        int tempAffin = artsAffinityFromPthreadValid(i, validCpus, validCpuCount-networkCores, workerCores, stride); //i % numCores;
        flat[i%workerCores].coreId = flat[i%workerCores].coreInfo->cpuId = tempAffin;
        addAThread(&flat[i%workerCores], 1, 0, 0, abstractWorker, workerThreadId++, config->pinThreads);
        max = (tempAffin > max) ? tempAffin : max;
        totalThreads++;
//        PRINTF("i: %u -> %u -> %u\n", i, i%workerCores, flat[i%workerCores].coreId);
        i+=stride;
        if(i >= workerCores && stride > 1)
        {
            strideLoop++;
            offset++;
            if(strideLoop == stride)
            {
                offset=strideLoop=0;
            }
            i=offset;
        }
        
    }
    
    unsigned int next = (max - max%stride) + stride;
    for(unsigned int i=0; i<numberOfSenders; i++)
    {
        for(; next<numCores; next+=config->coresPerNetworkThread)
        {
            if(validCpus[next] > -1)
            {
                flat[i+workerCores].on = 1;
                flat[i+workerCores].coreInfo = artsMalloc(sizeof(struct artsCoreInfo));
                flat[i+workerCores].coreId = flat[i+workerCores].coreInfo->cpuId =  validCpus[next];
                addAThread(&flat[i+workerCores], 0, 1, 0, abstractOutbound, networkOutThreadId++, config->pinThreads);
                next+=config->coresPerNetworkThread;
                break;
            }
                
        }
        
    }
    for(unsigned int i=0; i<numberOfReceivers; i++)
    {
        for(; next<numCores; next+=config->coresPerNetworkThread)
        {
            if(validCpus[next] > -1)
            {
                flat[i+workerCores+numberOfSenders].on = 1;
                flat[i+workerCores+numberOfSenders].coreInfo = artsMalloc(sizeof(struct artsCoreInfo));
                flat[i+workerCores+numberOfSenders].coreId = flat[i+workerCores+numberOfSenders].coreInfo->cpuId =  validCpus[next];
                addAThread(&flat[i+workerCores+numberOfSenders], 0, 0, 1, abstractInbound, networkInThreadId++, config->pinThreads);
                next+=config->coresPerNetworkThread;
                break;
            }
        }
        
    }
    if(validCpus)
        artsFree(validCpus);
}

unsigned int flattenMask(struct artsConfig * config, unsigned int numCores, struct unitMask * unit, struct threadMask ** flat)
{
    unsigned int maskSize=0;
    unsigned int threadId=0;

    for(int i =0; i< numCores; i++)
    {
        if(unit[i].on)
        {
            maskSize+=unit[i].threads;

        }
    }
    *flat = (struct threadMask*) artsCalloc(sizeof(struct threadMask)*maskSize);
    unsigned int * groupCount = artsCalloc(sizeof(unsigned int)*abstractMax);
    struct unitThread * next;
    unsigned int count = 0;
    for(int i =0; i< numCores; i++)
    {
        if(unit[i].on)
        {
            next = unit[i].listHead;

            while(next != NULL)
            {
                assert(count < maskSize);
                setThreadMask(&(*flat)[count], &unit[i], next);
                (*flat)[count].groupPos = groupCount[next->groupId]++;
                (*flat)[count].id = count;
                ++count;
                next = next->next;
            }
        }
    }
    artsFree(groupCount);

    return count;
}

struct threadMask * getThreadMask(struct artsConfig * config)
{
    if(config->senderCount > (artsGlobalRankCount-1)*config->ports)
        config->senderCount = (artsGlobalRankCount-1)*config->ports;
    if(config->recieverCount > (artsGlobalRankCount-1)*config->ports)
        config->recieverCount = (artsGlobalRankCount-1)*config->ports;

    unsigned int workerThreads = config->threadCount;
    unsigned int totalThreads = config->threadCount;

    bool networkOn = (artsGlobalRankCount>1);
    struct unitMask * unit;
    struct threadMask * flat;

    unsigned int coreCount = (config->coreCount) ? config->coreCount : sysconf(_SC_NPROCESSORS_ONLN);

    unit = artsCalloc(sizeof(struct unitMask) * coreCount);
    defaultPolicy(workerThreads, config->senderCount, config->recieverCount, unit, coreCount, config);

    totalThreads = flattenMask(config, coreCount, unit, &flat);

    if(config->printTopology)
        printMask(flat,totalThreads);
    artsRuntimeNodeInit(workerThreads, 1, config->senderCount, config->recieverCount, totalThreads, 0, config);
    return flat;
}

#endif

void printMask(struct threadMask * units, unsigned int numberOfUnits)
{
    unsigned int i;
    MASTER_PRINTF(" Id   GroupId  GroupPos  Cluster  Core  Unit    On  Worker  Send  Recv   Pin Status\n");
    for(i=0; i<numberOfUnits; i++)
    {
        MASTER_PRINTF("%3u    %3u     %3u       %3u     %3u    %3u     %1u     %1u     %1u     %1u      %1u    %1u\n",
            units[i].id, units[i].groupId, units[i].groupPos, units[i].clusterId, units[i].coreId, units[i].unitId,
            units[i].on, units[i].worker, units[i].networkSend, units[i].networkReceive, units[i].pin, units[i].statusSend);
    }
}
