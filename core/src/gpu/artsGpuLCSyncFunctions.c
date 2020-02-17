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
#include <cuda_runtime.h>
#include "artsGpuLCSyncFunctions.h"
#include "artsDbFunctions.h"
#include "artsGlobals.h"
#include "artsGpuRouteTable.h"
#include "artsDebug.h"
#include "artsAtomics.h"
#include "artsGpuStreamBuffer.h"

#define DPRINTF(...)
// #define DPRINTF(...) PRINTF(__VA_ARGS__)

//To use this lock the unlock must be an even number
unsigned int versionLock(artsLCMeta_t * meta)
{
    artsWriterLock(meta->readLock, meta->writeLock);
    return *meta->hostVersion;
}

bool tryVersionLock(artsLCMeta_t * meta)
{
    return artsWriterTryLock(meta->readLock, meta->writeLock);
}

void versionUnlock(artsLCMeta_t * meta)
{
    artsAtomicAdd(meta->hostVersion, 2U);
    artsWriterUnlock(meta->writeLock);
}

void * makeLCShadowCopy(struct artsDb * db)
{
    unsigned int size = db->header.size;
    void * dest = (void*)(((char*)db) + size);
    struct artsDb * shadowCopy = (struct artsDb*) dest;
    
    artsWriterLock(&db->reader, &db->writer);
    unsigned int hostVersion = db->version;
    if(!shadowCopy->version || hostVersion != shadowCopy->version)
    {
        memcpy(dest, (void*)db, size);
    }
    artsWriterUnlock(&db->writer);
    return dest;
}

inline void artsPrintDbMetaData(artsLCMeta_t * db) 
{ 
    DPRINTF("guid: %lu ptr: %p dataSize: %lu hostVersion: %u gpuVersion: %u gpuTimeStamp: %u gpu: %d\n",  
        db->guid,
        db->data, 
        db->dataSize,
        *db->hostVersion,
        *db->hostTimeStamp,
        db->gpuVersion, 
        db->gpuTimeStamp, 
        db->gpu);
}

void artsMemcpyGpuDb(artsLCMeta_t * host, artsLCMeta_t * dev)
{
    unsigned int hostVersion = versionLock(host);
    memcpy(host->data, dev->data, host->dataSize);
    *host->hostTimeStamp = dev->gpuTimeStamp;
    versionUnlock(host);
}

void artsGetLatestGpuDb(artsLCMeta_t * host, artsLCMeta_t * dev)
{
    unsigned int hostVersion = versionLock(host);
    if(*host->hostTimeStamp < dev->gpuTimeStamp)
    {
            memcpy(host->data, dev->data, host->dataSize);
            host->gpuVersion = dev->gpuVersion;
            host->gpuTimeStamp = dev->gpuTimeStamp;
            *host->hostTimeStamp = dev->gpuTimeStamp;
            host->gpu = dev->gpu;
    }
    versionUnlock(host);
}

void artsGetRandomGpuDb(artsLCMeta_t * host, artsLCMeta_t * dev)
{
    bool firstFlag = (host->gpu == -1);
    bool randomFlag = ((artsThreadSafeRandom() & 1) == 0);
    if(firstFlag || randomFlag)
    {
        if(tryVersionLock(host))
        {
            memcpy(host->data, dev->data, host->dataSize);
            host->gpuVersion = dev->gpuVersion;
            host->gpuTimeStamp = dev->gpuTimeStamp;
            *host->hostTimeStamp = dev->gpuTimeStamp;
            host->gpu = dev->gpu;
            // if(!firstFlag && randomFlag)
                // artsGpuInvalidateRouteTables(host->guid, (unsigned int) -1);
            versionUnlock(host);
        }
        
    }
}

void artsGetNonZerosUnsignedInt(artsLCMeta_t * host, artsLCMeta_t * dev)
{
    unsigned int numElem = host->dataSize/sizeof(unsigned int);
    unsigned int * dst = (unsigned int*) host->data;
    unsigned int * src = (unsigned int*) dev->data;
    unsigned int hostVersion = versionLock(host);
    for(unsigned int i=0; i<numElem; i++)
    {
        DPRINTF("src: %u dest: %u\n", src[i], dst[i]);
        if(src[i])
            dst[i] = src[i];
    }
    versionUnlock(host);
}

void artsGetMinDbUnsignedInt(artsLCMeta_t * host, artsLCMeta_t * dev)
{
    unsigned int count = 0;
    unsigned int count2 = 0;
    unsigned int numElem = host->dataSize / sizeof(unsigned int);
    unsigned int * dst = (unsigned int*) host->data;
    unsigned int * src = (unsigned int*) dev->data;
    unsigned int hostVersion = versionLock(host);
    for(unsigned int i=0; i<numElem; i++)
    {
        if(src[i]<dst[i])
        {
            DPRINTF("src: %u dst: %u\n", src[i], dst[i]);
            dst[i] = src[i];
            count++;
        }
        if(src[i] != (unsigned int)-1)
            count2++;
    }
    DPRINTF("%lu %u %u\n", host->guid, count, count2);
    versionUnlock(host);
}

void artsAddDbUnsignedInt(artsLCMeta_t * host, artsLCMeta_t * dev)
{
    unsigned int numElem = host->dataSize / sizeof(unsigned int);
    unsigned int * dst = (unsigned int*) host->data;
    unsigned int * src = (unsigned int*) dev->data;
    unsigned int hostVersion = versionLock(host);
    for(unsigned int i=0; i<numElem; i++)
    {
        dst[i] += src[i];
    }
    DPRINTF("%lu %u %u\n", host->guid, count, count2);
    versionUnlock(host);
}

void artsXorDbUint64(artsLCMeta_t * host, artsLCMeta_t * dev)
{
    unsigned int numElem = host->dataSize / sizeof(uint64_t);
    uint64_t * dst = (uint64_t*) host->data;
    uint64_t * src = (uint64_t*) dev->data;
    uint64_t hostVersion = versionLock(host);
    for(unsigned int i=0; i<numElem; i++)
    {
        DPRINTF("xor[%u]: %lu -- %lu = %lu\n", i, dst[i], src[i], dst[i] ^ src[i]);
        dst[i] ^= src[i];

    }
    DPRINTF("%lu %u %u\n", host->guid, count, count2);
    versionUnlock(host);
}

/***********************************************************************/

__global__ void artsCopyGpuDb(struct artsDb * sink, struct artsDb * src)
{
    unsigned int * srcData = (unsigned int*) (src+1);
    unsigned int * sinkData = (unsigned int*) (sink+1);

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    sinkData[index] = srcData[index];
}

__global__ void artsMinGpuDbUnsignedInt(struct artsDb * sink, struct artsDb * src)
{
    unsigned int * srcData = (unsigned int*) (src+1);
    unsigned int * sinkData = (unsigned int*) (sink+1);

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(srcData[index] < sinkData[index])
        sinkData[index] = srcData[index];
}

__global__ void artsNonZeroGpuDbUnsignedInt(struct artsDb * sink, struct artsDb * src)
{
    unsigned int * srcData = (unsigned int*) (src+1);
    unsigned int * sinkData = (unsigned int*) (sink+1);

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(sinkData > 0)
        sinkData[index] = srcData[index];
}

__global__ void artsAddGpuDbUnsignedInt(struct artsDb * sink, struct artsDb * src)
{
    unsigned int * srcData = (unsigned int*) (src+1);
    unsigned int * sinkData = (unsigned int*) (sink+1);

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    sinkData[index] += srcData[index];
}

__global__ void artsXorGpuDbUint64(struct artsDb * sink, struct artsDb * src)
{
    unsigned long long * srcData = (unsigned long long*) (src+1);
    unsigned long long * sinkData = (unsigned long long*) (sink+1);

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    sinkData[index] ^= srcData[index];
}

/***********************************************************************/

#define GPUGROUPSIZE 4
#define GPUNUMGROUP 2

void gpuReductionLaunch(int root, int a, int b, unsigned int * remMask, artsGuid_t guid, unsigned int size, artsLCSyncFunctionGpu_t fnPtr)
{
    if(a < 0 || b < 0 )
        return;

    if(root != a && root != b)
    {
        PRINTF("LC Reduction tree invalid root! %d %d %d\n", root, a, b);
        artsDebugGenerateSegFault(); 
    }

    DPRINTF("A: %d B: %d -> Root: %d guid: %lu\n", a, b, root, guid);
    unsigned int toRemove = (root == a) ? b : a;
    *remMask &= ~(1 << toRemove);

    void * dbData = artsGpuRouteTableLookupDbRes(guid, root, NULL, NULL, false);
    void * dst = (void*)(((char*)dbData) + size);
    DPRINTF("%d %p %p\n", root, dbData, dst);

    void * src = artsGpuRouteTableLookupDbRes(guid, toRemove, NULL, NULL, false);
    DPRINTF("%d %p\n", toRemove, src);

    DPRINTF("src: %p dst: %p size: %u\n", src, dst, size);
    reduceDatafromGpus(dst, root, src, toRemove, size, fnPtr, lcSyncElementSize[artsNodeInfo.gpuLCSync], dbData);
}

void gpuShadowReductionLaunch(int root, artsGuid_t guid, unsigned int size, artsLCSyncFunctionGpu_t fnPtr)
{
    void * sink = artsGpuRouteTableLookupDbRes(guid, root, NULL, NULL, false);
    void * src = (void*)(((char*)sink) + size);

    doReductionNow(root, sink, src, fnPtr, sizeof(unsigned int), size);
}

void gpuCopyLaunch(int root, int a, int b, bool srcShadow, bool dstShadow, artsGuid_t guid, unsigned int size)
{
    
    if(a < 0 || b < 0 )
        return;

    if(root != a && root != b)
    {
        PRINTF("LC Reduction tree invalid root! %d %d %d\n", root, a, b);
        artsDebugGenerateSegFault(); 
    }

    DPRINTF("A: %d B: %d -> Root: %d\n", a, b, root);
    unsigned int toRemove = (root == a) ? b : a;

    void * dst = artsGpuRouteTableLookupDbRes(guid, root, NULL, NULL, false);
    if(dstShadow)
        dst = (void*)(((char*)dst) + size);
    DPRINTF("%d %p %p\n", root, dbData, dst);

    void * src = artsGpuRouteTableLookupDbRes(guid, toRemove, NULL, NULL, false);
    if(srcShadow)
        src = (void*)(((char*)src) + size);
    DPRINTF("%d %p\n", toRemove, src);

    DPRINTF("src: %p dst: %p size: %u\n", src, dst, size);
    copyGputoGpu(dst, root, src, toRemove, size);
}

void findRoots(unsigned int local, int * roots)
{
    for(unsigned int i=0; i<GPUNUMGROUP; i++)
        roots[i] = -1;

    //Make a mask of 4 bits (GPUGROUPSIZE)
    unsigned int mask = 0;
    for(unsigned int j=0; j<GPUGROUPSIZE; j++)
    {
        unsigned int bit = 1 << j;
        mask |= bit;
    }

    //Assumes grid... Add shifted local mask with mask and or results
    unsigned int localRoots = (unsigned int) -1;
    for(unsigned int i=0; i<GPUNUMGROUP; i++)
    {
        unsigned int tempLocal = local >> (i*GPUGROUPSIZE);
        unsigned int temp = mask & tempLocal;
        localRoots &= temp;
    }
    
    //Recover the roots
    for(int i=0; i<GPUGROUPSIZE; i++)
    {
        if(localRoots & (1 << i))
        {   
            DPRINTF("FOUND MATCHING ROOTS\n");
            for(unsigned int j=0; j<GPUNUMGROUP; j++)
                roots[j] = i + j*GPUGROUPSIZE;
            return;
        }
    }

    for(unsigned int i=0; i<GPUNUMGROUP; i++)
    {
        // PRINTF("i: %u\n", i);
        for(unsigned int j=0; j<GPUGROUPSIZE; j++)
        {
            unsigned int bit = i*GPUGROUPSIZE + j;
            // PRINTF("bit: %u\n", bit);
            if(local & (1 << bit))
            {
                roots[i] = bit;
                break;
            }
        }
    }
}

typedef struct
{
    int a;
    int b;
    int root;
    int level;
} trav;

void addToTrav(int root, int a, int b, unsigned int level, unsigned int * size, trav * ds, unsigned int * maxLevel)
{
    if(a < 0 || b < 0 )
        return;

    unsigned int index = (*size);
    *size = *size + 1;
    ds[index].a = a;
    ds[index].b = b;
    ds[index].root = root;
    ds[index].level = level;

    *maxLevel = (*maxLevel < level) ? level : *maxLevel;
}

int gpuTreeReductionRec(int root, unsigned int start, unsigned int stop, unsigned int mask, unsigned int level, unsigned int * listSize, trav * list, unsigned int * maxLevel)
{
    int localRoot = -1;
    // PRINTF("root: %u start: %u stop: %u\n", root, start, stop);
    int gpuId[2] = {start, stop};

    if(stop - start > 1) //Recursive call
    {
        unsigned int middle = (1 + stop - start) / 2;
        gpuId[0] = gpuTreeReductionRec(root, start, start + middle - 1, mask, level+1, listSize, list, maxLevel);
        gpuId[1] = gpuTreeReductionRec(root, start + middle, stop, mask, level+1, listSize, list, maxLevel);
    }

    bool startFound = (gpuId[0] < 0) ? false : ((mask & (1 << gpuId[0])) != 0);
    bool stopFound =  (gpuId[1] < 0) ? false : ((mask & (1 << gpuId[1])) != 0);

    if(startFound && stopFound) //Both are in the mask
    {
        if(root == gpuId[0] || root == gpuId[1])
            localRoot = root;
        else
            localRoot = gpuId[0]; //This is the min
    }
    else if(startFound && !stopFound) //Only start is in the mask
    {
        gpuId[1] = -1;
        localRoot = gpuId[0];
    }
    else if(!startFound && stopFound) //Only stop is in the mask
    {
        gpuId[0] = -1;
        localRoot = gpuId[1];
    }
    else //Neither start or stop is in the mask
    {
        gpuId[1] = -1;
        gpuId[0] = -1;
        // localRoot = -1;
    }

    addToTrav(localRoot, gpuId[0], gpuId[1], level, listSize, list, maxLevel);
    return localRoot;
}

void gpuTreeReductionStart(unsigned int mask, unsigned int * listSize, trav * list, unsigned int * maxLevel)
{
    int root[GPUNUMGROUP];
    findRoots(mask, root);
    for(unsigned int i=0; i<GPUNUMGROUP; i++)
    {
        DPRINTF("Root[%d]: %d\n", i, root[i]);
        gpuTreeReductionRec(root[i], i * GPUGROUPSIZE, ((i+1) * GPUGROUPSIZE) - 1, mask, 2, listSize, list, maxLevel);
    }
    addToTrav(root[0], root[0], root[1], 1, listSize, list, maxLevel);
}

unsigned int gpuTreeReduction(unsigned int mask, artsGuid_t guid, unsigned int dbSize, artsLCSyncFunctionGpu_t dbFn)
{
    DPRINTF("mask: %u\n", mask);
    unsigned int maxLevel = 0;
    unsigned int listSize = 0;
    trav list[GPUNUMGROUP * GPUGROUPSIZE];

    gpuTreeReductionStart(mask, &listSize, list, &maxLevel);

    unsigned int remMask = mask;
    
    for(unsigned int i=maxLevel; i>0; i--)
    {
        for(unsigned int j=0; j<listSize; j++)
        {
            if(list[j].level == i)
                gpuReductionLaunch(list[j].root, list[j].a, list[j].b, &remMask, guid, dbSize, dbFn);
        }
    }
    DPRINTF("remMask: %u\n", remMask);
    return remMask;
}

/***********************************************************/

bool checkMax(unsigned int currentSize, unsigned int *visited, unsigned int * maxSize, unsigned int * maxVisited, unsigned int cycleSize)
{
    if(*maxSize < currentSize)
    {
        *maxSize = currentSize;
        memcpy(maxVisited, visited, sizeof(unsigned int)*currentSize);
        return (currentSize == cycleSize) && (maxVisited[0] == maxVisited[cycleSize - 1]);
    }
    return false;
}

extern bool ** gpuAdjList;
unsigned int gpuDepthFirstRec( unsigned int vertex,
                            unsigned int cycleSize, unsigned int mask, 
                            unsigned int current, unsigned int * visited,
                            unsigned int * maxSize, unsigned int * maxVisited)
{
    unsigned int order = artsGetTotalGpus();
    visited[current++] = vertex; //Record order visited

    bool ret = checkMax(current, visited, maxSize, maxVisited, cycleSize);

    unsigned int temp = ~(1 << vertex); //Mark off list
    mask &= temp;

    if(current + 1 == cycleSize) //This means the next iteration is the final one... Lets look for a cycle to make a ring
        mask |= 1 << visited[0];

    if(current < cycleSize)
    {
        for(unsigned int i=0; i<order; i++)
        {
            if((mask & (1 << i)) && gpuAdjList[vertex][i])
            {
                PRINTF("%u -> %u\n", vertex, i);
                if(gpuDepthFirstRec(i, cycleSize, mask, current, visited, maxSize, maxVisited))
                    return true;
            }
        }
    }
    return ret;
}

unsigned int * gpuDepthFirst(unsigned int mask, unsigned int * maxSize)
{
    unsigned int * ret = NULL;
    unsigned int cycleSize = 1; //Add one for the backedge
    for(unsigned int i=0; i<sizeof(mask)*8; i++)
    {
        if(mask & (1 << i))
            cycleSize++;
    }
    
    unsigned int * visited = (unsigned int*) artsCalloc(sizeof(unsigned int)*cycleSize);
    unsigned int * maxVisited = (unsigned int*) artsCalloc(sizeof(unsigned int)*cycleSize);
    for(unsigned int i=0; i<artsGetTotalGpus(); i++)
    {
        if(mask & (1 << i))
        {
            PRINTF("i: %u\n", i);
            if(gpuDepthFirstRec(i, cycleSize, mask, 0, visited, maxSize, maxVisited))
            {
                ret = maxVisited;
                break;
            }
        }
    }
    artsFree(visited);
    if(!ret)
        artsFree(maxVisited);
    return ret;
}

bool gpuRingReduction(unsigned int mask, unsigned int guid, unsigned int dbSize, artsLCSyncFunctionGpu_t fnPtr)
{
    unsigned int remMask = mask;
    unsigned int cycleSize = 0;
    unsigned int * cycle = gpuDepthFirst(mask, &cycleSize);
    if(cycle && cycleSize > 1)
    {
        unsigned int numGpus = cycleSize - 1;
        PRINTF("Cycle Size: %u\n", cycleSize);
        for(unsigned int i=0; i<1; i++)
        {
            for(unsigned int j=1; j<cycleSize; j++)
                gpuCopyLaunch(cycle[j], cycle[j-1], cycle[j], (i==0) ? false : true, true, guid, dbSize);

            for(unsigned int j=0; j<numGpus; j++)
                gpuShadowReductionLaunch(cycle[j], guid, dbSize, fnPtr);
        }
        return true;
    }
    return false;
}

void gpuLCInvalidate(unsigned int mask, artsGuid_t guid)
{
    for(unsigned int i=0; i<artsGetTotalGpus(); i++)
    {
        if(mask & (1 << i))
        {
            artsGpuInvalidateOnRouteTable(guid, i);
            artsGpuRouteTableReturnDb(guid, true, i);
        }
    }
}

unsigned int gpuLCReturnDb(unsigned int mask, artsGuid_t guid)
{
    unsigned int remMask = 0;
    for(unsigned int i=0; i<artsGetTotalGpus(); i++)
    {
        if(mask & (1 << i))
        {
            if(!remMask && i==2)
                remMask = 1 << i;
            else
                artsGpuRouteTableReturnDb(guid, false, i);
        }
    }
    return remMask;
}

unsigned int gpuLCReduce(artsGuid_t guid, struct artsDb * db, artsLCSyncFunctionGpu_t dbFn, bool * copyOnly)
{
    *copyOnly = false;
    unsigned int remMask = 0;
    unsigned int size = db->header.size;
    struct artsDb * shadowCopy = (struct artsDb*)(((char*)db) + size);
    
    artsWriterLock(&db->reader, &db->writer);
    unsigned int mask = artsGpuLookupDbFix(guid);
    if(mask)
    {
        //RING IS NOT WORKING...
        // if(db->version == shadowCopy->version && gpuRingReduction(mask, guid, size, dbFn))
        // {
        //     remMask = gpuLCReturnDb(mask, guid);
        //     *copyOnly = true;
        // }
        // else
        {
            remMask = gpuTreeReduction(mask, guid, size, dbFn);
            gpuLCInvalidate(mask & ~remMask, guid);
        }
    }
    remMask = mask;
    artsWriterUnlock(&db->writer);
    return remMask;
}