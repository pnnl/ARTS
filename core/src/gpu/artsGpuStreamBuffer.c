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

//Some help https://devblogs.nvidia.com/how-overlap-data-transfers-cuda-cc/
//and https://github.com/NVIDIA-developer-blog/code-samples/blob/master/series/cuda-cpp/overlap-data-transfers/async.cu
//Once this *class* works we will put a stream(s) in create a thread local
//stream.  Then we will push stuff!
#include "artsDbFunctions.h"
#include "artsGpuStream.h"
#include "artsGpuRuntime.h"
#include "artsGlobals.h"
#include "artsDeque.h"
#include "artsGpuRouteTable.h"
#include "artsDebug.h"
#include "artsEdtFunctions.h"
#include "artsGpuStreamBuffer.h"

#define DPRINTF( ... )
// #define DPRINTF( ... ) PRINTF( __VA_ARGS__ )

#define CHECKSTREAM 4096
#define MAXSTREAM 32
#define MAXBUFFER 128

volatile unsigned int streamCheckCount[MAXSTREAM] = {0};

volatile unsigned int buffLock[MAXSTREAM] = {0};
unsigned int hostToDevCount[MAXSTREAM] = {0};
unsigned int kernelToDevCount[MAXSTREAM] = {0};
unsigned int devToHostCount[MAXSTREAM] = {0};
unsigned int wrapUpCount[MAXSTREAM] = {0};

artsBufferMemMove_t hostToDevBuff[MAXSTREAM][MAXBUFFER];
artsBufferKernel_t kernelToDevBuff[MAXSTREAM][MAXBUFFER];
artsBufferMemMove_t devToHostBuff[MAXSTREAM][MAXBUFFER];
void * wrapUpBuff[MAXSTREAM][MAXBUFFER];

bool pushDataToStream(unsigned int gpuId, void * dst, void * src, size_t count, bool buff)
{
    if(buff)
    {
        artsLock(&buffLock[gpuId]);
        hostToDevBuff[gpuId][hostToDevCount[gpuId]].dst = dst;
        hostToDevBuff[gpuId][hostToDevCount[gpuId]].src = src;
        hostToDevBuff[gpuId][hostToDevCount[gpuId]].count = count;
        hostToDevCount[gpuId]++;
        
        bool ret = false;
        if(hostToDevCount[gpuId] == MAXBUFFER)
            ret = flushStream(gpuId);
        artsUnlock(&buffLock[gpuId]);
        return ret;
    }
    CHECKCORRECT(cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, artsGpus[gpuId].stream));
    return true;
}

bool getDataFromStream(unsigned int gpuId, void * dst, void * src, size_t count, bool buff)
{
    if(buff)
    {
        artsLock(&buffLock[gpuId]);
        devToHostBuff[gpuId][devToHostCount[gpuId]].dst = dst;
        devToHostBuff[gpuId][devToHostCount[gpuId]].src = src;
        devToHostBuff[gpuId][devToHostCount[gpuId]].count = count;
        devToHostCount[gpuId]++;

        bool ret = false;
        if(devToHostCount[gpuId] == MAXBUFFER)
            ret = flushStream(gpuId);
        artsUnlock(&buffLock[gpuId]);
        return ret;
    }
    CHECKCORRECT(cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, artsGpus[gpuId].stream));
    return true;
}

bool pushKernelToStream(unsigned int gpuId, uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t * depv, artsEdt_t fnPtr, dim3 grid, dim3 block, bool buff)
{
    if(buff)
    {
        artsLock(&buffLock[gpuId]);
        kernelToDevBuff[gpuId][kernelToDevCount[gpuId]].paramc = paramc;
        kernelToDevBuff[gpuId][kernelToDevCount[gpuId]].paramv = paramv;
        kernelToDevBuff[gpuId][kernelToDevCount[gpuId]].depc = depc;
        kernelToDevBuff[gpuId][kernelToDevCount[gpuId]].depv = depv;
        kernelToDevBuff[gpuId][kernelToDevCount[gpuId]].fnPtr = fnPtr;
        kernelToDevBuff[gpuId][kernelToDevCount[gpuId]].grid[0] = grid.x;
        kernelToDevBuff[gpuId][kernelToDevCount[gpuId]].grid[1] = grid.y;
        kernelToDevBuff[gpuId][kernelToDevCount[gpuId]].grid[2] = grid.z;
        kernelToDevBuff[gpuId][kernelToDevCount[gpuId]].block[0] = block.x;
        kernelToDevBuff[gpuId][kernelToDevCount[gpuId]].block[1] = block.y;
        kernelToDevBuff[gpuId][kernelToDevCount[gpuId]].block[2] = block.z;
        kernelToDevCount[gpuId]++;
        
        bool ret = false;
        if(kernelToDevCount[gpuId] == MAXBUFFER)
            ret = flushStream(gpuId);
        artsUnlock(&buffLock[gpuId]);
        return ret;
    }

    void * kernelArgs[] = { &paramc, &paramv, &depc, &depv };
    int maxActiveBlocks, blockSize;
    float occupancy;
    blockSize = (int) block.x * block.y * block.z;
    struct cudaDeviceProp prop = artsGpus[gpuId].prop;
    // TODO: Shared Memory should be unset from 0 when supported
    CHECKCORRECT(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, (const void*) fnPtr, (int) blockSize, 0));
    occupancy = (maxActiveBlocks * blockSize / prop.warpSize) / (float) (prop.maxThreadsPerMultiProcessor / prop.warpSize);

    // Moving average of occupancy
    artsLock(&artsGpus[gpuId].deviceLock);
    artsGpus[gpuId].occupancy = (occupancy + (artsGpus[gpuId].totalEdts-1) * artsGpus[gpuId].occupancy) / (++artsGpus[gpuId].totalEdts);
    artsUnlock(&artsGpus[gpuId].deviceLock);

    CHECKCORRECT(cudaLaunchKernel((const void *)fnPtr, grid, block, (void**)kernelArgs, (size_t)0, artsGpus[gpuId].stream));
    return true;
}

bool pushWrapUpToStream(unsigned int gpuId, void * hostClosure, bool buff)
{
    if(buff)
    {
        artsLock(&buffLock[gpuId]);
        wrapUpBuff[gpuId][wrapUpCount[gpuId]] = hostClosure;
        wrapUpCount[gpuId]++;
        
        bool ret = false;
        if(wrapUpCount[gpuId] == MAXBUFFER)
            ret = flushStream(gpuId);
        artsUnlock(&buffLock[gpuId]);
        return ret;
    }

    #if CUDART_VERSION >= 10000
        CHECKCORRECT(cudaLaunchHostFunc(artsGpus[gpuId].stream, artsWrapUp, hostClosure));
    #else
        CHECKCORRECT(cudaStreamAddCallback(artsGpus[gpuId].stream, artsWrapUp, hostClosure, 0));
    #endif
    return true;
}

bool flushMemStream(unsigned int gpuId, unsigned int * count, artsBufferMemMove_t * buff, enum cudaMemcpyKind kind)
{
    unsigned int max = *count;
    bool ret = (max > 0);
    for(unsigned int i=0; i<max; i++)
    {
        // PRINTF("i: %u %p %p %u %p\n", i, buff[i].dst, buff[i].src, buff[i].count,  &artsGpus[gpuId].stream);
        CHECKCORRECT(cudaMemcpyAsync(buff[i].dst, buff[i].src, buff[i].count, kind, artsGpus[gpuId].stream));
    }
    *count = 0;
    return ret;
}

bool flushKernelStream(unsigned int gpuId)
{
    bool ret = (kernelToDevCount > 0);
    for(unsigned int i=0; i<kernelToDevCount[gpuId]; i++)
    {
        void * kernelArgs[] = { 
            &kernelToDevBuff[gpuId][i].paramc, 
            &kernelToDevBuff[gpuId][i].paramv, 
            &kernelToDevBuff[gpuId][i].depc, 
            &kernelToDevBuff[gpuId][i].depv };
        dim3 grid(kernelToDevBuff[gpuId][i].grid[0], kernelToDevBuff[gpuId][i].grid[1], kernelToDevBuff[gpuId][i].grid[2]);
        dim3 block(kernelToDevBuff[gpuId][i].block[0], kernelToDevBuff[gpuId][i].block[1], kernelToDevBuff[gpuId][i].block[2]);
        CHECKCORRECT(cudaLaunchKernel((const void *)kernelToDevBuff[gpuId][i].fnPtr, grid, block, (void**)kernelArgs, (size_t)0, artsGpus[gpuId].stream));
    }
    kernelToDevCount[gpuId] = 0;
    return ret;
}

bool flushWrapUpStream(unsigned int gpuId)
{
    bool ret = (wrapUpCount > 0);
    for(unsigned int i=0; i<wrapUpCount[gpuId]; i++)
    {
        #if CUDART_VERSION >= 10000
            CHECKCORRECT(cudaLaunchHostFunc(artsGpus[gpuId].stream, artsWrapUp, wrapUpBuff[gpuId][i]));
        #else
            CHECKCORRECT(cudaStreamAddCallback(artsGpus[gpuId].stream, artsWrapUp, wrapUpBuff[gpuId][i], 0));
        #endif
    }
    wrapUpCount[gpuId] = 0;
    return ret;
}

bool flushStream(unsigned int gpuId)
{
    DPRINTF("%u %u %u %u\n", hostToDevCount[gpuId], kernelToDevCount[gpuId], devToHostCount[gpuId], wrapUpCount[gpuId]);
    if(hostToDevCount[gpuId] || kernelToDevCount[gpuId] || devToHostCount[gpuId] || wrapUpCount[gpuId])
    {
        int savedDevice;
        CHECKCORRECT(cudaGetDevice(&savedDevice));
        CHECKCORRECT(cudaSetDevice(gpuId));

        flushMemStream(gpuId, &hostToDevCount[gpuId], hostToDevBuff[gpuId], cudaMemcpyHostToDevice);
        flushKernelStream(gpuId);
        flushMemStream(gpuId, &devToHostCount[gpuId], devToHostBuff[gpuId], cudaMemcpyDeviceToHost);
        flushWrapUpStream(gpuId);

        CHECKCORRECT(cudaSetDevice(savedDevice));
        return true;
    }
    return false;
}

bool checkStreams(bool buffOn)
{
    if(buffOn)
    {
        bool ret = false;
        for(unsigned int i=0; i<artsNodeInfo.gpu; i++)
        {
            if(hostToDevCount[i] || kernelToDevCount[i] || devToHostCount[i] || wrapUpCount[i])
            {
                artsAtomicFetchAdd(&streamCheckCount[i], 1U);
                if(streamCheckCount[i] % CHECKSTREAM == 0)
                {
                    artsLock(&buffLock[i]);
                    ret |= flushStream(i);
                    artsUnlock(&buffLock[i]);
                }
            }
        }
        return ret;
    }
    return false;
}