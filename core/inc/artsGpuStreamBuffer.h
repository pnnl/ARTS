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
#ifndef ARTSGPUSTREAMBUFFER_H
#define ARTSGPUSTREAMBUFFER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <cuda_runtime.h>    
#include "artsRT.h"

typedef struct 
{
    void * dst;
    void * src;
    size_t count;
} artsBufferMemMove_t;


typedef struct
{
    uint32_t paramc;
    uint64_t * paramv; 
    uint32_t depc;
    artsEdtDep_t * depv;
    artsEdt_t fnPtr;
    unsigned int grid[3];
    unsigned int block[3];
} artsBufferKernel_t;

// CHECKCORRECT(cudaMemcpyAsync(dataPtr, depv[i].ptr, size, cudaMemcpyHostToDevice, artsGpu->stream));
bool pushDataToStream(unsigned int gpuId, void * dst, void * src, size_t count, bool buff);
bool getDataFromStream(unsigned int gpuId, void * dst, void * src, size_t count, bool buff);

//  void * kernelArgs[] = { &paramc, &devParamv, &depc, &devDepv };
// CHECKCORRECT(cudaLaunchKernel((const void *)fnPtr, grid, block, (void**)kernelArgs, (size_t)0, artsGpu->stream));
bool pushKernelToStream(unsigned int gpuId, uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t * depv, artsEdt_t fnPtr, dim3 grid, dim3 block, bool buff);

// #if CUDART_VERSION >= 10000
//     CHECKCORRECT(cudaLaunchHostFunc(artsGpu->stream, artsWrapUp, hostClosure));
// #else
//     CHECKCORRECT(cudaStreamAddCallback(artsGpu->stream, artsWrapUp, hostClosure, 0));
// #endif
bool pushWrapUpToStream(unsigned int gpuId, void * hostClosure, bool buff);

bool flushMemStream(unsigned int gpuId, unsigned int * count, artsBufferMemMove_t * buff, enum cudaMemcpyKind kind);
bool flushKernelStream(unsigned int gpuId);
bool flushWrapUpStream(unsigned int gpuId);

bool flushStream(unsigned int gpuId);
bool checkStreams(bool buffOn);

#ifdef __cplusplus
}
#endif

#endif /* ARTSGPUSTREAMBUFFER_H */

