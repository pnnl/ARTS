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
#ifndef ARTSGPURUNTIME_H
#define ARTSGPURUNTIME_H

#ifdef __cplusplus
extern "C" {
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "artsRT.h"
#include "artsGpuStream.h"
    
typedef struct
{
    struct artsEdt wrapperEdt;
    dim3 grid;
    dim3 block;
    artsGuid_t endGuid;
    artsGuid_t dataGuid;
    uint32_t slot;
    bool passthrough;
    bool lib;
} artsGpuEdt_t;

void * artsCudaMallocHost(unsigned int size);
void artsCudaFreeHost(void * ptr);
void * artsCudaMalloc(unsigned int size);
void artsCudaFree(void * ptr);
artsGuid_t artsEdtCreateGpuDep(artsEdt_t funcPtr, unsigned int route, uint32_t paramc, uint64_t * paramv, uint32_t depc, dim3 grid, dim3 block, artsGuid_t endGuid, uint32_t slot, artsGuid_t dataGuid, bool hasDepv);
artsGuid_t artsEdtCreateGpu(artsEdt_t funcPtr, unsigned int route, uint32_t paramc, uint64_t * paramv, uint32_t depc, dim3 grid, dim3 block, artsGuid_t endGuid, uint32_t slot, artsGuid_t dataGuid);
artsGuid_t artsEdtCreateGpuPT(artsEdt_t funcPtr, unsigned int route, uint32_t paramc, uint64_t * paramv, uint32_t depc, dim3 grid, dim3 block, artsGuid_t endGuid, uint32_t slot, unsigned int passSlot);
artsGuid_t artsEdtCreateGpuPTDep(artsEdt_t funcPtr, unsigned int route, uint32_t paramc, uint64_t * paramv, uint32_t depc, dim3 grid, dim3 block, artsGuid_t endGuid, uint32_t slot, unsigned int passSlot, bool hasDepv);
artsGuid_t artsEdtCreateGpuLib(artsEdt_t funcPtr, unsigned int route, uint32_t paramc, uint64_t * paramv, uint32_t depc, dim3 grid, dim3 block);
artsGuid_t artsEdtCreateGpuPTWithGuid(artsEdt_t funcPtr, artsGuid_t guid, uint32_t paramc, uint64_t * paramv, uint32_t depc, dim3 grid, dim3 block, artsGuid_t endGuid, uint32_t slot, unsigned int passSlot);

dim3 * artsGetGpuGrid();
dim3 * artsGetGpuBlock();
cudaStream_t * artsGetGpuStream();
int artsGetGpuId();
unsigned int artsGetNumGpus();
void artsPutInDbFromGpu(void * ptr, artsGuid_t dbGuid, unsigned int offset, unsigned int size, bool freeData);

void artsGpuHostWrapUp(void * edtPacket, artsGuid_t toSignal, uint32_t slot, artsGuid_t dataGuid);
void artsRunGpu(void * edtPacket, artsGpu_t * artsGpu);
bool artsGpuSchedulerLoop();

#ifdef __cplusplus
}
#endif

#endif /* ARTSGPURUNTIME_H */

