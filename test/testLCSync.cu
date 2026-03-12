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

#include "arts/arts.h"
#include "arts/gpu/GpuRuntime.cuh"

__global__ void temp(uint32_t paramc, uint64_t *paramv, uint32_t depc,
                     artsEdtDep_t depv[]) {
  uint64_t gpuId = getGpuIndex();
  // printf("Hello from %lu\n", gpuId);
  unsigned int *addr = (unsigned int *)depv[0].ptr;
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  addr[index] = gpuId + 1;
}

void done(uint32_t paramc, uint64_t *paramv, uint32_t depc,
          artsEdtDep_t depv[]) {
  unsigned int *tile = (unsigned int *)depv[0].ptr;
  for (unsigned int j = 0; j < artsGetTotalGpus(); j++)
    printf("%u, ", tile[j]);
  printf("\n");
  artsShutdown();
}

extern "C" void initPerNode(unsigned int nodeId, int argc, char **argv) {}

extern "C" void initPerGpu(unsigned int nodeId, int devId, cudaStream_t *stream,
                           int argc, char *argv) {}

extern "C" void initPerWorker(unsigned int nodeId, unsigned int workerId,
                              int argc, char **argv) {
  if (!workerId) {
    unsigned int *addr = NULL;
    PRINTF("creating size: %u\n", sizeof(unsigned int) * artsGetTotalGpus());
    artsGuid_t dbGuid = artsDbCreate(
        (void **)&addr, sizeof(unsigned int) * artsGetTotalGpus(), ARTS_DB_LC);
    for (uint64_t i = 0; i < artsGetTotalGpus(); i++)
      addr[i] = (unsigned int)-1;

    artsGuid_t doneGuid =
        artsEdtCreate(done, 0, 0, NULL, artsGetTotalGpus() + 1);
    artsLCSync(doneGuid, 0, dbGuid);
    // artsSignalEdt(doneGuid, 0, dbGuid);

    dim3 threads(artsGetTotalGpus(), 1, 1);
    dim3 grid(1, 1, 1);
    for (uint64_t i = 0; i < artsGetTotalGpus(); i++) {
      if (i == 3 || i == 4 || i == 7) {
        PRINTF("CREATING EDT for GPU: %lu\n", i);
        artsGuid_t edtGuid =
            artsEdtCreateGpuDirect(temp, nodeId, i, 0, NULL, 1, grid, threads,
                                   doneGuid, i + 1, NULL_GUID, true);
        artsSignalEdt(edtGuid, 0, dbGuid);
      } else
        artsSignalEdt(doneGuid, i + 1, NULL_GUID);
    }
  }
}

extern "C" void cleanPerGpu(unsigned int nodeId, int devId,
                            cudaStream_t *stream) {}

int main(int argc, char **argv) {
  artsRT(argc, argv);
  return 0;
}