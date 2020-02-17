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
#include "artsGpuRuntime.h"
#include "cublas_v2.h"
#include "cublas_api.h"
#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#define GPULISTLEN 32

unsigned int ** devPtrRaw; //This is a list of the search frontier in global memory on each gpu

//This will probably be where you want to do the actual traversal
__global__ void temp(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    // unsigned int gpuId = (unsigned int) paramv[0]; //The current gpu we are on
    uint64_t gpuId = getGpuIndex();
    unsigned int ** addr = (unsigned int **)depv[0].ptr; //This is the devPtrRaw -> tells us where current frontier is on device
    unsigned int * local = addr[gpuId]; //We need the one corresponding to our gpu

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    local[(GPULISTLEN - 1) - index] = (unsigned int)gpuId; //index; //Just writing some blah blah value to sort
}

//This should be where we do the sorting and should launch the next iteration
void thrustSort(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    artsGuid_t doneGuid = paramv[0]; //This can be the end if the frontier is empty
    unsigned int gpuIndex = paramv[1]; //gpuIndex
    unsigned int * rawPtr = devPtrRaw[gpuIndex]; //The corresponding dev pointer (frontier) to our gpu

    unsigned int * tile = NULL; //This will hold a tile of the new frontier
    artsGuid_t tileGuid = artsDbCreate((void**) &tile, sizeof(unsigned int) * GPULISTLEN, ARTS_DB_GPU_READ);

    thrust::device_ptr<unsigned int> devPtr(rawPtr);
    thrust::sort(devPtr, devPtr+GPULISTLEN); //Do the sorting
    
    //Copy the data from the gpu to the host
    artsPutInDbFromGpu(thrust::raw_pointer_cast(devPtr), tileGuid, 0, sizeof(unsigned int) * GPULISTLEN, false);

    //Probably should make some new edts and signal them with the data!
    //Or signal the end if we are done
    artsSignalEdt(doneGuid, gpuIndex, tileGuid); //don't really need tileGuid just doing it for testing
}

void done(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    //This is just for testing...
    //We should see it is sorted
    for(unsigned int i=0; i<depc; i++)
    {
        unsigned int * tile = (unsigned int *) depv[i].ptr;
        printf("GPU %u: ", i);
        for(unsigned int j=0; j<GPULISTLEN; j++)
            printf("%u, ", tile[j]);
        printf("\n");
    }
    artsShutdown();
}

extern "C"
void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    devPtrRaw = (unsigned int**) artsCalloc(sizeof(unsigned int*) * artsGetTotalGpus());
}

extern "C"
void initPerGpu(unsigned int nodeId, int devId, cudaStream_t * stream, int argc, char * argv)
{
    devPtrRaw[devId] = (unsigned int*) artsCudaMalloc(sizeof(unsigned int) * GPULISTLEN);
}

extern "C"
void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{
    if(!workerId)
    {
        unsigned int ** addr;
        artsGuid_t dbGuid = artsDbCreate((void**)&addr, sizeof(unsigned int*) * artsGetTotalGpus(), ARTS_DB_GPU_READ);
        for(uint64_t i=0; i<artsGetTotalGpus(); i++)
            addr[i] = devPtrRaw[i];

        artsGuid_t doneGuid = artsEdtCreate(done, 0, 0, NULL, artsGetTotalGpus());

        dim3 threads (GPULISTLEN, 1, 1);
        dim3 grid (1, 1, 1);
        for(uint64_t i=0; i<artsGetTotalGpus(); i++)
        {
            uint64_t args[] = {doneGuid, i}; 
            artsGuid_t edtGuid = artsEdtCreateGpuLibDirect(thrustSort, nodeId, i, 2, args, 1, grid, threads);
            artsGuid_t edtGuid2 = artsEdtCreateGpuDirect(temp, nodeId, i, 1, &i, 1, grid, threads, edtGuid, 0, dbGuid, true);
            artsSignalEdt(edtGuid2, 0, dbGuid);
        }
    }
}

extern "C"
void cleanPerGpu(unsigned int nodeId, int devId, cudaStream_t * stream)
{
    artsCudaFree(devPtrRaw[devId]);
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}