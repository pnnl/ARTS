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

#define IDX2C(i, j, ld) (((j) * (ld)) + (i))
#define m 6    // a - mxk matrix
#define n 4    // b - kxn matrix
#define k 5    // c - mxn matrix

#define CHECKCUBLASERROR(x) {                   \
  cublasStatus_t err;                           \
  if( (err = (x)) != CUBLAS_STATUS_SUCCESS )    \
    PRINTF("FAILED %s: %s\n", #x, err);         \
}

cublasHandle_t * handle;

void work(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    int            i, j;                          // i-row index ,j- column index
    float*         a;                             // mxk matrix a on the host
    float*         b;                             // kxn matrix b on the host
    float*         c;                             // mxn matrix c on the host

    a = (float*)malloc(m * k * sizeof(float));    // host memory for a
    b = (float*)malloc(k * n * sizeof(float));    // host memory for b
    c = (float*)malloc(m * n * sizeof(float));    // host memory for c

    // define an mxk matrix a column by column
    int ind = 11;                               // a:
    for (j = 0; j < k; j++) {                   // 11 ,17 ,23 ,29 ,35
        for (i = 0; i < m; i++) {               // 12 ,18 ,24 ,30 ,36
            a[IDX2C(i, j, m)] = (float)ind++;   // 13 ,19 ,25 ,31 ,37
        }                                       // 14 ,20 ,26 ,32 ,38
    }                                           // 15 ,21 ,27 ,33 ,39

    // 16 ,22 ,28 ,34 ,40
    // print a row by row
    printf("a:\n");
    for (i = 0; i < m; i++) {
        for (j = 0; j < k; j++) {
            printf(" %5.0f", a[IDX2C(i, j, m)]);
        }
        printf("\n");
    }

    // define a kxn matrix b column by column
    ind = 11;                                   // b:
    for (j = 0; j < n; j++) {                   // 11 ,16 ,21 ,26
        for (i = 0; i < k; i++) {               // 12 ,17 ,22 ,27
            b[IDX2C(i, j, k)] = (float)ind++;   // 13 ,18 ,23 ,28
        }                                       // 14 ,19 ,24 ,29
    }                                           // 15 ,20 ,25 ,30

    // print b row by row
    printf("b:\n");
    for (i = 0; i < k; i++) {
        for (j = 0; j < n; j++) {
        printf(" %5.0f", b[IDX2C(i, j, k)]);
        }
        printf("\n");
    }

    // define an mxn matrix c column by column
    ind = 11;                                   // c:
    for (j = 0; j < n; j++) {                   // 11 ,17 ,23 ,29
        for (i = 0; i < m; i++) {               // 12 ,18 ,24 ,30
            c[IDX2C(i, j, m)] = (float)ind++;   // 13 ,19 ,25 ,31
        }                                       // 14 ,20 ,26 ,32
    }                                           // 15 ,21 ,27 ,33

    // 16 ,22 ,28 ,34
    // print c row by row
    printf("c:\n");
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            printf(" %5.0f", c[IDX2C(i, j, m)]);
        }
        printf("\n");
    }
    // on the device
    float* d_a;                                                      // d_a - a on the device
    float* d_b;                                                      // d_b - b on the device
    float* d_c;                                                      // d_c - c on the device
    CHECKCORRECT(cudaMalloc((void**)&d_a, m * k * sizeof(*a)));    // device
    // memory alloc for a
    CHECKCORRECT(cudaMalloc((void**)&d_b, k * n * sizeof(*b)));         // device
    // memory alloc for b
    CHECKCORRECT(cudaMalloc((void**)&d_c, m * n * sizeof(*c)));         // device
    // memory alloc for c
        // initialize CUBLAS context

    // copy matrices from the host to the device
    CHECKCUBLASERROR(cublasSetMatrix(m, k, sizeof(*a), a, m, d_a, m));    //a -> d_a
    CHECKCUBLASERROR(cublasSetMatrix(k, n, sizeof(*b), b, k, d_b, k));    //b -> d_b
    CHECKCUBLASERROR(cublasSetMatrix(m, n, sizeof(*c), c, m, d_c, m));    //c -> d_c
    float al  = 1.0f;                                              // al =1
    float bet = 1.0f;                                              // bet =1

    // matrix - matrix multiplication : d_c = al*d_a *d_b + bet *d_c
    // d_a -mxk matrix , d_b -kxn matrix , d_c -mxn matrix ;
    // al ,bet -scalars
    CHECKCUBLASERROR(cublasSgemm(handle[artsGetGpuId()], CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &al, d_a, m, d_b, k, &bet, d_c, m));
    
    float * finalData;
    artsGuid_t finalGuid = artsDbCreate((void**)&finalData, sizeof(float) * m * n, ARTS_DB_READ);
    artsPutInDbFromGpu(d_c, finalGuid, 0, sizeof(float) * m * n, true);
    // stat = cublasGetMatrix(m, n, sizeof(*c), d_c, m, c, m);    // cp d_c - >c
    
    
    cudaFree(d_a);            // free device memory
    cudaFree(d_b);            // free device memory
    // cudaFree(d_c);            // free device memory
    
    free(a);                  // free host memory
    free(b);                  // free host memory
    free(c);                  // free host memory
    
    artsGuid_t toSignal = (artsGuid_t) paramv[0];
    artsSignalEdt(toSignal, 0, finalGuid);
}

void done(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    float * c = (float*) depv[0].ptr;
    printf("c after Sgemm :\n");
    for (unsigned int i = 0; i < m; i++) {
        for (unsigned int j = 0; j < n; j++) {
            printf(" %7.0f", c[IDX2C(i, j, m)]);    // print c after Sgemm
        }
        printf("\n");
    }
    artsShutdown();
}

extern "C"
void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    
}

extern "C"
void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{
    if(!workerId)
    {
        dim3 threads (1, 1);
        dim3 grid (1, 1);

        artsGuid_t doneGuid = artsEdtCreate(done, 0, 0, NULL, 1);
        artsGuid_t workGuid = artsEdtCreateGpuLib(work, 0, 1, (uint64_t*)&doneGuid, 0, grid, threads);
    }
}

extern "C"
void initPerGpu(int devId, cudaStream_t * stream)
{
    PRINTF("DevId: %d\n", devId);
    if(!devId)
    {
        handle = (cublasHandle_t*) artsCalloc(sizeof(cublasHandle_t) * artsGetNumGpus());
        PRINTF("NUM GPUS: %u\n", artsGetNumGpus());
    }
    cublasStatus_t stat = cublasCreate(&handle[devId]);
}

extern "C"
void cleanPerGpu(int devId, cudaStream_t * stream)
{
    PRINTF("DevId: %d\n", devId);
    cublasStatus_t stat = cublasDestroy(handle[devId]);
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}