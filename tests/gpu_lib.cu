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

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "arts.h"
#include "arts/gpu.h"
#include "arts/gpu/gpu_stream.h"

#define IDX2C(i, j, ld) (((j) * (ld)) + (i))
#define M 6 // a - mxk matrix
#define N 4 // b - kxn matrix
#define K 5 // c - mxn matrix

#define CHECKCUBLASERROR(x)                                                    \
  {                                                                            \
    cublasStatus_t err;                                                        \
    if ((err = (x)) != CUBLAS_STATUS_SUCCESS) {                                \
      arts_printf("FAILED %s: %s\n", #x, err);                                 \
    }                                                                          \
  }

cublasHandle_t *handle;

void work(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
          arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  int i;    // i-row index
  int j;    // j- column index
  float *a; // mxk matrix a on the host
  float *b; // kxn matrix b on the host
  float *c; // mxn matrix c on the host

  a = (float *)malloc((size_t)M * K * sizeof(float)); // host memory for a
  b = (float *)malloc((size_t)K * N * sizeof(float)); // host memory for b
  c = (float *)malloc((size_t)M * N * sizeof(float)); // host memory for c

  // define an mxk matrix a column by column
  int ind = 11;                         // a:
  for (j = 0; j < K; j++) {             // 11 ,17 ,23 ,29 ,35
    for (i = 0; i < M; i++) {           // 12 ,18 ,24 ,30 ,36
      a[IDX2C(i, j, M)] = (float)ind++; // 13 ,19 ,25 ,31 ,37
    } // 14 ,20 ,26 ,32 ,38
  } // 15 ,21 ,27 ,33 ,39

  // 16 ,22 ,28 ,34 ,40
  // print a row by row
  printf("a:\n");
  for (i = 0; i < M; i++) {
    for (j = 0; j < K; j++) {
      printf(" %5.0f", a[IDX2C(i, j, M)]);
    }
    printf("\n");
  }

  // define a kxn matrix b column by column
  ind = 11;                             // b:
  for (j = 0; j < N; j++) {             // 11 ,16 ,21 ,26
    for (i = 0; i < K; i++) {           // 12 ,17 ,22 ,27
      b[IDX2C(i, j, K)] = (float)ind++; // 13 ,18 ,23 ,28
    } // 14 ,19 ,24 ,29
  } // 15 ,20 ,25 ,30

  // print b row by row
  printf("b:\n");
  for (i = 0; i < K; i++) {
    for (j = 0; j < N; j++) {
      printf(" %5.0f", b[IDX2C(i, j, K)]);
    }
    printf("\n");
  }

  // define an mxn matrix c column by column
  ind = 11;                             // c:
  for (j = 0; j < N; j++) {             // 11 ,17 ,23 ,29
    for (i = 0; i < M; i++) {           // 12 ,18 ,24 ,30
      c[IDX2C(i, j, M)] = (float)ind++; // 13 ,19 ,25 ,31
    } // 14 ,20 ,26 ,32
  } // 15 ,21 ,27 ,33

  // 16 ,22 ,28 ,34
  // print c row by row
  printf("c:\n");
  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j++) {
      printf(" %5.0f", c[IDX2C(i, j, M)]);
    }
    printf("\n");
  }
  // on the device
  float *d_a; // d_a - a on the device
  float *d_b; // d_b - b on the device
  float *d_c; // d_c - c on the device
  CHECKCORRECT(cudaMalloc((void **)&d_a, (size_t)M * K * sizeof(*a))); // device
  // memory alloc for a
  CHECKCORRECT(cudaMalloc((void **)&d_b, (size_t)K * N * sizeof(*b))); // device
  // memory alloc for b
  CHECKCORRECT(cudaMalloc((void **)&d_c, (size_t)M * N * sizeof(*c))); // device
  // memory alloc for c
  // initialize CUBLAS context

  // copy matrices from the host to the device
  CHECKCUBLASERROR(cublasSetMatrix(M, K, sizeof(*a), a, M, d_a, M)); // a -> d_a
  CHECKCUBLASERROR(cublasSetMatrix(K, N, sizeof(*b), b, K, d_b, K)); // b -> d_b
  CHECKCUBLASERROR(cublasSetMatrix(M, N, sizeof(*c), c, M, d_c, M)); // c -> d_c
  float al = 1.0f;                                                   // al =1
  float bet = 1.0f;                                                  // bet =1

  // matrix - matrix multiplication : d_c = al*d_a *d_b + bet *d_c
  // d_a -mxk matrix , d_b -kxn matrix , d_c -mxn matrix ;
  // al ,bet -scalars
  CHECKCUBLASERROR(cublasSgemm(handle[arts_get_gpu_id()], CUBLAS_OP_N,
                               CUBLAS_OP_N, M, N, K, &al, d_a, M, d_b, K, &bet,
                               d_c, M));

  float *final_data;
  arts_guid_t final_guid =
      arts_db_create((void **)&final_data, sizeof(float) * (size_t)M * N,
                     ARTS_DB_DEFAULT, NULL);
  arts_put_in_db_from_gpu(d_c, final_guid, 0, sizeof(float) * (size_t)M * N,
                          true);
  // stat = cublasGetMatrix(M, N, sizeof(*c), d_c, M, c, M);    // cp d_c - >c

  cudaFree(d_a); // free device memory
  cudaFree(d_b); // free device memory
  // cudaFree(d_c);            // free device memory

  free(a); // free host memory
  free(b); // free host memory
  free(c); // free host memory

  arts_guid_t to_signal = (arts_guid_t)paramv[0];
  arts_signal_edt(to_signal, 0, final_guid, DB_MODE_EW);
}

void done(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
          arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  float *c = (float *)depv[0].ptr;
  /*
   * Verify C = alpha*A*B + beta*C_init where alpha=1, beta=1.
   * A(6x5) col-major starting at 11, B(5x4) col-major starting at 11,
   * C_init(6x4) col-major starting at 11.
   * C[0][0] = 11*11 + 17*12 + 23*13 + 29*14 + 35*15 + 11 = 1566.
   */
  float expected_c00 = 1566.0f;
  bool ok = true;
  if (c == NULL) {
    arts_printf("  FAIL: gpu_lib cuBLAS result is NULL\n");
    ok = false;
  } else if (c[IDX2C(0, 0, M)] != expected_c00) {
    arts_printf("  FAIL: gpu_lib C[0][0] = %.0f, expected %.0f\n",
                c[IDX2C(0, 0, M)], expected_c00);
    ok = false;
  }
  if (ok) {
    arts_printf("  PASS: gpu_lib cuBLAS Sgemm result verified\n");
  }
  arts_shutdown();
}

extern "C" void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  dim3 threads(1, 1);
  dim3 grid(1, 1);

  arts_hint_t hint_0 = {0, 0};
  arts_guid_t done_guid = arts_edt_create(done, 0, NULL, 1, &hint_0);
  arts_gpu_hint_t gpu_hint = {};
  gpu_hint.gpu = -1;
  gpu_hint.route = 0;
  gpu_hint.lib = true;
  arts_guid_t work_guid = arts_edt_create_gpu(
      work, 1, (uint64_t *)&done_guid, 0, arts_from_dim3(grid),
      arts_from_dim3(threads), &gpu_hint);
  (void)work_guid;
}

extern "C" void arts_init_per_gpu(unsigned int node_id, int dev_id,
                                  cudaStream_t *stream, int argc, char **argv) {
  (void)node_id;
  (void)stream;
  (void)argc;
  (void)argv;
  arts_printf("DevId: %d\n", dev_id);
  if (!dev_id) {
    handle =
        (cublasHandle_t *)calloc(arts_get_num_gpus(), sizeof(cublasHandle_t));
    arts_printf("NUM GPUS: %u\n", arts_get_num_gpus());
  }
  cublasStatus_t stat = cublasCreate(&handle[dev_id]);
  (void)stat;
}

extern "C" void arts_fini_per_gpu(unsigned int node_id, int dev_id,
                                  cudaStream_t *stream) {
  (void)node_id;
  (void)stream;
  arts_printf("DevId: %d\n", dev_id);
  cublasStatus_t stat = cublasDestroy(handle[dev_id]);
  (void)stat;
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
