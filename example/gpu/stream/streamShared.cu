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

/*-----------------------------------------------------------------------*/
/* Program: Stream                                                       */
/* Revision: $Id: stream_omp.c,v 5.4 2009/02/19 13:57:12 mccalpin Exp mccalpin $ */
/* Original code developed by John D. McCalpin                           */
/* Programmers: John D. McCalpin                                         */
/*              Joe R. Zagar                                             */
/*                                                                       */
/* This program measures memory transfer rates in MB/s for simple        */
/* computational kernels coded in C.                                     */
/*-----------------------------------------------------------------------*/
/* Copyright 1991-2003: John D. McCalpin                                 */
/*-----------------------------------------------------------------------*/
/* License:                                                              */
/*  1. You are free to use this program and/or to redistribute           */
/*     this program.                                                     */
/*  2. You are free to modify this program for your own use,             */
/*     including commercial use, subject to the publication              */
/*     restrictions in item 3.                                           */
/*  3. You are free to publish results obtained from running this        */
/*     program, or from works that you derive from this program,         */
/*     with the following limitations:                                   */
/*     3a. In order to be referred to as "STREAM benchmark results",     */
/*         published results must be in conformance to the STREAM        */
/*         Run Rules, (briefly reviewed below) published at              */
/*         http://www.cs.virginia.edu/stream/ref.html                    */
/*         and incorporated herein by reference.                         */
/*         As the copyright holder, John McCalpin retains the            */
/*         right to determine conformity with the Run Rules.             */
/*     3b. Results based on modified source code or on runs not in       */
/*         accordance with the STREAM Run Rules must be clearly          */
/*         labelled whenever they are published.  Examples of            */
/*         proper labelling include:                                     */
/*         "tuned STREAM benchmark results"                              */
/*         "based on a variant of the STREAM benchmark code"             */
/*         Other comparable, clear and reasonable labelling is           */
/*         acceptable.                                                   */
/*     3c. Submission of results to the STREAM benchmark web site        */
/*         is encouraged, but not required.                              */
/*  4. Use of this program or creation of derived works based on this    */
/*     program constitutes acceptance of these licensing restrictions.   */
/*  5. Absolutely no warranty is expressed or implied.                   */
/*-----------------------------------------------------------------------*/
# include <stdio.h>
# include <math.h>
# include <float.h>
# include <limits.h>
# include <sys/time.h>
#include "cublas_v2.h"
#include <cuda_runtime.h>

#ifdef STREAMN
#define N STREAMN
#else
#define N 134217728
#endif

# define NTIMES	10
# define OFFSET	0

# define HLINE "-------------------------------------------------------------\n"

# ifndef MIN
# define MIN(x,y) ((x)<(y)?(x):(y))
# endif
# ifndef MAX
# define MAX(x,y) ((x)>(y)?(x):(y))
# endif

#define MAXTHREADS 32
#define MAXTHREADBLOCKSPERSM 2
#define NUMBEROFSM 80
#define MAXGRID MAXTHREADBLOCKSPERSM * NUMBEROFSM

extern double mysecond();
extern void checkSTREAMresults();

unsigned int tileSize = N;
unsigned int tiles = 0;
unsigned int numGpus = 1;

static double avgtime[4] = {0};
static double maxtime[4] = {0};
static double mintime[4] = {FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX};

static const char	* label[4] = {
    "Copy:      ", 
    "Scale:     ", 
    "Add:       ", 
    "Triad:     "};

static double bytes[4] = {
    2 * sizeof(double) * N,
    2 * sizeof(double) * N,
    3 * sizeof(double) * N,
    3 * sizeof(double) * N
    };

int quantum;

double *a = NULL;
double *b = NULL;
double *c = NULL;

__global__ void copyKernel(int index, unsigned int len, double * a, double * b)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < len)
        b[idx+index] = a[idx+index];
}

__global__ void scaleKernel(int index, unsigned int len, double scale, double * a, double * b)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < len) 
        b[idx+index] = scale * a[idx+index];
}

__global__ void addKernel(int index, unsigned int len, double * a, double * b, double * c)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < len) 
        c[idx+index] = a[idx+index] + b[idx+index];
}

__global__ void triadKernel(int index, unsigned int len, double scale, double * a, double * b, double * c)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < len) 
        c[idx+index] = a[idx+index] + scale * b[idx+index];
}

int main(int argc, char** argv)
{
    if (argc > 1)
        tileSize = (unsigned int) atoi(argv[1]);
    
    if (argc > 2)
        numGpus = (unsigned int) atoi(argv[2]);

    tiles = N / tileSize;
    if(N % tileSize)
        tiles++;

    printf("N: %d Tiles: %u Tile Size: %u\n", N, tiles, tileSize);

    int quantum, checktick();
    int	BytesPerWord;
    register int j, k;
    double scalar, t, times[4][NTIMES];

    /* --- SETUP --- determine precision and check timing --- */

    printf(HLINE);
    BytesPerWord = sizeof(double);
    printf("This system uses %d bytes per DOUBLE PRECISION word.\n", BytesPerWord);

    printf(HLINE);
    printf("Array size = %d, Offset = %d\n" , N, OFFSET);
    printf("Total memory required = %.1f MB.\n", (3.0 * BytesPerWord) * ( (double) N / 1048576.0));
    printf("Each test is run %d times, but only\n", NTIMES);
    printf("the *best* time for each is used.\n");

    cudaMallocManaged(&a, N * sizeof(double));
    cudaMallocManaged(&b, N * sizeof(double));
    cudaMallocManaged(&c, N * sizeof(double));

    unsigned int numThreads = (MAXTHREADS < tileSize) ? MAXTHREADS : tileSize;
    dim3 blockSize(numThreads, 1, 1);
    dim3 numBlocks((tileSize+numThreads-1)/numThreads, 1, 1);

    /* Get initial value for system clock. */
    for (j=0; j<N; j++) {
        a[j] = 1.0;
        b[j] = 2.0;
        c[j] = 0.0;
    }

    printf(HLINE);

    if  ( (quantum = checktick()) >= 1) 
	    printf("Your clock granularity/precision appears to be %d microseconds.\n", quantum);
    else
	    printf("Your clock granularity appears to be less than one microsecond.\n");

    t = mysecond();
    for (j = 0; j < N; j++)
        a[j] = 2.0E0 * a[j];
    t = 1.0E6 * (mysecond() - t);

    printf("Each test below will take on the order of %d microseconds.\n", (int) t  );
    printf("   (= %d clock ticks)\n", (int) (t/quantum) );
    printf("Increase the size of the arrays if this shows that\n");
    printf("you are not getting at least 20 clock ticks per test.\n");

    printf(HLINE);

    printf("WARNING -- The above is only a rough guideline.\n");
    printf("For best results, please be sure you know the\n");
    printf("precision of your system timer.\n");
    printf(HLINE);
    
    /*	--- MAIN LOOP --- repeat test cases NTIMES times --- */

    scalar = 3.0;
    for (k=0; k<NTIMES; k++)
	{
	    times[0][k] = mysecond();
        for (j=0; j<tiles; j++)
        {
            cudaSetDevice(j % numGpus);
            // c[j] = a[j];
            copyKernel<<<numBlocks, blockSize>>>(j*tileSize, (j+1 < tiles) ? tileSize : N - j*tileSize, a, c);
        }

        for(j=0; j<numGpus; j++)
        {
            cudaSetDevice(j);
            cudaDeviceSynchronize();
        }
	    times[0][k] = mysecond() - times[0][k];
	
	    times[1][k] = mysecond();
        for (j=0; j<tiles; j++)
        {
            cudaSetDevice(j % numGpus);
            // b[j] = scalar*c[j];
            scaleKernel<<<numBlocks, blockSize>>>(j*tileSize, (j+1 < tiles) ? tileSize : N - j*tileSize, scalar, c, b);
        }

        for(j=0; j<numGpus; j++)
        {
            cudaSetDevice(j);
            cudaDeviceSynchronize();
        }
	    times[1][k] = mysecond() - times[1][k];
	
	    times[2][k] = mysecond();
	    for (j=0; j<tiles; j++)
        {
            cudaSetDevice(j % numGpus);
	        // c[j] = a[j]+b[j];
            addKernel<<<numBlocks, blockSize>>>(j*tileSize, (j+1 < tiles) ? tileSize : N - j*tileSize, a, b, c);
        }

        for(j=0; j<numGpus; j++)
        {
            cudaSetDevice(j);
            cudaDeviceSynchronize();
        }
	    times[2][k] = mysecond() - times[2][k];
	
	    times[3][k] = mysecond();
	    for (j=0; j<tiles; j++)
        {
            cudaSetDevice(j % numGpus);
	        // a[j] = b[j]+scalar*c[j];
            triadKernel<<<numBlocks, blockSize>>>(j*tileSize, (j+1 < tiles) ? tileSize : N - j*tileSize, scalar, b, c, a);
        }

        for(j=0; j<numGpus; j++)
        {
            cudaSetDevice(j);
            cudaDeviceSynchronize();
        }
	    times[3][k] = mysecond() - times[3][k];
    }

    /*	--- SUMMARY --- */
    for (k=1; k<NTIMES; k++) /* note -- skip first iteration */
	{
        for (j=0; j<4; j++)
        {
            avgtime[j] = avgtime[j] + times[j][k];
            mintime[j] = MIN(mintime[j], times[j][k]);
            maxtime[j] = MAX(maxtime[j], times[j][k]);
        }
	}
    
    printf("Function      Rate (MB/s)   Avg time     Min time     Max time\n");
    for (j=0; j<4; j++) {
	    avgtime[j] = avgtime[j]/(double)(NTIMES-1);

        printf("%s%11.4f  %11.4f  %11.4f  %11.4f\n", label[j],
            1.0E-06 * bytes[j]/mintime[j],
            avgtime[j],
            mintime[j],
            maxtime[j]);
    }
    printf(HLINE);

    /* --- Check Results --- */
    checkSTREAMresults();
    printf(HLINE);
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    return 0;
}

# define	M	20

int checktick()
{
    int	i, minDelta, Delta;
    double t1, t2, timesfound[M];

/*  Collect a sequence of M unique time values from the system. */

    for (i = 0; i < M; i++) {
        t1 = mysecond();
        while( ((t2=mysecond()) - t1) < 1.0E-6 );
        timesfound[i] = t1 = t2;
    }

/*
 * Determine the minimum difference between these M values.
 * This result will be our estimate (in microseconds) for the
 * clock granularity.
 */

    minDelta = 1000000;
    for (i = 1; i < M; i++) {
        Delta = (int)( 1.0E6 * (timesfound[i]-timesfound[i-1]));
        minDelta = MIN(minDelta, MAX(Delta,0));
	}

    return(minDelta);
}

/* A gettimeofday routine to give access to the wall
   clock timer on most UNIX-like systems.  */

#include <sys/time.h>

double mysecond()
{
        struct timeval tp;
        struct timezone tzp;
        int i;

        i = gettimeofday(&tp,&tzp);
        return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

void checkSTREAMresults ()
{
	double aj,bj,cj,scalar;
	double asum,bsum,csum;
	double epsilon;
	int	j,k;

    /* reproduce initialization */
	aj = 1.0;
	bj = 2.0;
	cj = 0.0;
    /* a[] is modified during timing check */
	aj = 2.0E0 * aj;
    /* now execute timing loop */
	scalar = 3.0;
	for (k=0; k<NTIMES; k++)
    {
        cj = aj;
        bj = scalar*cj;
        cj = aj+bj;
        aj = bj+scalar*cj;
    }
	aj = aj * (double) (N);
	bj = bj * (double) (N);
	cj = cj * (double) (N);

	asum = 0.0;
	bsum = 0.0;
	csum = 0.0;
	for (j=0; j<N; j++) {
		asum += a[j];
		bsum += b[j];
		csum += c[j];
	}
#ifdef VERBOSE
	printf ("Results Comparison: \n");
	printf ("        Expected  : %f %f %f \n",aj,bj,cj);
	printf ("        Observed  : %f %f %f \n",asum,bsum,csum);
#endif

#define abs(a) ((a) >= 0 ? (a) : -(a))
	epsilon = 1.e-8;

	if (abs(aj-asum)/asum > epsilon) {
		printf ("Failed Validation on array a[]\n");
		printf ("        Expected  : %f \n",aj);
		printf ("        Observed  : %f \n",asum);
	}
	else if (abs(bj-bsum)/bsum > epsilon) {
		printf ("Failed Validation on array b[]\n");
		printf ("        Expected  : %f \n",bj);
		printf ("        Observed  : %f \n",bsum);
	}
	else if (abs(cj-csum)/csum > epsilon) {
		printf ("Failed Validation on array c[]\n");
		printf ("        Expected  : %f \n",cj);
		printf ("        Observed  : %f \n",csum);
	}
	else {
		printf ("Solution Validates\n");
	}
}
