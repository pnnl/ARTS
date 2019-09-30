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
#include "cublas_v2.h"
#include <cuda_runtime.h>
#include "arts.h"
#include "artsGpuRuntime.h"
#include "streamUtil.h"

void launch2KernelEdt(artsEdt_t funPtr, unsigned int tileSize, unsigned int totalSize, double scalar, artsGuidRange * aGuid, artsGuidRange * bGuid)
{ 
    unsigned int tiles = totalSize / tileSize;
    if(totalSize % tileSize) 
        tiles++;

    artsGuid_t toSignal = artsAllocateLocalBuffer(NULL, 0, tiles+1, NULL_GUID);

    unsigned int numThreads = (THREADSPERBLOCK < tileSize) ? THREADSPERBLOCK : tileSize;
    dim3 threads = {numThreads, 1, 1};
    dim3 grid = {tileSize/numThreads, 1, 1};

    uint64_t args[] = { 0, (uint64_t) scalar };
    if(scalar != 0)
    {
        for(unsigned int i=0; i<tiles; ++i)
        {
            args[0] = (i+1 < tiles) ? tileSize : totalSize - i*tileSize;
            artsGuid_t edtGuid = artsEdtCreateGpu(funPtr, artsGetCurrentNode(), 2, args, 2, grid, threads, toSignal, 0, NULL_GUID);
            artsSignalEdt(edtGuid, 0, artsGetGuid(aGuid, i));
            artsSignalEdt(edtGuid, 1, artsGetGuid(bGuid, i));
        }
    }
    else
    {
        for(unsigned int i=0; i<tiles; ++i)
        {
            args[0] = (i+1 < tiles) ? tileSize : totalSize - i*tileSize;
            artsGuid_t edtGuid = artsEdtCreateGpu(funPtr, artsGetCurrentNode(), 1, args, 2, grid, threads, toSignal, 0, NULL_GUID);
            artsSignalEdt(edtGuid, 0, artsGetGuid(aGuid, i));
            artsSignalEdt(edtGuid, 1, artsGetGuid(bGuid, i));
        }
    }
    artsBlockForBuffer(toSignal);
}

void launch3KernelEdt(artsEdt_t funPtr, unsigned int tileSize, unsigned int totalSize, double scalar, artsGuidRange * aGuid, artsGuidRange * bGuid, artsGuidRange * cGuid)
{
    unsigned int tiles = totalSize / tileSize;
    if(totalSize % tileSize) 
        tiles++;
    
    artsGuid_t toSignal = artsAllocateLocalBuffer(NULL, 0, tiles+1, NULL_GUID);

    unsigned int numThreads = (THREADSPERBLOCK < tileSize) ? THREADSPERBLOCK : tileSize;
    unsigned int remThreads = tileSize;
    dim3 threads = {numThreads, 1, 1};
    dim3 grid = {tileSize/numThreads, 1, 1};

    uint64_t args[] = { 0, (uint64_t) scalar };
    if(scalar != 0)
    {
        for(unsigned int i=0; i<tiles; ++i)
        {
            args[0] = (i+1 < tiles) ? tileSize : totalSize - i*tileSize;
            artsGuid_t edtGuid = artsEdtCreateGpu(funPtr, artsGetCurrentNode(), 2, args, 3, grid, threads, toSignal, 0, NULL_GUID);
            artsSignalEdt(edtGuid, 0, artsGetGuid(aGuid, i));
            artsSignalEdt(edtGuid, 1, artsGetGuid(bGuid, i));
            artsSignalEdt(edtGuid, 2, artsGetGuid(cGuid, i));
        }
    }
    else
    {
        for(unsigned int i=0; i<tiles; ++i)
        {
            args[0] = (i+1 < tiles) ? tileSize : totalSize - i*tileSize;
            artsGuid_t edtGuid = artsEdtCreateGpu(funPtr, artsGetCurrentNode(), 1, args, 3, grid, threads, toSignal, 0, NULL_GUID);
            artsSignalEdt(edtGuid, 0, artsGetGuid(aGuid, i));
            artsSignalEdt(edtGuid, 1, artsGetGuid(bGuid, i));
            artsSignalEdt(edtGuid, 2, artsGetGuid(cGuid, i));
        }
    }
    artsBlockForBuffer(toSignal);
}

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

double mysecond()
{
    struct timeval tp;
    struct timezone tzp;
    int i;

    i = gettimeofday(&tp,&tzp);
    return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}
void checkSTREAMresults(unsigned int tileSize, unsigned int totalSize, double **aTile, double **bTile, double **cTile)
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

    unsigned int numTiles = totalSize / tileSize;
    if(totalSize % tileSize)
        numTiles++;

    unsigned int temp = 0;
    for(unsigned int i=0; i<numTiles; i++)
    {
        unsigned int end = (i+1 < numTiles) ? tileSize : totalSize - i*tileSize;
        for(unsigned int j=0; j<end; j++)
        {
            asum+=aTile[i][j];
            bsum+=bTile[i][j];
            csum+=cTile[i][j];
            temp++;
        }
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