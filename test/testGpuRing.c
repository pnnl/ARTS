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
#include <stdio.h>
#include <stdlib.h>
#include <arts.h>
#include <artsArrayList.h>

#define CHECKCORRECT(x) {                                   \
  cudaError_t err;                                          \
  if( (err = (x)) != cudaSuccess )                          \
    PRINTF("FAILED %s: %s\n", #x, cudaGetErrorString(err)); \
}

#define MAX(a, b) ((a > b) ? a : b)

unsigned int * maxCycle = NULL;
artsArrayList * gpuList = NULL;
bool ** adjList = NULL;
unsigned int order = 0;

bool setMaxCycle(unsigned int * cycle, unsigned int cycleSize)
{
    uint64_t length = artsLengthArrayList(gpuList);
    if(cycle[0] == cycle[cycleSize-1])
    {
        for(unsigned int i=0; i<cycleSize-1; i++)
        {
            unsigned int found = 0;
            for(uint64_t j=0; j<length; j++)
            {
                int * temp = (int*) artsGetFromArrayList(gpuList, i);
                if(cycle[j] == *temp)
                    found++;
            }
            if(found != 1)
                return false;
        }
        
        maxCycle = (unsigned int*) artsCalloc(sizeof(unsigned int)*(cycleSize));
        for(unsigned int i=0; i<cycleSize; i++)
            maxCycle[i] = cycle[i];
        return true;
    }
    return false;
}

bool depthFirstRec(unsigned int vertex, unsigned int current, unsigned int cycleSize, unsigned int * cycle)
{
    cycle[current] = vertex;

    if(current + 1 == cycleSize)
        return setMaxCycle(cycle, cycleSize);
    else
    {
        for(unsigned int i=0; i<order; i++)
        {
            if(adjList[vertex][i])
                if(depthFirstRec(i, current+1, cycleSize, cycle))
                    return true;
        }
    }
    return false;
}

void depthFirst(unsigned int cycleSize)
{
    unsigned int * cycle = (unsigned int*) artsCalloc(sizeof(unsigned int)*(order+1));
    for(unsigned int i=0; i<order; i++)
    {
        if(depthFirstRec(i, 0, cycleSize, cycle))
        {
            for(unsigned int i=0; i<cycleSize; i++)
                printf("%u ", maxCycle[i]);
            printf("\n");
            return;
        }
    }
}

bool ** fullyConnect()
{
    bool ** adjList = (bool**) artsCalloc(sizeof(bool*)*order);
    for(unsigned int i=0; i<order; i++)
        adjList[i] = (bool*) artsCalloc(sizeof(bool)*order);
        
    uint64_t length = artsLengthArrayList(gpuList);
    for(uint64_t i=0; i<length; i++)
    {
        int * src = (int*) artsGetFromArrayList(gpuList, i);
        CHECKCORRECT(cudaSetDevice(*src));
        for(uint64_t j=0; j<length; j++)
        {
            if(i != j)
            {
                int hasAccess = 0;
                int * dst = (int*) artsGetFromArrayList(gpuList, j);
                CHECKCORRECT(cudaDeviceCanAccessPeer(&hasAccess, *src, *dst));
                if(hasAccess)
                {
                    adjList[*src][*dst] = 1;
                    CHECKCORRECT(cudaDeviceEnablePeerAccess(*dst, 0));
                }
            }
        }
    }
    return adjList;
}

void printAdjList()
{
    for(unsigned int i=0; i<order; i++)
    {
        PRINTF("%u: ", i);
        for(unsigned int j=0; j<order; j++)
        {
            printf("%u ", adjList[i][j]);
        }
        printf("\n");
    }
}

int main(int argc, char ** argv)
{
    if(argc < 4)
    {
        PRINTF("usage: bw gpu1 gpu2 ...\n");
        return 0;
    }

    unsigned int bw = atoi(argv[1]);
    gpuList = artsNewArrayList(sizeof(int), 8);
    
    for(unsigned int i=0; i<argc-2; i++)
    {
        unsigned int gpu = atoi(argv[2+i]);
        order = MAX(order, gpu);
        artsPushToArrayList(gpuList, &gpu);
    }
    unsigned int cycleSize = (unsigned int) artsLengthArrayList(gpuList) + 1;
    order++;

    adjList = fullyConnect();
    printAdjList();
    depthFirst(cycleSize);
    return 0;
}