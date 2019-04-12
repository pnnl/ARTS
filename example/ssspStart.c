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

#define NUMVERT 100
#define NUMEDGE 10

artsGuid_t * graphGuid;

typedef struct
{
    unsigned int target;
    unsigned int weight;
} edge;

typedef struct
{
    unsigned int index;
    unsigned int distance;
    edge edgeList[NUMEDGE];
} vertex;

artsGuid_t indexToGuid(unsigned int index)
{
    return graphGuid[index/NUMVERT];
}

unsigned int globalIndexToOffset(unsigned int index)
{
    return index % NUMVERT;
}

void visit(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    unsigned int index = paramv[0];
    uint64_t * neighbors = &paramv[1];
    vertex * home = depv[0].ptr;
    unsigned int * distance = &home[index].distance;
    
    for(unsigned int i=1; i<paramc; i++)
    {
        vertex * current = depv[i].ptr;
        unsigned int offset = globalIndexToOffset(paramc[i]);
        unsigned int temp = current[offset].distance + ;
    }
    
    
}

void getDistances(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    vertex * v = depv[0].ptr;
    for(unsigned int i=0; i<NUMVERT; i++)
    {
        uint64_t edges[NUMEDGE+1];
        edges[0] = i;
        for(unsigned int j=0; j<NUMEDGE; j++)
            edges[j+1] = v[i].edgeList[j].target;
        
        artsGuid_t guid = artsEdtCreate(visit, 0, NUMEDGE+1, edges, NUMEDGE+1, NULL);
        
        for(unsigned int j=0; j<NUMEDGE; j++)
            artsSignalEdt(guid, indexToGuid(v[i].edgeList[j].target), j+1, DB_MODE_NON_COHERENT_READ);
        
        artsSignalEdt(guid, indexToGuid(v[i].index), 0, DB_MODE_NON_COHERENT_READ);
    }
    artsShutdown();
}

void shutDown(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    for(unsigned int i=0; i<depc; i++)
    {
        vertex * v = depv[i].ptr;
        for(unsigned int j=0; j<NUMVERT; j++)
        {
            for(unsigned int k=0; k<NUMEDGE; k++)
            {
                PRINTF("Index: %u Dist: %u Edge Target: %u Edge Weight: %u guid: %lu\n", v[j].index, v[j].distance, v[j].edgeList[k].target, v[j].edgeList[k].weight, indexToGuid(v[j].index));
            }
        }
    }
    artsShutdown();
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    srand(42);
    graphGuid = artsMalloc(sizeof(artsGuid_t)*artsGetTotalNodes());
    for(unsigned int i=0; i<artsGetTotalNodes(); i++)
    {
        graphGuid[i] = artsReserveGuidRoute(ARTS_DB_READ, i % artsGetTotalNodes());
    }
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{   
    if(!workerId)
    {
        for(unsigned int i=0; i<artsGetTotalNodes(); i++)
        {
            if(artsIsGuidLocal(graphGuid[i]))
            {
                PRINTF("SIZE ALLOC: %u vert %u edge %u\n", sizeof(vertex) * NUMVERT, sizeof(vertex), sizeof(edge));
                vertex * v = artsDbCreateWithGuid(graphGuid[i], sizeof(vertex) * NUMVERT);
                for(unsigned int j=0; j<NUMVERT; j++)
                {
                    
                    v[j].index = i*NUMVERT+j;
                    v[j].distance = 0;
                    for(unsigned int k=0; k<NUMEDGE; k++)
                    {
                        v[j].edgeList[k].target = rand() % (NUMVERT*artsGetTotalNodes());
                        v[j].edgeList[k].weight = (rand() % 25) + 1;
                        
                    }
                }
            }
        }
    }
    
    if(!nodeId && !workerId)
    {
        artsGuid_t guid = artsEdtCreate(shutDown, 0, 0, NULL, artsGetTotalNodes(), NULL);
        for(unsigned int i=0; i<artsGetTotalNodes(); i++)
        {
            artsSignalEdt(guid, graphGuid[i], i);
        }
    }
    
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}
