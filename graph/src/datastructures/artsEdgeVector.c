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
#include <assert.h>
#include <stdlib.h>
#include <inttypes.h>
#include "arts.h"
#include "artsEdgeVector.h"

#define INCREASE_SZ_BY 2

//comparators
int compareBySource(const void * e1, const void * e2) 
{
    edge_t  * pe1 = (edge_t*)e1;
    edge_t  * pe2 = (edge_t*)e2;

    if (pe1->source < pe2->source)
        return -1;
    else if (pe1->source == pe2->source)
        return 0;
    else
        return 1;
}

int compareBySourceAndTarget(const void * e1, const void * e2) 
{
    edge_t * pe1 = (edge_t*)e1;
    edge_t * pe2 = (edge_t*)e2;

    if (pe1->source < pe2->source)
        return -1;
    else if (pe1->source == pe2->source) 
    {
        if (pe1->target < pe2->target)
            return -1;
        else if (pe1->target == pe2->target) 
            return 0;
    else
        return 1;
    } 
    else
        return 1;
}

// end comparators

void initEdgeVector(artsEdgeVector *v, graph_sz_t initialSize) 
{
    v->edge_array = artsMalloc(initialSize * sizeof(edge_t));
    v->used = 0;
    v->size = initialSize;
}

void pushBackEdge(artsEdgeVector *v, vertex_t s, vertex_t t, edge_data_t d) 
{
    if (v->used == v->size) 
    {
        v->size *= INCREASE_SZ_BY;
        void* new = artsRealloc(v->edge_array, v->size * sizeof(edge_t));
        if (!new) 
        {
            PRINTF("[ERROR] Unable to reallocate memory. Cannot continue\n.");
            assert(false);
            return;
        } 
        v->edge_array = new;
    }

    v->edge_array[v->used].source = s;
    v->edge_array[v->used].target = t;
    v->edge_array[v->used++].data = d;
}

void printEdgeVector(const artsEdgeVector *v) 
{
    for (uint64_t i = 0; i < v->used; ++i) 
    {
        //PRINTF("(%" PRIu64 ", %" PRIu64 ", %" PRIu64 ")", v->edge_array[i].source, v->edge_array[i].target, v->edge_array[i].data);
        PRINTF("(%" PRIu64 ", %" PRIu64 ")", v->edge_array[i].source, 
        v->edge_array[i].target);
    }
}

void freeEdgeVector(artsEdgeVector *v) 
{
    artsFree(v->edge_array);
    v->edge_array = NULL;
    v->used = 0; 
    v->size = 0;
}

void sortBySource(artsEdgeVector *v) 
{
    qsort((void*)v->edge_array, v->used, sizeof(edge_t), compareBySource);
}

void sortBySourceAndTarget(artsEdgeVector *v) 
{
    qsort((void*)v->edge_array, v->used, sizeof(edge_t), compareBySourceAndTarget);
}
