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
#define _GNU_SOURCE
#include <stdio.h>
#include <pthread.h>
#include <sys/resource.h>
#if !defined(__APPLE__)
  #include <sys/prctl.h>
#endif
#include <sys/types.h>
#include <unistd.h>
#include "arts.h"
#include "artsAtomics.h"
#include "artsOutOfOrderList.h"
#include "artsDebug.h"

#define NUMTHREADS 2

typedef void (* callback )( void *, void *);

volatile unsigned int count = 0;
struct artsOutOfOrderList list;

void printer(void * id, void * arg)
{
    unsigned int * Id = (unsigned int *)id;
    printf("Print %u\n", *Id);
    artsFree(Id);
}

void * adder(void * data)
{
    for(unsigned int i=0; i<20; i++)
    {
        unsigned int * id = (unsigned int*) artsMalloc(sizeof(unsigned int));
        *id = i;
        while(!artsOutOfOrderListAddItem(&list, id))
        {
            PRINTF("RESET\n");
            artsOutOfOrderListReset(&list);
        }
//        printf("Print Added %u\n", i);
        artsAtomicAdd(&count, 1U);
    }
}

void * firer(void * data)
{
    while(count < 10);
    PRINTF("FIRE 1\n");
    artsOutOfOrderListFireCallback(&list, 0,  printer);
    while(count < 20);
    PRINTF("FIRE 2\n");
    artsOutOfOrderListFireCallback(&list, 0,  printer);
}

//pthread_create(&nodeThreadList[i], &attr, &artsThreadLoop, &mask[i]);
//pthread_join(nodeThreadList[i], NULL);

int main(void)
{
    list.head.next = 0;
    for(unsigned int i=0; i<OOPERELEMENT; i++)
    {
        list.head.array[i] = 0;
    }

    pthread_t thread[NUMTHREADS];
    for(unsigned int t=0; t<NUMTHREADS-1; t++)
    {
       if(pthread_create(&thread[t], NULL, adder, 0))
       {
          printf("ERROR on create\n");
          exit(-1);
       }
    }

    if(pthread_create(&thread[NUMTHREADS-1], NULL, firer, 0))
    {
       printf("ERROR on create\n");
       exit(-1);
    }

    void * status;
    for(unsigned int t=0; t<NUMTHREADS; t++)
    {
        if(pthread_join(thread[t], &status))
        {
           printf("ERROR on join\n");
           exit(-1);
        }
    }
    return 0;
}
