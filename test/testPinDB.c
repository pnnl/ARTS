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

unsigned int node = 0;
artsGuid_t someDbGuid = NULL_GUID;

//This will hang but print a warning if the edt is not on the same node as the pinned DBs
void edtFunc(uint32_t paramc, uint64_t * paramv, uint32_t depc, artsEdtDep_t depv[])
{
    unsigned int * ptr = depv[0].ptr;
    unsigned int * ptr2 = depv[1].ptr;
    
    if(*ptr == 1234)
        PRINTF("artsDbCreate Check\n");
    else
        PRINTF("artsDBCreate Fail\n");
    
    if(*ptr2 == 9876)
        PRINTF("artsDbCreateWithGuid Check\n");
    else
        PRINTF("artsDbCreateWithGuid Fail\n");
    
    artsShutdown();
}

void initPerNode(unsigned int nodeId, int argc, char** argv)
{
    //This is the node we are going to pin to
    node = atoi(argv[1]);
    //Allocate some DB to test artsDbCreateWithGuid
    someDbGuid = artsReserveGuidRoute(ARTS_DB_PIN, node);
}

void initPerWorker(unsigned int nodeId, unsigned int workerId, int argc, char** argv)
{   
    
    if(!workerId && nodeId == node)
    {
        int * ptr = NULL;
        //Set pin to true to pin to node given by command line
        //It is pinned to the node creating the DB
        artsGuid_t dbGuid = artsDbCreate((void**)&ptr, sizeof(unsigned int), ARTS_DB_PIN);
        *ptr = 1234;
        
        //EDT is going to run on node given by command line
        artsGuid_t edtGuid = artsEdtCreate(edtFunc, node, 0, NULL, 2);
        
        //Put both signals up front forcing one to be out of order to test the OO code path
        artsSignalEdt(edtGuid, 0, dbGuid); //Note the mode
        artsSignalEdt(edtGuid, 1, someDbGuid); //Note the mode
        
        //This is the delayed DB 
        int * ptr2 = artsDbCreateWithGuid(someDbGuid, sizeof(unsigned int));
        *ptr2 = 9876;
    }
}

int main(int argc, char** argv)
{
    artsRT(argc, argv);
    return 0;
}
