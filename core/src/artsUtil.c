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
#include "arts.h"
#include "artsGuid.h"
#include "artsRemote.h"
#include "artsRemoteFunctions.h"
#include "artsGlobals.h"
#include "artsAtomics.h"
#include "artsCounter.h"
#include "artsRuntime.h"
#include "artsEdtFunctions.h"
#include "artsOutOfOrder.h"
#include "artsRouteTable.h"
#include "artsDebug.h"
#include "artsEdtFunctions.h"
#include "artsDbFunctions.h"
#include "artsIntrospection.h"
#include <stdarg.h>
#include <string.h>
#include <inttypes.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>

#define DPRINTF( ... )

extern __thread struct artsEdt * currentEdt;
extern unsigned int numNumaDomains;

artsGuid_t artsGetCurrentGuid()
{
    if(currentEdt)
    {
        return  currentEdt->currentEdt;
    }
    return NULL_GUID;
}

unsigned int artsGetCurrentNode()
{
    return artsGlobalRankId;
}

unsigned int artsGetTotalNodes()
{
   return artsGlobalRankCount;
}

unsigned int artsGetTotalWorkers()
{
    return artsNodeInfo.workerThreadCount;
}

unsigned int artsGetCurrentWorker()
{
    return artsThreadInfo.groupId;
}

unsigned int artsGetCurrentCluster()
{
    return artsThreadInfo.clusterId;
}

unsigned int artsGetTotalClusters()
{
    return numNumaDomains;
}

void artsStopLocalWorker()
{
    artsThreadInfo.alive = false;
}

void artsStopLocalNode()
{
    artsRuntimeStop();
}

uint64_t artsThreadSafeRandom()
{
    long int temp = jrand48(artsThreadInfo.drand_buf);
    return (uint64_t) temp;
}

unsigned int artsGetTotalGpus()
{
    return artsNodeInfo.gpu;
}

//char * artsParallelCreateMMAP( char * pathToFile)
//{
//    int fd, offset;
//    char *data;
//    struct stat sbuf;
//    if ((fd = open(pathToFile, O_RDONLY)) == -1)
//    {
//        perror("open");
//        exit(1);
//    }
//    if (stat(pathToFile, &sbuf) == -1)
//    {
//        perror("stat");
//        exit(1);
//    }
//
//    return mmap((caddr_t)0, sbuf.st_size, PROT_READ, MAP_SHARED, fd, 0);
//}
//
//void artsParallelStartRead(artsGuid_t * arrayOfGuids, unsigned int numberOfGuids,
//             artsRecordReader_t reader, char * pathToFile)
//{
//    unsigned int blocksPerNode = 0;
//    FILE * file = NULL;
//    for(unsigned int i=0; i<numberOfGuids; i++)
//    {
//        if(artsIsGuidLocalExt(arrayOfGuids[i]))
//        {
//            if(blocksPerNode%artsNodeInfo.workerThreadCount == artsThreadInfo.groupId)
//            {
//                if(file)
//                    fseek(file, 0, SEEK_SET);
//                else
//                    file = fopen(pathToFile, "r");
//
//                if(file)
//                {
//                    void * dbTempPtr = NULL;
//                    unsigned int dbSize = 0;
//                    reader(file, i, &dbSize, &dbTempPtr);
//                    if(dbTempPtr && dbSize)
//                    {
//                        ARTSSETMEMSHOTTYPE(artsDbMemorySize);
//                        struct ocrDb * ptr = ocrCalloc(sizeof(struct ocrDb) + dbSize);
//                        OCRSETMEMSHOTTYPE(ocrDefaultMemorySize);
//                        memcpy(ptr+1, dbTempPtr, dbSize);
//                        ocrFree(dbTempPtr);
//                        ocrDbCreateInternal((void*)ptr, dbSize, DB_PROP_NONE, NULL_GUID, sizeof(struct ocrDb)+dbSize, false);
//                        ocrRouteTableUpdateItem(ptr, arrayOfGuids[i], ocrGlobalRankId, 0);
//                        ocrRouteTableFireOO(arrayOfGuids[i], ocrOutOfOrderHandler);
//                    }
//                }
//                else
//                {
//                    PRINTF("Unable to open file %s\n", pathToFile);
//                }
//            }
//            blocksPerNode++;
//        }
//    }
//    if(file)
//        fclose(file);
//}
//
//void ocrParallelStartReadFixedSizeMMAP(ocrGuid_t * arrayOfGuids, unsigned int numberOfGuids,
//                              unsigned int size, ocrRecordReader_t reader, char * pathToFile)
//{
//    unsigned int blocksPerNode = 0;
//    for(unsigned int i=0; i<numberOfGuids; i++)
//    {
//        if(ocrIsGuidLocalExt(arrayOfGuids[i]))
//        {
//            if(blocksPerNode%ocrNodeInfo.workerThreadCount == ocrThreadInfo.groupId)
//            {
//                {
//                    OCRSETMEMSHOTTYPE(ocrDbMemorySize);
//                    struct ocrDb * ptr = ocrCalloc(sizeof(struct ocrDb) + size);
//                    OCRSETMEMSHOTTYPE(ocrDefaultMemorySize);
//                    void * temp = ptr + 1;
//                    unsigned int dbSize = size;
//                    reader(pathToFile, i, &dbSize, &temp);
//                    ocrDbCreateInternal((void*)ptr, dbSize, DB_PROP_NONE, NULL_GUID, sizeof(struct ocrDb) + size, false);
//                    ocrRouteTableUpdateItem(ptr, arrayOfGuids[i], ocrGlobalRankId, 0);
//                    ocrRouteTableFireOO(arrayOfGuids[i], ocrOutOfOrderHandler);
//                }
//            }
//            blocksPerNode++;
//        }
//    }
//}
//
//
//void ocrParallelStartReadFixedSizeLine(ocrGuid_t * arrayOfGuids, unsigned int numberOfGuids, unsigned int size, ocrRecordReaderLine_t reader, char * pathToFile, ocrMapper_t mapper, unsigned int mapperLength)
//{
//    unsigned int blocksPerNode = 0;
//    FILE * file = NULL;
//    unsigned int MAX_BUFFER_LENGTH = 128;
//    file = fopen(pathToFile, "r");
//    char buffer[MAX_BUFFER_LENGTH];
//    unsigned int localDbArrayLength=0;
//
//    for(unsigned int i=0; i<numberOfGuids; i++)
//    {
//        if(ocrIsGuidLocalExt(arrayOfGuids[i]))
//        {
//            if(blocksPerNode%ocrNodeInfo.workerThreadCount == ocrThreadInfo.groupId)
//                localDbArrayLength++;
//            blocksPerNode++;
//        }
//    }
//    struct ocrDb ** localDbArray = ocrMalloc(sizeof(struct ocrDb*) * localDbArrayLength);
//    unsigned int * localDbArrayMap = ocrMalloc(sizeof(unsigned int) * localDbArrayLength*mapperLength);
//    unsigned int localOffset=0;
//    blocksPerNode=0;
//    for(unsigned int i=0; i<numberOfGuids; i++)
//    {
//        if(ocrIsGuidLocalExt(arrayOfGuids[i]))
//        {
//            if(blocksPerNode%ocrNodeInfo.workerThreadCount == ocrThreadInfo.groupId)
//            {
//                OCRSETMEMSHOTTYPE(ocrDbMemorySize);
//                localDbArray[localOffset] = ocrCalloc(sizeof(struct ocrDb) + size);
//                for(int j=0; j< mapperLength; j++)
//                    localDbArrayMap[localOffset*mapperLength+j] = mapper(i, j);
//                OCRSETMEMSHOTTYPE(ocrDefaultMemorySize);
//                localOffset++;
//            }
//            blocksPerNode++;
//        }
//    }
//    bool firstRead=0;
//    if(file)
//    {
//        while (fgets(buffer, MAX_BUFFER_LENGTH, file))
//        {
//            localOffset=0;
//            blocksPerNode=0;
//            firstRead=true;
//            for(unsigned int i=0; i<localDbArrayLength; i++)
//            {
//                struct ocrDb * ptr = localDbArray[i];
//                void * temp = ptr + 1;
//                unsigned int dbSize = size;
//                if(firstRead)
//                    reader(buffer, i, &dbSize, &temp, localDbArrayMap+i*mapperLength);
//                else
//                    reader(NULL, i, &dbSize, &temp, localDbArrayMap+i*mapperLength);
//
//                firstRead = false;
//            }
//        }
//        fclose(file);
//        localOffset=0;
//        blocksPerNode=0;
//        for(unsigned int i=0; i<numberOfGuids; i++)
//        {
//            if(ocrIsGuidLocalExt(arrayOfGuids[i]))
//            {
//                if(blocksPerNode%ocrNodeInfo.workerThreadCount == ocrThreadInfo.groupId)
//                {
//                    struct ocrDb * ptr = localDbArray[localOffset];
//                    unsigned int dbSize = size;
//                    ocrDbCreateInternal((void*)ptr, dbSize, DB_PROP_NONE, NULL_GUID, sizeof(struct ocrDb) + size, false);
//                    ocrRouteTableUpdateItem(ptr, arrayOfGuids[i], ocrGlobalRankId, 0);
//                    ocrRouteTableFireOO(arrayOfGuids[i], ocrOutOfOrderHandler);
//                    localOffset++;
//                }
//                blocksPerNode++;
//            }
//        }
//    }
//    else
//    {
//        PRINTF("Unable to open file %s\n", pathToFile);
//    }
//    ocrFree(localDbArray);
//    ocrFree(localDbArrayMap);
//}
//
//void ocrParallelStartReadFixedSize(ocrGuid_t * arrayOfGuids, unsigned int numberOfGuids,
//                              unsigned int size, ocrRecordReader_t reader, char * pathToFile)
//{
//    unsigned int blocksPerNode = 0;
//    FILE * file = NULL;
//    for(unsigned int i=0; i<numberOfGuids; i++)
//    {
//        if(ocrIsGuidLocalExt(arrayOfGuids[i]))
//        {
//            if(blocksPerNode%ocrNodeInfo.workerThreadCount == ocrThreadInfo.groupId)
//            {
//                if(file)
//                    fseek(file, 0, SEEK_SET);
//                else
//                    file = fopen(pathToFile, "r");
//                if(file)
//                {
//                    OCRSETMEMSHOTTYPE(ocrDbMemorySize);
//                    struct ocrDb * ptr = ocrCalloc(sizeof(struct ocrDb) + size);
//                    OCRSETMEMSHOTTYPE(ocrDefaultMemorySize);
//                    void * temp = ptr + 1;
//                    unsigned int dbSize = size;
//                    reader(file, i, &dbSize, &temp);
//                    ocrDbCreateInternal((void*)ptr, dbSize, DB_PROP_NONE, NULL_GUID, sizeof(struct ocrDb) + size, false);
//                    ocrRouteTableUpdateItem(ptr, arrayOfGuids[i], ocrGlobalRankId, 0);
//                    ocrRouteTableFireOO(arrayOfGuids[i], ocrOutOfOrderHandler);
//                }
//                else
//                {
//                    PRINTF("Unable to open file %s\n", pathToFile);
//                }
//            }
//            blocksPerNode++;
//        }
//    }
//    if(file)
//        fclose(file);
//}
