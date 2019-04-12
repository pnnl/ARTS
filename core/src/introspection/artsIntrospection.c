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

#include "artsIntrospection.h"
#include "artsAtomics.h"
#include "artsGlobals.h"
#include "artsRemoteFunctions.h"
#include "artsDebug.h"
#include <sys/stat.h>

#define DPRINTF( ... )
#define NANOSECS 1000000000
#define localTimeStamp artsGetTimeStamp
#define globalTimeStamp artsGetTimeStamp

artsMETRICNAME;
uint64_t ** countWindow;
uint64_t ** timeWindow;
uint64_t ** maxTotal;

char * printTotalsToFile = NULL;
volatile unsigned int inspectorOn = 0;
artsInspector * inspector = NULL;
artsInspectorStats * stats = NULL;
artsInspectorShots * inspectorShots = NULL;
artsPacketInspector * packetInspector = NULL;

uint64_t artsGetInspectorTime(void)
{
    return inspector->startTimeStamp;
}

bool artsInternalInspecting(void)
{
    return (inspectorOn);
}

void artsInternalStartInspector(unsigned int startPoint)
{
    if(inspector && inspector->startPoint == startPoint)
    {
        inspectorOn = 1;
        inspector->startTimeStamp = globalTimeStamp();
//        PRINTF("TURNING INSPECTION ON Folder: %s %ld\n", printTotalsToFile, inspector->startTimeStamp);
    }
}

void artsInternalStopInspector(void)
{
    if(inspector)
    {
        inspectorOn = 0;
        inspector->endTimeStamp = globalTimeStamp();
    }
}

void printMetrics(void)
{
    for(unsigned int i=0; i<artsLastMetricType; i++)
    {
        PRINTF("%35s %ld %ld %ld %ld %ld %ld %ld %ld %ld\n",
                artsMetricName[i], 
                countWindow[i][0], countWindow[i][1], countWindow[i][2], 
                timeWindow[i][0], timeWindow[i][1], timeWindow[i][2],
                maxTotal[i][0], maxTotal[i][1], maxTotal[i][2]);
    }
}

void artsInternalInitIntrospector(struct artsConfig * config)
{
    char * inspFileName = config->introspectiveConf;
    char * inspOutputPrefix = config->introspectiveFolder;
    unsigned int traceLevel = config->introspectiveTraceLevel;
    unsigned int startPoint = config->introspectiveStartPoint;
    
    if(inspFileName)
    {
        DPRINTF("countWindow %u\n", sizeof(uint64_t*) * artsLastMetricType);
        countWindow = artsMalloc(sizeof(uint64_t*) * artsLastMetricType);
        DPRINTF("timeWindow %u\n", sizeof(uint64_t*) * artsLastMetricType);
        timeWindow = artsMalloc(sizeof(uint64_t*) * artsLastMetricType);
        DPRINTF("maxTotal %u\n", sizeof(uint64_t*) * artsLastMetricType);
        maxTotal = artsMalloc(sizeof(uint64_t*) * artsLastMetricType);
        
        for(unsigned int i=0; i<artsLastMetricType; i++)
        {
            DPRINTF("countWindow[%u] %u\n",i, sizeof(uint64_t) * artsMETRICLEVELS);
            countWindow[i] = artsMalloc(sizeof(uint64_t) * artsMETRICLEVELS);
            DPRINTF("timeWindow[%u] %u\n",i, sizeof(uint64_t) * artsMETRICLEVELS);
            timeWindow[i] = artsMalloc(sizeof(uint64_t) * artsMETRICLEVELS);
            DPRINTF("maxTotal[%u] %u\n",i, sizeof(uint64_t) * artsMETRICLEVELS);
            maxTotal[i] = artsMalloc(sizeof(uint64_t) * artsMETRICLEVELS);
            for(unsigned int j=0; j<artsMETRICLEVELS; j++)
            {
                countWindow[i][j] = -1;
                timeWindow[i][j] = -1;
                maxTotal[i][j] = -1;
            }
        }

        artsInternalReadInspectorConfigFile(inspFileName);
        if(!artsGlobalRankId)
            printMetrics();
        DPRINTF("inspector %u\n", sizeof(artsInspector));
        inspector = artsCalloc(sizeof(artsInspector));
        inspector->startPoint = startPoint;
        DPRINTF("inspector->coreMetric %u\n", sizeof(artsPerformanceUnit) * artsLastMetricType * artsNodeInfo.totalThreadCount);
        inspector->coreMetric = artsCalloc(sizeof(artsPerformanceUnit) * artsLastMetricType * artsNodeInfo.totalThreadCount);
        for(unsigned int i=0; i<artsNodeInfo.totalThreadCount; i++)
        {
            for(unsigned int j=0; j<artsLastMetricType; j++)
            {
                inspector->coreMetric[i*artsLastMetricType + j].maxTotal = maxTotal[j][0];
                inspector->coreMetric[i*artsLastMetricType + j].timeMethod = localTimeStamp;
            }
        }
        
        inspector->nodeMetric = artsCalloc(sizeof(artsPerformanceUnit) * artsLastMetricType);
        for(unsigned int j=0; j<artsLastMetricType; j++)
        {
            inspector->nodeMetric[j].maxTotal = maxTotal[j][1];
            inspector->nodeMetric[j].timeMethod = globalTimeStamp;
        }
        
        inspector->systemMetric = artsCalloc(sizeof(artsPerformanceUnit) * artsLastMetricType);
        for(unsigned int j=0; j<artsLastMetricType; j++)
        {
            inspector->systemMetric[j].maxTotal = maxTotal[j][2];
            inspector->systemMetric[j].timeMethod = globalTimeStamp;
        }
        
        DPRINTF("stats %u\n", sizeof(artsInspectorStats));
        stats = artsCalloc(sizeof(artsInspectorStats));
        DPRINTF("packetInspector %u\n", sizeof(artsPacketInspector));
        packetInspector = artsCalloc(sizeof(artsPacketInspector));
        packetInspector->minPacket = (uint64_t) -1;
        packetInspector->maxPacket = 0;
        packetInspector->intervalMin = (uint64_t) -1;
        packetInspector->intervalMax = 0;
        
        if(inspOutputPrefix && traceLevel < artsMETRICLEVELS)
        {
            if(traceLevel<=artsSystem)
            {
                inspectorShots = artsMalloc(sizeof(artsInspectorShots));
                DPRINTF("inspectorShots->coreMetric\n");
                inspectorShots->coreMetric = artsCalloc(sizeof(artsArrayList*) * artsLastMetricType * artsNodeInfo.totalThreadCount);
                for(unsigned int i = 0; i < artsLastMetricType * artsNodeInfo.totalThreadCount; i++)
                    inspectorShots->coreMetric[i] = artsNewArrayList(sizeof(artsMetricShot), 1024);
                DPRINTF("inspectorShots->nodeMetric\n");
                inspectorShots->nodeMetric = artsCalloc(sizeof(artsArrayList*) * artsLastMetricType);
                for(unsigned int i = 0; i < artsLastMetricType; i++)
                    inspectorShots->nodeMetric[i] = artsNewArrayList(sizeof(artsMetricShot), 1024);
                DPRINTF("inspectorShots->systemMetric\n");
                inspectorShots->systemMetric = artsCalloc(sizeof(artsArrayList*) * artsLastMetricType);
                for(unsigned int i = 0; i < artsLastMetricType; i++)
                    inspectorShots->systemMetric[i] = artsNewArrayList(sizeof(artsMetricShot), 1024);
                DPRINTF("inspectorShots->nodeLock %u\n", sizeof(unsigned int) * artsLastMetricType);
                inspectorShots->nodeLock = artsCalloc(sizeof(unsigned int) * artsLastMetricType);
                DPRINTF("inspectorShots->systemLock %u\n", sizeof(unsigned int) * artsLastMetricType);
                inspectorShots->systemLock = artsCalloc(sizeof(unsigned int) * artsLastMetricType);
                inspectorShots->prefix = inspOutputPrefix;
                inspectorShots->traceLevel = (artsMetricLevel) traceLevel;
            }
        }
        if(inspOutputPrefix && traceLevel <= artsMETRICLEVELS)
            printTotalsToFile = inspOutputPrefix;
    }
}

bool metricTryLock(artsMetricLevel level, artsPerformanceUnit * metric)
{
    if(level == artsThread)
        return true;
    
    unsigned int local;
    while(1)
    {
        local = artsAtomicCswap(&metric->lock, 0U, 1U);
        if(local != 2U)
            break;
    }
    return (local == 0U);
}

void metricLock(artsMetricLevel level, artsPerformanceUnit * metric)
{
    if(level == artsThread)
        return;
    while(!artsAtomicCswap(&metric->lock, 0U, 1U));
}

void metricUnlock(artsPerformanceUnit * metric)
{
    metric->lock=0U;
}

artsPerformanceUnit * getMetric(artsMetricType type, artsMetricLevel level)
{
    artsPerformanceUnit * metric = NULL;
    if(inspector)
    {
        switch(level)
        {
            case artsThread:
                metric = &inspector->coreMetric[artsThreadInfo.threadId*artsLastMetricType + type];
                break;
            case artsNode:
                metric = &inspector->nodeMetric[type];
                break;
            case artsSystem:
                metric = &inspector->systemMetric[type];
                break;
            default:
                metric = NULL;
                break;
        }
    }
    return metric;
}

uint64_t artsInternalGetPerformanceMetricTotal(artsMetricType type, artsMetricLevel level)
{
    
    artsPerformanceUnit * metric = getMetric(type, level);
    return (metric) ? metric->totalCount : 0;
}

double artsInternalGetPerformanceMetricRate(artsMetricType type, artsMetricLevel level, bool last)
{
    artsPerformanceUnit * metric = getMetric(type, level);
    if(metric)
    {
        uint64_t localWindowTimeStamp;
        uint64_t localWindowCountStamp;
        uint64_t localCurrentCountStamp;
        uint64_t localCurrentTimeStamp;
        
        metricLock(level, metric);
        if(last)
        {
            localWindowTimeStamp = metric->lastWindowTimeStamp;
            localWindowCountStamp = metric->lastWindowCountStamp;
            localCurrentCountStamp = metric->windowCountStamp;
            localCurrentTimeStamp = metric->windowTimeStamp;
            metricUnlock(metric);
        }
        else
        {
            localWindowTimeStamp = metric->windowTimeStamp;
            localWindowCountStamp = metric->windowCountStamp;
            metricUnlock(metric);
            localCurrentCountStamp = metric->totalCount;
            localCurrentTimeStamp = metric->timeMethod();
        }       

        if(localCurrentCountStamp && localCurrentTimeStamp)
        {
            double num = (double)(localCurrentCountStamp - localWindowCountStamp);
            double den = (double)(localCurrentTimeStamp - localWindowTimeStamp);
            PRINTF("%u %s %lf / %lf\n", level, artsMetricName[type], num, den);
            return  num / den / 1E9;
        }
    }
    return 0;
}

double artsInternalGetPerformanceMetricTotalRate(artsMetricType type, artsMetricLevel level)
{
    artsPerformanceUnit * metric = getMetric(type, level);
    if(metric)
    {
        double num = (double)metric->totalCount;
        double den = (double)metric->timeMethod() - inspector->startTimeStamp;
        PRINTF("%u %s %lf / %lf\n", level, artsMetricName[type], num, den);
        return  num / den;
    }
    return 0;
}

double artsMetricTest(artsMetricType type, artsMetricLevel level, uint64_t num)
{
    artsPerformanceUnit * metric = getMetric(type, level);
    if(metric && num)
    {
        double tot = (double)metric->totalCount;
        if(tot)
        {
            double dif = (double)metric->timeMethod() - inspector->startTimeStamp;
            double temp = ((double)num * dif) / tot;
//            PRINTF("%u %s (%lu * %lf / %lf\n", level, artsMetricName[type], num, dif, tot);
            return  temp;
        }
        return 100000;
    }
    return 0;
}

uint64_t artsInternalGetPerformanceMetricRateU64(artsMetricType type, artsMetricLevel level, bool last)
{
    artsPerformanceUnit * metric = getMetric(type, level);
    if(metric)
    {
        uint64_t localWindowTimeStamp;
        uint64_t localWindowCountStamp;
        uint64_t localCurrentCountStamp;
        uint64_t localCurrentTimeStamp;
        
        metricLock(level, metric);
        if(last)
        {
            localWindowTimeStamp = metric->lastWindowTimeStamp;
            localWindowCountStamp = metric->lastWindowCountStamp;
            localCurrentCountStamp = metric->windowCountStamp;
            localCurrentTimeStamp = metric->windowTimeStamp;
            metricUnlock(metric);
        }
        else
        {
            localWindowTimeStamp = metric->windowTimeStamp;
            localWindowCountStamp = metric->windowCountStamp;
            metricUnlock(metric);
            localCurrentCountStamp = metric->totalCount;
            localCurrentTimeStamp = metric->timeMethod();
        }       
       
        if(localCurrentCountStamp && localCurrentTimeStamp && localCurrentCountStamp > localWindowCountStamp)
        {
//            PRINTF("%lu / %lu = %lu\n", (localCurrentTimeStamp - localWindowTimeStamp), (localCurrentCountStamp - localWindowCountStamp), ((localCurrentTimeStamp - localWindowTimeStamp) / (localCurrentCountStamp - localWindowCountStamp)));
            return (localCurrentTimeStamp - localWindowTimeStamp) / (localCurrentCountStamp - localWindowCountStamp);
        }
    }
    return 0;
}

uint64_t artsInternalGetPerformanceMetricRateU64Diff(artsMetricType type, artsMetricLevel level, uint64_t * total)
{
    artsPerformanceUnit * metric = getMetric(type, level);
    if(metric)
    {
        metricLock(level, metric);
        uint64_t localWindowTimeStamp =  metric->windowTimeStamp;
        uint64_t localWindowCountStamp =  metric->windowCountStamp;
        uint64_t lastWindowTimeStamp =  metric->lastWindowTimeStamp;
        uint64_t lastWindowCountStamp =  metric->lastWindowCountStamp;
        metricUnlock(metric);
        
        uint64_t localCurrentCountStamp = metric->totalCount;
        uint64_t localCurrentTimeStamp = metric->timeMethod();
        *total = localCurrentCountStamp;
        if(localCurrentCountStamp)
        {
            uint64_t diff = localCurrentCountStamp - localWindowCountStamp;
            if(diff && localWindowTimeStamp)
            {
                return (localCurrentTimeStamp - localWindowTimeStamp) / diff;
            }
            else 
            {
                diff = localWindowCountStamp - lastWindowCountStamp;
                if(diff && localWindowCountStamp && lastWindowTimeStamp)
                {
                    return (localWindowCountStamp - lastWindowTimeStamp) / diff;
                }
            }
        }
        
    }
    return 0;
}

uint64_t artsInternalGetTotalMetricRateU64(artsMetricType type, artsMetricLevel level, uint64_t * total, uint64_t * timeStamp)
{
    artsPerformanceUnit * metric = getMetric(type, level);
    if(metric)
    {
        uint64_t localCurrentCountStamp = *total = metric->totalCount;
        uint64_t localCurrentTimeStamp = metric->timeMethod();
        *timeStamp = localCurrentTimeStamp;
        uint64_t startTime = metric->firstTimeStamp;
        if(startTime && localCurrentCountStamp)
        {
//            if(!artsGlobalRankId && !artsThreadInfo.threadId)
//                PRINTF("TIME: %lu COUNT: %lu RATE: %lu\n", (localCurrentTimeStamp - startTime), localCurrentCountStamp, (localCurrentTimeStamp - startTime) / localCurrentCountStamp);
            return (localCurrentTimeStamp - startTime) / localCurrentCountStamp;
        }
    }
    return 0;
}

void artsInternalHandleRemoteMetricUpdate(artsMetricType type, artsMetricLevel level, uint64_t toAdd, bool sub)
{
    artsPerformanceUnit * metric = getMetric(type, level);
    if(metric)
    {
        metricLock(level, metric);
        if(sub)
        {
            metric->windowCountStamp -= toAdd;
            metric->totalCount -= toAdd;
        }
        else
        {
            metric->windowCountStamp += toAdd;
            metric->totalCount += toAdd;
        }
        metricUnlock(metric);
        artsAtomicAddU64(&stats->remoteUpdates, 1);
    }
}

void internalUpdateMax(artsMetricLevel level, artsPerformanceUnit * metric, uint64_t total)
{
    uint64_t entry = metric->maxTotal;
    uint64_t localMax = metric->maxTotal;
    if(localMax>total)
        return;
    if(level==artsThread)
        metric->maxTotal = total;
    else
    {
        while(localMax < total)
        {
            localMax = artsAtomicCswapU64(&metric->maxTotal, localMax, total);
        }
    }
}

uint64_t internalObserveMax(artsMetricLevel level, artsPerformanceUnit * metric)
{
    uint64_t max = -1;
    if(metric->maxTotal!=-1)
    {
        if(level==artsThread)
        {
            max = metric->maxTotal;
            metric->maxTotal = metric->totalCount;
        }
        else
        {
            max = artsAtomicSwapU64(&metric->maxTotal, metric->totalCount);
        }
    }
    return max;
}

bool artsInternalSingleMetricUpdate(artsMetricType type, artsMetricLevel level, uint64_t *toAdd, bool *sub, artsPerformanceUnit * metric)
{
    if(!countWindow[type][level] || !timeWindow[type][level])
        return true;
    
    if(countWindow[type][level] == -1 && timeWindow[type][level] ==-1)
        return false;
    
    uint64_t totalStamp;
    if(*toAdd)
    {
        if(*sub)
        {
            //Subtraction assumes a zero total sum at the level it is evaluating (you can skip a level by setting a window to 0)
            //It is unclear what a negative result means...
            if(metric->totalCount < *toAdd)
            {
                PRINTF("Potential Inspection Underflow Detected! Level: %s Type: %s\n", level, artsMetricName[type]);
                artsDebugPrintStack();
            }
            totalStamp = (level==artsThread) ? metric->totalCount-=*toAdd : artsAtomicSubU64(&metric->totalCount, *toAdd);
    //        totalStamp = artsAtomicSubU64(&metric->totalCount, *toAdd);
        }
        else
        {
            totalStamp = (level==artsThread) ? metric->totalCount+=*toAdd : artsAtomicAddU64(&metric->totalCount, *toAdd);
    //        totalStamp = artsAtomicAddU64(&metric->totalCount, *toAdd);
        }
        internalUpdateMax(level, metric, totalStamp);
    }
    //Read local values to see if we need to update
    uint64_t localWindowTimeStamp = metric->windowTimeStamp;
    uint64_t localWindowCountStamp = metric->windowCountStamp;
    
    uint64_t timeStamp = metric->timeMethod();
    //Check if it is the first timeStamp
    if(!localWindowTimeStamp)
    {
        if(!artsAtomicCswapU64(&metric->windowTimeStamp, 0, timeStamp))
            metric->firstTimeStamp = metric->windowTimeStamp;
        return false;
    } 
    //Compute the difference in time and counts
    uint64_t elapsed = (timeStamp > localWindowTimeStamp) ? timeStamp - localWindowTimeStamp : 0;
    uint64_t last = (totalStamp > localWindowCountStamp) ? totalStamp - localWindowCountStamp : localWindowCountStamp - totalStamp;
        
    if(last >= countWindow[type][level] || elapsed >= timeWindow[type][level])
    {
        if(!metricTryLock(level, metric))
            return false;
        //Check and see if someone else already updated...
        if(localWindowTimeStamp!=metric->windowTimeStamp)
        {
            metricUnlock(metric);
            return false;
        }
        DPRINTF("Check metric %d %d %" PRIu64 " %" PRIu64 " vs %" PRIu64 " %" PRIu64 "\n", level, type, last, elapsed, countWindow[type][level], timeWindow[type][level]);
        DPRINTF("Updating metric %d %d %" PRIu64 " %" PRIu64 "\n", level, type, metric->windowCountStamp, metric->windowTimeStamp);
        //temp store the old
        metric->lastWindowTimeStamp = metric->windowTimeStamp;
        metric->lastWindowCountStamp = metric->windowCountStamp;
        metric->lastWindowMaxTotal = metric->windowMaxTotal;
        //updated to the latest
        metric->windowCountStamp = metric->totalCount;
        metric->windowMaxTotal = internalObserveMax(level, metric);
        metric->windowTimeStamp = metric->timeMethod(); //timeStamp;
        //determine the waterfall
        if(metric->windowCountStamp > metric->lastWindowCountStamp)
        {
            *toAdd = metric->windowCountStamp - metric->lastWindowCountStamp;
            *sub = false;
        }
        else
        {
            *toAdd = metric->lastWindowCountStamp - metric->windowCountStamp;
            *sub = true;
        }
        metricUnlock(metric);
        return true;
    }
    return false;
}

void takeRateShot(artsMetricType type, artsMetricLevel level, bool last)
{
    if(inspectorShots && level >= inspectorShots->traceLevel)
    {
        if(!countWindow[type][level] || !timeWindow[type][level])
            return;
        DPRINTF("TRACING LEVEL %d\n", level);
        
        int traceOn = artsThreadInfo.mallocTrace;
        artsThreadInfo.mallocTrace = 0;
        artsPerformanceUnit * metric = metric = getMetric(type, level);   
        if(metric)
        {
            artsArrayList * list = NULL;
            unsigned int * lock = NULL;
            switch(level)
            {
                case artsThread:
                    list = inspectorShots->coreMetric[artsThreadInfo.threadId*artsLastMetricType + type];
                    lock = NULL;
                    break;

                case artsNode:
                    list = inspectorShots->nodeMetric[type];
                    lock = &inspectorShots->nodeLock[type];
                    break;

                case artsSystem:
                    list = inspectorShots->systemMetric[type];
                    lock = &inspectorShots->systemLock[type];
                    break;

                default:
                    list = NULL;
                    lock = NULL;
                    break;
            }

            if(list)
            {
                if(lock) 
                {
                    unsigned int local;
                    while(1)
                    {
                        local = artsAtomicCswap(lock, 0U, 2U );
                        if(local == 2U)
                        {
                            artsMEMTRACEON;
                            return;
                        }
                        if(!local)
                            break;
                    }
                }
                artsMetricShot shot;
                if(last)
                {
                    metricLock(level, metric);
                    shot.maxTotal = metric->windowMaxTotal;
                    shot.windowTimeStamp = metric->lastWindowTimeStamp;
                    shot.windowCountStamp = metric->lastWindowCountStamp;
                    shot.currentTimeStamp = metric->windowTimeStamp;
                    shot.currentCountStamp = metric->windowCountStamp;
                    metricUnlock(metric);
                }
                else
                {
                    metricLock(level, metric);
                    shot.windowTimeStamp = metric->windowTimeStamp;
                    shot.windowCountStamp = metric->windowCountStamp;
                    metricUnlock(metric);
                    
                    shot.maxTotal = metric->maxTotal;
                    shot.currentCountStamp = metric->totalCount;
                    shot.currentTimeStamp = metric->timeMethod();
                }

//                if(shot.windowCountStamp && shot.windowTimeStamp)
                artsThreadInfo.mallocTrace = 0;
                artsPushToArrayList(list, &shot);
                artsThreadInfo.mallocTrace = 1;
                
                if(lock) 
                    *lock = 0U;
            }
        }
        artsThreadInfo.mallocTrace = traceOn;
    }
}

artsMetricLevel artsInternalUpdatePerformanceCoreMetric(unsigned int core, artsMetricType type, artsMetricLevel level, uint64_t toAdd, bool sub)
{
    if(type <= artsFirstMetricType || type >= artsLastMetricType)
    {
        PRINTF("Wrong Introspection Type %d\n", type);
        artsDebugGenerateSegFault();
    }
     
    artsMetricLevel updatedLevel = artsNoLevel;
    if(inspectorOn)
    {
        switch(level)
        {
            case artsThread:
                DPRINTF("Thread updated up to %d %" PRIu64 " %u %s\n", updatedLevel, toAdd, sub, artsMetricName[type]);
                if(!artsInternalSingleMetricUpdate(type, artsThread, &toAdd, &sub, &inspector->coreMetric[core*artsLastMetricType + type]))
                    break;
                takeRateShot(type, artsThread, true);
                updatedLevel = artsThread;

            case artsNode:
                DPRINTF("Node   updated up to %d %" PRIu64 " %u %s\n", updatedLevel, toAdd, sub, artsMetricName[type]);
                if(!artsInternalSingleMetricUpdate(type, artsNode, &toAdd, &sub, &inspector->nodeMetric[type]))
                    break;
                artsAtomicAddU64(&stats->nodeUpdates, 1);
                takeRateShot(type, artsNode, true);
                updatedLevel = artsNode;

            case artsSystem:
                DPRINTF("System updated up to %d %" PRIu64 " %u %s\n", updatedLevel, toAdd, sub, artsMetricName[type]);
                if(artsInternalSingleMetricUpdate(type, artsSystem, &toAdd, &sub, &inspector->systemMetric[type]))
                {
                    uint64_t timeToSend = inspector->systemMetric[type].timeMethod();
                    int traceOn = artsThreadInfo.mallocTrace;
                    artsThreadInfo.mallocTrace = 0;
                    for(unsigned int i=0; i<artsGlobalRankCount; i++)
                        if(i!=artsGlobalRankId)
                            artsRemoteMetricUpdate(i, type, level, timeToSend, toAdd, sub);
                    artsThreadInfo.mallocTrace = traceOn;
                    artsAtomicAddU64(&stats->systemUpdates, 1);
                    if(artsGlobalRankCount>1)
                        artsAtomicAddU64(&stats->systemMessages, artsGlobalRankCount-1);
                    takeRateShot(type, artsSystem, true);
                    updatedLevel = artsSystem;
                }
            default:
                break;
        }
    }
    return updatedLevel;
}

void artsInternalSetThreadPerformanceMetric(artsMetricType type, uint64_t value)
{   
    
    if(countWindow[type][artsThread] == -1 && timeWindow[type][artsThread] ==-1)
        return;
    
    artsPerformanceUnit * metric = getMetric(type, artsThread);
    if(metric)
    {
        bool shot = true;
        
        metric->lastWindowCountStamp = metric->windowCountStamp;
        metric->lastWindowTimeStamp = metric->windowTimeStamp;
        metric->lastWindowMaxTotal = metric->maxTotal;
        
        uint64_t localTime = metric->timeMethod();
        if(!metric->firstTimeStamp)
        {
            shot = false;
            metric->firstTimeStamp = localTime;
        }
        
        metric->totalCount = value;
        metric->windowCountStamp = value;
        metric->windowTimeStamp = localTime;
        if(metric->maxTotal < value)
            metric->maxTotal = value;
        if(shot) // && (elapsed >= timeWindow[type][artsThread] || last >= countWindow[type][artsThread]))
        {
//            PRINTF("TAKING SHOT %s %lu\n", artsMetricName[type], value);
            takeRateShot(type, artsThread, true); 
        }
    }   
}

artsMetricLevel artsInternalUpdatePerformanceMetric(artsMetricType type, artsMetricLevel level, uint64_t toAdd, bool sub)
{
    return artsInternalUpdatePerformanceCoreMetric(artsThreadInfo.threadId, type, level, toAdd, sub);
}

void metricPrint(artsMetricType type, artsMetricLevel level, artsMetricShot * shot, FILE * stream)
{
    fprintf(stream, "%d,%d,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n", type, level, shot->windowCountStamp, shot->windowTimeStamp, shot->currentCountStamp, shot->currentTimeStamp, shot->maxTotal);
}

void internalMetricWriteToFile(artsMetricType type, artsMetricLevel level, char * filename, artsArrayList * list)
{
    if(!artsLengthArrayList(list))
    {
        remove(filename);
        return;
    }
    
    FILE * fp = fopen(filename,"w");
    if(fp)
    {
        DPRINTF("FILE: %s\n", filename);
        uint64_t last = 0;        
        int traceOn = artsThreadInfo.mallocTrace;
        artsThreadInfo.mallocTrace = 0;
        artsArrayListIterator * iter = artsNewArrayListIterator(list);
        artsMetricShot * shot;
        while(artsArrayListHasNext(iter))
        {
            shot = artsArrayListNext(iter);
            metricPrint(type, level, shot, fp);
            if(shot->currentTimeStamp < last)
            {
                PRINTF("Out of order snap shot: %s %" PRIu64 "\n", filename, last);
                last = shot->currentTimeStamp;
            }
        }
        artsDeleteArrayListIterator(iter);
        if(last < inspector->endTimeStamp)
        {
            shot->windowCountStamp = shot->currentCountStamp;
            shot->windowTimeStamp = shot->currentTimeStamp;
            shot->currentTimeStamp = inspector->endTimeStamp;
            metricPrint(type, level, shot, fp);
        }
        artsThreadInfo.mallocTrace = traceOn;
        fclose(fp);
    }
    else
        PRINTF("Couldn't open %s\n", filename);
}

void artsInternalWriteMetricShotFile(unsigned int threadId, unsigned int nodeId)
{
    if(inspectorShots)
    {
        struct stat st = {0};
        if (stat(inspectorShots->prefix, &st) == -1)
            mkdir(inspectorShots->prefix, 0755);
        
        artsArrayList * list;
        char filename[1024];
        
        switch(inspectorShots->traceLevel)
        {
            case artsThread:
                for(unsigned int i=0; i<artsLastMetricType; i++)
                {
                    list = inspectorShots->coreMetric[threadId*artsLastMetricType + i];
                    sprintf(filename,"%s/%s_%s_%u_%u.ct", inspectorShots->prefix, "threadMetric", artsMetricName[i], nodeId, threadId);
                    internalMetricWriteToFile(i, artsThread, filename, list);
                }
                
            case artsNode:
                if(!threadId)
                {
                    for(unsigned int i=0; i<artsLastMetricType; i++)
                    {
                        list = inspectorShots->nodeMetric[i];
                        sprintf(filename,"%s/%s_%s_%u.ct", inspectorShots->prefix, "nodeMetric", artsMetricName[i], nodeId);
                        internalMetricWriteToFile(i, artsNode, filename, list);
                    }
                }
                
            case artsSystem:
                if(!threadId)// && !nodeId)
                {
                    for(unsigned int i=0; i<artsLastMetricType; i++)
                    {
                        list = inspectorShots->systemMetric[i];
                        sprintf(filename,"%s/%s_%s_%u.ct", inspectorShots->prefix, "systemMetric", artsMetricName[i], nodeId);
                        internalMetricWriteToFile(i, artsSystem, filename, list);
                    }
                }
                
            default:
                break;
        }        
    }
}

void artsInternalReadInspectorConfigFile(char * filename)
{
    char * line = NULL;
    size_t length = 0;
    FILE * fp = fopen(filename,"r");
    if(!fp)
        return;
    
    char temp[artsMAXMETRICNAME];
    
    while (getline(&line, &length, fp) != -1) 
    {
        DPRINTF("%s", line);
        if(line[0]!='#')
        {
            int paramRead = 0;
            sscanf(line, "%s", temp);
            size_t offset = strlen(temp);
            unsigned int metricIndex = -1;
            for(unsigned int i=0; i<artsLastMetricType; i++)
            {
                if(!strcmp(temp, artsMetricName[i]))
                {
                    metricIndex = i;
                    break;
                }
            }

            if(metricIndex >= 0 && metricIndex < artsLastMetricType)
            {
                for(unsigned int i=0; i<artsMETRICLEVELS; i++)
                {
                    while(line[offset] == ' ') offset++;
                    paramRead+=sscanf(&line[offset], "%" SCNu64 "", &countWindow[metricIndex][i]);
                    sscanf(&line[offset], "%s", temp);
                    offset += strlen(temp);
                    DPRINTF("temp: %s %u %u %u %" PRIu64 "\n", temp, metricIndex, i, offset, countWindow[metricIndex][i]);
                }

                for(unsigned int i=0; i<artsMETRICLEVELS; i++)
                {
                    while(line[offset] == ' ') offset++;
                    paramRead+=sscanf(&line[offset], "%" SCNu64 "", &timeWindow[metricIndex][i]);
                    sscanf(&line[offset], "%s", temp);
                    offset += strlen(temp);
                    DPRINTF("temp: %s %u %u %u %" PRIu64 "\n", temp, metricIndex, i, offset, timeWindow[metricIndex][i]);
                }
                
                for(unsigned int i=0; i<artsMETRICLEVELS; i++)
                {
                    while(line[offset] == ' ') offset++;
                    paramRead+=sscanf(&line[offset], "%" SCNu64 "", &maxTotal[metricIndex][i]);
                    sscanf(&line[offset], "%s", temp);
                    offset += strlen(temp);
                    DPRINTF("temp: %s %u %u %u %" PRIu64 "\n", temp, metricIndex, i, offset, maxTotal[metricIndex][i]);
                }
            }
            
            if(metricIndex < 0 || metricIndex >= artsLastMetricType || paramRead < artsMETRICLEVELS * 2)
            {
                PRINTF("FAILED to init metric %s\n", temp);
            }
        }
    }
    fclose(fp);
    
    if (line)
        free(line);
}

void internalPrintTotals(unsigned int nodeId)
{
    if(printTotalsToFile)
    {
        struct stat st = {0};
        if (stat(printTotalsToFile, &st) == -1)
            mkdir(printTotalsToFile, 0755);
        
        char filename[1024];
        sprintf(filename,"%s/finalCounts_%u.ct", printTotalsToFile, nodeId);
        FILE * fp = fopen(filename,"w");
        if(fp)
        {
            for(unsigned int i=0; i<artsLastMetricType; i++)
            {
                fprintf(fp, "%s, System, %" PRIu64 ", %" PRIu64 "\n", artsMetricName[i], inspector->systemMetric[i].totalCount, inspector->systemMetric[i].maxTotal);  
                fprintf(fp, "%s, Node, %" PRIu64 ", %" PRIu64 "\n", artsMetricName[i], inspector->nodeMetric[i].totalCount, inspector->nodeMetric[i].maxTotal); 
                for(unsigned int j=0; j<artsNodeInfo.totalThreadCount; j++)
                    fprintf(fp, "%s, Core_%u, %" PRIu64 ", %" PRIu64 "\n", artsMetricName[i], j, inspector->coreMetric[j*artsLastMetricType + i].totalCount, inspector->coreMetric[j*artsLastMetricType + i].maxTotal);
            }

            uint64_t counted = 0;
            uint64_t posNotCounted = 0;
            uint64_t negNotCounted = 0;
            artsPerformanceUnit * metric;
            for(unsigned int i=0; i<artsLastMetricType; i++)
            {
                counted = posNotCounted = negNotCounted = 0;
                for(unsigned int j=0; j<artsNodeInfo.totalThreadCount; j++)
                {   
                    metric = &inspector->coreMetric[j*artsLastMetricType + i];
                    counted += metric->windowCountStamp;
                    if(metric->totalCount > metric->windowCountStamp)
                        posNotCounted += (metric->totalCount - metric->windowCountStamp);
                    else
                        negNotCounted += (metric->windowCountStamp - metric->totalCount);
                }
                metric = &inspector->nodeMetric[i]; 
                uint64_t sum = metric->totalCount + posNotCounted - negNotCounted;
                if(metric->totalCount == counted && counted + posNotCounted >= negNotCounted)
                    fprintf(fp, "%s, Match, Sum, %" PRIu64 ", Total Counted, %" PRIu64 ", +Rem, %" PRIu64 ", -Rem, %" PRIu64 "\n", artsMetricName[i], sum, metric->totalCount, posNotCounted, negNotCounted);
                else
                    fprintf(fp, "%s, Error, Sum, %" PRIu64 ", Total Counted, %" PRIu64 ", +Rem, %" PRIu64 ", -Rem, %" PRIu64 "\n", artsMetricName[i], sum, metric->totalCount, posNotCounted, negNotCounted);
            }
            fprintf(fp, "Node Updates, %" PRIu64 ", System Updates, %" PRIu64 ", Remote Updates,  %" PRIu64 ", System Messages, %" PRIu64 "\n", stats->nodeUpdates, stats->systemUpdates, stats->remoteUpdates, stats->systemMessages);
        }
        else
            PRINTF("Couldn't open %s\n", filename);
    }
}

void printInspectorTime(void)
{
    printf("Stat 0 Node %u Start %" PRIu64 " End %" PRIu64 "\n", artsGlobalRankId, inspector->startTimeStamp, inspector->endTimeStamp);
}

void printInspectorStats(void)
{
    printf("Stat 3 Node %u Node_Updates %" PRIu64 " System_Updates %" PRIu64 " Remote_Updates  %" PRIu64 " System_Messages %" PRIu64 "\n", artsGlobalRankId, stats->nodeUpdates, stats->systemUpdates, stats->remoteUpdates, stats->systemMessages);
}

void printModelTotalMetrics(artsMetricLevel level)
{
    if(level==artsNode)
        printf("Stat 1 Node %u edt %" PRIu64 " edt_signal %" PRIu64 " event_signal %" PRIu64 " network_sent %" PRIu64 " network_recv %" PRIu64 " malloc %" PRIu64 " free %" PRIu64 "\n",
               artsGlobalRankId,
               artsInternalGetPerformanceMetricTotal(artsEdtThroughput, level),
               artsInternalGetPerformanceMetricTotal(artsEdtSignalThroughput, level),
               artsInternalGetPerformanceMetricTotal(artsEventSignalThroughput, level),
               artsInternalGetPerformanceMetricTotal(artsNetworkSendBW, level),
               artsInternalGetPerformanceMetricTotal(artsNetworkRecieveBW, level),
               artsInternalGetPerformanceMetricTotal(artsMallocBW, level),
               artsInternalGetPerformanceMetricTotal(artsFreeBW, level));
    else if(level==artsThread)
    {
        PRINTF("Stat 1 Thread %u edt %" PRIu64 " edt_signal %" PRIu64 " event_signal %" PRIu64 " network_sent %" PRIu64 " network_recv %" PRIu64 " malloc %" PRIu64 " free %" PRIu64 "\n",
               artsThreadInfo.threadId,
               artsInternalGetPerformanceMetricTotal(artsEdtThroughput, level),
               artsInternalGetPerformanceMetricTotal(artsEdtSignalThroughput, level),
               artsInternalGetPerformanceMetricTotal(artsEventSignalThroughput, level),
               artsInternalGetPerformanceMetricTotal(artsNetworkSendBW, level),
               artsInternalGetPerformanceMetricTotal(artsNetworkRecieveBW, level),
               artsInternalGetPerformanceMetricTotal(artsMallocBW, level),
               artsInternalGetPerformanceMetricTotal(artsFreeBW, level));
    }
}

static inline void readerLock(volatile unsigned int * reader, volatile unsigned int * writer)
{
    while(1)
    {
        while(*writer);
        artsAtomicFetchAdd(reader, 1U);
        if(*writer==0)
            break;
        artsAtomicSub(reader, 1U);
    }
}

static inline void readerUnlock(volatile unsigned int * reader)
{
    artsAtomicSub(reader, 1U);
}

static inline void writerLock(volatile unsigned int * reader, volatile unsigned int * writer)
{
    while(artsAtomicCswap(writer, 0U, 1U) == 0U);
    while(*reader);
    return;
}

static inline void writeUnlock(volatile unsigned int * writer)
{
    artsAtomicSwap(writer, 0U);
}


static inline void updatePacketExtreme(uint64_t val, volatile uint64_t * old, bool min)
{
    uint64_t local = *old;
    uint64_t res;
    if(min)
    {
        while(val < local)
        {
            res = artsAtomicCswapU64(old, local, val);
            if(res==local)
                break;
            else
                local = res;
        }
    }
    else
    {
        while(val > local)
        {
            res = artsAtomicCswapU64(old, local, val);
            if(res==local)
                break;
            else
                local = res;
        }
    }
}

void artsInternalUpdatePacketInfo(uint64_t bytes)
{
    if(packetInspector)
    {
        readerLock(&packetInspector->reader, &packetInspector->writer);
        artsAtomicAddU64(&packetInspector->totalBytes, bytes);
        artsAtomicAddU64(&packetInspector->totalPackets, 1U);
        updatePacketExtreme(bytes, &packetInspector->maxPacket, false);
        updatePacketExtreme(bytes, &packetInspector->minPacket, true);
        readerUnlock(&packetInspector->reader);

        readerLock(&packetInspector->intervalReader, &packetInspector->intervalWriter);
        artsAtomicAddU64(&packetInspector->intervalBytes, bytes);
        artsAtomicAddU64(&packetInspector->intervalPackets, 1U);
        updatePacketExtreme(bytes, &packetInspector->intervalMax, false);
        updatePacketExtreme(bytes, &packetInspector->intervalMin, true);
        readerUnlock(&packetInspector->intervalReader);
    }
}

void artsInternalPacketStats(uint64_t * totalBytes, uint64_t * totalPackets, uint64_t * minPacket, uint64_t * maxPacket)
{
    if(packetInspector)
    {
        writerLock(&packetInspector->reader, &packetInspector->writer);
        (*totalBytes) = packetInspector->totalBytes;
        (*totalPackets) = packetInspector->totalPackets;
        (*minPacket) = packetInspector->minPacket;
        (*maxPacket) = packetInspector->maxPacket;
        writeUnlock(&packetInspector->writer);
    }
}

void artsInternalIntervalPacketStats(uint64_t * totalBytes, uint64_t * totalPackets, uint64_t * minPacket, uint64_t * maxPacket)
{
    if(packetInspector)
    {
        writerLock(&packetInspector->intervalReader, &packetInspector->intervalWriter);
        (*totalBytes) = artsAtomicSwapU64(&packetInspector->totalBytes, 0);
        (*totalPackets) = artsAtomicSwapU64(&packetInspector->totalPackets, 0);
        (*minPacket) = artsAtomicSwapU64(&packetInspector->minPacket, 0);
        (*maxPacket) = artsAtomicSwapU64(&packetInspector->maxPacket, 0);
        writeUnlock(&packetInspector->intervalWriter);
    }
}
