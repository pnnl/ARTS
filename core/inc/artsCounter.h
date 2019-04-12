/******************************************************************************
 * ** This material was prepared as an account of work sponsored by an agency   **
 * ** of the United States Government.  Neither the United States Government    **
 * ** nor the United States Department of Energy, nor Battelle, nor any of      **
 * ** their employees, nor any jurisdiction or organization that has cooperated **
 * ** in the development of these materials, makes any warranty, express or     **
 * ** implied, or assumes any legal liability or responsibility for the accuracy,* 
 * ** completeness, or usefulness or any information, apparatus, product,       **
 * ** software, or process disclosed, or represents that its use would not      **
 * ** infringe privately owned rights.                                          **
 * **                                                                           **
 * ** Reference herein to any specific commercial product, process, or service  **
 * ** by trade name, trademark, manufacturer, or otherwise does not necessarily **
 * ** constitute or imply its endorsement, recommendation, or favoring by the   **
 * ** United States Government or any agency thereof, or Battelle Memorial      **
 * ** Institute. The views and opinions of authors expressed herein do not      **
 * ** necessarily state or reflect those of the United States Government or     **
 * ** any agency thereof.                                                       **
 * **                                                                           **
 * **                      PACIFIC NORTHWEST NATIONAL LABORATORY                **
 * **                                  operated by                              **
 * **                                    BATTELLE                               **
 * **                                     for the                               **
 * **                      UNITED STATES DEPARTMENT OF ENERGY                   **
 * **                         under Contract DE-AC05-76RL01830                  **
 * **                                                                           **
 * ** Copyright 2019 Battelle Memorial Institute                                **
 * ** Licensed under the Apache License, Version 2.0 (the "License");           **
 * ** you may not use this file except in compliance with the License.          **
 * ** You may obtain a copy of the License at                                   **
 * **                                                                           **
 * **    https://www.apache.org/licenses/LICENSE-2.0                            **
 * **                                                                           **
 * ** Unless required by applicable law or agreed to in writing, software       **
 * ** distributed under the License is distributed on an "AS IS" BASIS, WITHOUT **
 * ** WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the  **
 * ** License for the specific language governing permissions and limitations   **
 * ******************************************************************************/
#ifndef ARTSCOUNTER_H
#define	ARTSCOUNTER_H
#ifdef __cplusplus
extern "C" {
#endif
    
#include "arts.h"
    
#ifdef JUSTCOUNT
#define COUNTERTIMESTAMP 0
#elif defined(COUNT) || defined(MODELCOUNT)
#define COUNTERTIMESTAMP artsGetTimeStamp()
#else
#define COUNTERTIMESTAMP 0
#endif
    
#ifdef MODELCOUNT
    
#define COUNT_edtCounter(x) x
#define COUNT_sleepCounter(x) x
#define COUNT_totalCounter(x) x
#define COUNT_signalEdtCounter(x) x
#define COUNT_signalEventCounter(x) x 
#define COUNT_mallocMemory(x) x
#define COUNT_callocMemory(x) x
#define COUNT_freeMemory(x) x
#define COUNT_edtFree(x) x
#define COUNT_emptyTime(x) x


#define CAT(a, ...) PRIMITIVE_CAT(a, __VA_ARGS__)
#define PRIMITIVE_CAT(a, ...) a ## __VA_ARGS__
    
#define IIF(c) PRIMITIVE_CAT(IIF_, c)
#define IIF_0(t, ...) __VA_ARGS__
#define IIF_1(t, ...) t  
    
#define COMPL(b) PRIMITIVE_CAT(COMPL_, b)
#define COMPL_0 1
#define COMPL_1 0
    
#define BITAND(x) PRIMITIVE_CAT(BITAND_, x)
#define BITAND_0(y) 0
#define BITAND_1(y) y
    
#define CHECK_N(x, n, ...) n
#define CHECK(...) CHECK_N(__VA_ARGS__, 0,)
#define PROBE(x) x, 1
    
#define NOT(x) CHECK(PRIMITIVE_CAT(NOT_, x))
#define NOT_0 PROBE(~)
    
#define BOOL(x) COMPL(NOT(x))
#define IF(c) IIF(BOOL(c))
#define EAT(...)
  
#define IS_PAREN(x) CHECK(IS_PAREN_PROBE x)
#define IS_PAREN_PROBE(...) PROBE(~)

#define COUNTER_ON(x) IS_PAREN( CAT(COUNT_, x) (()) )
    
#define INITCOUNTERLIST(threadId, nodeId, folder, startPoint) artsInitCounterList(threadId, nodeId, folder, startPoint)
#define ARTSSTARTCOUNTING(startPoint) artsStartCounters(startPoint)
#define ARTSCOUNTERSON() artsCountersOn()
#define ARTSCOUNTERSOFF() artsEndCounters()
#define ARTSCREATECOUNTER(counter) artsCreateCounter(threadId, nodeId, counterName)
#define ARTSGETCOUNTER(counter) artsGetCounter(counter)
    
#define ARTSCOUNTERINCREMENT(counter)         IF(COUNTER_ON(counter)) ( artsCounterIncrement,         EAT ) (artsGetCounter(counter))
#define ARTSCOUNTERINCREMENTBY(counter, num)  IF(COUNTER_ON(counter)) ( artsCounterIncrementBy,       EAT ) (artsGetCounter(counter), num)
#define ARTSCOUNTERTIMERSTART(counter)        IF(COUNTER_ON(counter)) ( artsCounterTimerStart,        EAT ) (artsGetCounter(counter))
#define ARTSCOUNTERTIMERENDINCREMENT(counter) IF(COUNTER_ON(counter)) ( artsCounterTimerEndIncrement, EAT ) (artsGetCounter(counter)) 
#define ARTSCOUNTERTIMERENDINCREMENTBY(counter, num) IF(COUNTER_ON(counter)) ( artsCounterTimerEndIncrementBy, EAT ) (artsGetCounter(counter), num) 
#define ARTSCOUNTERADDTIME(counter, time)     IF(COUNTER_ON(counter)) ( artsCounterAddTime,           EAT ) (artsGetCounter(counter), time)
#define ARTSCOUNTERSETENDTOTIME(counter)      IF(COUNTER_ON(counter)) ( artsCounterSetEndTime,        EAT ) (artsGetCounter(counter), artsExtGetTimeStamp())
#define ARTSCOUNTERADDENDTIME(counter)        IF(COUNTER_ON(counter)) ( artsCounterAddEndTime,        EAT ) (artsGetCounter(counter))
#define ARTSCOUNTERNONEMPTY(counter)          IF(COUNTER_ON(counter)) ( artsCounterNonEmtpy,          EAT ) (artsGetCounter(counter))

#define INTERNAL_ARTSEDTCOUNTERTIMERSTART(counter) artsCounterTimerStart(artsGetCounter((artsThreadInfo.currentEdtGuid) ? (counter ## On) : (counter)))
#define ARTSEDTCOUNTERTIMERSTART(counter) IF(COUNTER_ON(counter)) ( INTERNAL_ARTSEDTCOUNTERTIMERSTART, EAT)(counter)
    
#define INTERNAL_ARTSEDTCOUNTERTIMERENDINCREMENT(counter) artsCounterTimerEndIncrement(artsGetCounter((artsThreadInfo.currentEdtGuid) ? (counter ## On) : (counter)))    
#define ARTSEDTCOUNTERTIMERENDINCREMENT(counter) IF(COUNTER_ON(counter)) ( INTERNAL_ARTSEDTCOUNTERTIMERENDINCREMENT, EAT)(counter)

#define ARTSRESETCOUNTER(counter)
#define ARTSCOUNTERTIMERENDOVERWRITE(counter)
#define ARTSCOUNTERPRINT(counter, stream)
#define ARTSCOUNTERSETSTARTTIME(counter, start)
#define ARTSCOUNTERSETENDTIME(counter, end)
#define ARTSCOUNTERGETSTARTTIME(counter)
#define ARTSCOUNTERGETENDTIME(counter)
#define ARTSUSERGETCOUNTER(counter)
#define ARTSEDTCOUNTERINCREMENT(counter)
#define ARTSEDTCOUNTERINCREMENTBY(counter, num)
#define ARTSUSERCOUNTERINCREMENT(counter)
#define ARTSUSERCOUNTERINCREMENTBY(counter, num)
#define ARTSUSERCOUNTERTIMERSTART(counter) artsCounterTimerStart(artsUserGetCounter(counter, #counter))
#define ARTSUSERCOUNTERTIMERENDINCREMENT(counter) artsCounterTimerEndIncrement(artsUserGetCounter(counter, #counter))
#define USERCOUNTERS(first, ...) enum __userCounters{ first=lastCounter, __VA_ARGS__}
#define USERCOUNTERINIT(counter) artsUserGetCounter(counter, #counter)

#elif COUNT    
#define INITCOUNTERLIST(threadId, nodeId, folder, startPoint) artsInitCounterList(threadId, nodeId, folder, startPoint)
#define ARTSSTARTCOUNTING(startPoint) artsStartCounters(startPoint)
#define ARTSCOUNTERSON() artsCountersOn()
#define ARTSCOUNTERSOFF() artsEndCounters()
#define ARTSCREATECOUNTER(counter) artsCreateCounter(threadId, nodeId, counterName)
#define ARTSGETCOUNTER(counter) artsGetCounter(counter)
#define ARTSRESETCOUNTER(counter) artsResetCounter(artsGetCounter(counter))
#define ARTSCOUNTERINCREMENT(counter) artsCounterIncrement(artsGetCounter(counter))
#define ARTSCOUNTERINCREMENTBY(counter, num) artsCounterIncrementBy(artsGetCounter(counter), num)
#define ARTSCOUNTERTIMERSTART(counter) artsCounterTimerStart(artsGetCounter(counter))
#define ARTSCOUNTERTIMERENDINCREMENT(counter) artsCounterTimerEndIncrement(artsGetCounter(counter))
#define ARTSCOUNTERTIMERENDINCREMENTBY(counter, num) artsCounterTimerEndIncrementBy(artsGetCounter(counter), num)
#define ARTSCOUNTERTIMERENDOVERWRITE(counter) artsCounterTimerEndOverwrite(artsGetCounter(counter))
#define ARTSCOUNTERPRINT(counter, stream) artsCounterPrint(artsGetCounter(counter), stream)
#define ARTSCOUNTERSETSTARTTIME(counter, start) artsCounterSetStartTime (artsGetCounter(counter), start)
#define ARTSCOUNTERSETENDTIME(counter, end) artsCounterSetEndTime(artsGetCounter(counter), end)
#define ARTSCOUNTERGETSTARTTIME(counter) artsCounterGetStartTime(artsGetCounter(counter))
#define ARTSCOUNTERGETENDTIME(counter) artsCounterGetEndTime(artsGetCounter(counter))  
#define ARTSCOUNTERADDTIME(counter, time) artsCounterAddTime(artsGetCounter(counter), time)
#define ARTSUSERGETCOUNTER(counter) artsUserGetCounter(counter, #counter)
#define ARTSCOUNTERSETENDTOTIME(counter) artsCounterSetEndTime(artsGetCounter(counter), artsExtGetTimeStamp())
#define ARTSCOUNTERADDENDTIME(counter) artsCounterAddEndTime(artsGetCounter(counter))
#define ARTSCOUNTERNONEMPTY(counter) artsCounterNonEmtpy(artsGetCounter(counter))
    
#define ARTSEDTCOUNTERINCREMENT(counter) artsCounterIncrement(artsGetCounter((artsThreadInfo.currentEdtGuid) ? (counter ## On) : (counter)))
#define ARTSEDTCOUNTERINCREMENTBY(counter, num) artsCounterIncrementBy(artsGetCounter((artsThreadInfo.currentEdtGuid) ? (counter ## On) : (counter)), num)
#define ARTSEDTCOUNTERTIMERSTART(counter) artsCounterTimerStart(artsGetCounter((artsThreadInfo.currentEdtGuid) ? (counter ## On) : (counter)))
#define ARTSEDTCOUNTERTIMERENDINCREMENT(counter) artsCounterTimerEndIncrement(artsGetCounter((artsThreadInfo.currentEdtGuid) ? (counter ## On) : (counter)))    
    
#define ARTSUSERCOUNTERINCREMENT(counter) artsCounterIncrement(artsUserGetCounter(counter, #counter))
#define ARTSUSERCOUNTERINCREMENTBY(counter, num) artsCounterIncrementBy(artsUserGetCounter(counter, #counter), num)
#define ARTSUSERCOUNTERTIMERSTART(counter) artsCounterTimerStart(artsUserGetCounter(counter, #counter))
#define ARTSUSERCOUNTERTIMERENDINCREMENT(counter) artsCounterTimerEndIncrement(artsUserGetCounter(counter, #counter))
#define USERCOUNTERINIT(counter) artsUserGetCounter(counter, #counter)
#define USERCOUNTERS(first, ...) enum __userCounters{ first=lastCounter, __VA_ARGS__}

#else

#define INITCOUNTERLIST(threadId, nodeId, folder, startPoint)
#define ARTSSTARTCOUNTING(startPoint)
//This seems backwards but it is for artsMallocFOA
#define ARTSCOUNTERSON() 1
#define ARTSCOUNTERSOFF()
#define ARTSCREATECOUNTER(counter)
#define ARTSGETCOUNTER(counter)
#define ARTSRESETCOUNTER(counter)
#define ARTSCOUNTERINCREMENT(counter)
#define ARTSCOUNTERINCREMENTBY(counter, num)
#define ARTSCOUNTERTIMERSTART(counter)
#define ARTSCOUNTERTIMERENDINCREMENT(counter)
#define ARTSCOUNTERTIMERENDINCREMENTBY(counter, num)
#define ARTSCOUNTERTIMERENDOVERWRITE(counter)
#define ARTSCOUNTERPRINT(counter, stream)
#define ARTSCOUNTERSETSTARTTIME(counter, start)
#define ARTSCOUNTERSETENDTIME(counter, end)
#define ARTSCOUNTERGETSTARTTIME(counter)
#define ARTSCOUNTERGETENDTIME(counter)  
#define ARTSCOUNTERADDTIME(counter, time)
#define ARTSUSERGETCOUNTER(counter)
#define ARTSEDTCOUNTERINCREMENT(counter)
#define ARTSEDTCOUNTERINCREMENTBY(counter, num)
#define ARTSEDTCOUNTERTIMERSTART(counter)
#define ARTSEDTCOUNTERTIMERENDINCREMENT(counter)
#define ARTSUSERCOUNTERINCREMENT(counter)
#define ARTSUSERCOUNTERINCREMENTBY(counter, num)
#define ARTSUSERCOUNTERTIMERSTART(counter)
#define ARTSUSERCOUNTERTIMERENDINCREMENT(counter)
#define USERCOUNTERINIT(counter)
#define USERCOUNTERS(first, ...)
#define ARTSCOUNTERSETENDTOTIME(counter)
#define ARTSCOUNTERADDENDTIME(counter)
#define ARTSCOUNTERNONEMPTY(counter)

#endif
    
#include "artsArrayList.h"
    
#define COUNTERNAMES const char * const __counterName[] = { \
"edtCounter", \
"sleepCounter", \
"totalCounter", \
"signalEventCounter", \
"signalEventCounterOn", \
"signalEdtCounter", \
"signalEdtCounterOn", \
"edtCreateCounter", \
"edtCreateCounterOn", \
"eventCreateCounter", \
"eventCreateCounterOn", \
"dbCreateCounter", \
"dbCreateCounterOn", \
"mallocMemory", \
"mallocMemoryOn", \
"callocMemory", \
"callocMemoryOn", \
"freeMemory", \
"freeMemoryOn", \
"guidAllocCounter", \
"guidAllocCounterOn", \
"guidLookupCounter", \
"guidLookupCounterOn", \
"getDbCounter", \
"getDbCounterOn", \
"putDbCounter", \
"putDbCounterOn", \
"contextSwitch", \
"yield" \
}
    
#define GETCOUNTERNAME(x) __counterName[x] 
#define COUNTERARRAYBLOCKSIZE 128
#define COUNTERPREFIXSIZE 1024
#define COUNTERPREFIX "counters"
#define FIRSTCOUNTER edtCounter
#define LASTCOUNTER lastCounter
        
    enum artsCounterType { 
        edtCounter=0, 
        sleepCounter, 
        totalCounter,
        signalEventCounter,
        signalEventCounterOn,
        signalEdtCounter,
        signalEdtCounterOn,
        edtCreateCounter,
        edtCreateCounterOn,
        eventCreateCounter,
        eventCreateCounterOn,
        dbCreateCounter,
        dbCreateCounterOn,
        mallocMemory,
        mallocMemoryOn,
        callocMemory,
        callocMemoryOn,
        freeMemory,
        freeMemoryOn,
        guidAllocCounter,
        guidAllocCounterOn,
        guidLookupCounter,
        guidLookupCounterOn,
        getDbCounter,
        getDbCounterOn,
        putDbCounter,
        putDbCounterOn,
        contextSwitch,
        yield,
        lastCounter
    };
    typedef enum artsCounterType artsCounterType;
    
    typedef struct {
        unsigned int threadId;
        unsigned int nodeId;
        const char * name;
        uint64_t count;
        uint64_t totalTime;
        uint64_t startTime;
        uint64_t endTime;
    } artsCounter;
    
    void artsInitCounterList(unsigned int threadId, unsigned int nodeId, char * folder, unsigned int startPoint);
    void artsStartCounters(unsigned int startPoint);
    void artsEndCounters();
    unsigned int artsCountersOn();
    artsCounter * artsCreateCounter(unsigned int threadId, unsigned int nodeId, const char * counterName);
    artsCounter * artsGetCounter(artsCounterType counter);
    artsCounter * artsUserGetCounter(unsigned int index, char * name);
    void artsResetCounter(artsCounter * counter);
    void artsCounterIncrement(artsCounter * counter);
    void artsCounterIncrementBy(artsCounter * counter, uint64_t num);
    void artsCounterTimerStart(artsCounter * counter);
    void artsCounterTimerEndIncrement(artsCounter * counter);
    void artsCounterTimerEndIncrementBy(artsCounter * counter, uint64_t num);
    void artsCounterTimerEndOverwrite(artsCounter * counter);
    void artsCounterPrint(artsCounter * counter, FILE * stream);
    void artsCounterSetStartTime(artsCounter * counter, uint64_t start);
    void artsCounterSetEndTime(artsCounter * counter, uint64_t end);
    void artsCounterAddEndTime(artsCounter * counter);
    void artsCounterNonEmtpy(artsCounter * counter);
    uint64_t artsCounterGetStartTime(artsCounter * counter);
    uint64_t artsCounterGetEndTime(artsCounter * counter);   
    void artsCounterAddTime(artsCounter * counter, uint64_t time);
    void artsWriteCountersToFile(unsigned int threadId, unsigned int nodeId);
#ifdef __cplusplus
}
#endif

#endif	/* ARTSCOUNTER_H */

