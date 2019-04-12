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
#include <unistd.h>
#include <signal.h>
#include <execinfo.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

# if !defined(__APPLE__)
#include <sys/resource.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/prctl.h>

void artsTurnOnCoreDumps()
{
    unsigned int res = prctl(PR_SET_DUMPABLE, 1);

    struct rlimit limit;

    limit.rlim_cur = RLIM_INFINITY ;
    limit.rlim_max = RLIM_INFINITY;
    pid_t pid = getpid();
    if(setrlimit(RLIMIT_CORE, &limit) != 0)
        printf("Failed to force core dumps\n");
}

#else

void artsTurnOnCoreDumps()
{
    printf("Core dumps not supported on OS X.\n");
}

#endif

void artsDebugPrintStack()
{
        void *array[10];
        size_t size = backtrace(array, 10);
        backtrace_symbols_fd(array, size, STDOUT_FILENO);
}

void artsDebugGenerateSegFault()
{
        raise(SIGSEGV);
}
