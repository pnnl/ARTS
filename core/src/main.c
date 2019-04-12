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
#define _FILE_OFFSET_BITS 64
#include "arts.h"
#include "artsThreads.h"
#include "artsConfig.h"
#include "artsGlobals.h"
#include "artsRemote.h"
#include "artsGuid.h"
#include "artsRuntime.h"
#include "artsRemoteLauncher.h"
#include "artsIntrospection.h"
#include "artsDebug.h"

extern struct artsConfig * config;

int mainArgc = 0;
char ** mainArgv = NULL;

int artsRT(int argc, char **argv)
{
    mainArgc = argc;
    mainArgv = argv;
    artsRemoteTryToBecomePrinter();
    config = artsConfigLoad(0, NULL, NULL);

    if(config->coreDump)
        artsTurnOnCoreDumps();

    artsGlobalRankId = 0;
    artsGlobalRankCount = config->tableLength;
    if(strncmp(config->launcher, "local", 5) != 0)
        artsServerSetup(config);
    artsGlobalMasterRankId= config->masterRank;
    if(artsGlobalRankId == config->masterRank && config->masterBoot)
        config->launcherData->launchProcesses(config->launcherData);

    if(artsGlobalRankCount>1)
    {
        artsRemoteSetupOutgoing();
        if(!artsRemoteSetupIncoming())
            return -1;
    }

    artsThreadInit(config);
    artsThreadZeroNodeStart();
       
    artsThreadMainJoin();

    if(artsGlobalRankId == config->masterRank && config->masterBoot)
    {
        config->launcherData->cleanupProcesses(config->launcherData);
    }
    artsConfigDestroy(config);
    artsRemoteTryToClosePrinter();
    return 0;
}
