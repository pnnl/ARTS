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
#ifndef ARTSCONFIG_H
#define ARTSCONFIG_H
#ifdef __cplusplus
extern "C" {
#endif

#include "arts.h"
#include "artsRemoteLauncher.h"

struct artsConfigTable
{
    unsigned int rank;
    char * ipAddress;
};

struct artsConfigVariable
{
    unsigned int size;
    struct artsConfigVariable * next;
    char variable[255];
    char value[];
};

struct artsConfig
{
    unsigned int myRank;
    char * masterNode;
    char * myIPAddress;
    char * netInterface;
    char * protocol;
    char * launcher;
    unsigned int ports;
    unsigned int osThreadCount;
    unsigned int threadCount;
    unsigned int coreCount;
    unsigned int recieverCount;
    unsigned int senderCount;
    unsigned int socketCount;
    unsigned int nodes;
    unsigned int masterRank;
    unsigned int port;
    unsigned int killMode;
    unsigned int routeTableSize;
    unsigned int routeTableEntries;
    unsigned int dequeSize;
    unsigned int introspectiveTraceLevel;
    unsigned int introspectiveStartPoint;
    unsigned int counterStartPoint;
    unsigned int printNodeStats;
    unsigned int scheduler;
    unsigned int shutdownEpoch;
    char * prefix;
    char * suffix;
    bool ibNames;
    bool masterBoot;
    bool coreDump;
    unsigned int pinStride;
    bool printTopology;
    bool pinThreads;
    unsigned int firstEdt;
    unsigned int shadLoopStride;
    uint64_t stackSize;
    struct artsRemoteLauncher * launcherData;
    char * introspectiveFolder;
    char * introspectiveConf;
    char * counterFolder;
    unsigned int tableLength;
    unsigned int tMT;  // @awmm temporal MT; # of MT aliases per core thread; 0 if disabled
    unsigned int coresPerNetworkThread;
    unsigned int gpu;
    unsigned int gpuLocality;
    unsigned int gpuFit;
    unsigned int gpuLCSync;
    unsigned int gpuMaxEdts;
    uint64_t gpuMaxMemory;
    bool gpuP2P;
    bool gpuBuffOn;
    unsigned int gpuRouteTableSize;
    unsigned int gpuRouteTableEntries;
    bool freeDbAfterGpuRun;
    bool runGpuGcPreEdt;
    bool runGpuGcIdle;
    bool deleteZerosGpuGc;
    struct artsConfigTable * table;
};

struct artsConfig * artsConfigLoad();
void artsConfigDestroy( void * config );
unsigned int artsConfigGetNumberOfThreads(char * location);
#ifdef __cplusplus
}
#endif

#endif
