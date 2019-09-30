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

#include <stdio.h>              //For FILE, popen
#include <string.h>             //For strncpy
#include <unistd.h>             //For getcwd
#include "arts.h"
#include "artsConfig.h"         //For struct artsConfig
#include "artsGlobals.h"        //For artsGloablMessageTable
#include "artsRemoteLauncher.h" //For struct artsLauncher

#define DPRINTF(...)
//#define DPRINTF( ... ) PRINTF( __VA_ARGS__ )

void artsRemoteLauncherSSHStartupProcesses( struct artsRemoteLauncher * launcher )
{
    unsigned int argc = launcher->argc;
    char ** argv = launcher->argv;
    struct artsConfig * config = launcher->config;
    unsigned int killMode = launcher->killStuckProcesses;
        
    FILE ** sshExecutions = NULL;
    int i,j,k;
    int startNode = config->myRank;
    
    char cwd[1024];
    int cwdLength;
    if (getcwd(cwd, sizeof(cwd)) == NULL)
      PRINTF("Getcwd Error.");
    cwdLength = strlen(cwd);
    
    sshExecutions = artsMalloc( sizeof( FILE *  ) * config->tableLength-1 );
    launcher->launcherMemory = sshExecutions;
    DPRINTF("%s\n", argv[0]);
    char command[4096];
    char directory[4096];
    DPRINTF("%d \n", config->tableLength);
    pid_t child;
    for(k=startNode+1; k< config->tableLength+startNode; k++ )
    {
        i=k%config->tableLength;
        unsigned int finalLength=0;
        unsigned int len = strlen(config->table[i].ipAddress);

        if(killMode)
        {
            strncpy(command+finalLength, "\"\"pkill ", 8);
            finalLength+=8;
            

            if(k==startNode+1)
            {
                len = strlen(argv[0]);
                char* lastSlash;
                for(j=0; j<len; j++)
                    if(argv[0][j]=='/')
                        lastSlash=argv[0]+j;

                *lastSlash = '\0';
            }
                
            len = strlen(argv[0]);
            int lastLen = len;
            len = strlen(argv[0]+len+1);
            len = (len>15) ? 15 : len;
            strncpy(command+finalLength, argv[0]+lastLen+1, len);
            finalLength+=len;
        }
        else
        {
            strncpy(command+finalLength, "\"\"cd ", 5);
            finalLength+=5;
            strncpy(command+finalLength, cwd, cwdLength);
            finalLength+=cwdLength;
            strncpy(command+finalLength, ";", 1);
            finalLength+=1;
            
            for(j=0; j<argc; j++)
            {
                *(command+finalLength++)=' ';
                len = strlen( argv[j] );
                strncpy(command+finalLength, argv[j], len);
                finalLength+=len;
            }
        }

        strncpy(command+finalLength, "\"\"\0", 3);
        finalLength+=3;

        child = fork();

        if(child == 0)
        {
            execlp("ssh", "-f", config->table[i].ipAddress, command, (char *)NULL); 
            //execlp("ssh", "-f", config->table[i].ipAddress, "cd runLocal; /home/land350/intel/test/intel/inspector_xe_2016.1.1.435552/bin64/inspxe-cl -c mi3 /home/land350/new/dtcp/test/artsFib 20", (char *)NULL); 
        }
    }
    if(killMode)
        exit(0);
}


void artsRemoteLauncherSSHCleanupProcesses( struct artsRemoteLauncher * launcher )
{
}

