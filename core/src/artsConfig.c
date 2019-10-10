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
#include "artsConfig.h"
#include "artsGlobals.h"
#include "artsRemoteLauncher.h"
#include "unistd.h"
#include <ctype.h>

#define DPRINTF(...)
//#define DPRINTF( ... ) PRINTF( __VA_ARGS__ )


char * extract_nodelist_lsf(char *envr, int stride, unsigned int *cnt){
   char *lsfNodes;
   char *resString;
   char *last;
   int ele = 0;
   lsfNodes = getenv(envr);
   if(lsfNodes == NULL) return NULL;
   if(stride <=0 ) stride =1;
   unsigned int nodesStrLen = strlen(lsfNodes) + 1;
   char *nodeList = malloc(sizeof(char) * nodesStrLen);
   unsigned int count = 0;
   unsigned int listStrLength = 0;
   last = resString = strtok(lsfNodes, " ");
   while (resString){
      resString = strtok(NULL, " ");
      ele ++;
      if(resString && strcmp(resString, last) && !(ele % stride)){
         strcpy(&nodeList[listStrLength], resString);
         listStrLength += strlen(resString);
         nodeList[listStrLength++] = ',';
         last = resString;
         count++;
      }
   }
   nodeList[listStrLength-1] = '\0';
   *cnt = count;
   return nodeList;
}

struct artsConfigVariable * artsConfigFindVariable(struct artsConfigVariable ** head, char * string)
{
    struct artsConfigVariable * found = NULL;
    struct artsConfigVariable * last = NULL;
    struct artsConfigVariable * next = *head;
    
    while(next != NULL)
    {
        if(strcmp(string, next->variable) == 0)
        {
            found = next;
            break;
        }
        last = next;
        next = next->next;
    }
    
    char * overide = getenv(string);
    if(overide)
    {
        unsigned int size = strlen(overide);
        struct artsConfigVariable * newVar = artsMalloc(sizeof(struct artsConfigVariable) + size);

        newVar->size = size;
        strcpy(newVar->variable, string);
        strcpy(newVar->value, overide);

        if(last)
            last->next = newVar;
        else
        {
            newVar->next = *head;
            *head = newVar;
        }

        if(next)
        {
            newVar->next = next->next;
            artsFree(next);
        }
        return newVar;
    }
    return found;
}

char * artsConfigFindVariableChar(struct artsConfigVariable * head, char * string)
{
    struct artsConfigVariable * found = NULL;
    char * overide = getenv(string);
    
    if(overide)
        return overide;

    while(head != NULL)
    {
        if(strcmp(string, head->variable) == 0)
        {
            found = head;
            break;
        }
        head = head->next;
    }
    
    if(found)
        return found->value;

    return NULL;
}

unsigned int artsConfigGetVariable(FILE * config, char * lookForMe)
{
    char * line;
    size_t len=0;
    ssize_t read;
    char * var;
    char * val;
    int size;
    struct artsConfigVariable * cVar;
    struct artsConfigVariable * head, * next=NULL;

    while ((read = getline(&line, &len, config)) != -1)
    {
        var = strtok(line, "=");
        val = strtok(NULL, "=");

        if( strcmp(lookForMe, var) == 0 )
        {
            if(val==NULL)
                return 4;
            size= strlen(val);

            if(val[size-1]=='\n')
                val[size-1]='\0';
                        
            return strtol(val, NULL, 10); 
        }
    }
    return 4;
}

void removeWhiteSpaces(char * str)
{
    char *write = str, *read = str;
    do {
           if (*read != ' ')
                      *write++ = *read;
    } while (*read++);
}

struct artsConfigVariable * artsConfigGetVariables(FILE * config)
{
    char * line = NULL;
    size_t len=0;
    ssize_t read;
    char * var;
    char * val;
    int size;
    struct artsConfigVariable * cVar;
    struct artsConfigVariable * head, * next=NULL;

    while ((read = getline(&line, &len, config)) != -1)
    {
        var = strtok(line, "=");
        val = strtok(NULL, "=");

        if( val != NULL )
        {
            size= strlen(val);

            if(val[size-1]=='\n')
                val[size-1]='\0';

            cVar = artsMalloc( sizeof( struct artsConfigVariable ) + size );
            cVar->size = size;

            strncpy(cVar->variable, var, 255);
            strcpy(cVar->value, val);
            
            removeWhiteSpaces(cVar->variable);
            removeWhiteSpaces(cVar->value);

            cVar->next=NULL;

            if(next!=NULL)
                next->next= cVar;
            else
                head=cVar;
            next=cVar;
        }
    }
    return head;
}

char * artsConfigMakeNewVar(char * var)
{
    char * newVar;
    unsigned int size;
    size = strlen(var);
    newVar = artsMalloc(size+1);
    strncpy(newVar, var, size);
    newVar[size] = '\0';
    DPRINTF("%s l\n", newVar);
    return var;
}

unsigned int artsConfigGetValue(char * start, char * stop)
{
    int value, size = stop - start;
    for(int i=0; i<size; i++)
    {
        if(isdigit(start[i]))
        {
            if(*stop==':')
            {
                *stop='\0';
                value = strtol(start+i,NULL, 10);
                *stop = ':';
            }
            else
                value = strtol(start+i,NULL, 10);
            break;
        }
    }
    return value;
}

char * artsConfigGetNodeName(char * start, char * stop)
{
    int value, size = stop - start;
    char * name;

    for(int i=0; i<size; i++)
    {
        if(isdigit(start[i]))
        {
            name = artsMalloc( (stop-start) );
            strncpy(name, start, stop-start);
            break;
        }
    }
    return name;
}
char * artsConfigGetHostname( char * name, unsigned int value )
{
    unsigned int length = strlen(name);
    unsigned int digits=1;
    unsigned int temp = value;
    unsigned int stop;
    char * outName = artsMalloc(length);

    while(temp>9)
    {
        temp /= 10;
        digits++;
    }

    temp = value;
    for(unsigned int i=0; i<length; i++)
    {
        if(isdigit(name[i]))
        {
            stop = i;
            while( stop < length  )
            {
                if(!isdigit(name[stop]))
                {
                    break;
                }
                stop++;
            }

            for(unsigned int j=stop-1; j>(stop-1) - digits; j--)
            {
                //name[j]= itoa( value%10 );
                //sprintf(name+j,"%d",value%10);
                name[j] = ((int)'0')+value%10;
                value /=10;
            }

            for(unsigned int j=(stop-1) - digits; j>=i; j--)
            {
                name[j]='0';
            }
            break;
        }
    }
    strncpy(outName, name, length);
    return outName;
}

char * artsConfigGetSlurmHostname( char * name, char * digitSample, unsigned int value, bool ib, char * prefix, char * suffix )
{
    unsigned int length = strlen(name);
    unsigned int digitLength = strlen(digitSample);
    unsigned int suffixLength = 0;
    unsigned int prefixLength = 0;
    unsigned int nameLength;
    
    if(suffix!= NULL)
        suffixLength = strlen(suffix);
    
    if(prefix!=NULL)
        prefixLength = strlen(prefix);

    nameLength = length + digitLength+1+prefixLength+suffixLength;
    char * outName = artsMalloc(nameLength);

    if(prefix!=NULL)
    {
        strncpy(outName, prefix, prefixLength );
        strncpy(outName+prefixLength, name, length );
    }
    else
        strncpy(outName, name, length );

    for(unsigned int i=digitLength; i>0; i--)
    {
        outName[prefixLength+length+i-1]= ((int)'0')+value%10;
        value /=10;
    }

    if(suffix!= NULL)
    {
        strncpy(outName+prefixLength+digitLength+length,suffix,suffixLength);
        strncpy(outName+prefixLength+digitLength+length+suffixLength,"\0",1);
    }
    else
        strncpy(outName+prefixLength+digitLength+length,"\0",1);
    return outName;
}

unsigned int artsConfigCountNodes( char * nodeList )
{
    unsigned int length = strlen(nodeList);
    unsigned int rangeCount=0;
    unsigned int nodes = 0;

    for(unsigned int i=0; i<length; i++)
        if(nodeList[i] == ',')
            nodes++;
    nodes++;

    char * begin, * end, * search;
    begin = nodeList;

    for(unsigned int i=0; i<length; i++)
    {
        if(nodeList[i] == ',')
            begin = nodeList + i;
        else if(nodeList[i] == ':')
        {
            nodes--;
            search = nodeList+i;
            end = nodeList+length;

            for(unsigned int j=0; j < end - search; j++)
            {
                if(nodeList[i+j] == ',')
                {
                    end = nodeList+i+j;
                }
            }

            unsigned int front = artsConfigGetValue(begin,nodeList+i);
            unsigned int back = artsConfigGetValue(nodeList+i+1,end);
            if(front>back)
                nodes+=(front-back)+1;
            else
                nodes+=(back-front)+1;
        }
    }
    return nodes;
}

char * artsGetNextPartition(char ** remainder)
{
    char * input = (*remainder);
    (*remainder) = NULL;
    if(input) 
    {
        unsigned int length = strlen(input);
        if(length)
        {
            bool flag = false;
            bool found = false;
            for(unsigned int i=0; i<=length; i++)
            {
                if(input[i] == '[')
                    flag = true;
                else if(input[i] == ']')
                    flag = false;
                else if(!flag && input[i] == ',')
                {
                    input[i] = '\0';
                    (*remainder) = &input[i+1];
                    found = true;
                    break;
                }
            }
            return input;
        }
    }
    return NULL;
}

void artsConfigCreateRoutingTable( struct artsConfig ** config, char* nodeList)
{
    unsigned int nodeCount;
    struct artsConfigTable * table;
    unsigned int currentNode = 0, strLength;
    char * temp, * next;
    unsigned int start;
    unsigned int stop;
    unsigned int direction, listLength;

    listLength = strlen(nodeList);

    unsigned int suffixLength = 0;
    unsigned int prefixLength = 0;
    unsigned int totalLength = 0;
    
    char* prefix = (*config)->prefix;
    char* suffix = (*config)->suffix;

    if(suffix!= NULL)
        suffixLength = strlen(suffix);
    if(prefix!=NULL)
        prefixLength = strlen(prefix);
    
    nodeCount = (*config)->nodes;
    (*config)->tableLength = nodeCount;
    table = artsMalloc(sizeof(struct artsConfigTable) * nodeCount);
    
    if(!(*config)->masterBoot)
    {
        char * part;
        while((part = artsGetNextPartition(&nodeList)))
        {
            char * nodeBegin = strtok(part, "[");
            char * next  = strtok(NULL, "[");
            if(next)
            {
                bool done = false;
                char * name = nodeBegin;
                nodeBegin = nodeBegin + strlen(nodeBegin) + 1;
                do 
                {
                    nodeBegin = strtok(nodeBegin, ",");
                    next = nodeBegin + strlen(nodeBegin) + 1;
                    if (nodeBegin) 
                    {
                        nodeBegin = strtok(nodeBegin, "-");
                        char * nodeEnd = strtok(NULL, "-");

                        if (nodeEnd) 
                        {
                            if (nodeEnd[strlen(nodeEnd) - 1] == ']') 
                            {
                                nodeEnd[strlen(nodeEnd) - 1] = '\0';
                                done = true;
                            }
                            
                            start = strtol(nodeBegin, NULL, 10);
                            stop = strtol(nodeEnd, NULL, 10);

                            if (start < stop)
                                direction = 1;
                            else
                                direction = -1;

                            while (start != stop + 1) 
                            {
                                table[currentNode].rank = currentNode;
                                table[currentNode].ipAddress = artsConfigGetSlurmHostname(name, nodeBegin, start, (*config)->ibNames, (*config)->prefix, (*config)->suffix);
                                start += direction;
                                currentNode++;
                            }
                        } 
                        else 
                        {
                            if (nodeBegin[strlen(nodeBegin) - 1] == ']') 
                            {
                                nodeBegin[strlen(nodeBegin) - 1] = '\0';
                                done = true;
                            }

                            unsigned int nameLength = strlen(name);
                            strLength = strlen(nodeBegin) + nameLength;
                            totalLength = strLength + 1 + prefixLength + suffixLength;
                            temp = artsMalloc(totalLength);

                            if (prefix != NULL)
                                strncpy(temp, prefix, prefixLength);
                            strncpy(temp + prefixLength, name, nameLength);

                            strncpy(temp + prefixLength + nameLength, nodeBegin, strlen(nodeBegin));

                            if (suffix != NULL)
                                strncpy(temp + prefixLength + strLength, suffix, suffixLength);
                            strncpy(temp + totalLength - 1, "\0", 1);

                            table[currentNode].rank = currentNode;
                            table[currentNode].ipAddress = temp;
                            currentNode++;
                        }
                    }
                    nodeBegin = next;
                } while (!done);
            }
            else
            {
                //Single node
                strLength = strlen(nodeBegin);
                totalLength = strLength + 1 + prefixLength + suffixLength;
                temp = artsMalloc(totalLength);

                if(prefix != NULL)
                    strncpy(temp, prefix, prefixLength);
                strncpy(temp + prefixLength, nodeBegin, strLength);

                if(suffix != NULL)
                    strncpy(temp + prefixLength + strLength, suffix, suffixLength);
                strncpy(temp + totalLength - 1, "\0", 1);

                table[currentNode].rank = currentNode;
                table[currentNode].ipAddress = temp;
                currentNode++;
            }            
        }
    }
    else {
        //This is the ssh path...
        char * nodeBegin = nodeList;
        char * next;
        do {
            nodeBegin = strtok(nodeBegin, ",");
            next = nodeBegin + strlen(nodeBegin) + 1;

            if (nodeBegin != NULL) {
                nodeBegin = strtok(nodeBegin, ":");
                char * nodeEnd = strtok(NULL, ":");

                if (nodeBegin[strlen(nodeBegin) - 1] == '\n')
                    nodeBegin[strlen(nodeBegin) - 1] = '\0';

                if (nodeEnd != NULL) {
                    start = artsConfigGetValue(nodeBegin, nodeEnd - 1);
                    stop = artsConfigGetValue(nodeEnd, nodeEnd + strlen(nodeEnd));

                    char * name = artsConfigGetNodeName(nodeBegin, nodeEnd);

                    if (start < stop)
                        direction = 1;
                    else
                        direction = -1;

                    strLength = strlen(nodeBegin);
                    while (start != stop + 1) {
                        table[currentNode].rank = currentNode;
                        table[currentNode].ipAddress = artsConfigGetHostname(name, start);
                        start += direction;
                        currentNode++;
                    }
                } 
                else
                {
                    table[currentNode].rank = currentNode;
                    strLength = strlen(nodeBegin);
                    temp = artsMalloc(strLength + 1);
                    strncpy(temp, nodeBegin, strLength + 1);
                    table[currentNode].ipAddress = temp;
                    currentNode++;
                }
            }
            nodeBegin = next;
        } while (nodeBegin < nodeList + listLength);
    }

    (*config)->table = table;
}

unsigned int artsConfigGetNumberOfThreads(char * location)
{
    FILE * configFile = NULL;
    if(location == NULL)
        configFile = fopen( "arts.cfg", "r" );
    else
        configFile = fopen( location, "r" );

    if(configFile == NULL)
    {
        return 4;
    }

    return artsConfigGetVariable( configFile, "threads");
}

struct artsConfig * artsConfigLoad()
{
    FILE * configFile = NULL;
    struct artsConfig * config;
    struct artsConfigVariable * configVariables;
    struct artsConfigVariable * foundVariable;
    char * foundVariableChar;

    char * end = NULL;

    config = artsMalloc( sizeof( struct artsConfig ) );

    char * location = getenv("artsConfig");
    if(location)
        configFile = fopen( location, "r" );
    else    
        configFile = fopen( "arts.cfg", "r" );

    if(configFile == NULL)
    {
        PRINTF("No Config file found (./arts.cfg).\n");
        configVariables = NULL;
    }
    else
        configVariables = artsConfigGetVariables(configFile);

    foundVariable = artsConfigFindVariable(&configVariables, "launcher");
    if (strncmp(foundVariable->value, "slurm", 5) == 0)
        config->launcher = artsConfigMakeNewVar("slurm"); 
    else if(strncmp(foundVariable->value, "lsf", 5) == 0)
        config->launcher = artsConfigMakeNewVar("lsf");
    else if (strncmp(foundVariable->value, "local", 5) == 0)
        config->launcher = artsConfigMakeNewVar("local");
    else
        config->launcher = artsConfigMakeNewVar("ssh");

    char * killSet = getenv("killMode");
    if(killSet == NULL)
    {
        if( (foundVariable = artsConfigFindVariable(&configVariables,"killMode")) != NULL)
        {
            config->killMode = strtol( foundVariable->value, &end , 10);

            if(config->killMode )
            {
                ONCE_PRINTF("Killmode set: Attempting to kill remote proccesses.\n");
            }
        } 
        else
        {
            config->killMode = 0;
        }
    }
    else
    {
        config->killMode = strtol( killSet, &end , 10);
    }
    
    if( (foundVariable = artsConfigFindVariable(&configVariables,"coreDump")) != NULL)
    {
        config->coreDump = strtol( foundVariable->value, &end , 10);
    }
    else
    {
        config->coreDump = 0;
    }
    
    if( (foundVariable = artsConfigFindVariable(&configVariables,"pinStride")) != NULL)
    {
        config->pinStride = strtol( foundVariable->value, &end , 10);
    }
    else
    {
        config->pinStride = 1;
    }
    
    if( (foundVariable = artsConfigFindVariable(&configVariables,"printTopology")) != NULL)
    {
        config->printTopology = strtol( foundVariable->value, &end , 10);
    }
    else
    {
        config->printTopology = 0;
    }
    
    if( (foundVariable = artsConfigFindVariable(&configVariables,"threads")) != NULL)
        config->threadCount = strtol( foundVariable->value, &end , 10);
    else
    {
        ONCE_PRINTF("Defaulting to 4 threads\n");
        config->threadCount = 4;
    }
    
    if( (foundVariable = artsConfigFindVariable(&configVariables,"osThreads")) != NULL)
        config->osThreadCount = strtol( foundVariable->value, &end , 10);
    else
    {
        config->osThreadCount = 0;
    }
    
    if( (foundVariable = artsConfigFindVariable(&configVariables,"coresPerNetworkThread")) != NULL)
        config->coresPerNetworkThread = strtol( foundVariable->value, &end , 10);
    else
    {
        config->coresPerNetworkThread = 1;
    }
    
    if( (foundVariable = artsConfigFindVariable(&configVariables,"ports")) != NULL)
        config->ports = strtol( foundVariable->value, &end , 10);
    else if (strncmp(config->launcher, "local", 5) != 0) 
    {
        ONCE_PRINTF("Defaulting to 1 connection per node\n");
        config->ports = 1;
    }
    
    if( (foundVariable = artsConfigFindVariable(&configVariables,"outgoing")) != NULL)
        config->senderCount = strtol( foundVariable->value, &end , 10);
    else if (strncmp(config->launcher, "local", 5) != 0) 
    {
        ONCE_PRINTF("Defaulting to 1 sender\n");
        config->senderCount = 1;
    }
    
    if( (foundVariable = artsConfigFindVariable(&configVariables,"incoming")) != NULL)
        config->recieverCount = strtol( foundVariable->value, &end , 10);
    else if (strncmp(config->launcher, "local", 5) != 0) 
    {
        ONCE_PRINTF("Defaulting to 1 reciever\n");
        config->recieverCount = 1;
    }
    
    if( (foundVariable = artsConfigFindVariable(&configVariables,"sockets")) != NULL)
        config->socketCount = strtol( foundVariable->value, &end , 10);
    else
    {
        //ONCE_PRINTF("Defaulting to 1 sockets\n");
        config->socketCount = 1;
    }

    if( (foundVariable = artsConfigFindVariable(&configVariables, "netInterface")) != NULL)
    {
        config->netInterface = artsConfigMakeNewVar( foundVariable->value );
        
        if(config->netInterface[0] == 'i')
            config->ibNames=true;
        else
            config->ibNames=false;
    }
    else
    {
        //ONCE_PRINTF("No network interface given: defaulting to eth0\n");

        config->netInterface = NULL;//artsConfigMakeNewVar( "eth0" );
        config->ibNames=false;
    }

    if( (foundVariable = artsConfigFindVariable(&configVariables, "protocol")) != NULL)
        config->protocol = artsConfigMakeNewVar( foundVariable->value );
    else
    {
        //ONCE_PRINTF("No protocol given: defaulting to tcp\n");

        config->protocol = artsConfigMakeNewVar( "tcp" );
    }

    if( (foundVariable = artsConfigFindVariable(&configVariables, "masterNode")) != NULL)
        config->masterNode = artsConfigMakeNewVar( foundVariable->value );
    else if (strncmp(config->launcher, "local", 5) != 0) 
    {
        if(strncmp(config->launcher, "slurm", 5 )!=0)
            ONCE_PRINTF("No master given: defaulting to first node in node list\n");

        config->masterNode = NULL;
    }
    
    if( (foundVariable = artsConfigFindVariable(&configVariables, "prefix")) != NULL)
        config->prefix = artsConfigMakeNewVar( foundVariable->value );
    else
        config->prefix = NULL;
    
    if( (foundVariable = artsConfigFindVariable(&configVariables, "suffix")) != NULL)
        config->suffix = artsConfigMakeNewVar( foundVariable->value );
    else
        config->suffix = NULL;
    
    if( (foundVariable = artsConfigFindVariable(&configVariables, "introspectiveConf")) != NULL)
        config->introspectiveConf = artsConfigMakeNewVar( foundVariable->value );
    else
        config->introspectiveConf = NULL;
    
    if( (foundVariable = artsConfigFindVariable(&configVariables, "introspectiveFolder")) != NULL)
        config->introspectiveFolder = artsConfigMakeNewVar( foundVariable->value );
    else
        config->introspectiveFolder = NULL;
    
    if( (foundVariableChar = artsConfigFindVariableChar(configVariables,"introspectiveTraceLevel")) != NULL)
        config->introspectiveTraceLevel = strtol( foundVariableChar, &end , 10);
    else
        config->introspectiveTraceLevel = 1;
    
    if( (foundVariableChar = artsConfigFindVariableChar(configVariables,"introspectiveStartPoint")) != NULL)
        config->introspectiveStartPoint = strtol( foundVariableChar, &end , 10);
    else
        config->introspectiveStartPoint = 1;

    if( (foundVariable = artsConfigFindVariable(&configVariables, "counterFolder")) != NULL)
        config->counterFolder = artsConfigMakeNewVar( foundVariable->value );
    else
        config->counterFolder = NULL;
    
    if( (foundVariableChar = artsConfigFindVariableChar(configVariables,"counterStartPoint")) != NULL)
        config->counterStartPoint = strtol( foundVariableChar, &end , 10);
    else
    {
        config->counterStartPoint = 1;
    }
    
    if( (foundVariableChar = artsConfigFindVariableChar(configVariables,"printNodeStats")) != NULL)
        config->printNodeStats = strtol( foundVariableChar, &end , 10);
    else
        config->printNodeStats = 0;
    
    if( (foundVariableChar = artsConfigFindVariableChar(configVariables,"scheduler")) != NULL)
        config->scheduler = strtol( foundVariableChar, &end , 10);
    else
        config->scheduler = 0;
    
    if( (foundVariableChar = artsConfigFindVariableChar(configVariables,"shutdownEpoch")) != NULL)
        config->shutdownEpoch = strtol( foundVariableChar, &end , 10);
    else
        config->shutdownEpoch = 0;
    
    if( (foundVariableChar = artsConfigFindVariableChar(configVariables,"shadLoopStride")) != NULL)
        config->shadLoopStride = strtol( foundVariableChar, &end , 10);
    else
        config->shadLoopStride = 32;
    
    // @awmm tMT
    if( (foundVariable = artsConfigFindVariable(&configVariables,"tMT")) != NULL)
        config->tMT = strtol( foundVariable->value, &end , 10);
    else
        config->tMT = 0;
    
    if( (foundVariable = artsConfigFindVariable(&configVariables,"coreCount")) != NULL)
        config->coreCount = strtol( foundVariable->value, &end , 10);
    else
        config->coreCount = 0;
    
    if( (foundVariable = artsConfigFindVariable(&configVariables,"gpu")) != NULL)
        config->gpu = strtol( foundVariable->value, &end , 10);
    else
        config->gpu = 0;
    
    if( (foundVariable = artsConfigFindVariable(&configVariables,"gpuLocality")) != NULL)
        config->gpuLocality = strtol( foundVariable->value, &end , 10);
    else
        config->gpuLocality = 0;

    if( (foundVariable = artsConfigFindVariable(&configVariables,"gpuFit")) != NULL)
        config->gpuFit = strtol( foundVariable->value, &end , 10);
    else
        config->gpuFit = 0;

    if( (foundVariable = artsConfigFindVariable(&configVariables,"gpuMaxEdts")) != NULL)
        config->gpuMaxEdts = strtol( foundVariable->value, &end , 10);
    else
        config->gpuMaxEdts = (unsigned int)-1;

    if( (foundVariable = artsConfigFindVariable(&configVariables,"gpuMaxMemory")) != NULL)
        config->gpuMaxMemory = strtol( foundVariable->value, &end , 10);
    else
        config->gpuMaxMemory = (size_t)-1;

    if( (foundVariable = artsConfigFindVariable(&configVariables,"gpuP2P")) != NULL)
        config->gpuP2P = strtol( foundVariable->value, &end , 10) > 0;
    else
        config->gpuP2P = false;

    if( (foundVariable = artsConfigFindVariable(&configVariables,"gpuRouteTableSize")) != NULL)
        config->gpuRouteTableSize = strtol( foundVariable->value, &end , 10);
    else
        config->gpuRouteTableSize = 12; //2^12
    
    if( (foundVariable = artsConfigFindVariable(&configVariables,"freeDbAfterGpuRun")) != NULL)
        config->freeDbAfterGpuRun = strtol( foundVariable->value, &end , 10) > 0;
    else
        config->freeDbAfterGpuRun = false;

    if( (foundVariable = artsConfigFindVariable(&configVariables,"runGpuGcIdle")) != NULL)
        config->runGpuGcIdle = strtol( foundVariable->value, &end , 10) > 0;
    else
        config->runGpuGcIdle = true;

    if( (foundVariable = artsConfigFindVariable(&configVariables,"runGpuGcPreEdt")) != NULL)
        config->runGpuGcPreEdt = strtol( foundVariable->value, &end , 10) > 0;
    else
        config->runGpuGcPreEdt = false;

    if( (foundVariable = artsConfigFindVariable(&configVariables,"deleteZerosGpuGc")) != NULL)
        config->deleteZerosGpuGc = strtol( foundVariable->value, &end , 10) > 0;
    else
        config->deleteZerosGpuGc = true;

    if( (foundVariable = artsConfigFindVariable(&configVariables,"gpuBuffOn")) != NULL)
        config->gpuBuffOn = strtol( foundVariable->value, &end , 10) > 0;
    else
        config->gpuBuffOn = false;

    //WARNING: Slurm Launcher Set!  
    if (strncmp(config->launcher, "slurm", 5) == 0) 
    {
        ONCE_PRINTF("Using Slurm\n");
        config->masterBoot = false;
        
        char *threadsTemp = getenv("SLURM_CPUS_PER_TASK");
        if (threadsTemp != NULL)
            config->threadCount = strtol(threadsTemp, &end, 10);

        char *slurmNodes;
        slurmNodes = getenv("SLURM_NNODES");
        config->nodes = strtol(slurmNodes, &end, 10);
        
        char *nodeList = getenv("SLURM_STEP_NODELIST");
        artsConfigCreateRoutingTable(&config, nodeList);

        unsigned int length = strlen(config->table[0].ipAddress) + 1;
        config->masterNode = artsMalloc(sizeof (char) * length);
        strncpy(config->masterNode, config->table[0].ipAddress, length);
        
        for (int i = 0; i < config->tableLength; i++) 
        {
            config->table[i].rank = i;
            if (strcmp(config->masterNode, config->table[i].ipAddress) == 0) 
            {
                DPRINTF("%d %s\n", i, config->table[i].ipAddress);
                config->masterRank = i;
            }
        }
    } 
    else if (strncmp(config->launcher, "lsf", 5) == 0) 
    {
        ONCE_PRINTF("Using LSF\n");
        config->masterBoot = false;
        unsigned int count =0;
        char *nodeList = extract_nodelist_lsf("LSB_HOSTS", 1, &count);
        if(!nodeList){
           nodeList = extract_nodelist_lsf("LSB_MCPU_HOSTS", 2, &count);
        }
        config->nodes = count;

        artsConfigCreateRoutingTable(&config, nodeList);

        unsigned int length = strlen(config->table[0].ipAddress) + 1;
        config->masterNode = artsMalloc(sizeof (char) * length);

        strncpy(config->masterNode, config->table[0].ipAddress, length);

        for (int i = 0; i < config->tableLength; i++) {
            config->table[i].rank = i;
            if (strcmp(config->masterNode, config->table[i].ipAddress) == 0) {
                DPRINTF("%d %s\n", i, config->table[i].ipAddress);
                config->masterRank = i;
            }
        }
    } 
    else if (strncmp(config->launcher, "ssh", 5) == 0) 
    {
        config->launcherData =
                artsRemoteLauncherCreate(0, NULL, config, config->killMode,
                artsRemoteLauncherSSHStartupProcesses,
                artsRemoteLauncherSSHCleanupProcesses);
        config->masterBoot = true;

        char * nodeList = 0;
        if ((foundVariable = artsConfigFindVariable(&configVariables, "nodes")) != NULL) 
        {
            nodeList = foundVariable->value;
            
            if ((foundVariable = artsConfigFindVariable(&configVariables, "nodeCount")) != NULL)
                config->nodes = strtol(foundVariable->value, &end, 10);
            else 
                config->nodes = artsConfigCountNodes(nodeList);
        } 
        else 
        {
            ONCE_PRINTF("No nodes given: defaulting to 1 node\n");
            nodeList = artsMalloc(sizeof (char)*strlen("localhost\0"));
            strncpy(nodeList, "localhost\0", strlen("localhost\0") + 1);
            config->nodes = 1;
        }

        DPRINTF("nodes: %s\n", nodeList);
        artsConfigCreateRoutingTable(&config, nodeList);

        if (config->masterNode == NULL) 
        {
            unsigned int length = strlen(config->table[0].ipAddress) + 1;
            config->masterNode = artsMalloc(sizeof (char)*length);
            strncpy(config->masterNode, config->table[0].ipAddress, length);
        }
        
        for (int i = 0; i < config->tableLength; i++) 
        {
            config->table[i].rank = i;
            if (strcmp(config->masterNode, config->table[i].ipAddress) == 0) 
            {
                DPRINTF("Here %d\n", config->tableLength);
                config->masterRank = i;
            }
        }
    } 
    else if (strncmp(config->launcher, "local", 5) == 0) 
    {
        ONCE_PRINTF("Running in Local Mode.\n");
        config->masterBoot = false;
        config->masterNode = NULL;
        // OS Threads
        char * threadsOS = getenv("OS_THREAD_COUNT");
        if (threadsOS != NULL)
            config->osThreadCount = strtol(threadsOS, &end, 10);
        else if (!config->osThreadCount)
            config->osThreadCount = 0; // Default to single thread.
        // OS Threads
        char *threadsUSER = getenv("USER_THREAD_COUNT");
        if (threadsUSER != NULL)
            config->threadCount = strtol(threadsUSER, &end, 10);
        else if (!config->threadCount)
            config->threadCount = 4; // Default to single thread.
        config->nodes = 1;
        config->tableLength = 1; // for GUID
        config->masterRank = 0;
    } 
    else 
    {
        ONCE_PRINTF("Unknown launcher: %s\n", config->launcher);
        exit(1);
    }

    if( (foundVariable = artsConfigFindVariable(&configVariables,"stackSize")) != NULL)
        config->stackSize = strtoull( foundVariable->value, &end , 10);
    else
    {
        config->stackSize = 0;
    }
    
    if( (foundVariable = artsConfigFindVariable(&configVariables,"workerInitDequeSize")) != NULL)
        config->dequeSize = strtol( foundVariable->value, &end , 10);
    else
    {
        ONCE_PRINTF("Defaulting the worker queue length to 4096\n");
        config->dequeSize = 4096;
    }

    if( (foundVariable = artsConfigFindVariable(&configVariables,"port")) != NULL)
        config->port = strtol( foundVariable->value, &end , 10);
    else if (strncmp(config->launcher, "local", 5) != 0) 
    {
        ONCE_PRINTF("Defaulting port to %d\n", 75563);
        config->port = 75563;
    }
    if( (foundVariable = artsConfigFindVariable(&configVariables,"routeTableSize")) != NULL)
        config->routeTableSize = strtol( foundVariable->value, &end , 10);
    else {
        ONCE_PRINTF("Defaulting routing table size to 2^20\n");
        config->routeTableSize = 20;
    }

    int routeTableEntries = 1;
    for (int i = 0; i < config->routeTableSize; i++)
        routeTableEntries *= 2;
    config->routeTableEntries = routeTableEntries;
    
    int gpuRouteTableEntries = 1;
    for (int i = 0; i < config->gpuRouteTableSize; i++)
        gpuRouteTableEntries *= 2;
    config->gpuRouteTableEntries = gpuRouteTableEntries;

    if( (foundVariable = artsConfigFindVariable(&configVariables,"pin")) != NULL)
        config->pinThreads = strtol( foundVariable->value, &end , 10);
    else
    {
        config->pinThreads = 1;
    }
    
    DPRINTF("Config Parsed\n");

    return config;
}

void artsConfigDestroy( void * config )
{
    artsFree( config );
}

