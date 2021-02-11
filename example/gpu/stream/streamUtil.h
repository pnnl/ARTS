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
#ifndef STREAMUTIL_H
#define STREAMUTIL_H
#ifdef __cplusplus
extern "C" {
#endif

# include <math.h>
# include <float.h>
# include <limits.h>
# include <sys/time.h>
# include "artsGlobals.h"

#ifdef STREAMN
#define N STREAMN
#else
#define N 134217728
#endif

# define NTIMES	100
# define OFFSET	0

#define M 20

#define THREADSPERBLOCK 1

# define HLINE "-------------------------------------------------------------\n"

# ifndef MIN
# define MIN(x,y) ((x)<(y)?(x):(y))
# endif
# ifndef MAX
# define MAX(x,y) ((x)>(y)?(x):(y))
# endif

void launch2KernelEdt(artsEdt_t funPtr, unsigned int tileSize, unsigned int totalSize, double scalar, artsGuidRange * aGuid, artsGuidRange * bGuid);
void launch3KernelEdt(artsEdt_t funPtr, unsigned int tileSize, unsigned int totalSize, double scalar, artsGuidRange * aGuid, artsGuidRange * bGuid, artsGuidRange * cGuid);

int checktick();
double mysecond();
void checkSTREAMresults(unsigned int tileSize, unsigned int totalSize, double **aTiles, double **bTiles, double **cTiles);

#ifdef __cplusplus
}
#endif

#endif