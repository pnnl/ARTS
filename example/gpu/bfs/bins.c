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
#include "bins.h"

#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "arts/arts.h"
#include "arts/utils/ArrayList.h"

static artsArrayList **list;

void initListRecord() {
  unsigned int size = artsGetTotalGpus() + 1;
  list = (artsArrayList **)artsCalloc(size, sizeof(artsArrayList *));
  for (unsigned int i = 0; i < size; i++)
    list[i] = artsNewArrayList(sizeof(unsigned int), 32);
}

void addToList(unsigned int frontierSize, unsigned int index) {
  artsPushToArrayList(list[index], &frontierSize);
}

void writeBinsToFile(unsigned int index) {
  // Do the CPU bins
  if (index + 1 == artsGetTotalGpus())
    writeBinsToFile(index + 1);

  char filename[1024];
  sprintf(filename, "bins_%u_%u.ct", artsGetCurrentNode(), index);
  if (artsLengthArrayList(list[index])) {
    FILE *fp = fopen(filename, "w");
    if (fp) {
      PRINTF("Writing to %s\n", filename);
      artsArrayListIterator *iter = artsNewArrayListIterator(list[index]);
      unsigned int *bin;
      while (artsArrayListHasNext(iter)) {
        bin = (unsigned int *)artsArrayListNext(iter);
        fprintf(fp, "%u\n", *bin);
      }
      artsDeleteArrayListIterator(iter);
      fclose(fp);
    } else
      PRINTF("Couldn't open %s\n", filename);
  }
}
