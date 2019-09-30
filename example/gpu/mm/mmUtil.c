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
#include <stdio.h>
#include <stdlib.h>
#include "arts.h"

void printMatrix(unsigned int rowSize, double * mat)
{
    unsigned int columnSize = rowSize;
    for(unsigned int i=0; i<columnSize; i++)
    {
        for(unsigned int j=0; j<rowSize; j++)
        {
            printf("%5.2f ", mat[i*rowSize + j]);
        }
        printf("\n");
    }
}

void initMatrix(unsigned int rowSize, double * mat, bool identity, bool zero)
{
    unsigned int columnSize = rowSize;
    for(unsigned int i=0; i<columnSize; i++)
    {
        for(unsigned int j=0; j<rowSize; j++)
        {
            if(zero)
                mat[i*rowSize + j] = 0;
            else if(identity)
            {
                if(i==j)
                    mat[i*rowSize + j] = 1;
                else
                    mat[i*rowSize + j] = 0;
            }
            else
                mat[i*rowSize + j] = rand()%10;
        }
    }
}

void copyBlock(unsigned int x, unsigned int y, unsigned int tileRowSize, double * tile, unsigned int rowSize, double * mat, bool toTile)
{
    unsigned int tileColumnSize = tileRowSize;
    
    unsigned int xOffset = tileRowSize    * y;
    unsigned int yOffset = tileColumnSize * x;
    
    if(toTile)
    {
        for(unsigned int i=0; i<tileColumnSize; i++)
            memcpy(&tile[ i * tileRowSize ], &mat[ (i + yOffset) * rowSize + xOffset ], tileRowSize * sizeof(double));
    }
    else
    {
        for(unsigned int i=0; i<tileColumnSize; i++)
            memcpy(&mat[ (i + yOffset) * rowSize + xOffset ], &tile[ i * tileRowSize ], tileRowSize * sizeof(double));
    }

}