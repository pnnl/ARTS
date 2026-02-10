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
#include "mm_util.h"

#include <stdlib.h>
#include <string.h>

#include "arts/runtime/rt.h"

void print_matrix(unsigned int row_size, double *mat) {
  unsigned int column_size = row_size;
  for (unsigned int i = 0; i < column_size; i++) {
    for (unsigned int j = 0; j < row_size; j++) {
      ARTS_PRINTF("%5.2f ", mat[(i * row_size) + j]);
    }
    ARTS_PRINTF("\n");
  }
}

void init_matrix(unsigned int row_size, double *mat, bool identity, bool zero) {
  unsigned int column_size = row_size;
  for (unsigned int i = 0; i < column_size; i++) {
    for (unsigned int j = 0; j < row_size; j++) {
      if (zero) {
        mat[(i * row_size) + j] = 0;
      } else if (identity) {
        if (i == j) {
          mat[(i * row_size) + j] = 1;
        } else {
          mat[(i * row_size) + j] = 0;
}
      } else {
        mat[(i * row_size) + j] = rand() % 10;
}
    }
  }
}

void copy_block(unsigned int x, unsigned int y, unsigned int tile_row_size,
               double *tile, unsigned int row_size, double *mat, bool to_tile) {
  unsigned int tile_column_size = tile_row_size;

  unsigned int x_offset = tile_row_size * y;
  unsigned int y_offset = tile_column_size * x;

  if (to_tile) {
    for (unsigned int i = 0; i < tile_column_size; i++) {
      memcpy(&tile[i * tile_row_size], &mat[((i + y_offset) * row_size) + x_offset],
             tile_row_size * sizeof(double));
}
  } else {
    for (unsigned int i = 0; i < tile_column_size; i++) {
      memcpy(&mat[((i + y_offset) * row_size) + x_offset], &tile[i * tile_row_size],
             tile_row_size * sizeof(double));
}
  }
}