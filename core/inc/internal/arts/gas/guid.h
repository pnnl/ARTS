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
#ifndef ARTS_GAS_GUID_H
#define ARTS_GAS_GUID_H
#ifdef __cplusplus
extern "C" {
#endif

#include "arts/runtime/rt.h"

typedef union {
  intptr_t bits : 64;
  struct __attribute__((packed)) {
    uint8_t type : 8;
    uint16_t rank : 16;
    uint64_t key : 40;
  } fields;
} arts_guid_bits_t;

arts_guid_t arts_guid_create_for_rank(unsigned int route, unsigned int type);
void arts_guid_key_generator_init();
void set_global_guid_on();
void set_guid_generator_after_parallel_start();
uint64_t arts_get_guid_key(arts_guid_t guid);
uint64_t arts_hash_guid_key(arts_guid_t guid);
arts_guid_range_t *arts_new_guid_range_node_hash(arts_type_t type, unsigned int size,
                                        unsigned int route,
                                        unsigned int hash_size);

#ifdef __cplusplus
}
#endif

#endif
