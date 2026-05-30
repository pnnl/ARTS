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
#ifndef ARTS_ARRAY_DB_H
#define ARTS_ARRAY_DB_H
#ifdef __cplusplus
extern "C" {
#endif

#include "arts.h"

/** Distributed array DataBlock spanning multiple nodes. */
struct arts_array_db_s {
  unsigned int element_size;       /**< Size of each element in bytes. */
  unsigned int elements_per_block; /**< Elements per node-local block. */
  unsigned int num_blocks;         /**< Total number of blocks. */
  char head[];                     /**< Flexible array of block GUIDs. */
};
typedef struct arts_array_db_s arts_array_db_t;

/**
 * @brief Create a distributed array DB spread equally across all nodes.
 *
 * @param[out] addr         Receives the local portion pointer.
 * @param      element_size Size of each element in bytes.
 * @param      num_elements Total number of elements.
 * @return GUID for accessing the array DB.
 */
arts_guid_t arts_new_array_db(arts_array_db_t **addr, unsigned int element_size,
                              unsigned int num_elements);

/**
 * @brief Create a distributed array DB with a pre-reserved @p guid.
 *
 * The GUID can target any node but must be of type @c ARTS_DB_PIN.
 *
 * @param guid         Pre-reserved GUID.
 * @param element_size Size of each element in bytes.
 * @param num_elements Total number of elements.
 * @return Pointer to the local arts_array_db_t.
 */
arts_array_db_t *arts_new_array_db_with_guid(arts_guid_t guid,
                                             unsigned int element_size,
                                             unsigned int num_elements);

/**
 * @brief Create a node-local array DB (not shared across nodes).
 *
 * @param guid         Pre-reserved GUID.
 * @param element_size Size of each element in bytes.
 * @param num_elements Total number of elements.
 * @param data         Optional initial data (may be @c NULL).
 * @return Pointer to the local arts_array_db_t.
 */
arts_array_db_t *arts_new_local_array_db_with_guid(arts_guid_t guid,
                                                   unsigned int element_size,
                                                   unsigned int num_elements,
                                                   void *data);

/**
 * @brief Signal an EDT with all blocks of an array DB.
 *
 * @param array    Array DB.
 * @param edt_guid Target EDT.
 * @param slot     Starting dependency slot.
 */
void arts_signal_array_db(arts_array_db_t *array, arts_guid_t edt_guid,
                          unsigned int slot);

/**
 * @brief Read an element from an array DB at @p index.
 *
 * Delivered to @p edt_guid at @p slot as a pointer dependency.
 *
 * @param edt_guid Destination EDT.
 * @param slot     Dependency slot.
 * @param array    Array DB.
 * @param index    Element index.
 */
void arts_get_from_array_db(arts_guid_t edt_guid, unsigned int slot,
                            arts_array_db_t *array, unsigned int index);

/**
 * @brief Write data into an array DB element and signal an EDT.
 *
 * The put falls within the current epoch.
 *
 * @param ptr      Source data.
 * @param edt_guid EDT to signal.
 * @param slot     Dependency slot.
 * @param array    Array DB.
 * @param index    Element index.
 */
void arts_put_in_array_db(void *ptr, arts_guid_t edt_guid, unsigned int slot,
                          arts_array_db_t *array, unsigned int index);

/**
 * @brief Launch an EDT for each element locally.
 *
 * Data is acquired read-only as a pointer dependency.
 *
 * @param array    Array DB.
 * @param func_ptr Function to execute per element.
 * @param paramc   Number of static parameters.
 * @param paramv   Array of parameters.
 */
void arts_for_each_in_array_db(arts_array_db_t *array, arts_edt_t func_ptr,
                               uint32_t paramc, const uint64_t *paramv);

/**
 * @brief Launch an EDT for each element across all nodes.
 *
 * Data is acquired read-only as a pointer dependency.
 *
 * @param array    Array DB.
 * @param stride   Number of elements per EDT.
 * @param func_ptr Function to execute.
 * @param paramc   Number of static parameters.
 * @param paramv   Array of parameters.
 */
void arts_for_each_in_array_db_at_data(arts_array_db_t *array,
                                       unsigned int stride, arts_edt_t func_ptr,
                                       uint32_t paramc, const uint64_t *paramv);

/**
 * @brief Gather all chunks of an array DB on one node and run an EDT.
 *
 * @param array    Array DB.
 * @param func_ptr Function to execute after gathering.
 * @param route    Node to gather on.
 * @param paramc   Number of static parameters.
 * @param paramv   Array of parameters.
 * @param depc     Number of dependency slots (usually num_blocks).
 */
void arts_gather_array_db(arts_array_db_t *array, arts_edt_t func_ptr,
                          unsigned int rank, uint32_t paramc,
                          const uint64_t *paramv, uint32_t depc);

/** @brief Gather array DB within a specific epoch. */
void arts_gather_array_db_epoch(arts_array_db_t *array, arts_edt_t func_ptr,
                                unsigned int rank, uint32_t paramc,
                                const uint64_t *paramv, uint32_t depc,
                                arts_guid_t epoch_guid);

/** @brief Gather array DB chunks into an existing EDT. */
void arts_gather_array_db_in_edt(arts_array_db_t *array,
                                 arts_guid_t to_edt_guid, uint64_t slot_offset);

/** @brief Get total element count of an array DB. */
unsigned int arts_get_size_array_db(arts_array_db_t *array);

#ifdef __cplusplus
}
#endif
#endif /* ARTS_ARRAY_DB_H */
