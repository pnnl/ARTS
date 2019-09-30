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

/*

GPU Scheduling policies
-----------------------

1. Fitting based (MU) : Find the best fit with respect to available memory in a GPU.
2. Locality based
    a. All Hits (AND) : if (Find the GPU with all the required data blocks) else (random policy).
    b. At least one hit (OR) : if (Find the GPU with at least one of the required data blocks) else (random policy).
    c. AND-MU : if (Find the GPU with all the required data blocks) else if (MU) else (random policy).
    d. OR-MU : if (Find the GPU with at least one of the required data blocks) else if (MU) else (random policy).
3. Inter-device Fetching
    a. D2D Memcpy : if (data block is a hit in one GPU and Edt is scheduled in another) Move data block DeviceToDevice instead of HostToDevice.
    b. D2D Load-Store : if (data block is a hit in one GPU and Edt is scheduled in another) Access data blocks using P2P L/S.
    c. Prefetch : Move data on to the device for future Edts peeking up the stack.
4. Random : 
    a. LRU
    b. Round Robin
    c. Time-bomb
    d. Greedy than oldest
    e. FIFO
    f. Prediction : Evict a data block based on static analysis or compile-time percolated knowledge.
*/