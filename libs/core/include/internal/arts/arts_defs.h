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
#ifndef ARTS_DEFS_H
#define ARTS_DEFS_H

/**
 * @file arts_defs.h
 * @brief Compiler attribute abstraction layer for GCC/Clang.
 *
 * ARTS targets strict @c -std=c17 (@c CMAKE_C_EXTENSIONS=OFF).
 * Double-underscore attribute spellings (@c __packed__, @c __weak__, …)
 * are silently accepted by GCC/Clang under @c -std=c17 without needing
 * @c __extension__.  These wrapper macros allow attributes to appear in
 * any position (before declarations, between @c struct and tag name,
 * after parameter lists, etc.).
 */

#if !defined(__GNUC__) && !defined(__clang__)
#error "ARTS requires GCC or Clang."
#endif

/** Generic attribute wrapper — passes @p x to @c __attribute__. */
#define ARTS_ATTRIBUTE(x) __attribute__((x))

/** Structure packing — place between @c struct and the tag name.
 *  @code struct ARTS_PACKED foo { ... }; @endcode */
#define ARTS_PACKED ARTS_ATTRIBUTE(__packed__)

/** Weak symbol — allows user programs to override default implementations. */
#define ARTS_WEAK ARTS_ATTRIBUTE(__weak__)

/** Weak-import attribute — uses @c __weak_import__ on macOS, @c __weak__
 * elsewhere. */
#ifdef __APPLE__
#define ARTS_WEAK_IMPORT ARTS_ATTRIBUTE(__weak_import__)
#else
#define ARTS_WEAK_IMPORT ARTS_WEAK
#endif

/** Pure function — return value depends only on arguments and global state.
 *  Enables additional compiler optimizations for idempotent functions. */
#define ARTS_PURE ARTS_ATTRIBUTE(__pure__)

/** Custom alignment — wraps @c __attribute__((aligned(n))).
 *  @param n Required alignment in bytes (must be a power of two). */
#define ARTS_ALIGNED(n) ARTS_ATTRIBUTE(__aligned__(n))

/** Max natural alignment (replaces bare @c __attribute__((aligned))). */
#define ARTS_ALIGNED_MAX ARTS_ATTRIBUTE(__aligned__)

/** Branch prediction hint — indicates the condition is likely true. */
#define ARTS_LIKELY(x) __extension__ __builtin_expect(!!(x), 1)

/** Branch prediction hint — indicates the condition is likely false. */
#define ARTS_UNLIKELY(x) __extension__ __builtin_expect(!!(x), 0)

/** Thread-local storage — C11 @c _Thread_local in C, C++11 @c thread_local
 *  in C++/CUDA.  Replaces the non-standard @c __thread throughout ARTS. */
#ifdef __cplusplus
#define ARTS_THREAD_LOCAL thread_local
#else
#define ARTS_THREAD_LOCAL _Thread_local
#endif

/** 128-bit unsigned integer type (used for 128-bit CAS in lock-free queues).
 *  Not available under NVCC — CUDA does not support 128-bit integers. */
#ifndef __CUDACC__
__extension__ typedef unsigned __int128 arts_uint128_t;
#endif

#endif /* ARTS_DEFS_H */
