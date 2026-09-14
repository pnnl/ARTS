/* SPDX-License-Identifier: Apache-2.0 */

/* Compile the same layout probe as C++ so runtime_types.h selects its
 * qualifier-free C++/nvcc declarations. */
#include "runtime_edt_event_layout.c"
