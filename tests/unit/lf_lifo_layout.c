/* SPDX-License-Identifier: Apache-2.0
 *
 * T004 — arts_lf_link_t / arts_lf_stack_t layout parity (lockfree_lifo.h).
 *
 * lockfree_lifo.h defines TWO struct definitions guarded by #ifdef __cplusplus:
 * the C path uses C11 _Atomic on `next`/`head`; the C++/nvcc path drops
 * _Atomic for layout-only visibility (the inline helpers are C-only).  A
 * struct-drift between the two would corrupt cross-language compilation (the
 * header is pulled into .cu TUs via runtime_types.h -> arts_event_s).  Census
 * 29.md §2 GAP: "No C++/nvcc layout-compile test."
 *
 * This file asserts the invariants the layout must satisfy.  It is compiled
 * BOTH as C and as C++ by CMake (same source, two languages); the static
 * asserts must hold under each.  Because _Atomic(T*) and plain T* have the
 * same size/alignment/representation on the supported ISAs (lock-free
 * pointer-width atomics), the two struct definitions are layout-identical.
 *
 * Pure compile-time test: success is simply that it compiles + runs and prints
 * PASS.  No threads.
 */

#include "arts/utils/lockfree_lifo.h"

#include <stddef.h>
#include <stdio.h>

#if defined(__cplusplus)
#define LF_STATIC_ASSERT(c, m) static_assert((c), m)
#define LF_ALIGNOF(t) alignof(t)
#else
#define LF_STATIC_ASSERT(c, m) _Static_assert((c), m)
#define LF_ALIGNOF(t) _Alignof(t)
#endif

typedef void *lf_voidp_t; /* a type name so alignof applies cleanly */

/* The intrusive link is a single pointer-width cell. */
LF_STATIC_ASSERT(sizeof(arts_lf_link_t) == sizeof(lf_voidp_t),
                 "arts_lf_link_t must be one pointer wide");
LF_STATIC_ASSERT(offsetof(arts_lf_link_t, next) == 0,
                 "arts_lf_link_t.next must be at offset 0");
LF_STATIC_ASSERT(LF_ALIGNOF(arts_lf_link_t) == LF_ALIGNOF(lf_voidp_t),
                 "arts_lf_link_t alignment must match a pointer");

/* The stack is a single head cell, same width/alignment as a pointer. */
LF_STATIC_ASSERT(sizeof(arts_lf_stack_t) == sizeof(lf_voidp_t),
                 "arts_lf_stack_t must be one pointer wide");
LF_STATIC_ASSERT(offsetof(arts_lf_stack_t, head) == 0,
                 "arts_lf_stack_t.head must be at offset 0");
LF_STATIC_ASSERT(LF_ALIGNOF(arts_lf_stack_t) == LF_ALIGNOF(lf_voidp_t),
                 "arts_lf_stack_t alignment must match a pointer");

int main(void) {
#if defined(__cplusplus)
  const char *lang = "C++";
#else
  const char *lang = "C";
#endif
  printf("PASS lf_lifo_layout (%s): link=%zu stack=%zu both pointer-wide, "
         "offset0\n",
         lang, sizeof(arts_lf_link_t), sizeof(arts_lf_stack_t));
  return 0;
}
