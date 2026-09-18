#include "fake_cxl_internal.h"
#include <stdlib.h>

/* All allocation functions are thin wrappers around the bump allocator.
 * Free is a no-op: the bump allocator does not support individual frees.
 * This matches the Rapid API's FAM-memory semantics where the region is
 * reclaimed on process exit, not per-object. */

void *fake_cxl_shared_malloc(size_t size) {
    return fake_cxl_bump_alloc(size);
}

void *fake_cxl_global_malloc(size_t size) {
    return fake_cxl_bump_alloc(size);
}

/* On a single-device fake, device-specific allocation is identical to global. */
void *fake_cxl_global_malloc_dev(size_t size, uint64_t dev_id) {
    (void)dev_id; /* TODO: sub-region per device once multi-device is needed */
    return fake_cxl_bump_alloc(size);
}

void fake_cxl_global_free(void *ptr) {
    (void)ptr; /* no-op: bump allocator */
}

void fake_cxl_shared_free(void *ptr) {
    (void)ptr; /* no-op: bump allocator */
}

/* On a single node, "last rank" semantics are the same as a plain alloc. */
void *fake_cxl_last_shared_malloc(size_t size) {
    return fake_cxl_bump_alloc(size);
}

int fake_cxl_is_fam_ptr(const void *ptr) {
    uintptr_t p = (uintptr_t)ptr;
    uintptr_t base = (uintptr_t)g_cxl_base;
    return g_cxl_base && p >= base && p < base + g_cxl_size;
}

uint64_t fake_cxl_get_fam_dev_id(const void *ptr) {
    (void)ptr;
    return 0; /* single device */
}

uint64_t fake_cxl_get_fam_region_dev_id(void) {
    return 0; /* single device */
}

unsigned fake_cxl_get_fam_dev_count(void) {
    return 1;
}
