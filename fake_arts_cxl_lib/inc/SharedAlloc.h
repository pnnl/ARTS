/* Fake Rapid SharedAlloc.h — single-node shm implementation */
#ifndef FAKE_CXL_SHARED_ALLOC_H
#define FAKE_CXL_SHARED_ALLOC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void    *fake_cxl_shared_malloc(size_t size);
void    *fake_cxl_global_malloc(size_t size);
void    *fake_cxl_global_malloc_dev(size_t size, uint64_t dev_id);
void     fake_cxl_global_free(void *ptr);
void     fake_cxl_shared_free(void *ptr);
void    *fake_cxl_last_shared_malloc(size_t size);
int      fake_cxl_is_fam_ptr(const void *ptr);
uint64_t fake_cxl_get_fam_dev_id(const void *ptr);
uint64_t fake_cxl_get_fam_region_dev_id(void);
unsigned fake_cxl_get_fam_dev_count(void);

#ifdef __cplusplus
}
#endif

#define SHARED_CXL_MALLOC(sz)               fake_cxl_shared_malloc(sz)
#define GLOBAL_CXL_MALLOC(sz)               fake_cxl_global_malloc(sz)
#define GLOBAL_CXL_MALLOC_DEV(sz, dev)      fake_cxl_global_malloc_dev((sz), (uint64_t)(dev))
#define GLOBAL_CXL_FREE(ptr)                fake_cxl_global_free((void *)(ptr))
#define SHARED_CXL_FREE(ptr)                fake_cxl_shared_free((void *)(ptr))
#define LAST_SHARED_CXL_MALLOC(sz)          fake_cxl_last_shared_malloc(sz)
#define SHARED_CXL_MALLOC_INITIALIZED(...)  ((void)0)
#define IS_FAM_PTR(ptr)                     fake_cxl_is_fam_ptr((const void *)(uintptr_t)(ptr))
#define GET_FAM_DEV_ID(ptr)                 fake_cxl_get_fam_dev_id((const void *)(uintptr_t)(ptr))
#define GET_FAM_REGION_DEV_ID()             fake_cxl_get_fam_region_dev_id()
#define GET_FAM_DEV_COUNT()                 fake_cxl_get_fam_dev_count()

#endif /* FAKE_CXL_SHARED_ALLOC_H */
