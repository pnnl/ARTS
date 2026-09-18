/* Fake Rapid MemOps.h — producer/consumer flush with tracking */
#ifndef FAKE_CXL_MEM_OPS_H
#define FAKE_CXL_MEM_OPS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void fake_cxl_flush_producer(const void *ptr, size_t size);
void fake_cxl_flush_consumer(const void *ptr, size_t size);

#ifdef __cplusplus
}
#endif

#define FLUSH_FENCE_PRODUCER(ptr, size) \
    fake_cxl_flush_producer((const void *)(ptr), (size_t)(size))
#define FLUSH_FENCE_CONSUMER(ptr, size) \
    fake_cxl_flush_consumer((const void *)(ptr), (size_t)(size))

#endif /* FAKE_CXL_MEM_OPS_H */
