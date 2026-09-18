#include "fake_cxl_internal.h"

#include <string.h>

/* x86 cache-line flush instructions via inline asm — no -mclflushopt needed.
 * On non-x86 platforms these compile away to compiler barriers, which is fine:
 * the fake region is regular DRAM and the CPU handles cache coherence itself. */
#if defined(__x86_64__) || defined(__i386__)
static inline void _clflushopt(const void *p) {
    __asm__ volatile("clflushopt %0" : "+m"(*(volatile char *)(uintptr_t)p) :: "memory");
}
static inline void _sfence(void) {
    __asm__ volatile("sfence" ::: "memory");
}
static inline void _clflush(const void *p) {
    __asm__ volatile("clflush %0" : "+m"(*(volatile char *)(uintptr_t)p) :: "memory");
}
#else
static inline void _clflushopt(const void *p) { (void)p; __asm__ volatile("" ::: "memory"); }
static inline void _sfence(void)               { __asm__ volatile("" ::: "memory"); }
static inline void _clflush(const void *p)     { (void)p; __asm__ volatile("" ::: "memory"); }
#endif

void fake_cxl_flush_producer(const void *ptr, size_t size) {
    flush_log_record(ptr, size, 0);
    /* Flush every cache line in the range, then store fence. */
    const char *p   = (const char *)((uintptr_t)ptr & ~(uintptr_t)63);
    const char *end = (const char *)ptr + size;
    while (p < end) {
        _clflushopt(p);
        p += 64;
    }
    _sfence();
}

void fake_cxl_flush_consumer(const void *ptr, size_t size) {
    flush_log_record(ptr, size, 1);
    const char *p   = (const char *)((uintptr_t)ptr & ~(uintptr_t)63);
    const char *end = (const char *)ptr + size;
    while (p < end) {
        _clflush(p);
        p += 64;
    }
}
