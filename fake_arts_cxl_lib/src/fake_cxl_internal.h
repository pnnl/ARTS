#pragma once
#include <stddef.h>
#include <stdint.h>

/* FAM region layout */
#define FAKE_CXL_BASE_ADDR    0x200000000000ULL   /* must match ARTS_CXL_BASE_ADDR */
#define FAKE_CXL_DEFAULT_SIZE (32ULL << 30)        /* 32 GiB */
#define FAKE_CXL_SHM_NAME     "/arts_fake_cxl"

/* First 64 bytes of the FAM region are the header (allocator cursor). */
typedef struct {
    volatile uint64_t alloc_cursor;
    uint8_t           _pad[56];
} cxl_region_header_t;

/* Shared state set by fake_cxl_init_shm() */
extern void  *g_cxl_base;
extern size_t g_cxl_size;

/* Bump-allocate @size bytes (cacheline-aligned) from the FAM region. */
void *fake_cxl_bump_alloc(size_t size);

/* Flush log entry */
typedef struct {
    uint64_t  timestamp_ns;
    uint64_t  thread_id;
    uintptr_t ptr;
    size_t    size;
    uint8_t   type;   /* 0 = producer, 1 = consumer */
    uint8_t   _pad[7];
} flush_entry_t;

#define FLUSH_LOG_CAPACITY (1U << 20)  /* 1 M entries = 32 MiB */
void flush_log_record(const void *ptr, size_t size, uint8_t type);
