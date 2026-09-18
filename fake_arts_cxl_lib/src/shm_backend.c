#define _GNU_SOURCE
#include "fake_cxl_internal.h"

#include <errno.h>
#include <fcntl.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

void *g_cxl_base = NULL;
size_t g_cxl_size = 0;

static void shm_cleanup(void) {
    shm_unlink(FAKE_CXL_SHM_NAME);
}

__attribute__((constructor))
static void fake_cxl_init_shm(void) {
    if (g_cxl_base != NULL)
        return;

    const char *env = getenv("ARTS_FAKE_CXL_REGION_SIZE");
    g_cxl_size = env ? (size_t)strtoull(env, NULL, 0) : FAKE_CXL_DEFAULT_SIZE;
    if (g_cxl_size < (64ULL << 20)) {
        fprintf(stderr, "fake_cxl: ARTS_FAKE_CXL_REGION_SIZE too small (min 64 MiB)\n");
        abort();
    }

    /* Rank 0 creates; other ranks (same node, different processes) open. */
    int fd = shm_open(FAKE_CXL_SHM_NAME, O_CREAT | O_RDWR, 0600);
    if (fd < 0) {
        fprintf(stderr, "fake_cxl: shm_open failed: %s\n", strerror(errno));
        abort();
    }
    if (ftruncate(fd, (off_t)g_cxl_size) < 0) {
        fprintf(stderr, "fake_cxl: ftruncate(%zu) failed: %s\n",
                g_cxl_size, strerror(errno));
        close(fd);
        abort();
    }

#ifdef MAP_FIXED_NOREPLACE
    int mflags = MAP_SHARED | MAP_FIXED_NOREPLACE;
#else
    int mflags = MAP_SHARED | MAP_FIXED;
#endif
    void *base = mmap((void *)FAKE_CXL_BASE_ADDR, g_cxl_size,
                      PROT_READ | PROT_WRITE, mflags, fd, 0);
    if (base == MAP_FAILED || base != (void *)FAKE_CXL_BASE_ADDR) {
        fprintf(stderr,
                "fake_cxl: mmap at 0x%llx (size %zu) failed: %s\n"
                "  Hint: set ARTS_FAKE_CXL_REGION_SIZE to a smaller value,\n"
                "  or verify the address window is free.\n",
                (unsigned long long)FAKE_CXL_BASE_ADDR, g_cxl_size,
                base == MAP_FAILED ? strerror(errno) : "wrong address");
        if (base != MAP_FAILED)
            munmap(base, g_cxl_size);
        close(fd);
        abort();
    }
    close(fd);
    g_cxl_base = base;

    /* Initialize the allocator cursor on the first process to map the region.
     * Use a compare-and-swap so subsequent processes don't reset it. */
    cxl_region_header_t *hdr = (cxl_region_header_t *)g_cxl_base;
    uint64_t expected = 0;
    uint64_t initial  = sizeof(cxl_region_header_t);
    __atomic_compare_exchange_n(&hdr->alloc_cursor, &expected, initial,
                                0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);

    atexit(shm_cleanup);
}

void *fake_cxl_bump_alloc(size_t size) {
    if (!g_cxl_base)
        fake_cxl_init_shm();

    const size_t align = 64;  /* cacheline */
    size_t aligned = (size + align - 1) & ~(size_t)(align - 1);

    cxl_region_header_t *hdr = (cxl_region_header_t *)g_cxl_base;
    uint64_t off = __atomic_fetch_add(&hdr->alloc_cursor, (uint64_t)aligned,
                                     __ATOMIC_SEQ_CST);
    if (off + aligned > g_cxl_size) {
        fprintf(stderr,
                "fake_cxl: FAM region exhausted (size %zu, requested %zu at +%llu)\n"
                "  Increase ARTS_FAKE_CXL_REGION_SIZE (current: %zu bytes)\n",
                g_cxl_size, size, (unsigned long long)off, g_cxl_size);
        abort();
    }
    return (char *)g_cxl_base + off;
}
