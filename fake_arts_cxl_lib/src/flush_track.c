#define _GNU_SOURCE
#include "fake_cxl_internal.h"

#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static flush_entry_t g_flush_log[FLUSH_LOG_CAPACITY];
static _Atomic uint32_t g_flush_log_idx = 0;

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

void flush_log_record(const void *ptr, size_t size, uint8_t type) {
    uint32_t idx = atomic_fetch_add_explicit(&g_flush_log_idx, 1u,
                                             memory_order_relaxed);
    if (idx >= FLUSH_LOG_CAPACITY)
        return; /* ring full — stop recording */
    flush_entry_t *e = &g_flush_log[idx];
    e->timestamp_ns = now_ns();
    e->thread_id    = (uint64_t)(uintptr_t)pthread_self();
    e->ptr          = (uintptr_t)ptr;
    e->size         = size;
    e->type         = type;
    memset(e->_pad, 0, sizeof(e->_pad));
}

static void write_all(int fd, const void *buf, size_t n) {
    const char *p = buf;
    while (n > 0) {
        ssize_t w = write(fd, p, n);
        if (w <= 0) break;
        p += (size_t)w;
        n -= (size_t)w;
    }
}

__attribute__((destructor))
static void flush_log_write(void) {
    uint32_t count = atomic_load_explicit(&g_flush_log_idx, memory_order_relaxed);
    if (count > FLUSH_LOG_CAPACITY)
        count = FLUSH_LOG_CAPACITY;
    if (count == 0)
        return;

    const char *path = getenv("ARTS_FLUSH_LOG");
    if (!path)
        path = "arts_flush_trace.bin";

    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        perror("fake_cxl: cannot write flush log");
        return;
    }

    /* Write a 16-byte file header: magic + entry count + entry size */
    uint64_t header[2] = { 0x4152545346434C58ULL /* ARTSFCLX */, count };
    write_all(fd, header, sizeof(header));
    write_all(fd, g_flush_log, count * sizeof(flush_entry_t));
    close(fd);

    fprintf(stderr,
            "fake_cxl: flush log written to %s (%u entries, %zu bytes)\n",
            path, count, (size_t)count * sizeof(flush_entry_t));
}
