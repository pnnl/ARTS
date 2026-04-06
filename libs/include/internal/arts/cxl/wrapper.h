/******************************************************************************
 * Copyright 2019 Battelle Memorial Institute
 * Licensed under the Apache License, Version 2.0
 ******************************************************************************/
#ifndef ARTS_CXL_WRAPPER_H
#define ARTS_CXL_WRAPPER_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>

#define CACHELINE_SIZE 64

/** Round @p x up to the nearest multiple of @p a (power of two). */
#define ALIGN_UP(x, a) (((x) + ((a) - 1)) & ~((a) - 1))

/*
 * Two-level CXL configuration:
 *
 *   ARTS_USE_CXL      — Enables CXL code paths (DB type, scheduler, GUID
 *                        encoding).  Always safe to set.
 *
 *   ARTS_CXL_NATIVE   — Set by CMake when the Rapid API + arts_cxl_lib are
 *                        actually found.  Maps SHARED_MALLOC / FLUSH_FENCE
 *                        to real CXL library calls.
 *
 * When ARTS_USE_CXL is defined but ARTS_CXL_NATIVE is NOT, the CXL code
 * compiles and runs with regular heap memory (stubs).  This is useful for
 * development, testing, and CI without CXL hardware.
 */

#if defined(ARTS_USE_CXL) && defined(ARTS_CXL_NATIVE)
/* ── Real CXL library (Rapid API + arts_cxl_lib) ─────────────────────────── */
#include <SharedAlloc.h>
#include <MemOps.h>
#define SHARED_MALLOC( ... )              SHARED_CXL_MALLOC(__VA_ARGS__)
#define GLOBAL_MALLOC( ... )              GLOBAL_CXL_MALLOC(__VA_ARGS__)
#define GLOBAL_MALLOC_DEV( ... )          GLOBAL_CXL_MALLOC_DEV(__VA_ARGS__)
#define GLOBAL_FREE( ... )                GLOBAL_CXL_FREE(__VA_ARGS__)
#define SHARED_FREE( ... )                SHARED_CXL_FREE(__VA_ARGS__)
#define SHARED_MALLOC_INITIALIZED(...)  SHARED_CXL_MALLOC_INITIALIZED(__VA_ARGS__)
#define LAST_SHARED_MALLOC(...)         LAST_SHARED_CXL_MALLOC(__VA_ARGS__)
#define IS_CXL_PTR(...)                 IS_FAM_PTR(__VA_ARGS__)
#define GET_CXL_DEV_ID( ... )           GET_FAM_DEV_ID( __VA_ARGS__ )
#define GET_CXL_REGION_DEV_ID( ... )    GET_FAM_REGION_DEV_ID( __VA_ARGS__ )
#define GET_CXL_DEV_COUNT( ... )        GET_FAM_DEV_COUNT( __VA_ARGS__ )
#else
/* ── Stubs (no CXL, or CXL code paths without hardware) ──────────────────── */
#define SHARED_MALLOC(...)              malloc(__VA_ARGS__)
#define GLOBAL_MALLOC(...)              malloc(__VA_ARGS__)
#define GLOBAL_FREE(...)                free(__VA_ARGS__)
#define SHARED_FREE(...)                free(__VA_ARGS__)
#define SHARED_MALLOC_INITIALIZED(...)
#define LAST_SHARED_MALLOC(...)         malloc(__VA_ARGS__)
#define IS_CXL_PTR(...)                 ((void)0, 0)
#define GET_CXL_DEV_ID( ... )           ((void)0, 0)
#define GET_CXL_DEV_COUNT( ... )        ((void)0, 0)
#define GET_CXL_REGION_DEV_ID( ... )    ((void)0, 0)
#ifndef FLUSH_FENCE_PRODUCER
#define FLUSH_FENCE_PRODUCER(...)
#endif
#ifndef FLUSH_FENCE_CONSUMER
#define FLUSH_FENCE_CONSUMER(...)
#endif
#endif

#ifdef __cplusplus
}
#endif
#endif /* ARTS_CXL_WRAPPER_H */
