#include <SharedAlloc.h>
#include <MemOps.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>

#define FAKE_CXL_BASE_ADDR 0x200000000000ULL

static int errors = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); errors++; } \
    else         { printf("PASS: %s\n", msg); } \
} while(0)

int main(void) {
    printf("=== fake_arts_cxl_lib smoke test ===\n\n");

    /* --- IS_FAM_PTR on a stack address should be false --- */
    int stack_var;
    CHECK(!IS_FAM_PTR(&stack_var), "IS_FAM_PTR(stack) == false");

    /* --- Allocate a small object in FAM --- */
    size_t alloc_size = 1024;
    void *p = SHARED_CXL_MALLOC(alloc_size);
    CHECK(p != NULL, "SHARED_CXL_MALLOC returns non-NULL");

    /* --- The returned pointer must be inside the FAM window --- */
    uint64_t addr = (uint64_t)(uintptr_t)p;
    CHECK(addr >= FAKE_CXL_BASE_ADDR, "allocation is at or above FAM base");
    CHECK(IS_FAM_PTR(p), "IS_FAM_PTR(FAM ptr) == true");

    /* --- Write into the FAM region, do a producer flush --- */
    memset(p, 0xAB, alloc_size);
    FLUSH_FENCE_PRODUCER(p, alloc_size);
    CHECK(1, "FLUSH_FENCE_PRODUCER did not crash");

    /* --- Read back, do a consumer flush --- */
    FLUSH_FENCE_CONSUMER(p, alloc_size);
    CHECK(((unsigned char *)p)[0] == 0xAB, "FAM memory round-trips correctly");

    /* --- A second allocation must be strictly above the first --- */
    void *q = GLOBAL_CXL_MALLOC(64);
    CHECK(q != NULL, "GLOBAL_CXL_MALLOC returns non-NULL");
    CHECK((uintptr_t)q > (uintptr_t)p, "second alloc is above first (bump)");
    CHECK(IS_FAM_PTR(q), "IS_FAM_PTR(second alloc) == true");

    /* --- Device helpers --- */
    CHECK(GET_FAM_DEV_COUNT() == 1, "GET_FAM_DEV_COUNT() == 1");
    CHECK(GET_FAM_DEV_ID(p) == 0, "GET_FAM_DEV_ID returns 0");
    CHECK(GET_FAM_REGION_DEV_ID() == 0, "GET_FAM_REGION_DEV_ID returns 0");

    /* --- GLOBAL_CXL_FREE is a no-op (should not crash) --- */
    GLOBAL_CXL_FREE(p);
    CHECK(1, "GLOBAL_CXL_FREE did not crash");

    /* --- Free on a stack pointer is also a no-op --- */
    SHARED_CXL_FREE(&stack_var);
    CHECK(1, "SHARED_CXL_FREE(stack) did not crash");

    /* --- LAST_SHARED_CXL_MALLOC --- */
    void *r = LAST_SHARED_CXL_MALLOC(256);
    CHECK(r != NULL && IS_FAM_PTR(r), "LAST_SHARED_CXL_MALLOC returns FAM ptr");

    /* --- SHARED_CXL_MALLOC_INITIALIZED is a no-op macro --- */
    SHARED_CXL_MALLOC_INITIALIZED(p);
    CHECK(1, "SHARED_CXL_MALLOC_INITIALIZED did not crash");

    printf("\n");
    if (errors == 0)
        printf("ALL TESTS PASSED\n");
    else
        printf("%d TEST(S) FAILED\n", errors);

    return errors ? 1 : 0;
}
