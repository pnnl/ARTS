/******************************************************************************
 * Copyright 2019 Battelle Memorial Institute
 * Licensed under the Apache License, Version 2.0
 ******************************************************************************/
#ifndef ARTS_CXL_SHARED_ALLOC_H
#define ARTS_CXL_SHARED_ALLOC_H
#ifdef __cplusplus
extern "C" {
#endif

#define COMPILER_DO_NOT_REORDER_WRITES() __asm__ volatile("" : : : "memory")
#define HW_MEMORY_FENCE() __sync_synchronize()

#ifdef __cplusplus
}
#endif
#endif /* ARTS_CXL_SHARED_ALLOC_H */
