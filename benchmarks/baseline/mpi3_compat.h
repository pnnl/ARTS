/* MPI-3.0 compatibility wrapper for legacy code using removed MPI-1.x
 * functions. Include this BEFORE mpi.h to suppress removal errors in Open
 * MPI 5.x. */
#ifndef BASELINE_MPI3_COMPAT_H
#define BASELINE_MPI3_COMPAT_H

/* Tell Open MPI to keep MPI-1.x compatibility declarations instead of
 * replacing them with _Static_assert errors. When OMPI_OMIT_MPI1_COMPAT_DECLS
 * is already defined, mpi.h skips the C11/_Static_assert path entirely. */
#define OMPI_OMIT_MPI1_COMPAT_DECLS 0
#define OMPI_REMOVED_USE_STATIC_ASSERT 0

#endif /* BASELINE_MPI3_COMPAT_H */
