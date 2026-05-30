/******************************************************************************
** Copyright 2019 Battelle Memorial Institute
** Licensed under the Apache License, Version 2.0
******************************************************************************/
#ifndef ARTS_TRANSPORT_STDIO_FORWARD_H
#define ARTS_TRANSPORT_STDIO_FORWARD_H

#include <stdbool.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * arts_stdio_forwarder_make_pipe — Create a pipe pair and spawn a reader
 * thread that copies the read-end's bytes to @p sink verbatim.  Raw
 * read(2)/write(2) is used (no FILE* on the read side) so the forwarder's
 * fd never joins glibc's stream chain — otherwise a blocked fgets would
 * hold a FILE* lock that an fflush(NULL) on the master could deadlock on.
 * There is no fixed limit on the number of live forwarders.
 *
 * @param rank         Child rank (diagnostics only).
 * @param stream_label "stdout" or "stderr" (diagnostics only; borrowed).
 * @param sink         Master's FILE* to write forwarded bytes to.
 * @return Write-end fd (>=0) on success — caller dup2's this into the
 *         child before exec, then closes their copy.  Returns -1 only on
 *         genuine resource exhaustion (pipe/pthread/malloc); the caller
 *         then leaves the stream on the fd inherited from the master so
 *         output still reaches the launch point — it is never discarded.
 */
int arts_stdio_forwarder_make_pipe(unsigned int rank, const char *stream_label,
                                   FILE *sink);

/**
 * arts_stdio_forwarder_shutdown_all — Join all reader threads and close
 * their fds.  Call AFTER the launcher has waitpid'd all child processes
 * (so pipe EOF has propagated and threads have exited their read loop).
 * Idempotent.
 */
void arts_stdio_forwarder_shutdown_all(void);

#ifdef __cplusplus
}
#endif
#endif /* ARTS_TRANSPORT_STDIO_FORWARD_H */
