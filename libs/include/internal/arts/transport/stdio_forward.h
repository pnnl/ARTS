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
 * thread that line-buffers the read-end and emits each line to @p sink
 * with the prefix "[<rank> <stream_label>] ".
 *
 * @param rank         Child rank used in the prefix.
 * @param stream_label "stdout" or "stderr" (borrowed; must stay valid for
 *                     the forwarder's lifetime — always a string literal
 *                     in our uses).
 * @param sink         Master's FILE* to write forwarded lines to.
 * @return Write-end fd (>=0) on success — caller dup2's this into the
 *         child before exec, then closes their copy. Returns -1 on
 *         failure (pipe/pthread errors); caller should fall back to the
 *         legacy redirect (/dev/null or log file).
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
