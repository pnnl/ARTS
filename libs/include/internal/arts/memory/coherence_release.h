/* SPDX-License-Identifier: Apache-2.0
 *
 * Coherence-protocol release path.
 *
 * release_rw is the user-visible RW release entry point.  It bumps
 * the in-buffer version, decrements writer_count, and depending on
 * (rest_count, home == self) issues the appropriate wire message:
 *
 *   R1 (home == self,  rest > 0):  version++; nothing else
 *   R2 (home == self,  rest == 0): version++; local_transfer
 *   R3 (home != self,  rest > 0):  version++; WRITEBACK_NORMAL + await ACK
 *   R4 (home != self,  rest == 0): version++; WRITEBACK_AND_TRANSFER + await
 * ACK
 *
 * If destroy is locally marked when release_rw enters, all wire
 * sends are skipped — destroy commits the runtime to teardown and
 * any state we would have written back is moot.
 */

#ifndef ARTS_MEMORY_COHERENCE_RELEASE_H
#define ARTS_MEMORY_COHERENCE_RELEASE_H

#ifdef __cplusplus
extern "C" {
#endif

#include "arts/memory/coherence.h"

/* Release a RW acquire.  Cache-only signature: the dual-stack
 * model keeps user data at cache->user_data and the buf is just a
 * coherence handle owned by the cache, so callers don't track a
 * per-acquire buf pointer. */
void arts_coh_release_rw(struct arts_db_cache_s *cache);

/* Release a RO acquire — currently a no-op (no held ref to drop)
 * but kept as a separate symbol for symmetry and future cutover. */
void arts_coh_release_ro(struct arts_db_cache_s *cache);

#ifdef __cplusplus
}
#endif

#endif /* ARTS_MEMORY_COHERENCE_RELEASE_H */
