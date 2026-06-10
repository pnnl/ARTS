/* Stub implementations for OCR utility functions absent from the distributed
 * (vdm.h) build.  These symbols are defined in the single-node ocr_api.cpp
 * but not in the distributed source tree.  The distributed runtime handles
 * its own abort path; apps that call these just need linkable symbols. */

#include "extensions/ocr-hints.h"
#include "ocr-std.h"
#include "ocr.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>

extern "C" {

void _ocrAssert(bool val, const char *str, const char *file, u32 line) {
  if (!val) {
    fprintf(stderr, "OCR assert failed: %s (%s:%u)\n", str, file,
            (unsigned)line);
    abort();
  }
}

void ocrAbort(u8 errorCode) {
  fprintf(stderr, "ocrAbort called with code %u\n", (unsigned)errorCode);
  exit((int)errorCode);
}

/* ocrSetHint: advisory placement hints — no-op in the distributed runtime
 * where task placement is controlled by the MPI rank topology. */
u8 ocrSetHint(ocrGuid_t /*guid*/, ocrHint_t * /*hint*/) { return 0; }

} /* extern "C" */
