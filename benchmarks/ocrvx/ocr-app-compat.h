/* ocr-app-compat.h — compatibility shim for OCR benchmark apps compiled
 * against the ocr-vx vdm.h API.  Force-included for all _ocrvx targets.
 *
 * The ocr-apps benchmark suite was written against the xsocr OCR API.
 * The ocr-vx distributed runtime (vdm.h) exposes equivalent functions under
 * different names.  This header maps the xsocr names to the vdm equivalents.
 */
#ifndef OCR_APP_COMPAT_H
#define OCR_APP_COMPAT_H

/* ocrPrintf(fmt, ...) → PRINTF(fmt, ...) */
#define ocrPrintf(...) PRINTF(__VA_ARGS__)

/* ocrGetArgc / ocrGetArgv: vdm drops the "ocr" prefix */
#define ocrGetArgc(ptr) getArgc(ptr)
#define ocrGetArgv(ptr, n) getArgv((ptr), (n))

/* ocrAssert is absent from vdm.h; map to the underlying _ocrAssert helper */
#define ocrAssert(a)                                                           \
  do {                                                                         \
    _ocrAssert((bool)((a) != 0), #a, __FILE__, __LINE__);                      \
  } while (0)

/* EDT_PROP_OEVT_VALID: property bit telling the runtime that an output event
 * GUID has been pre-created by the caller.  ocr-vx defines and honors the
 * same bit (0x8) in its ocr-types.h; keep a fallback for older trees. */
#ifndef EDT_PROP_OEVT_VALID
#define EDT_PROP_OEVT_VALID ((u16) 0x8)
#endif

#endif /* OCR_APP_COMPAT_H */
