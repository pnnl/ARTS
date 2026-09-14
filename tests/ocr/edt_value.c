#include "arts.h"

static void check(uint32_t pc, const uint64_t *pv, uint32_t dc,
                  arts_edt_dep_t dv[]) {
  if (pc != dc)
    arts_abort(1);
  for (uint32_t i = 0; i < dc; ++i) {
    if (dv[i].guid != pv[i] || dv[i].ptr != NULL ||
        dv[i].mode != DB_MODE_NULL) {
      arts_printf("FAIL edt_value: slot %u changed value, mode, or pointer\n", i);
      arts_abort(1);
    }
  }
  arts_printf("PASS edt_value: opaque uint64 values preserved\n");
  arts_shutdown();
}

void main_edt(uint32_t pc, const uint64_t *pv, uint32_t dc,
              arts_edt_dep_t dv[]) {
  (void)pc;
  (void)pv;
  (void)dc;
  (void)dv;
  const uint64_t values[] = {0, 1, UINT64_C(0xDEADBEEFCAFEBABE), UINT64_MAX,
                             UINT64_C(0x4000000000000001),
                             UINT64_C(0x8000000000000001)};
  const uint32_t count = sizeof(values) / sizeof(values[0]);
  arts_guid_t sink = arts_edt_create(check, count, values, count, NULL);
  for (uint32_t i = 0; i < count; ++i)
    arts_edt_satisfy_slot(sink, i, values[i], DB_MODE_NULL);
}

int main(int argc, char **argv) { return arts_rt(argc, argv) != 0; }
