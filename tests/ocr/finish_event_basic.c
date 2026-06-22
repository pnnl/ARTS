#include "arts.h"

void leaf_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("PASS: leaf ran (no finish-scope inherited)\n");
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_edt_create(leaf_edt, 0, NULL, 0, NULL);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
