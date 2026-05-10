/* tests/finish_event_inherit.c */
#include "arts.h"

static volatile int finish_signal_ran = 0;

void term_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  finish_signal_ran = 1;
  arts_printf("PASS: termination EDT fired after finish-scope drained\n");
  arts_shutdown();
}

void inner_leaf(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  /* Just exit — DECR will fire on completion. */
}

void finish_root(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  /* Spawn 3 leaves under this finish-scope. */
  for (int i = 0; i < 3; i++) {
    arts_edt_create(inner_leaf, 0, NULL, 0, NULL);
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  /* Create term_edt with depc=1, attach to finish_root's finish_event. */
  arts_guid_t term = arts_edt_create(term_edt, 0, NULL, 1, NULL);
  arts_edt_hint_t hint = ARTS_EDT_HINT_FINISH;
  arts_guid_t root = arts_edt_create(finish_root, 0, NULL, 0, &hint);
  arts_guid_t fe = arts_edt_get_finish_event(root);
  arts_add_dependence(fe, term, 0, DB_MODE_NULL);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return finish_signal_ran ? 0 : 1;
}
