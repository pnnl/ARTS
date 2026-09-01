/* Hermetic tests for the registered pool's per-node availability estimate
 * (arts_regpool_parse_node_avail): the parser is pure over a stdio stream,
 * so every policy claim is checked here against synthetic meminfo text —
 * including the two ends that matter: a cache-heavy node must be admitted
 * (its file LRU is reclaimable on demand) and an anon-full node must still
 * report ~MemFree (the constrained-OOM refusal guard). */
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "arts/memory/regpool.h"

static size_t parse(const char *text) {
  FILE *f = fmemopen((void *)text, strlen(text), "r");
  assert(f != NULL);
  size_t r = arts_regpool_parse_node_avail(f);
  fclose(f);
  return r;
}

int main(void) {
  /* Cache-heavy healthy node (the incident shape): tiny MemFree, tens of
   * GiB on the file LRUs, negligible writeback.  Expected: MemFree plus
   * half the clean file pages. */
  {
    const char *t = "Node 6 MemTotal:       130829040 kB\n"
                    "Node 6 MemFree:        150000 kB\n"
                    "Node 6 MemUsed:        130679040 kB\n"
                    "Node 6 Active(anon):   1000000 kB\n"
                    "Node 6 Inactive(anon): 2000000 kB\n"
                    "Node 6 Active(file):   24661016 kB\n"
                    "Node 6 Inactive(file): 47900732 kB\n"
                    "Node 6 Unevictable:    32 kB\n"
                    "Node 6 Dirty:          84 kB\n"
                    "Node 6 Writeback:      0 kB\n"
                    "Node 6 NFS_Unstable:   0 kB\n"
                    "Node 6 WritebackTmp:   0 kB\n"
                    "Node 6 HugePages_Total: 0\n";
    size_t want_kb =
        150000u + (24661016u + 47900732u - 84u) / 2u;
    assert(parse(t) == (size_t)want_kb * 1024);
  }

  /* Anon-full node: file LRUs empty — the estimate must collapse to
   * MemFree so the caller's clamp still refuses it. */
  {
    const char *t = "Node 2 MemFree:        150000 kB\n"
                    "Node 2 Active(file):   0 kB\n"
                    "Node 2 Inactive(file): 0 kB\n"
                    "Node 2 Dirty:          0 kB\n";
    assert(parse(t) == (size_t)150000 * 1024);
  }

  /* Dirty-heavy: only the clean remainder counts, halved. */
  {
    const char *t = "Node 0 MemFree:        100000 kB\n"
                    "Node 0 Active(file):   5242880 kB\n"
                    "Node 0 Inactive(file): 5242880 kB\n"
                    "Node 0 Dirty:          9437184 kB\n"
                    "Node 0 Writeback:      0 kB\n";
    assert(parse(t) == ((size_t)100000 + (10485760u - 9437184u) / 2u) * 1024);
  }

  /* Writeback-bound exceeds the file LRUs (transient counter skew):
   * clamp at zero extra, never underflow. */
  {
    const char *t = "Node 0 MemFree:        100000 kB\n"
                    "Node 0 Active(file):   500 kB\n"
                    "Node 0 Inactive(file): 500 kB\n"
                    "Node 0 Dirty:          5000 kB\n";
    assert(parse(t) == (size_t)100000 * 1024);
  }

  /* File-LRU fields absent: MemFree alone (older field sets). */
  {
    const char *t = "Node 0 MemTotal:       1000000 kB\n"
                    "Node 0 MemFree:        123456 kB\n"
                    "Node 0 MemUsed:        876544 kB\n";
    assert(parse(t) == (size_t)123456 * 1024);
  }

  /* Only one of the two LRU fields: incomplete pair falls back to
   * MemFree alone rather than half-counting. */
  {
    const char *t = "Node 0 MemFree:        100 kB\n"
                    "Node 0 Active(file):   999999 kB\n";
    assert(parse(t) == (size_t)100 * 1024);
  }

  /* No MemFree at all / empty stream: unknown must not veto growth. */
  assert(parse("Node 0 MemTotal: 1 kB\n") == SIZE_MAX);
  assert(parse("") == SIZE_MAX);

  /* Field-name discrimination: Active(anon)/plain Active must not feed the
   * file counters, and suffix-less lines must not derail the scan. */
  {
    const char *t = "Node 1 Active:         777777 kB\n"
                    "Node 1 Active(anon):   888888 kB\n"
                    "Node 1 HugePages_Total: 0\n"
                    "Node 1 MemFree:        1000 kB\n"
                    "Node 1 Active(file):   2000 kB\n"
                    "Node 1 Inactive(file): 2000 kB\n";
    assert(parse(t) == ((size_t)1000 + (2000u + 2000u) / 2u) * 1024);
  }

  /* A complete real per-node meminfo (captured verbatim): the else-if
   * chain's field discrimination must hold against the full production
   * field set — HugePages_Free, SReclaimable, FilePages, Shmem, plain
   * Active/Inactive and the (anon) variants must all be ignored. */
  {
    const char *t =
        "Node 0 MemTotal:       131792596 kB\n"
        "Node 0 MemFree:        108191140 kB\n"
        "Node 0 MemUsed:        23601456 kB\n"
        "Node 0 SwapCached:            0 kB\n"
        "Node 0 Active:          1510648 kB\n"
        "Node 0 Inactive:       18994656 kB\n"
        "Node 0 Active(anon):      63292 kB\n"
        "Node 0 Inactive(anon):  4427004 kB\n"
        "Node 0 Active(file):    1447356 kB\n"
        "Node 0 Inactive(file): 14567652 kB\n"
        "Node 0 Unevictable:        3072 kB\n"
        "Node 0 Mlocked:               0 kB\n"
        "Node 0 Dirty:                 0 kB\n"
        "Node 0 Writeback:             0 kB\n"
        "Node 0 FilePages:      16317620 kB\n"
        "Node 0 Mapped:           190428 kB\n"
        "Node 0 AnonPages:       4188748 kB\n"
        "Node 0 Shmem:            302652 kB\n"
        "Node 0 KernelStack:       22612 kB\n"
        "Node 0 PageTables:        24916 kB\n"
        "Node 0 SecPageTables:         0 kB\n"
        "Node 0 NFS_Unstable:          0 kB\n"
        "Node 0 Bounce:                0 kB\n"
        "Node 0 WritebackTmp:          0 kB\n"
        "Node 0 KReclaimable:    1479140 kB\n"
        "Node 0 Slab:            1955976 kB\n"
        "Node 0 SReclaimable:    1479140 kB\n"
        "Node 0 SUnreclaim:       476836 kB\n"
        "Node 0 AnonHugePages:   3686400 kB\n"
        "Node 0 ShmemHugePages:        0 kB\n"
        "Node 0 ShmemPmdMapped:        0 kB\n"
        "Node 0 FileHugePages:        0 kB\n"
        "Node 0 FilePmdMapped:        0 kB\n"
        "Node 0 HugePages_Total:     0\n"
        "Node 0 HugePages_Free:      0\n"
        "Node 0 HugePages_Surp:      0\n";
    size_t want_kb = 108191140u + (1447356u + 14567652u) / 2u;
    assert(parse(t) == (size_t)want_kb * 1024);
  }

  printf("regpool_node_avail: all cases passed\n");
  return 0;
}
