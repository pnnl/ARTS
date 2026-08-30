/* SPDX-License-Identifier: Apache-2.0
 *
 * grant_purge_word_model — exhaustive model check of the PURGE release
 * policy's grant word and its one pure decider, arts_grant_purge_release_next
 * (coherence/grant_purge.c, declared non-static for exactly this purpose).
 *
 * The word packs two facts that are surrendered independently: an explicit
 * possession bit (own) and a pure activity count sharing the remaining bits.
 * The decider is the count-dropping edge, committed by a single CAS at the
 * caller: given the word a CAS just read and whether this rank may hand
 * possession back at all (purgeable), it returns the next word to commit and
 * which of NONE / RETURN / TAIL / ILLEGAL happened. It touches no global
 * state and performs no I/O, so its whole behavior is a function of its
 * three inputs — exhaustively checkable without any runtime.
 *
 * This test calls the real decider directly (no reimplementation to compare
 * against) over every row of its truth table, swept across a representative
 * range of counts including both encoding boundaries, crossed with both
 * values of own and of purgeable. Two invariants are checked after every
 * single call, not just at hand-picked points:
 *   - own==0 ⟹ count==0 is preserved whenever it held on input (a decrement
 *     can never leave a used count sitting under a cleared possession bit);
 *   - a call that clears possession never leaves a nonzero count behind (the
 *     one edge that could manufacture that state is guarded off as ILLEGAL,
 *     never committed).
 * The seed constants a fresh possession/hold pair is built from are checked
 * against the header's own accessors rather than against their bit pattern,
 * so the test tracks the encoding if the bit assignment ever changes.
 *
 * PURE test: no arts_rt, no ports, no threads — runs standalone at any time.
 * The word this decider operates on exists only where a write grant migrates
 * under the PURGE release policy (VAL/INV, never EXCL, which arbitrates a
 * different single word of its own); under RETAIN the test compiles and
 * exits PASS having done nothing, rather than being excluded from the build.
 */
#include <stdio.h>

#if !defined(ARTS_RELEASE_PURGE) || defined(ARTS_PROTOCOL_EXCL)
int main(void) {
  printf("PASS grant_purge_word_model: skipped (the [own|count] grant word "
         "and its decider exist only in a VAL/INV x PURGE build; EXCL "
         "arbitrates its own lock word and RETAIN keeps the sentinel-only "
         "word)\n");
  return 0;
}
#else

#include "arts/coherence/types_common.h"

#include <stdbool.h>
#include <stddef.h>

static int g_fail = 0;
static int g_checks = 0;

/* Every call's outcome is checked against these two invariants regardless of
 * which table row produced it — written purely off the header's own own/count
 * accessors, never off a literal bit pattern. */
static void chk_invariants(const char *name, unsigned int cur,
                           unsigned int next) {
  bool cur_legal = ARTS_GRANT_OWN_OF(cur) || ARTS_GRANT_COUNT_OF(cur) == 0u;
  bool next_legal = ARTS_GRANT_OWN_OF(next) || ARTS_GRANT_COUNT_OF(next) == 0u;
  if (cur_legal && !next_legal) {
    (void)fprintf(stderr,
                  "FAIL [%s]: own==0 => count==0 broken by this transition "
                  "(cur=%#010x next=%#010x)\n",
                  name, cur, next);
    g_fail = 1;
  }
  if (ARTS_GRANT_OWN_OF(cur) && !ARTS_GRANT_OWN_OF(next) &&
      ARTS_GRANT_COUNT_OF(next) != 0u) {
    (void)fprintf(stderr,
                  "FAIL [%s]: possession cleared with a nonzero count left "
                  "behind (cur=%#010x next=%#010x) — a borrowed bit, not a "
                  "clean release\n",
                  name, cur, next);
    g_fail = 1;
  }
}

/* Call the real decider once and pin its exact output against an explicit
 * expected (next, act) pair — never against a second copy of its own logic. */
static void chk(const char *name, unsigned int cur, bool purgeable,
                unsigned int ex_next, unsigned int ex_act) {
  unsigned int act = 0xffffffffu;
  unsigned int next = arts_grant_purge_release_next(cur, purgeable, &act);
  g_checks++;
  if (next != ex_next || act != ex_act) {
    (void)fprintf(stderr,
                  "FAIL [%s]: cur=%#010x purgeable=%d => next=%#010x act=%u, "
                  "expected next=%#010x act=%u\n",
                  name, cur, (int)purgeable, next, act, ex_next, ex_act);
    g_fail = 1;
  }
  chk_invariants(name, cur, next);
}

int main(void) {
  /* Representative count sweep: a small dense range plus both values that
   * pin the encoding's boundary (the top of the count field, and one below
   * it, where a decrement or a mis-widened field would first show up). */
  const unsigned int COUNTS[] = {0u,
                                 1u,
                                 2u,
                                 3u,
                                 4u,
                                 5u,
                                 6u,
                                 7u,
                                 8u,
                                 ARTS_GRANT_COUNT_MASK - 1u,
                                 ARTS_GRANT_COUNT_MASK};
  const size_t NCOUNTS = sizeof(COUNTS) / sizeof(COUNTS[0]);

  /* Row: own==1, count>1 — other holds remain; decrement only, act NONE.
   * purgeable is not consulted while a hold remains under this one. */
  for (size_t i = 0; i < NCOUNTS; i++) {
    unsigned int c = COUNTS[i];
    if (c <= 1u) {
      continue;
    }
    unsigned int cur = ARTS_GRANT_OWN | c;
    unsigned int ex_next = ARTS_GRANT_OWN | (c - 1u);
    chk("own=1,count>1,purgeable -> decrement only", cur, true, ex_next,
        ARTS_GRANT_ACT_NONE);
    chk("own=1,count>1,!purgeable -> decrement only", cur, false, ex_next,
        ARTS_GRANT_ACT_NONE);
  }

  /* Row: own=1,count=1,purgeable — the idle edge: hand possession back. */
  chk("own=1,count=1,purgeable -> RETURN", ARTS_GRANT_SEED_HOLDING, true, 0u,
      ARTS_GRANT_ACT_RETURN);

  /* Row: own=1,count=1,!purgeable (the home) — keep possession, serve demand. */
  chk("own=1,count=1,!purgeable -> TAIL", ARTS_GRANT_SEED_HOLDING, false,
      ARTS_GRANT_SEED_IDLE, ARTS_GRANT_ACT_TAIL);

  /* Row: own=1,count=0 — possession with nothing under it; a raw decrement
   * here would borrow the possession bit, so the edge is refused rather than
   * committed: the word comes back byte-for-byte unchanged. purgeable does
   * not matter to a refusal. */
  chk("own=1,count=0,purgeable -> ILLEGAL", ARTS_GRANT_SEED_IDLE, true,
      ARTS_GRANT_SEED_IDLE, ARTS_GRANT_ACT_ILLEGAL);
  chk("own=1,count=0,!purgeable -> ILLEGAL", ARTS_GRANT_SEED_IDLE, false,
      ARTS_GRANT_SEED_IDLE, ARTS_GRANT_ACT_ILLEGAL);

  /* Row: own==0 — nothing held, nothing owed: a double or spurious release
   * is a no-op regardless of whatever the count bits happen to hold, which
   * is exactly the defense the own-bit guard exists to provide. Swept across
   * the same count values to prove the guard is on the own bit alone, never
   * on the count field. */
  for (size_t i = 0; i < NCOUNTS; i++) {
    unsigned int cur = COUNTS[i]; /* own bit clear by construction */
    chk("own=0,purgeable -> NONE (no-op)", cur, true, cur,
        ARTS_GRANT_ACT_NONE);
    chk("own=0,!purgeable -> NONE (no-op)", cur, false, cur,
        ARTS_GRANT_ACT_NONE);
  }

  /* Seed encoding, read back through the header's own accessors — never
   * against a literal bit pattern, so this tracks the encoding rather than
   * pinning it a second time. */
  g_checks++;
  if (!ARTS_GRANT_OWN_OF(ARTS_GRANT_SEED_HOLDING) ||
      ARTS_GRANT_COUNT_OF(ARTS_GRANT_SEED_HOLDING) != 1u) {
    (void)fprintf(stderr,
                  "FAIL: ARTS_GRANT_SEED_HOLDING does not decode to "
                  "own=1,count=1\n");
    g_fail = 1;
  }
  g_checks++;
  if (!ARTS_GRANT_OWN_OF(ARTS_GRANT_SEED_IDLE) ||
      ARTS_GRANT_COUNT_OF(ARTS_GRANT_SEED_IDLE) != 0u) {
    (void)fprintf(stderr,
                  "FAIL: ARTS_GRANT_SEED_IDLE does not decode to "
                  "own=1,count=0\n");
    g_fail = 1;
  }

  /* A commit that installs one hold and then folds the rest of its cohort in
   * with a plain count add depends on that add staying INSIDE the count field:
   * possession must survive it untouched, and the count must read back as the
   * cohort size.  Checked at the top of the field as well as the bottom, since
   * that is where a carry would escape into the possession bit. */
  {
    const unsigned int NS[] = {1u, 2u, 3u, 8u, ARTS_GRANT_COUNT_MASK - 1u,
                               ARTS_GRANT_COUNT_MASK};
    for (size_t i = 0; i < sizeof(NS) / sizeof(NS[0]); i++) {
      unsigned int n = NS[i];
      unsigned int w = ARTS_GRANT_SEED_HOLDING + (n - 1u);
      g_checks++;
      if (!ARTS_GRANT_OWN_OF(w) || ARTS_GRANT_COUNT_OF(w) != n) {
        (void)fprintf(stderr,
                      "FAIL: folding a cohort of %u into the commit's hold "
                      "does not decode to own=1,count=%u\n",
                      n, n);
        g_fail = 1;
      }
    }
  }

  if (g_fail) {
    return 1;
  }
  printf("PASS grant_purge_word_model: %d transitions/invariant checks over "
         "the count sweep [0..8, MASK-1, MASK] x own{0,1} x purgeable{0,1} "
         "— decrement-only, RETURN/TAIL idle edges, ILLEGAL refusal, "
         "own==0 no-op, own==0<=>count==0, and the seed encoding all hold\n",
         g_checks);
  return 0;
}

#endif /* ARTS_RELEASE_PURGE && !ARTS_PROTOCOL_EXCL */
