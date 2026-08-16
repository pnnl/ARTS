/* SPDX-License-Identifier: Apache-2.0
 *
 * Probabilistic interleaving-window widener for race reproduction.
 *
 * A rare race needs two threads inside a nanosecond-scale window at once;
 * stress loops hit such windows at rates that make a failure observable but
 * not diagnosable.  Injecting a short randomized delay at the handful of
 * points where lock-free decisions commit widens those windows by three
 * orders of magnitude, trading a little wall-clock for reproduction rates
 * that turn a 500-iteration hunt into a 10-iteration one (the randomized-
 * scheduler family of testing techniques).
 *
 * Off unless ARTS_SCHED_FUZZ=<pct> is set in the environment (percentage
 * probability per fuzz point, 1-100).  Compiled out of NDEBUG (Release)
 * builds entirely — this is a reproduction aid, never a shipping code path.
 */
#ifndef ARTS_SYSTEM_SCHEDFUZZ_H
#define ARTS_SYSTEM_SCHEDFUZZ_H

#ifndef NDEBUG
/* Delay this thread ~0.5-5us with the configured probability.  Safe from
 * any thread context that may already block (worker or progress dispatch);
 * never call it under a CQ token or inside a signal handler. */
void arts_sched_fuzz_point(void);
#else
#define arts_sched_fuzz_point() ((void)0)
#endif

#endif /* ARTS_SYSTEM_SCHEDFUZZ_H */
