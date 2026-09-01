/* SPDX-License-Identifier: Apache-2.0
 *
 * T215 — handle_launcher precedence.
 * Documents B106 (default-launcher SSH foot-gun).
 *
 * handle_launcher resolves config->launcher with this precedence:
 *   0. cfg value == "flux"                    -> "flux" iff FLUX_TASK_RANK is
 *      set (a flux task env), else HARD ERROR.  Opt-in by explicit value,
 *      honored ahead of the env sniffs: a flux instance may run inside
 *      another scheduler's allocation whose leaked variables must not steal
 *      a declared flux run — and conversely a flux variable leaking into a
 *      local/ssh run must hijack nothing.
 *   1. SLURM_PROCID or SLURM_NNODES present   -> "slurm"  (env wins over cfg)
 *   2. LSB_HOSTS or LSB_MCPU_HOSTS present    -> "lsf"    (env wins over cfg)
 *   3. no scheduler env, cfg value == "local" -> "local"
 *   4. no scheduler env, value NULL or other  -> "ssh"   (DEFAULT, B106)
 *
 * Case 4 is the foot-gun: a bare cfg with no launcher= line and no scheduler
 * env silently becomes SSH (which spawns ssh children), NOT local — flux env
 * present or not.  This test PINS all outcomes, including that env overrides
 * an explicit value="local" (slurm/lsf win), that the bare default is "ssh",
 * and the flux opt-in contract in both directions.
 *
 * Pure unit: each case clears the scheduler env first, drives handle_launcher,
 * checks config->launcher, then frees it (handle_launcher mallocs the string).
 * The flux-without-env case is an ARTS_ERROR death path, probed through the
 * armed longjmp stub in config_test_common.h.
 */

#include "../../libs/src/core/system/config.c"
#include "config_test_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;

static void clear_sched_env(void) {
  unsetenv("SLURM_PROCID");
  unsetenv("SLURM_NNODES");
  unsetenv("LSB_HOSTS");
  unsetenv("LSB_MCPU_HOSTS");
  unsetenv("FLUX_TASK_RANK");
}

static void run_case(const char *desc, const char *cfg_value,
                     const char *want) {
  struct arts_config_s c;
  memset(&c, 0, sizeof(c));
  handle_launcher(&c, cfg_value, NULL);
  if (c.launcher == NULL || strcmp(c.launcher, want) != 0) {
    fprintf(stderr, "FAIL launcher[%s]: got '%s' want '%s'\n", desc,
            c.launcher ? c.launcher : "(null)", want);
    fails++;
  }
  arts_free(c.launcher);
}

int main(void) {
  /* 1. value="local" but SLURM_NNODES set -> slurm (env wins). */
  clear_sched_env();
  setenv("SLURM_NNODES", "4", 1);
  run_case("slurm-overrides-local", "local", "slurm");

  /* 1b. SLURM_PROCID alone also -> slurm. */
  clear_sched_env();
  setenv("SLURM_PROCID", "0", 1);
  run_case("slurm-procid", NULL, "slurm");

  /* 2. LSB_HOSTS set -> lsf (env wins over value="local"). */
  clear_sched_env();
  setenv("LSB_HOSTS", "h0 h1", 1);
  run_case("lsf-overrides-local", "local", "lsf");

  /* 3. no scheduler env, value="local" -> local. */
  clear_sched_env();
  run_case("explicit-local", "local", "local");

  /* 4a. no scheduler env, value NULL -> ssh (DEFAULT foot-gun, B106). */
  clear_sched_env();
  run_case("bare-default-ssh", NULL, "ssh");

  /* 4b. no scheduler env, value="cluster" (unknown) -> ssh. */
  clear_sched_env();
  run_case("unknown-value-ssh", "cluster", "ssh");

  /* 0a. value="flux" + flux task env -> flux. */
  clear_sched_env();
  setenv("FLUX_TASK_RANK", "0", 1);
  run_case("flux-opt-in", "flux", "flux");

  /* 0b. the declared value beats a leaked SLURM env (flux-inside-slurm:
     every task inherits its broker's SLURM_PROCID). */
  clear_sched_env();
  setenv("FLUX_TASK_RANK", "1", 1);
  setenv("SLURM_PROCID", "0", 1);
  run_case("flux-beats-leaked-slurm", "flux", "flux");

  /* 0c. flux env alone hijacks nothing: an explicit local run inside a
     flux allocation stays local, and the bare default stays ssh. */
  clear_sched_env();
  setenv("FLUX_TASK_RANK", "0", 1);
  run_case("flux-env-keeps-local", "local", "local");
  clear_sched_env();
  setenv("FLUX_TASK_RANK", "0", 1);
  run_case("flux-env-keeps-bare-ssh", NULL, "ssh");

  /* 0d. value="flux" WITHOUT the task env is a hard error, never a silent
     fall-through. */
  clear_sched_env();
  {
    struct arts_config_s c;
    memset(&c, 0, sizeof(c));
    g_abort_fired = 0;
    g_abort_armed = 1;
    if (setjmp(g_abort_jmp) == 0) {
      handle_launcher(&c, "flux", NULL);
      fprintf(stderr, "FAIL launcher[flux-no-env]: returned instead of "
                      "aborting\n");
      fails++;
    } else if (!g_abort_fired) {
      fprintf(stderr, "FAIL launcher[flux-no-env]: longjmp without abort\n");
      fails++;
    }
    g_abort_armed = 0;
  }

  clear_sched_env();

  if (fails) {
    return 1;
  }
  printf("PASS config_launcher_env_precedence: flux opt-in both ways, "
         "slurm/lsf env-win, bare->ssh (B106 foot-gun pinned)\n");
  return 0;
}
