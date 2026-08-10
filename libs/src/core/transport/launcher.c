/******************************************************************************
** This material was prepared as an account of work sponsored by an agency   **
** of the United States Government.  Neither the United States Government    **
** nor the United States Department of Energy, nor Battelle, nor any of      **
** their employees, nor any jurisdiction or organization that has cooperated **
** in the development of these materials, makes any warranty, express or     **
** implied, or assumes any legal liability or responsibility for the accuracy,*
** completeness, or usefulness or any information, apparatus, product,       **
** software, or process disclosed, or represents that its use would not      **
** infringe privately owned rights.                                          **
**                                                                           **
** Reference herein to any specific commercial product, process, or service  **
** by trade name, trademark, manufacturer, or otherwise does not necessarily **
** constitute or imply its endorsement, recommendation, or favoring by the   **
** United States Government or any agency thereof, or Battelle Memorial      **
** Institute. The views and opinions of authors expressed herein do not      **
** necessarily state or reflect those of the United States Government or     **
** any agency thereof.                                                       **
**                                                                           **
**                      PACIFIC NORTHWEST NATIONAL LABORATORY                **
**                                  operated by                              **
**                                    BATTELLE                               **
**                                     for the                               **
**                      UNITED STATES DEPARTMENT OF ENERGY                   **
**                         under Contract DE-AC05-76RL01830                  **
**                                                                           **
** Copyright 2019 Battelle Memorial Institute                                **
** Licensed under the Apache License, Version 2.0 (the "License");           **
** you may not use this file except in compliance with the License.          **
** You may obtain a copy of the License at                                   **
**                                                                           **
**    https://www.apache.org/licenses/LICENSE-2.0                            **
**                                                                           **
** Unless required by applicable law or agreed to in writing, software       **
** distributed under the License is distributed on an "AS IS" BASIS, WITHOUT **
** WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the  **
** License for the specific language governing permissions and limitations   **
******************************************************************************/

#include "arts/transport/launcher.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

#include "arts/system/config.h"
#include "arts/system/print.h"
#include "arts/transport/stdio_forward.h"

static int arts_shell_quote(const char *input, char *output,
                            size_t output_size) {
  size_t out_index = 0;

  if (output_size < 3) {
    return -1;
  }

  output[out_index++] = '\'';
  while (*input && out_index + 1 < output_size) {
    if (*input == '\'') {
      if (out_index + 5 >= output_size) {
        return -1;
      }
      output[out_index++] = '\'';
      output[out_index++] = '"';
      output[out_index++] = '\'';
      output[out_index++] = '"';
      output[out_index++] = '\'';
    } else {
      output[out_index++] = *input;
    }
    input++;
  }

  if (out_index + 2 > output_size) {
    return -1;
  }

  output[out_index++] = '\'';
  output[out_index] = '\0';
  return 0;
}

/* Saturating formatted append into a fixed-size buffer.
 *
 * `length` is the number of bytes already written (excluding the terminator)
 * and is updated in place to the new written length, always clamped to
 * `buf_size - 1` so the buffer stays null-terminatable at `buf[length]`.
 *
 * snprintf returns the length it WOULD have written had the buffer been
 * unbounded, which can exceed the space that remains.  Accumulating that raw
 * return value lets `length` run past `buf_size`, after which the remaining
 * size `buf_size - length` underflows and a later `buf[length]` indexes out of
 * bounds.  Guarding the remaining size against underflow and clamping the
 * running length keeps every write — including the terminating one — in
 * bounds even when the composed string is truncated. */
static void arts_cmd_appendf(char *buf, size_t buf_size, size_t *length,
                             const char *fmt, ...) {
  if (buf_size == 0 || *length >= buf_size) {
    if (buf_size != 0) {
      *length = buf_size - 1;
    }
    return;
  }

  size_t remaining = buf_size - *length;
  va_list args;
  va_start(args, fmt);
  int written = vsnprintf(buf + *length, remaining, fmt, args);
  va_end(args);

  if (written < 0) {
    return;
  }
  if ((size_t)written >= remaining) {
    /* Truncated: the buffer is now full up to its last writable byte. */
    *length = buf_size - 1;
  } else {
    *length += (size_t)written;
  }
}

void arts_launcher_ssh_startup_processes(struct arts_launcher_s *launcher) {
  unsigned int argc = launcher->argc;
  char **argv = launcher->argv;
  struct arts_config_s *config = launcher->config;
  unsigned int kill_mode = launcher->kill_stuck_processes;

  FILE **ssh_executions = NULL;
  int i;
  int j;
  int k;
  int start_node = (int)config->my_rank;

  char cwd[1024];
  if (getcwd(cwd, sizeof(cwd)) == NULL) {
    return;
  }

  // Derive the current executable path and basename.
  // self_exe holds the absolute path (used for launch mode — `cd CWD &&
  // /abs/path/to/exe`). binary_name holds the basename only (used for kill mode
  // — `pkill basename`).
  char self_exe[4096];
  char binary_name[256];
  self_exe[0] = '\0';
  binary_name[0] = '\0';

  // Try to get the current executable path
  ssize_t self_exe_len =
      readlink("/proc/self/exe", self_exe, sizeof(self_exe) - 1);
  if (self_exe_len != -1) {
    self_exe[self_exe_len] = '\0';
  } else if (argc > 0 && argv && argv[0]) {
    // Fallback to argv[0] if /proc/self/exe is unavailable
    strncpy(self_exe, argv[0], sizeof(self_exe) - 1);
    self_exe[sizeof(self_exe) - 1] = '\0';
  }

  // Extract basename (for pkill in kill mode)
  if (self_exe[0] != '\0') {
    const char *base_ptr = self_exe;
    char *slash_ptr = strrchr(self_exe, '/');
    if (slash_ptr) {
      base_ptr = slash_ptr + 1;
    }
    size_t base_len = strlen(base_ptr);
    size_t copy_len = (base_len < sizeof(binary_name) - 1)
                          ? base_len
                          : sizeof(binary_name) - 1;
    memcpy(binary_name, base_ptr, copy_len);
    binary_name[copy_len] = '\0';
  }

  // Allocate for all non-master nodes
  // ssh_executions =
  //     (FILE **)arts_malloc(sizeof(FILE *) * (config->table_length - 1));
  // launcher->launcherMemory = ssh_executions;

  char command[4096];
  char quoted_command[(sizeof(command) * 6) + 8];
  char wrapped_command[(sizeof(command) * 6) + 16];
  pid_t child;

  // Allocate PID tracking array for non-kill-mode launches
  if (!kill_mode) {
    unsigned int num_remotes = config->table_length - 1;
    launcher->child_pids = (pid_t *)arts_calloc(num_remotes, sizeof(pid_t));
    launcher->child_count = 0;
  }

  for (k = start_node + 1; k < (int)config->table_length + start_node; k++) {
    i = k % (int)config->table_length;
    size_t final_length = 0;

    if (kill_mode) {
      // Kill any previously running instance by process name (basename, up to
      // 15 chars)
      if (binary_name[0] != '\0') {
        arts_cmd_appendf(command, sizeof(command), &final_length, "pkill %s",
                         binary_name);
      } else if (argc > 0 && argv && argv[0]) {
        // Extract basename from argv[0] for pkill
        const char *argv_base = argv[0];
        char *argv_slash = strrchr(argv[0], '/');
        if (argv_slash) {
          argv_base = argv_slash + 1;
        }

        // Limit to 15 chars for pkill
        char pkill_name[16];
        strncpy(pkill_name, argv_base, 15);
        pkill_name[15] = '\0';

        arts_cmd_appendf(command, sizeof(command), &final_length, "pkill %s",
                         pkill_name);
      } else {
        continue;
      }
    } else {
      // Launch mode: ensure the remote shell changes to the same working
      // directory and launches the binary by absolute path (the binary may
      // live in a subdirectory of CWD, so basename + "./" would fail).
      if (self_exe[0] != '\0') {
        arts_cmd_appendf(command, sizeof(command), &final_length, "cd %s && ",
                         cwd);
        // Pass through arts_config environment variable if set
        char *arts_config_env = getenv("ARTS_CONFIG");
        if (arts_config_env) {
          arts_cmd_appendf(command, sizeof(command), &final_length,
                           "ARTS_CONFIG=%s ", arts_config_env);
        }
        // Pass through LD_LIBRARY_PATH if set
        char *ld_library_path = getenv("LD_LIBRARY_PATH");
        if (ld_library_path) {
          arts_cmd_appendf(command, sizeof(command), &final_length,
                           "LD_LIBRARY_PATH=%s ", ld_library_path);
        }
        // Pass ARTS_RANK to tell spawned process its rank (prevents recursive
        // spawning)
        arts_cmd_appendf(command, sizeof(command), &final_length,
                         "ARTS_RANK=%d ", i);
        arts_cmd_appendf(command, sizeof(command), &final_length, "%s",
                         self_exe);
        // Pass through any arguments beyond argv[0] if provided
        for (j = 1; j < (int)argc; j++) {
          arts_cmd_appendf(command, sizeof(command), &final_length, " %s",
                           argv[j]);
        }
      } else {
        // Fallback: attempt to use argv if available, otherwise just cd
        arts_cmd_appendf(command, sizeof(command), &final_length, "cd %s && ",
                         cwd);
        // Pass through arts_config environment variable if set
        char *arts_config_env = getenv("ARTS_CONFIG");
        if (arts_config_env) {
          arts_cmd_appendf(command, sizeof(command), &final_length,
                           "ARTS_CONFIG=%s ", arts_config_env);
        }
        // Pass through LD_LIBRARY_PATH if set
        char *ld_library_path = getenv("LD_LIBRARY_PATH");
        if (ld_library_path) {
          arts_cmd_appendf(command, sizeof(command), &final_length,
                           "LD_LIBRARY_PATH=%s ", ld_library_path);
        }
        // Pass ARTS_RANK to tell spawned process its rank (prevents recursive
        // spawning)
        arts_cmd_appendf(command, sizeof(command), &final_length,
                         "ARTS_RANK=%d ", i);
        for (j = 0; j < (int)argc; j++) {
          arts_cmd_appendf(command, sizeof(command), &final_length, " %s",
                           argv[j]);
        }
      }
    }

    // Null-terminate. final_length is clamped to sizeof(command)-1, so this
    // index is always in bounds.
    command[final_length] = '\0';

    ARTS_DEBUG("SSH command[%d]: %s", i, command);

    if (arts_shell_quote(command, quoted_command, sizeof(quoted_command)) !=
        0) {
      continue;
    }

    int wrapped_length = snprintf(wrapped_command, sizeof(wrapped_command),
                                  "sh -c %s", quoted_command);
    if (wrapped_length < 0 ||
        (size_t)wrapped_length >= sizeof(wrapped_command)) {
      continue;
    }

    ARTS_DEBUG("SSH wrapped[%d]: %s", i, wrapped_command);

    int forward_out_wfd = -1;
    int forward_err_wfd = -1;
    if (!kill_mode) {
      forward_out_wfd = arts_stdio_forwarder_make_pipe(i, "stdout", stdout);
      forward_err_wfd = arts_stdio_forwarder_make_pipe(i, "stderr", stderr);
    }

    child = fork();

    if (child == 0) {
      if (kill_mode) {
        /* Kill mode runs a remote pkill whose output carries no useful
         * information — silence it so it never clutters the launch point. */
        int devnull = open("/dev/null", O_RDWR);
        if (devnull >= 0) {
          dup2(devnull, STDOUT_FILENO);
          dup2(devnull, STDERR_FILENO);
          if (devnull > STDERR_FILENO) {
            close(devnull);
          }
        }
      } else {
        /* Pipe to master: SSH inherits these fds as its stdout/stderr and
         * relays the remote ARTS process's output.  If a forwarder is
         * missing, leave that stream on the fd inherited from the master so
         * the remote output still reaches the launch point — never discard.
         * Streams are handled independently. */
        if (forward_out_wfd >= 0) {
          dup2(forward_out_wfd, STDOUT_FILENO);
          if (forward_out_wfd > STDERR_FILENO) {
            close(forward_out_wfd);
          }
        }
        if (forward_err_wfd >= 0) {
          dup2(forward_err_wfd, STDERR_FILENO);
          if (forward_err_wfd > STDERR_FILENO) {
            close(forward_err_wfd);
          }
        }
      }

      // No -f flag — child stays alive as the SSH process, allowing the
      // master to waitpid during cleanup to ensure remote exits.
      // -n redirects stdin from /dev/null; BatchMode=yes prevents prompts.
      execlp("ssh", "ssh", "-n", "-o", "BatchMode=yes", "-o",
             "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10", "-o",
             "ServerAliveInterval=5", "-o", "ServerAliveCountMax=3",
             config->table[i].ip_address, wrapped_command, (char *)NULL);

      // If execlp fails
      _exit(127);
    } else if (child > 0) {
      /* Parent: close write-ends so EOF propagates when SSH child exits. */
      if (forward_out_wfd >= 0) {
        close(forward_out_wfd);
      }
      if (forward_err_wfd >= 0) {
        close(forward_err_wfd);
      }
      if (!kill_mode) {
        launcher->child_pids[launcher->child_count++] = child;
      }
    } else {
      if (forward_out_wfd >= 0) {
        close(forward_out_wfd);
      }
      if (forward_err_wfd >= 0) {
        close(forward_err_wfd);
      }
    }
  }

  if (kill_mode) {
    exit(0);
  }
}

void arts_launcher_ssh_cleanup_processes(struct arts_launcher_s *launcher) {
  if (!launcher || !launcher->child_pids) {
    return;
  }

  // Wait for SSH children to exit naturally. Without -f, the SSH process
  // stays alive until the remote ARTS process exits. This ensures remote
  // processes have fully released their ports before the master returns.
  // (Killing SSH does NOT kill the remote — no PTY means no SIGHUP.)
  for (unsigned int i = 0; i < launcher->child_count; i++) {
    if (launcher->child_pids[i] <= 0) {
      continue;
    }
    int status;
    bool exited = false;
    // Give remote up to 15 seconds to exit after shutdown (remote may
    // spend up to 5 seconds in the time-sync timeout during cleanup).
    for (int ms = 0; ms < 15000; ms += 10) {
      if (waitpid(launcher->child_pids[i], &status, WNOHANG) != 0) {
        exited = true;
        break;
      }
      usleep(10000);
    }
    if (!exited) {
      // Remote didn't exit in time — force-kill the SSH session
      kill(launcher->child_pids[i], SIGKILL);
      waitpid(launcher->child_pids[i], &status, 0);
    }
  }

  arts_free(launcher->child_pids);
  launcher->child_pids = NULL;
  launcher->child_count = 0;
}

/*--- Local launcher (multi-node on single machine) -------------------------*/

void arts_launcher_local_startup_processes(struct arts_launcher_s *launcher) {
  struct arts_config_s *config = launcher->config;

  /* Resolve the current executable path. */
  char self_exe[4096];
  ssize_t exe_len = readlink("/proc/self/exe", self_exe, sizeof(self_exe) - 1);
  if (exe_len < 0) {
    ARTS_ERROR("Local launcher: readlink(/proc/self/exe) failed: %s",
               strerror(errno));
    return;
  }
  self_exe[exe_len] = '\0';

  /* Build argv for child processes: [self_exe, argv[1..], NULL]. */
  unsigned int argc = launcher->argc;
  unsigned int new_argc = argc > 0 ? argc : 1;
  char **new_argv = (char **)arts_malloc((new_argc + 1) * sizeof(char *));
  new_argv[0] = self_exe;
  for (unsigned int j = 1; j < argc; j++) {
    new_argv[j] = launcher->argv[j];
  }
  new_argv[new_argc] = NULL;

  /* The port block this run settled on.  The spawned ranks re-read the same
   * cfg, which for a local run names no ports at all — this handoff is the only
   * way they learn what was claimed, and every rank must derive its peers'
   * ports from the identical base. */
  char port_spec[512];
  size_t spec_len = 0;
  port_spec[0] = '\0';
  for (unsigned int j = 0; j < config->port_count && config->ports; j++) {
    int written =
        snprintf(port_spec + spec_len, sizeof(port_spec) - spec_len, "%s%u",
                 j == 0 ? "" : ",", config->ports[j]);
    if (written < 0 || (size_t)written >= sizeof(port_spec) - spec_len) {
      port_spec[0] = '\0';
      break;
    }
    spec_len += (size_t)written;
  }
  /* Published once here rather than per child: every child takes the same base,
   * and the environment is inherited across fork.  Nothing re-reads the config
   * in this process, so exporting it here changes only what the children see. */
  if (port_spec[0] != '\0') {
    setenv(ARTS_RESOLVED_PORTS_ENV, port_spec, 1);
  }

  /* Allocate PID tracking for non-master nodes. */
  unsigned int num_children = config->table_length - 1;
  launcher->child_pids = (pid_t *)arts_calloc(num_children, sizeof(pid_t));
  launcher->child_count = 0;

  for (unsigned int i = 1; i < config->table_length; i++) {
    int forward_out_wfd = arts_stdio_forwarder_make_pipe(i, "stdout", stdout);
    int forward_err_wfd = arts_stdio_forwarder_make_pipe(i, "stderr", stderr);

    pid_t child = fork();

    if (child == 0) {
      /* Child: die when master dies. Without PDEATHSIG, an abnormally
       * killed master leaves the local-launcher children reparented to
       * init, surviving indefinitely as orphans (observed: rank-N processes
       * running 599% CPU long after the harness moved on). Race window
       * between fork() and prctl() is closed by the getppid() recheck.
       *
       * SIGKILL, not SIGTERM: this fires only when the master died without
       * running its cleanup (which already does waitpid -> SIGTERM ->
       * SIGKILL for the graceful path), i.e. the master itself was killed
       * abnormally — typically because a rank is hung.  A hung rank's
       * graceful SIGTERM path can hang too, leaving the orphan alive and
       * holding its ports, which poisons every subsequent run that binds
       * the same port range.  There is nothing left to shut down gracefully
       * at that point. */
      prctl(PR_SET_PDEATHSIG, SIGKILL);
      if (getppid() == 1) {
        _exit(0);
      }
      /* Child: set rank via environment, redirect I/O, exec. */
      char rank_str[16];
      (void)snprintf(rank_str, sizeof(rank_str), "%u", i);
      setenv("ARTS_RANK", rank_str, 1);

      /* Forward each stream to the master through its pipe (the master's
       * reader threads drain them).  If a forwarder could not be created,
       * leave that stream on the fd this child inherited from the master so
       * its output still reaches the launch point — a rank's stdout/stderr
       * is never discarded.  Streams are handled independently so one
       * failed forwarder does not take the other down. */
      if (forward_out_wfd >= 0) {
        dup2(forward_out_wfd, STDOUT_FILENO);
        if (forward_out_wfd > STDERR_FILENO) {
          close(forward_out_wfd);
        }
      }
      if (forward_err_wfd >= 0) {
        dup2(forward_err_wfd, STDERR_FILENO);
        if (forward_err_wfd > STDERR_FILENO) {
          close(forward_err_wfd);
        }
      }

      execv(self_exe, new_argv);
      _exit(127);
    } else if (child > 0) {
      /* Parent: close write-ends so only the child holds them; on child
       * exit the kernel closes its copies and the reader thread sees EOF. */
      if (forward_out_wfd >= 0) {
        close(forward_out_wfd);
      }
      if (forward_err_wfd >= 0) {
        close(forward_err_wfd);
      }
      launcher->child_pids[launcher->child_count++] = child;
      ARTS_INFO("Local launcher: spawned rank %u (pid %d)", i, (int)child);
    } else {
      if (forward_out_wfd >= 0) {
        close(forward_out_wfd);
      }
      if (forward_err_wfd >= 0) {
        close(forward_err_wfd);
      }
      ARTS_ERROR("Local launcher: fork() failed for rank %u: %s", i,
                 strerror(errno));
    }
  }

  arts_free(new_argv);
}

void arts_launcher_local_cleanup_processes(struct arts_launcher_s *launcher) {
  if (!launcher || !launcher->child_pids) {
    return;
  }

  /* Local-launcher cleanup. Children are direct ARTS processes (not SSH
   * wrappers), so we drive shutdown ourselves: a graceful waitpid window,
   * then SIGTERM, then a final SIGKILL backstop. Combined with the
   * PR_SET_PDEATHSIG installed in startup, this guarantees no rank
   * survives a master exit (graceful or otherwise). */
  const int graceful_ms = 5000;
  const int term_ms = 2000;

  for (unsigned int i = 0; i < launcher->child_count; i++) {
    pid_t pid = launcher->child_pids[i];
    if (pid <= 0) {
      continue;
    }
    int status;
    bool exited = false;

    for (int ms = 0; ms < graceful_ms; ms += 10) {
      if (waitpid(pid, &status, WNOHANG) != 0) {
        exited = true;
        break;
      }
      usleep(10000);
    }
    if (exited) {
      continue;
    }

    /* Graceful window expired — escalate to SIGTERM. */
    ARTS_WARN("Local launcher: rank %u (pid %d) did not exit within %d ms, "
              "sending SIGTERM",
              i + 1, (int)pid, graceful_ms);
    kill(pid, SIGTERM);
    for (int ms = 0; ms < term_ms; ms += 10) {
      if (waitpid(pid, &status, WNOHANG) != 0) {
        exited = true;
        break;
      }
      usleep(10000);
    }
    if (exited) {
      continue;
    }

    /* Still alive — last resort. */
    ARTS_WARN("Local launcher: rank %u (pid %d) ignored SIGTERM, sending "
              "SIGKILL",
              i + 1, (int)pid);
    kill(pid, SIGKILL);
    waitpid(pid, &status, 0);
  }

  arts_free(launcher->child_pids);
  launcher->child_pids = NULL;
  launcher->child_count = 0;
}
