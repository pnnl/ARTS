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

#include "arts/network/remote_launcher.h"

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "arts/system/print.h"
#include "arts/system/config.h"

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

void arts_remote_launcher_ssh_startup_processes(
    struct arts_remote_launcher_s *launcher) {
  unsigned int argc = launcher->argc;
  char **argv = launcher->argv;
  struct arts_config_s *config = launcher->config;
  unsigned int kill_mode = launcher->killStuckProcesses;

  FILE **ssh_executions = NULL;
  int i;
  int j;
  int k;
  int start_node = (int)config->my_rank;

  char cwd[1024];
  if (getcwd(cwd, sizeof(cwd)) == NULL) {
    return;
  }

  // Derive the current binary name so remote nodes can execute the same program
  char self_exe[4096];
  char binary_name[256];
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
  } else {
    self_exe[0] = '\0';
  }

  // Extract basename
  const char *base_ptr = self_exe;
  char *slash_ptr = (self_exe[0] != '\0') ? strrchr(self_exe, '/') : NULL;
  if (slash_ptr) {
    base_ptr = slash_ptr + 1;
  }
  if (base_ptr && base_ptr[0] != '\0') {
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

  for (k = start_node + 1; k < (int)config->table_length + start_node; k++) {
    i = k % (int)config->table_length;
    unsigned int final_length = 0;

    if (kill_mode) {
      // Kill any previously running instance by process name (basename, up to
      // 15 chars)
      if (binary_name[0] != '\0') {
        final_length +=
            snprintf(command + final_length, sizeof(command) - final_length,
                     "pkill %s", binary_name);
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

        final_length +=
            snprintf(command + final_length, sizeof(command) - final_length,
                     "pkill %s", pkill_name);
      } else {
        continue;
      }
    } else {
      // Launch mode: ensure the remote shell changes to the same working
      // directory and launches the binary
      if (binary_name[0] != '\0') {
        final_length +=
            snprintf(command + final_length, sizeof(command) - final_length,
                     "cd %s && ", cwd);
        // Pass through arts_config environment variable if set
        char *arts_config_env = getenv("ARTS_CONFIG");
        if (arts_config_env) {
          final_length +=
              snprintf(command + final_length, sizeof(command) - final_length,
                       "arts_config=%s ", arts_config_env);
        }
        // Pass through LD_LIBRARY_PATH if set
        char *ld_library_path = getenv("LD_LIBRARY_PATH");
        if (ld_library_path) {
          final_length +=
              snprintf(command + final_length, sizeof(command) - final_length,
                       "LD_LIBRARY_PATH=%s ", ld_library_path);
        }
        // Pass ARTS_RANK to tell spawned process its rank (prevents recursive
        // spawning)
        final_length +=
            snprintf(command + final_length, sizeof(command) - final_length,
                     "ARTS_RANK=%d ", i);
        final_length +=
            snprintf(command + final_length, sizeof(command) - final_length,
                     "./%s", binary_name);
        // Pass through any arguments beyond argv[0] if provided
        for (j = 1; j < (int)argc; j++) {
          final_length +=
              snprintf(command + final_length, sizeof(command) - final_length,
                       " %s", argv[j]);
        }
      } else {
        // Fallback: attempt to use argv if available, otherwise just cd
        final_length +=
            snprintf(command + final_length, sizeof(command) - final_length,
                     "cd %s && ", cwd);
        // Pass through arts_config environment variable if set
        char *arts_config_env = getenv("ARTS_CONFIG");
        if (arts_config_env) {
          final_length +=
              snprintf(command + final_length, sizeof(command) - final_length,
                       "arts_config=%s ", arts_config_env);
        }
        // Pass through LD_LIBRARY_PATH if set
        char *ld_library_path = getenv("LD_LIBRARY_PATH");
        if (ld_library_path) {
          final_length +=
              snprintf(command + final_length, sizeof(command) - final_length,
                       "LD_LIBRARY_PATH=%s ", ld_library_path);
        }
        // Pass ARTS_RANK to tell spawned process its rank (prevents recursive
        // spawning)
        final_length +=
            snprintf(command + final_length, sizeof(command) - final_length,
                     "ARTS_RANK=%d ", i);
        for (j = 0; j < (int)argc; j++) {
          final_length +=
              snprintf(command + final_length, sizeof(command) - final_length,
                       " %s", argv[j]);
        }
      }
    }

    // Null-terminate
    command[final_length] = '\0';

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

    child = fork();

    if (child == 0) {
      // Child process: execute SSH command
      // Use a non-interactive ssh invocation and run the command via a shell
      // Passing the command to `sh -c` avoids fragile local quoting
      execlp("ssh", "ssh", "-f", "-o", "StrictHostKeyChecking=no", "-o",
             "ConnectTimeout=10", "-o", "ServerAliveInterval=5", "-o",
             "ServerAliveCountMax=3", config->table[i].ip_address,
             wrapped_command, (char *)NULL);

      // If execlp fails
      ARTS_ERROR("SSH execlp failed: %s", strerror(errno));
    }
  }

  if (kill_mode) {
    exit(0);
  }
}

void arts_remote_launcher_ssh_cleanup_processes(
    struct arts_remote_launcher_s *launcher) {
  // if (launcher && launcher->launcherMemory) {
  //   arts_free(launcher->launcherMemory);
  //   launcher->launcherMemory = NULL;
  // }
}
