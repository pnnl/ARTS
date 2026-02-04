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

#include "arts/network/RemoteLauncher.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "arts/system/Config.h"

static int artsShellQuote(const char *input, char *output, size_t outputSize) {
  size_t outIndex = 0;

  if (outputSize < 3) {
    return -1;
  }

  output[outIndex++] = '\'';
  while (*input && outIndex + 1 < outputSize) {
    if (*input == '\'') {
      if (outIndex + 5 >= outputSize) {
        return -1;
      }
      output[outIndex++] = '\'';
      output[outIndex++] = '"';
      output[outIndex++] = '\'';
      output[outIndex++] = '"';
      output[outIndex++] = '\'';
    } else {
      output[outIndex++] = *input;
    }
    input++;
  }

  if (outIndex + 2 > outputSize) {
    return -1;
  }

  output[outIndex++] = '\'';
  output[outIndex] = '\0';
  return 0;
}

void artsRemoteLauncherSSHStartupProcesses(
    struct artsRemoteLauncher *launcher) {
  unsigned int argc = launcher->argc;
  char **argv = launcher->argv;
  struct artsConfig *config = launcher->config;
  unsigned int killMode = launcher->killStuckProcesses;

  FILE **sshExecutions = NULL;
  int i, j, k;
  int startNode = config->myRank;

  char cwd[1024];
  if (getcwd(cwd, sizeof(cwd)) == NULL) {
    return;
  }

  // Derive the current binary name so remote nodes can execute the same program
  char selfExe[4096];
  char binaryName[256];
  binaryName[0] = '\0';

  // Try to get the current executable path
  ssize_t selfExeLen = readlink("/proc/self/exe", selfExe, sizeof(selfExe) - 1);
  if (selfExeLen != -1) {
    selfExe[selfExeLen] = '\0';
  } else if (argc > 0 && argv && argv[0]) {
    // Fallback to argv[0] if /proc/self/exe is unavailable
    strncpy(selfExe, argv[0], sizeof(selfExe) - 1);
    selfExe[sizeof(selfExe) - 1] = '\0';
  } else {
    selfExe[0] = '\0';
  }

  // Extract basename
  const char *basePtr = selfExe;
  char *slashPtr = (selfExe[0] != '\0') ? strrchr(selfExe, '/') : NULL;
  if (slashPtr) {
    basePtr = slashPtr + 1;
  }
  if (basePtr && basePtr[0] != '\0') {
    size_t baseLen = strlen(basePtr);
    size_t copyLen =
        (baseLen < sizeof(binaryName) - 1) ? baseLen : sizeof(binaryName) - 1;
    memcpy(binaryName, basePtr, copyLen);
    binaryName[copyLen] = '\0';
  }

  // Allocate for all non-master nodes
  // sshExecutions =
  //     (FILE **)artsMalloc(sizeof(FILE *) * (config->tableLength - 1));
  // launcher->launcherMemory = sshExecutions;

  char command[4096];
  char quotedCommand[(sizeof(command) * 6) + 8];
  char wrappedCommand[(sizeof(command) * 6) + 16];
  pid_t child;

  for (k = startNode + 1; k < config->tableLength + startNode; k++) {
    i = k % config->tableLength;
    unsigned int finalLength = 0;

    if (killMode) {
      // Kill any previously running instance by process name (basename, up to
      // 15 chars)
      if (binaryName[0] != '\0') {
        finalLength +=
            snprintf(command + finalLength, sizeof(command) - finalLength,
                     "pkill %s", binaryName);
      } else if (argc > 0 && argv && argv[0]) {
        // Extract basename from argv[0] for pkill
        const char *argvBase = argv[0];
        char *argvSlash = strrchr(argv[0], '/');
        if (argvSlash)
          argvBase = argvSlash + 1;

        // Limit to 15 chars for pkill
        char pkillName[16];
        strncpy(pkillName, argvBase, 15);
        pkillName[15] = '\0';

        finalLength +=
            snprintf(command + finalLength, sizeof(command) - finalLength,
                     "pkill %s", pkillName);
      } else {
        continue;
      }
    } else {
      // Launch mode: ensure the remote shell changes to the same working
      // directory and launches the binary
      if (binaryName[0] != '\0') {
        finalLength +=
            snprintf(command + finalLength, sizeof(command) - finalLength,
                     "cd %s && ", cwd);
        // Pass through artsConfig environment variable if set
        char *artsConfigEnv = getenv("artsConfig");
        if (artsConfigEnv) {
          finalLength +=
              snprintf(command + finalLength, sizeof(command) - finalLength,
                       "artsConfig=%s ", artsConfigEnv);
        }
        // Pass through LD_LIBRARY_PATH if set
        char *ldLibraryPath = getenv("LD_LIBRARY_PATH");
        if (ldLibraryPath) {
          finalLength +=
              snprintf(command + finalLength, sizeof(command) - finalLength,
                       "LD_LIBRARY_PATH=%s ", ldLibraryPath);
        }
        // Pass ARTS_RANK to tell spawned process its rank (prevents recursive spawning)
        finalLength +=
            snprintf(command + finalLength, sizeof(command) - finalLength,
                     "ARTS_RANK=%d ", i);
        finalLength +=
            snprintf(command + finalLength, sizeof(command) - finalLength,
                     "./%s", binaryName);
        // Pass through any arguments beyond argv[0] if provided
        for (j = 1; j < (int)argc; j++) {
          finalLength +=
              snprintf(command + finalLength, sizeof(command) - finalLength,
                       " %s", argv[j]);
        }
      } else {
        // Fallback: attempt to use argv if available, otherwise just cd
        finalLength += snprintf(command + finalLength,
                                sizeof(command) - finalLength, "cd %s && ", cwd);
        // Pass through artsConfig environment variable if set
        char *artsConfigEnv = getenv("artsConfig");
        if (artsConfigEnv) {
          finalLength +=
              snprintf(command + finalLength, sizeof(command) - finalLength,
                       "artsConfig=%s ", artsConfigEnv);
        }
        // Pass through LD_LIBRARY_PATH if set
        char *ldLibraryPath = getenv("LD_LIBRARY_PATH");
        if (ldLibraryPath) {
          finalLength +=
              snprintf(command + finalLength, sizeof(command) - finalLength,
                       "LD_LIBRARY_PATH=%s ", ldLibraryPath);
        }
        // Pass ARTS_RANK to tell spawned process its rank (prevents recursive spawning)
        finalLength +=
            snprintf(command + finalLength, sizeof(command) - finalLength,
                     "ARTS_RANK=%d ", i);
        for (j = 0; j < (int)argc; j++) {
          finalLength +=
              snprintf(command + finalLength, sizeof(command) - finalLength,
                       " %s", argv[j]);
        }
      }
    }

    // Null-terminate
    command[finalLength] = '\0';

    if (artsShellQuote(command, quotedCommand, sizeof(quotedCommand)) != 0) {
      continue;
    }

    int wrappedLength = snprintf(wrappedCommand, sizeof(wrappedCommand),
                                 "sh -c %s", quotedCommand);
    if (wrappedLength < 0 || (size_t)wrappedLength >= sizeof(wrappedCommand)) {
      continue;
    }

    child = fork();

    if (child == 0) {
      // Child process: execute SSH command
      // Use a non-interactive ssh invocation and run the command via a shell
      // Passing the command to `sh -c` avoids fragile local quoting
      execlp("ssh", "ssh", "-f", "-o", "StrictHostKeyChecking=no", "-o",
             "ConnectTimeout=10", "-o", "ServerAliveInterval=5", "-o",
             "ServerAliveCountMax=3", config->table[i].ipAddress,
             wrappedCommand, (char *)NULL);

      // If execlp fails
      exit(1);
    }
  }

  if (killMode) {
    exit(0);
  }
}

void artsRemoteLauncherSSHCleanupProcesses(
    struct artsRemoteLauncher *launcher) {
  // if (launcher && launcher->launcherMemory) {
  //   artsFree(launcher->launcherMemory);
  //   launcher->launcherMemory = NULL;
  // }
}
