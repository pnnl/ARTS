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
#include "arts/system/config.h"

#include <ctype.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <unistd.h>

#include "arts.h"
#include "arts/utils/malloc.h"
#include "arts/network/remote_launcher.h"
#include "arts/system/arts_print.h"
#include "arts/system/debug.h"

char *extract_nodelist_lsf(const char *envr, int stride, unsigned int *cnt) {
  char *lsf_nodes;
  char *res_string;
  char *last;
  int ele = 0;
  lsf_nodes = getenv(envr);
  if (lsf_nodes == NULL) {
    return NULL;
}
  if (stride <= 0) {
    stride = 1;
}
  unsigned int nodes_str_len = strlen(lsf_nodes) + 1;
  char *node_list = (char *)arts_malloc(sizeof(char) * nodes_str_len);
  unsigned int count = 0;
  unsigned int list_str_length = 0;
  last = res_string = strtok(lsf_nodes, " ");
  while (res_string) {
    res_string = strtok(NULL, " ");
    ele++;
    if (res_string && strcmp(res_string, last) != 0 && !(ele % stride)) {
      unsigned int res_len = strlen(res_string);
      memcpy(&node_list[list_str_length], res_string, res_len + 1);
      list_str_length += strlen(res_string);
      node_list[list_str_length++] = ',';
      last = res_string;
      count++;
    }
  }
  node_list[list_str_length - 1] = '\0';
  *cnt = count;
  return node_list;
}

struct arts_config_variable_s *
arts_config_find_variable(struct arts_config_variable_s **head, const char *string) {
  struct arts_config_variable_s *found = NULL;
  struct arts_config_variable_s *last = NULL;
  struct arts_config_variable_s *next = *head;

  while (next != NULL) {
    if (strcmp(string, next->variable) == 0) {
      found = next;
      break;
    }
    last = next;
    next = next->next;
  }

  char *overide = getenv(string);
  if (overide) {
    unsigned int size = strlen(overide);
    struct arts_config_variable_s *new_var = (struct arts_config_variable_s *)arts_malloc(
        sizeof(struct arts_config_variable_s) + size);

    new_var->size = size;
    memcpy(new_var->variable, string, strlen(string) + 1);
    memcpy(new_var->value, overide, size + 1);

    if (last) {
      last->next = new_var;
    } else {
      new_var->next = *head;
      *head = new_var;
    }

    if (next) {
      new_var->next = next->next;
      arts_free(next);
    }
    return new_var;
  }
  return found;
}

static void remove_white_spaces(char *str) {
  char *write = str;
  char *read = str;
  do {
    if (*read != ' ') {
      *write++ = *read;
}
  } while (*read++);
}

struct arts_config_variable_s *arts_config_get_variables(FILE *config) {
  char *line = NULL;
  size_t len = 0;
  ssize_t read;
  char *var;
  char *val;
  int size;
  struct arts_config_variable_s *c_var;
  struct arts_config_variable_s *head = NULL;
  struct arts_config_variable_s *next = NULL;

  while ((read = getline(&line, &len, config)) != -1) {
    var = strtok(line, "=");
    val = strtok(NULL, "=");

    if (val != NULL) {
      size = (int)strlen(val);

      if (val[size - 1] == '\n') {
        val[size - 1] = '\0';
}

      c_var = (struct arts_config_variable_s *)arts_malloc(
          sizeof(struct arts_config_variable_s) + size);
      c_var->size = size;

      strncpy(c_var->variable, var, 255);
      memcpy(c_var->value, val, (unsigned int)size + 1);

      remove_white_spaces(c_var->variable);
      remove_white_spaces(c_var->value);

      c_var->next = NULL;

      if (next != NULL) {
        next->next = c_var;
      } else {
        head = c_var;
}
      next = c_var;
    }
  }
  if (line) {
    free(line);
}
  return head;
}

char *arts_config_make_new_var(const char *var) {
  char *new_var;
  unsigned int size;
  size = strlen(var);
  new_var = (char *)arts_malloc(size + 1);
  strncpy(new_var, var, size);
  new_var[size] = '\0';
  return new_var;
}

char *arts_config_get_slurm_hostname(char *name, char *digit_sample,
                                     unsigned int value) {
  unsigned int length = strlen(name);
  unsigned int digit_length = strlen(digit_sample);
  unsigned int name_length = length + digit_length + 1;
  char *out_name = (char *)arts_malloc(name_length);

  strncpy(out_name, name, length);

  for (unsigned int i = digit_length; i > 0; i--) {
    out_name[length + i - 1] = (char)('0' + (value % 10));
    value /= 10;
  }

  out_name[name_length - 1] = '\0';
  return out_name;
}

// Helper: Check if string contains only digits
static bool arts_config_is_all_digits(const char *str) {
  if (!str || !*str) {
    return false;
}
  while (*str) {
    if (!isdigit(*str)) {
      return false;
}
    str++;
  }
  return true;
}

// Count nodes in new syntax: node[01-10], hostname:port, hostname
unsigned int arts_config_count_nodes(char *node_list) {
  unsigned int length = strlen(node_list);
  unsigned int nodes = 0;

  // Count commas to get initial entry count
  for (unsigned int i = 0; i < length; i++) {
    if (node_list[i] == ',') {
      nodes++;
}
}
  nodes++;  // One more than comma count

  // Adjust for bracket ranges (each range is one entry but multiple nodes)
  unsigned int i = 0;
  while (i < length) {
    // Find bracket range
    if (node_list[i] == '[') {
      nodes--;  // This entry is a range, not a single node
      // Find the closing bracket
      unsigned int bracket_start = i + 1;
      while (i < length && node_list[i] != ']') {
        i++;
}
      if (i < length) {
        // Parse range inside brackets: "01-10" or "1-5"
        char range_spec[64];
        unsigned int range_len = i - bracket_start;
        if (range_len < sizeof(range_spec)) {
          strncpy(range_spec, node_list + bracket_start, range_len);
          range_spec[range_len] = '\0';

          // Find the dash
          char *dash = strchr(range_spec, '-');
          if (dash) {
            *dash = '\0';
            unsigned int start = strtol(range_spec, NULL, 10);
            unsigned int end = strtol(dash + 1, NULL, 10);
            if (start <= end) {
              nodes += (end - start + 1);
            } else {
              nodes += (start - end + 1);
}
          } else {
            // Single number in brackets (unusual but handle it)
            nodes += 1;
          }
        }
      }
    }
    // Note: colon after bracket (for port) or colon for hostname:port
    // doesn't change node count - it's just port specification
    i++;
  }

  return nodes;
}

char *arts_get_next_partition(char **remainder) {
  char *input = (*remainder);
  (*remainder) = NULL;
  if (input) {
    unsigned int length = strlen(input);
    if (length) {
      bool flag = false;
      bool found = false;
      for (unsigned int i = 0; i <= length; i++) {
        if (input[i] == '[') {
          flag = true;
        } else if (input[i] == ']') {
          flag = false;
        } else if (!flag && input[i] == ',') {
          input[i] = '\0';
          (*remainder) = &input[i + 1];
          found = true;
          break;
        }
      }
      return input;
    }
  }
  return NULL;
}

void arts_config_create_routing_table(struct arts_config_s **config, char *node_list) {
  unsigned int node_count;
  struct arts_config_table_s *table;
  unsigned int current_node = 0;
  unsigned int str_length;
  char *temp;
  char *next;
  unsigned int start;
  unsigned int stop;
  unsigned int direction;
  unsigned int list_length;

  if (node_list == NULL) {
    return;
  }
  list_length = strlen(node_list);

  unsigned int total_length = 0;

  node_count = (*config)->nodes;
  (*config)->table_length = node_count;
  table = (struct arts_config_table_s *)arts_calloc(node_count,
                                               sizeof(struct arts_config_table_s));

  if (!(*config)->master_boot) {
    char *part;
    while ((part = arts_get_next_partition(&node_list))) {
      char *node_begin = strtok(part, "[");
      char *next = strtok(NULL, "[");
      if (next) {
        bool done = false;
        char *name = node_begin;
        node_begin = node_begin + strlen(node_begin) + 1;
        do {
          node_begin = strtok(node_begin, ",");
          next = node_begin + strlen(node_begin) + 1;
          if (node_begin) {
            node_begin = strtok(node_begin, "-");
            char *node_end = strtok(NULL, "-");

            if (node_end) {
              if (node_end[strlen(node_end) - 1] == ']') {
                node_end[strlen(node_end) - 1] = '\0';
                done = true;
              }

              start = strtol(node_begin, NULL, 10);
              stop = strtol(node_end, NULL, 10);

              if (start < stop) {
                direction = 1;
              } else {
                direction = -1;
}

              while (start != stop + 1) {
                table[current_node].rank = current_node;
                table[current_node].ip_address =
                    arts_config_get_slurm_hostname(name, node_begin, start);
                start += direction;
                current_node++;
              }
            } else {
              if (node_begin[strlen(node_begin) - 1] == ']') {
                node_begin[strlen(node_begin) - 1] = '\0';
                done = true;
              }

              unsigned int name_length = strlen(name);
              unsigned int node_len = strlen(node_begin);
              total_length = name_length + node_len + 1;
              temp = (char *)arts_malloc(total_length);
              memcpy(temp, name, name_length);
              memcpy(temp + name_length, node_begin, node_len);
              temp[total_length - 1] = '\0';

              table[current_node].rank = current_node;
              table[current_node].ip_address = temp;
              current_node++;
            }
          }
          node_begin = next;
        } while (!done);
      } else {
        // Single node
        str_length = strlen(node_begin);
        temp = (char *)arts_malloc(str_length + 1);
        memcpy(temp, node_begin, str_length);
        temp[str_length] = '\0';

        table[current_node].rank = current_node;
        table[current_node].ip_address = temp;
        current_node++;
      }
    }
  } else {
    // SSH path: parse node[01-10]:port, hostname:port, or hostname
    char *node_begin = node_list;
    char *next;

    do {
      // Get next comma-separated entry
      node_begin = strtok(node_begin, ",");
      if (node_begin == NULL) {
        break;
}
      next = node_begin + strlen(node_begin) + 1;

      // Strip trailing newline if present
      str_length = strlen(node_begin);
      if (str_length > 0 && node_begin[str_length - 1] == '\n') {
        node_begin[str_length - 1] = '\0';
}

      // Check for bracket range: node[01-10] or node[01-10]:port
      char *bracket_open = strchr(node_begin, '[');
      char *bracket_close = bracket_open ? strchr(bracket_open, ']') : NULL;

      if (bracket_open && bracket_close && bracket_close > bracket_open) {
        // Bracket range syntax: base[start-end] or base[start-end]:port
        char base_name[256];
        unsigned int base_len = bracket_open - node_begin;
        if (base_len >= sizeof(base_name)) {
          base_len = sizeof(base_name) - 1;
}
        strncpy(base_name, node_begin, base_len);
        base_name[base_len] = '\0';

        // Extract range spec from brackets
        char range_spec[64];
        unsigned int range_len = bracket_close - bracket_open - 1;
        if (range_len >= sizeof(range_spec)) {
          range_len = sizeof(range_spec) - 1;
}
        strncpy(range_spec, bracket_open + 1, range_len);
        range_spec[range_len] = '\0';

        // Check for port after closing bracket: ]:port
        unsigned int node_port = 0;
        char *port_spec = strchr(bracket_close, ':');
        if (port_spec) {
          node_port = strtol(port_spec + 1, NULL, 10);
        }

        // Parse range: "01-10" or "1-5"
        char *dash = strchr(range_spec, '-');
        if (dash) {
          *dash = '\0';
          start = strtol(range_spec, NULL, 10);
          stop = strtol(dash + 1, NULL, 10);
          unsigned int pad_width = strlen(range_spec);  // Preserve padding width

          if (start <= stop) {
            direction = 1;
          } else {
            direction = -1;
}

          while (start != stop + direction) {
            // Create padded hostname: base + padded_number
            char hostname[512];
            (void)snprintf(hostname, sizeof(hostname), "%s%0*u", base_name, pad_width,
                     start);
            table[current_node].rank = current_node;
            table[current_node].ip_address = arts_config_make_new_var(hostname);
            table[current_node].port = node_port;
            start += direction;
            current_node++;
          }
        } else {
          // Single number in brackets (unusual)
          unsigned int num = strtol(range_spec, NULL, 10);
          char hostname[512];
          (void)snprintf(hostname, sizeof(hostname), "%s%s", base_name, range_spec);
          table[current_node].rank = current_node;
          table[current_node].ip_address = arts_config_make_new_var(hostname);
          table[current_node].port = node_port;
          current_node++;
        }
      } else {
        // No brackets - check for hostname:port or just hostname
        char *colon_pos = strchr(node_begin, ':');
        if (colon_pos && arts_config_is_all_digits(colon_pos + 1)) {
          // hostname:port format
          *colon_pos = '\0';  // Temporarily terminate hostname
          table[current_node].rank = current_node;
          table[current_node].ip_address = arts_config_make_new_var(node_begin);
          table[current_node].port = strtol(colon_pos + 1, NULL, 10);
          *colon_pos = ':';  // Restore for safety
          current_node++;
        } else {
          // Just hostname, no port
          table[current_node].rank = current_node;
          table[current_node].ip_address = arts_config_make_new_var(node_begin);
          table[current_node].port = 0;
          current_node++;
        }
      }

      node_begin = next;
    } while (node_begin < node_list + list_length);
  }

  (*config)->table = table;
}

/*=============================================================================
 * Table-Driven Configuration Infrastructure
 *
 * Each config keyword is registered in a declarative table. Simple keywords
 * (uint, uint64, bool, string) are auto-parsed via offsetof(). Complex
 * keywords use custom handler functions.
 *===========================================================================*/

enum arts_config_type {
  CONFIG_UINT,   /* unsigned int — strtol(value, NULL, 10) */
  CONFIG_UINT64, /* uint64_t — strtoull(value, NULL, 10) */
  CONFIG_BOOL,   /* bool — strtol(value, NULL, 10) > 0 */
  CONFIG_STRING, /* char* — arts_config_make_new_var(value) */
  CONFIG_CUSTOM  /* custom handler function */
};

typedef void (*config_handler_t)(struct arts_config_s *config, const char *value,
                                 struct arts_config_variable_s **vars);

struct arts_config_entry_s {
  const char *key;
  enum arts_config_type type;
  size_t offset;
  const char *default_value;
  config_handler_t handler;
};

/* Look up a config variable's value string, with env var override. */
static const char *config_lookup(struct arts_config_variable_s **vars,
                                 const char *key) {
  struct arts_config_variable_s *found = arts_config_find_variable(vars, key);
  return found ? found->value : NULL;
}

/* Auto-parse a config value into the struct field at entry->offset. */
static void config_auto_parse(struct arts_config_s *config,
                              const struct arts_config_entry_s *entry,
                              const char *value) {
  char *base = (char *)config;
  char *end = NULL;
  switch (entry->type) {
  case CONFIG_UINT:
    *(unsigned int *)(base + entry->offset) =
        (unsigned int)strtol(value, &end, 10);
    break;
  case CONFIG_UINT64:
    *(uint64_t *)(base + entry->offset) = strtoull(value, &end, 10);
    break;
  case CONFIG_BOOL:
    *(bool *)(base + entry->offset) = strtol(value, &end, 10) > 0;
    break;
  case CONFIG_STRING: {
    char **field = (char **)(base + entry->offset);
    if (*field) {
      arts_free(*field);
    }
    *field = arts_config_make_new_var(value);
    break;
  }
  case CONFIG_CUSTOM:
    break;
  }
}

/*--- Custom Handlers -------------------------------------------------------*/

static void handle_launcher(struct arts_config_s *config, const char *value,
                            struct arts_config_variable_s **vars) {
  (void)vars;
  if (!value) {
    /* Auto-detect from environment: SLURM > LSF > default SSH */
    if (getenv("SLURM_PROCID") || getenv("SLURM_NNODES")) {
      config->launcher = arts_config_make_new_var("slurm");
    } else if (getenv("LSB_HOSTS") || getenv("LSB_MCPU_HOSTS")) {
      config->launcher = arts_config_make_new_var("lsf");
    } else {
      config->launcher = arts_config_make_new_var("ssh");
    }
    return;
  }
  if (strncmp(value, "ssh", 3) == 0) {
    config->launcher = arts_config_make_new_var("ssh");
  } else if (strncmp(value, "slurm", 5) == 0) {
    config->launcher = arts_config_make_new_var("slurm");
  } else if (strncmp(value, "lsf", 3) == 0) {
    config->launcher = arts_config_make_new_var("lsf");
  } else if (strncmp(value, "local", 5) == 0) {
    config->launcher = arts_config_make_new_var("local");
  } else {
    config->launcher = arts_config_make_new_var("ssh");
  }
}

static void handle_net_interface(struct arts_config_s *config, const char *value,
                                 struct arts_config_variable_s **vars) {
  (void)vars;
  if (value) {
    config->net_interface = arts_config_make_new_var(value);
  }
}

static void handle_port(struct arts_config_s *config, const char *value,
                        struct arts_config_variable_s **vars) {
  (void)vars;
  if (!value) {
    /* Default applied later in config_compute_derived for non-local launchers */
    return;
  }
  if (value[0] == '[') {
    char *ptr = (char *)value + 1;
    char *endptr;
    unsigned long start_port = strtoul(ptr, &endptr, 10);
    if (endptr != ptr && *endptr == '-') {
      ptr = endptr + 1;
      unsigned long end_port = strtoul(ptr, &endptr, 10);
      if (endptr != ptr) {
        config->port_range = true;
        config->port_start = (unsigned int)start_port;
        config->port_end = (unsigned int)end_port;
        config->port = (unsigned int)start_port;
        return;
      }
    }
    config->port_range = false;
    config->port = 75563;
  } else {
    config->port_range = false;
    config->port = (unsigned int)strtol(value, NULL, 10);
  }
}

/*--- Config Entry Table ----------------------------------------------------*/

#define OFF(f) offsetof(struct arts_config_s, f)

static const struct arts_config_entry_s config_entries[] = {
    /* --- Threading --- */
    {"worker_threads",           CONFIG_UINT,   OFF(worker_thread_count),      "4",          NULL},
    {"stack_size",               CONFIG_UINT64, OFF(stack_size),               "0",          NULL},
    /* --- Pinning --- */
    {"pin",                      CONFIG_BOOL,   OFF(pin_threads),              "1",          NULL},
    {"pin_stride",               CONFIG_UINT,   OFF(pin_stride),               "1",          NULL},
    {"print_topology",           CONFIG_BOOL,   OFF(print_topology),           "0",          NULL},
    /* --- Scheduling --- */
    {"scheduler",                CONFIG_UINT,   OFF(scheduler),                "0",          NULL},
    {"worker_init_deque_size",   CONFIG_UINT,   OFF(deque_size),               "4096",       NULL},
    {"route_table_size",         CONFIG_UINT,   OFF(route_table_size),         "20",         NULL},
    {"auto_shutdown",            CONFIG_UINT,   OFF(auto_shutdown),            "0",          NULL},
    /* --- GPU --- */
    {"gpu",                      CONFIG_UINT,   OFF(gpu),                      "0",          NULL},
    {"gpu_locality",             CONFIG_UINT,   OFF(gpu_locality),             "0",          NULL},
    {"gpu_fit",                  CONFIG_UINT,   OFF(gpu_fit),                  "0",          NULL},
    {"gpu_lc_sync",              CONFIG_UINT,   OFF(gpu_lc_sync),              "0",          NULL},
    {"gpu_max_edts",             CONFIG_UINT,   OFF(gpu_max_edts),             NULL,         NULL},
    {"gpu_max_memory",           CONFIG_UINT64, OFF(gpu_max_memory),           NULL,         NULL},
    {"gpu_p2p",                  CONFIG_BOOL,   OFF(gpu_p2p),                  "0",          NULL},
    {"gpu_route_table_size",     CONFIG_UINT,   OFF(gpu_route_table_size),     "12",         NULL},
    {"free_db_after_gpu_run",    CONFIG_BOOL,   OFF(free_db_after_gpu_run),    "0",          NULL},
    {"run_gpu_gc_idle",          CONFIG_BOOL,   OFF(run_gpu_gc_idle),          "1",          NULL},
    {"run_gpu_gc_pre_edt",       CONFIG_BOOL,   OFF(run_gpu_gc_pre_edt),      "0",          NULL},
    {"delete_zeros_gpu_gc",      CONFIG_BOOL,   OFF(delete_zeros_gpu_gc),      "1",          NULL},
    {"gpu_buff_on",              CONFIG_BOOL,   OFF(gpu_buff_on),              "0",          NULL},
    /* --- Networking (conditional defaults applied in config_compute_derived) --- */
    {"sender_threads",           CONFIG_UINT,   OFF(sender_thread_count),      NULL,         NULL},
    {"receiver_threads",         CONFIG_UINT,   OFF(receiver_thread_count),    NULL,         NULL},
    {"num_ports",                CONFIG_UINT,   OFF(num_ports),                NULL,         NULL},
    {"master_node",              CONFIG_STRING, OFF(master_node),              NULL,         NULL},
    /* --- Debug --- */
    {"kill_mode",                CONFIG_UINT,   OFF(kill_mode),                "0",          NULL},
    {"core_dump",                CONFIG_BOOL,   OFF(core_dump),                "0",          NULL},
    {"print_node_stats",         CONFIG_UINT,   OFF(print_node_stats),         "0",          NULL},
    {"watchdog_timeout",         CONFIG_UINT,   OFF(watchdog_timeout),         "10",         NULL},
    /* --- Counters --- */
    {"counter_folder",           CONFIG_STRING, OFF(counter_folder),           "./counters", NULL},
    {"counter_capture_interval", CONFIG_UINT,   OFF(counter_capture_interval), "100",        NULL},
    /* --- Custom handlers --- */
    {"launcher",                 CONFIG_CUSTOM, 0,                             NULL,         handle_launcher},
    {"net_interface",            CONFIG_CUSTOM, 0,                             NULL,         handle_net_interface},
    {"default_port",             CONFIG_CUSTOM, 0,                             NULL,         handle_port},
    /* sentinel */
    {NULL, 0, 0, NULL, NULL}
};

#undef OFF

/*--- Table-Driven Parse Loop -----------------------------------------------*/

static void config_parse_table(struct arts_config_s *config,
                               struct arts_config_variable_s **vars) {
  for (int i = 0; config_entries[i].key != NULL; i++) {
    const struct arts_config_entry_s *entry = &config_entries[i];
    const char *value = config_lookup(vars, entry->key);
    if (entry->type == CONFIG_CUSTOM) {
      if (entry->handler) {
        entry->handler(config, value, vars);
      }
      continue;
    }
    if (value == NULL) {
      value = entry->default_value;
    }
    if (value != NULL) {
      config_auto_parse(config, entry, value);
    }
  }
}

/*--- Launcher Setup --------------------------------------------------------*/

/* Set master node from routing table[0] and find master rank. */
static void config_set_master_from_table(struct arts_config_s *config) {
  if (config->master_node) {
    arts_free(config->master_node);
  }
  config->master_node = arts_config_make_new_var(config->table[0].ip_address);
  for (unsigned int i = 0; i < config->table_length; i++) {
    config->table[i].rank = i;
    if (strcmp(config->master_node, config->table[i].ip_address) == 0) {
      config->master_rank = i;
    }
  }
}

static void config_setup_slurm(struct arts_config_s *config) {
  config->master_boot = false;

  char *threads_temp = getenv("SLURM_CPUS_PER_TASK");
  if (threads_temp != NULL) {
    config->thread_count = (unsigned int)strtol(threads_temp, NULL, 10);
  }

  char *slurm_nodes = getenv("SLURM_NNODES");
  if (slurm_nodes != NULL) {
    config->nodes = (unsigned int)strtol(slurm_nodes, NULL, 10);
  } else {
    config->nodes = 1;
  }

  char *node_list = getenv("SLURM_STEP_NODELIST");
  arts_config_create_routing_table(&config, node_list);
  config_set_master_from_table(config);
}

static void config_setup_lsf(struct arts_config_s *config) {
  config->master_boot = false;
  unsigned int count = 0;
  char *node_list = extract_nodelist_lsf("LSB_HOSTS", 1, &count);
  if (!node_list) {
    node_list = extract_nodelist_lsf("LSB_MCPU_HOSTS", 2, &count);
  }
  config->nodes = count;

  arts_config_create_routing_table(&config, node_list);
  config_set_master_from_table(config);
}

static void config_setup_ssh(struct arts_config_s *config,
                             struct arts_config_variable_s **vars) {
  config->launcher_data =
      arts_remote_launcher_create(0, NULL, config, config->kill_mode,
                                  arts_remote_launcher_ssh_startup_processes,
                                  arts_remote_launcher_ssh_cleanup_processes);
  config->master_boot = true;

  char *node_list = NULL;
  const char *nodes_value = config_lookup(vars, "nodes");
  if (nodes_value) {
    /* nodes_value points into the linked list — safe to use as strtok input
       since arts_config_create_routing_table will tokenize it. We need a
       mutable copy for count_nodes since it may also tokenize. */
    node_list = arts_config_make_new_var(nodes_value);

    const char *node_count_value = config_lookup(vars, "node_count");
    if (node_count_value) {
      config->nodes = (unsigned int)strtol(node_count_value, NULL, 10);
    } else {
      config->nodes = arts_config_count_nodes(node_list);
    }
  } else {
    node_list = arts_config_make_new_var("localhost");
    config->nodes = 1;
  }

  arts_config_create_routing_table(&config, node_list);

  if (config->master_node == NULL) {
    config->master_node = arts_config_make_new_var(config->table[0].ip_address);
  }
  for (unsigned int i = 0; i < config->table_length; i++) {
    config->table[i].rank = i;
    if (strcmp(config->master_node, config->table[i].ip_address) == 0) {
      config->master_rank = i;
    }
  }
}

static void config_setup_local(struct arts_config_s *config) {
  config->master_boot = false;
  if (config->master_node) {
    arts_free(config->master_node);
    config->master_node = NULL;
  }

  char *threads_user = getenv("USER_THREAD_COUNT");
  if (threads_user != NULL) {
    config->worker_thread_count = (unsigned int)strtol(threads_user, NULL, 10);
  }

  config->nodes = 1;
  config->table_length = 1;
  config->master_rank = 0;
}

static void config_setup_launcher(struct arts_config_s *config,
                                  struct arts_config_variable_s **vars) {
  if (strcmp(config->launcher, "slurm") == 0) {
    config_setup_slurm(config);
  } else if (strcmp(config->launcher, "lsf") == 0) {
    config_setup_lsf(config);
  } else if (strcmp(config->launcher, "ssh") == 0) {
    config_setup_ssh(config, vars);
  } else if (strcmp(config->launcher, "local") == 0) {
    config_setup_local(config);
  } else {
    arts_abort(1);
  }
}

/*--- Computed Fields & Warnings --------------------------------------------*/

static void config_set_pre_defaults(struct arts_config_s *config) {
  config->gpu_max_edts = (unsigned int)-1;
  config->gpu_max_memory = (uint64_t)-1;
}

static void config_compute_derived(struct arts_config_s *config) {
  /* Power-of-2 route table entries via bit shift. */
  config->route_table_entries = 1U << config->route_table_size;
  config->gpu_route_table_entries = 1U << config->gpu_route_table_size;

  /* Networking conditional defaults (non-local launcher only). */
  if (strcmp(config->launcher, "local") != 0) {
    if (!config->sender_thread_count) {
      config->sender_thread_count = 1;
    }
    if (!config->receiver_thread_count) {
      config->receiver_thread_count = 1;
    }
    if (!config->num_ports) {
      config->num_ports = 1;
    }
    if (!config->port) {
      config->port = 75563;
    }
  }

  /* Compute total thread count.
     If thread_count was set directly (SLURM/env), derive worker count from it.
     Otherwise compute total from worker + sender + receiver. */
  if (config->thread_count > 0) {
    config->worker_thread_count = config->thread_count
        - config->sender_thread_count - config->receiver_thread_count;
  }
  config->thread_count = config->worker_thread_count
      + config->sender_thread_count + config->receiver_thread_count;

  /* Assign per-node ports from port range to routing table.
     Each node gets config->num_ports consecutive ports, non-overlapping.
     e.g., ports=2, port=[10001-10004]: node0=10001,10002 node1=10003,10004 */
  if (config->port_range && config->table != NULL) {
    for (unsigned int i = 0; i < config->table_length; i++) {
      config->table[i].port = config->port_start + (i * config->num_ports);
    }
  }
}

static void config_print_warnings(struct arts_config_s *config) {
  if (config->free_db_after_gpu_run) {
    ARTS_INFO("free_db_after_gpu_run is on -- intended for testing, not "
              "performance.");
  }
  if (config->run_gpu_gc_pre_edt) {
    ARTS_INFO("run_gpu_gc_pre_edt is on -- intended for testing, not "
              "performance.");
  }
}

/*--- Config File Open / Variable Cleanup -----------------------------------*/

static FILE *config_open_file(void) {
  const char *location = getenv("ARTS_CONFIG");
  FILE *f = fopen(location ? location : "arts.cfg", "r");
  if (!f) {
    ARTS_INFO("No config file found (./arts.cfg).");
    arts_debug_generate_seg_fault();
  }
  return f;
}

static void config_free_variables(struct arts_config_variable_s *vars) {
  while (vars != NULL) {
    struct arts_config_variable_s *next_var = vars->next;
    arts_free(vars);
    vars = next_var;
  }
}

/*=============================================================================
 * arts_config_load — Phased config loading
 *
 * Phase 1: Open file, parse key=value pairs into linked list
 * Phase 2: Allocate config, set non-zero pre-defaults
 * Phase 3: Table-driven parse (replaces ~280 lines of if-else)
 * Phase 4: Launcher-specific setup (nodes, routing table, master)
 * Phase 5: Computed fields, warnings
 *===========================================================================*/

struct arts_config_s *arts_config_load(void) {
  /* Phase 1: Open config file, parse key=value pairs. */
  FILE *fp = config_open_file();
  struct arts_config_variable_s *vars = arts_config_get_variables(fp);
  (void)fclose(fp);

  /* Phase 2: Allocate config, set non-zero pre-defaults. */
  struct arts_config_s *config =
      (struct arts_config_s *)arts_calloc(1, sizeof(struct arts_config_s));
  config_set_pre_defaults(config);

  /* Phase 3: Table-driven parse. */
  config_parse_table(config, &vars);

  /* Phase 4: Launcher-specific setup (nodes, routing table, master). */
  config_setup_launcher(config, &vars);

  /* Phase 5: Computed fields, warnings. */
  config_compute_derived(config);
  config_print_warnings(config);

  /* Cleanup variable linked list. */
  config_free_variables(vars);
  return config;
}

void arts_config_destroy(struct arts_config_s *config) {
  arts_free(config->launcher);
  if (config->launcher_data) {
    arts_free(config->launcher_data);
  }
  if (config->table) {
    for (unsigned int i = 0; i < config->table_length; i++) {
      arts_free(config->table[i].ip_address);
    }
    arts_free(config->table);
  }
  if (config->master_node) {
    arts_free(config->master_node);
  }
  if (config->net_interface) {
    arts_free(config->net_interface);
  }
  if (config->counter_folder) {
    arts_free(config->counter_folder);
  }
  arts_free(config);
}
