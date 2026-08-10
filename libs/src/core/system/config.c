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
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <unistd.h>

#include "arts.h"
#include "arts/system/print.h"
#include "arts/transport/launcher.h"
#include "arts/utils/malloc.h"

/*--- Compiler-injected config overrides ------------------------------------*/
static char *arts_config_override_path = NULL;
static char *arts_config_override_data = NULL;


char *extract_nodelist_lsf(const char *envr, int stride, unsigned int *cnt) {
  char *lsf_nodes;
  char *res_string;
  char *last;
  int ele = 0;
  lsf_nodes = getenv(envr);
  if (lsf_nodes == NULL) {
    *cnt = 0; /* define the out-param on every path so callers never read
                 garbage */
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
  /* The loop appends "<host>," for each distinct non-first token, so
   * list_str_length is 0 when nothing was appended (e.g. a single host, the
   * common 1-node case).  Strip the trailing ',' only when one exists;
   * otherwise terminate at index 0 (an unsigned (0-1) index would be a wild
   * OOB write). */
  if (list_str_length > 0) {
    node_list[list_str_length - 1] = '\0';
  } else {
    node_list[0] = '\0';
  }
  *cnt = count;
  return node_list;
}

struct arts_config_variable_s *
arts_config_find_variable(struct arts_config_variable_s **head,
                          const char *string) {
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
    /* value[] is a flexible array and the copy below writes size+1 bytes
     * (including the terminating NUL), so the allocation must reserve size+1
     * (not size) trailing bytes. */
    struct arts_config_variable_s *new_var =
        (struct arts_config_variable_s *)arts_malloc(
            sizeof(struct arts_config_variable_s) + size + 1);

    new_var->size = size;
    /* Default the link to NULL so the not-found / append case below leaves a
     * well-formed tail; the found case overwrites it with next->next. */
    new_var->next = NULL;
    /* variable[] is a fixed char[255]; bound the name copy so an over-long key
     * cannot overflow it. */
    size_t name_len = strlen(string);
    if (name_len >= sizeof(new_var->variable)) {
      name_len = sizeof(new_var->variable) - 1;
    }
    memcpy(new_var->variable, string, name_len);
    new_var->variable[name_len] = '\0';
    memcpy(new_var->value, overide, size + 1);

    if (next) {
      /* Found: splice the replacement in place of `next`. */
      new_var->next = next->next;
      if (last) {
        last->next = new_var;
      } else {
        *head = new_var;
      }
      arts_free(next);
    } else if (last) {
      /* Not found, non-empty list: append as the new tail (next stays NULL). */
      last->next = new_var;
    } else {
      /* Not found, empty list: prepend. */
      *head = new_var;
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
  /* The sample fixes the minimum width (zero-padded lists keep their
     padding), but an unpadded range that crosses a digit-width boundary
     (e.g. 8..12) must widen per element — truncating to the sample's width
     would drop high-order digits. */
  unsigned int value_digits = 1;
  for (unsigned int v = value; v >= 10; v /= 10) {
    value_digits++;
  }
  if (value_digits > digit_length) {
    digit_length = value_digits;
  }
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
  nodes++; // One more than comma count

  // Adjust for bracket ranges (each range is one entry but multiple nodes)
  unsigned int i = 0;
  while (i < length) {
    // Find bracket range
    if (node_list[i] == '[') {
      nodes--; // This entry is a range, not a single node
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

static unsigned int *parse_port_spec(const char *spec, unsigned int *count);

void arts_config_create_routing_table(struct arts_config_s **config,
                                      char *node_list) {
  unsigned int node_count;
  struct arts_config_table_s *table;
  unsigned int current_node = 0;
  unsigned int str_length;
  char *temp;
  char *next;
  unsigned int start;
  unsigned int stop;
  int direction;
  unsigned int list_length;

  if (node_list == NULL) {
    return;
  }
  list_length = strlen(node_list);

  unsigned int total_length = 0;

  node_count = (*config)->nodes;
  (*config)->table_length = node_count;
  table = (struct arts_config_table_s *)arts_calloc(
      node_count, sizeof(struct arts_config_table_s));

  if (!(*config)->master_boot) {
    char *part;
    while ((part = arts_get_next_partition(&node_list))) {
      /* The table is calloc'd with exactly node_count entries; never index past
       * it even if the node_list string parses to more hosts than config->nodes
       * (a user misconfiguration).  current_node is the running write index. */
      if (current_node >= node_count) {
        break;
      }
      char *node_begin = strtok(part, "[");
      char *next = strtok(NULL, "[");
      if (next) {
        bool done = false;
        char *name = node_begin;
        node_begin = node_begin + strlen(node_begin) + 1;
        do {
          node_begin = strtok(node_begin, ",");
          next = node_begin + strlen(node_begin) + 1;
          if (current_node >= node_count) {
            break; /* table full — never write past node_count entries */
          }
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

              /* One-past-the-end in the walk direction; unsigned wraparound
                 makes stop + (-1) the correct sentinel for descending. */
              while (start != stop + direction) {
                if (current_node >= node_count) {
                  break;
                }
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
    // SSH path: parse node[01-10] range expansion, or a plain hostname
    char *node_begin = node_list;
    char *next;

    do {
      // Get next comma-separated entry
      node_begin = strtok(node_begin, ",");
      if (node_begin == NULL) {
        break;
      }
      if (current_node >= node_count) {
        break; /* table is full (node_count entries); never write past it */
      }
      next = node_begin + strlen(node_begin) + 1;

      // Strip trailing newline if present
      str_length = strlen(node_begin);
      if (str_length > 0 && node_begin[str_length - 1] == '\n') {
        node_begin[str_length - 1] = '\0';
      }

      // Check for bracket range: node[01-10]
      char *bracket_open = strchr(node_begin, '[');
      char *bracket_close = bracket_open ? strchr(bracket_open, ']') : NULL;

      if (bracket_open && bracket_close && bracket_close > bracket_open) {
        // Bracket range syntax: base[start-end]
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

        // Parse range: "01-10" or "1-5"
        char *dash = strchr(range_spec, '-');
        if (dash) {
          *dash = '\0';
          start = strtol(range_spec, NULL, 10);
          stop = strtol(dash + 1, NULL, 10);
          unsigned int pad_width = strlen(range_spec);

          if (start <= stop) {
            direction = 1;
          } else {
            direction = -1;
          }

          while (start != stop + direction) {
            if (current_node >= node_count) {
              break;
            }
            char hostname[512];
            (void)snprintf(hostname, sizeof(hostname), "%s%0*u", base_name,
                           pad_width, start);
            table[current_node].rank = current_node;
            table[current_node].ip_address = arts_config_make_new_var(hostname);
            start += direction;
            current_node++;
          }
        } else {
          // Single number in brackets (unusual)
          char hostname[512];
          (void)snprintf(hostname, sizeof(hostname), "%s%s", base_name,
                         range_spec);
          table[current_node].rank = current_node;
          table[current_node].ip_address = arts_config_make_new_var(hostname);
          current_node++;
        }
      } else {
        // Plain hostname; ports come from the shared base, never per node
        table[current_node].rank = current_node;
        table[current_node].ip_address = arts_config_make_new_var(node_begin);
        current_node++;
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

typedef void (*config_handler_t)(struct arts_config_s *config,
                                 const char *value,
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
  /* Environment always wins: SLURM/LSF env vars override config value */
  if (getenv("SLURM_PROCID") || getenv("SLURM_NNODES")) {
    config->launcher = arts_config_make_new_var("slurm");
    return;
  }
  if (getenv("LSB_HOSTS") || getenv("LSB_MCPU_HOSTS")) {
    config->launcher = arts_config_make_new_var("lsf");
    return;
  }
  /* No job scheduler env — use config value (ssh or local) */
  if (value && strncmp(value, "local", 5) == 0) {
    config->launcher = arts_config_make_new_var("local");
  } else {
    config->launcher = arts_config_make_new_var("ssh");
  }
}

static void handle_net_interface(struct arts_config_s *config,
                                 const char *value,
                                 struct arts_config_variable_s **vars) {
  (void)vars;
  if (value) {
    config->net_interface = arts_config_make_new_var(value);
  }
}

/*
 * parse_port_spec — Parse a port specification string into an array.
 *
 * Supported formats:
 *   "50000"              → [50000]              (single port)
 *   "[50000-50001]"      → [50000, 50001]       (range)
 *   "50000,50020,50040"  → [50000, 50020, 50040] (comma-separated)
 *
 * Returns: dynamically allocated array (caller must free), sets *count.
 * Returns NULL with *count=0 when the spec is absent, empty, or malformed —
 * every port must be a bare decimal number a TCP socket can carry.
 */
static bool parse_one_port(const char *text, unsigned int *out) {
  char *end = NULL;
  unsigned long value = strtoul(text, &end, 10);
  if (end == text || *end != '\0' || value == 0 || value > UINT16_MAX) {
    return false;
  }
  *out = (unsigned int)value;
  return true;
}

static unsigned int *parse_port_spec(const char *spec, unsigned int *count) {
  *count = 0;
  if (!spec || !*spec) {
    return NULL;
  }

  if (spec[0] == '[') {
    /* Range format: [start-end] */
    char *ptr = (char *)spec + 1;
    char *endptr;
    unsigned long start = strtoul(ptr, &endptr, 10);
    if (endptr != ptr && *endptr == '-') {
      ptr = endptr + 1;
      unsigned long end = strtoul(ptr, &endptr, 10);
      if (endptr != ptr && strcmp(endptr, "]") == 0 && start <= end &&
          start > 0 && end <= UINT16_MAX) {
        unsigned int n = (unsigned int)(end - start + 1);
        unsigned int *ports =
            (unsigned int *)arts_malloc(n * sizeof(unsigned int));
        for (unsigned int i = 0; i < n; i++) {
          ports[i] = (unsigned int)(start + i);
        }
        *count = n;
        return ports;
      }
    }
    /* Malformed range.  Reported as "no ports parsed" so the caller can fail
       the load: substituting a port here would bind something the operator
       never named. */
    return NULL;
  }

  /* Check for comma-separated list */
  if (strchr(spec, ',')) {
    unsigned int n = 1;
    for (const char *p = spec; *p; p++) {
      if (*p == ',') {
        n++;
      }
    }
    unsigned int *ports = (unsigned int *)arts_malloc(n * sizeof(unsigned int));
    char *copy = arts_config_make_new_var(spec);
    char *tok = strtok(copy, ",");
    unsigned int i = 0;
    bool well_formed = true;
    while (tok && i < n) {
      if (!parse_one_port(tok, &ports[i])) {
        well_formed = false;
        break;
      }
      i++;
      tok = strtok(NULL, ",");
    }
    arts_free(copy);
    if (!well_formed || i != n) {
      arts_free(ports);
      return NULL;
    }
    *count = i;
    return ports;
  }

  /* Single port */
  unsigned int *ports = (unsigned int *)arts_malloc(sizeof(unsigned int));
  if (!parse_one_port(spec, &ports[0])) {
    arts_free(ports);
    return NULL;
  }
  *count = 1;
  return ports;
}

static void handle_ports(struct arts_config_s *config, const char *value,
                         struct arts_config_variable_s **vars) {
  (void)vars;
  if (!value) {
    /* Absent: resolved (or rejected) in config_compute_derived, which knows
       whether this run is allowed to choose its own. */
    return;
  }
  if (config->ports) {
    arts_free(config->ports);
  }
  config->ports = parse_port_spec(value, &config->ports_count);
  if (config->ports == NULL) {
    ARTS_ERROR("Malformed ports '%s': expected a port (25000), a range "
               "([25000-25001]), or a comma-separated list (25000,25010), "
               "each entry in 1-65535",
               value);
  }
}

#ifdef ARTS_USE_CXL
static void
handle_cxl_db_allocation_strategy(struct arts_config_s *config,
                                  const char *value,
                                  struct arts_config_variable_s **vars) {
  /* Default strategy is "static". */
  config->cxl_db_allocation_strategy = ARTS_CXL_DB_ALLOC_STATIC;
  config->cxl_db_allocation_device = 0;

  if (value && strncmp(value, "round_robin", 11) == 0) {
    config->cxl_db_allocation_strategy = ARTS_CXL_DB_ALLOC_ROUND_ROBIN;
  } else {
    /* Static strategy: also read cxl_db_allocation_device. */
    const char *dev_value = config_lookup(vars, "cxl_db_allocation_device");
    if (dev_value) {
      config->cxl_db_allocation_device =
          (unsigned int)strtol(dev_value, NULL, 10);
    }
  }
}
#endif /* ARTS_USE_CXL */

/*--- Config Entry Table ----------------------------------------------------*/

#define OFF(f) offsetof(struct arts_config_s, f)

static const struct arts_config_entry_s config_entries[] = {
    /* --- Threading --- */
    {"worker_threads", CONFIG_UINT, OFF(worker_thread_count), "4", NULL},
    {"stack_size", CONFIG_UINT64, OFF(stack_size), "0", NULL},
    /* --- Pinning --- */
    {"pin", CONFIG_BOOL, OFF(pin_threads), "1", NULL},
    /* --- Scheduling --- */
    {"scheduler", CONFIG_UINT, OFF(scheduler), "0", NULL},
    {"deque_type", CONFIG_UINT, OFF(deque_type), "0", NULL},
    {"worker_init_deque_size", CONFIG_UINT, OFF(deque_size), "4096", NULL},
    {"route_table_size", CONFIG_UINT, OFF(route_table_size), "16", NULL},
    /* --- GPU --- */
    {"gpu", CONFIG_UINT, OFF(gpu), "0", NULL},
    {"gpu_locality", CONFIG_UINT, OFF(gpu_locality), "0", NULL},
    {"gpu_fit", CONFIG_UINT, OFF(gpu_fit), "0", NULL},
    {"gpu_lc_sync", CONFIG_UINT, OFF(gpu_lc_sync), "0", NULL},
    {"gpu_max_edts", CONFIG_UINT, OFF(gpu_max_edts), NULL, NULL},
    {"gpu_max_memory", CONFIG_UINT64, OFF(gpu_max_memory), NULL, NULL},
    {"gpu_p2p", CONFIG_BOOL, OFF(gpu_p2p), "0", NULL},
    {"gpu_route_table_size", CONFIG_UINT, OFF(gpu_route_table_size), "12",
     NULL},
    {"free_db_after_gpu_run", CONFIG_BOOL, OFF(free_db_after_gpu_run), "0",
     NULL},
    {"run_gpu_gc_idle", CONFIG_BOOL, OFF(run_gpu_gc_idle), "1", NULL},
    {"run_gpu_gc_pre_edt", CONFIG_BOOL, OFF(run_gpu_gc_pre_edt), "0", NULL},
    {"delete_zeros_gpu_gc", CONFIG_BOOL, OFF(delete_zeros_gpu_gc), "1", NULL},
    {"gpu_buff_on", CONFIG_BOOL, OFF(gpu_buff_on), "0", NULL},
    /* --- Networking (conditional defaults applied in config_compute_derived)
       --- */
    {"progress_threads", CONFIG_UINT, OFF(progress_thread_count), NULL, NULL},
    {"port_count", CONFIG_UINT, OFF(port_count), NULL, NULL},
    {"master_node", CONFIG_STRING, OFF(master_node), NULL, NULL},
    {"provider", CONFIG_STRING, OFF(provider), NULL, NULL},
    {"fabric_domain", CONFIG_STRING, OFF(fabric_domain), NULL, NULL},
    {"regpool_slab_mb", CONFIG_UINT, OFF(regpool_slab_mb), "64", NULL},
    /* --- Debug --- */
    {"kill_mode", CONFIG_UINT, OFF(kill_mode), "0", NULL},
    {"core_dump", CONFIG_BOOL, OFF(core_dump), "0", NULL},
    /* --- Counters --- */
    {"counter_folder", CONFIG_STRING, OFF(counter_folder), "./counters", NULL},
    {"counter_capture_interval", CONFIG_UINT, OFF(counter_capture_interval),
     "100", NULL},
    /* --- Custom handlers --- */
    {"launcher", CONFIG_CUSTOM, 0, NULL, handle_launcher},
    {"net_interface", CONFIG_CUSTOM, 0, NULL, handle_net_interface},
    {"ports", CONFIG_CUSTOM, 0, NULL, handle_ports},
#ifdef ARTS_USE_CXL
    /* --- CXL DB allocation --- */
    {"cxl_db_allocation_strategy", CONFIG_CUSTOM, 0, NULL,
     handle_cxl_db_allocation_strategy},
#endif /* ARTS_USE_CXL */
    /* sentinel */
    {NULL, 0, 0, NULL, NULL}};

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
      break;
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

  /* srun sets the step-scoped list; a bare sbatch shell (no srun) only has
     the job-scoped one.  Same compressed hostlist format either way. */
  char *node_list = getenv("SLURM_STEP_NODELIST");
  if (node_list == NULL) {
    node_list = getenv("SLURM_JOB_NODELIST");
  }
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
  config->launcher_data = arts_launcher_create(
      0, NULL, config, config->kill_mode, arts_launcher_ssh_startup_processes,
      arts_launcher_ssh_cleanup_processes);
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
      break;
    }
  }
}

static void config_setup_local(struct arts_config_s *config,
                               struct arts_config_variable_s **vars) {
  if (config->master_node) {
    arts_free(config->master_node);
    config->master_node = NULL;
  }

  char *threads_user = getenv("USER_THREAD_COUNT");
  if (threads_user != NULL) {
    config->worker_thread_count = (unsigned int)strtol(threads_user, NULL, 10);
  }

  /* Determine node count from config file. */
  unsigned int node_count = 1;
  const char *node_count_value = config_lookup(vars, "node_count");
  const char *nodes_value = config_lookup(vars, "nodes");

  if (node_count_value) {
    node_count = (unsigned int)strtol(node_count_value, NULL, 10);
  } else if (nodes_value) {
    char *tmp = arts_config_make_new_var(nodes_value);
    node_count = arts_config_count_nodes(tmp);
    arts_free(tmp);
  }

  if (node_count > 1) {
    /* Multi-node local: simulate cluster on a single machine. */
    config->master_boot = true;
    config->shared_pu_pool = true;
    config->nodes = node_count;

    /* Build routing table from nodes string (preserves per-node ports)
       or generate one with node_count entries of 127.0.0.1. */
    char *node_list = NULL;
    if (nodes_value) {
      node_list = arts_config_make_new_var(nodes_value);
    } else {
      /* Build "127.0.0.1, 127.0.0.1, ..." for node_count entries. */
      size_t len = (size_t)node_count * 12; /* "127.0.0.1, " per entry */
      node_list = (char *)arts_malloc(len);
      node_list[0] = '\0';
      for (unsigned int i = 0; i < node_count; i++) {
        if (i > 0) {
          strncat(node_list, ", ", len - strlen(node_list) - 1);
        }
        strncat(node_list, "127.0.0.1", len - strlen(node_list) - 1);
      }
    }

    arts_config_create_routing_table(&config, node_list);

    /* Replace all hostnames with 127.0.0.1 (user may have specified
       "localhost" or other aliases). */
    for (unsigned int i = 0; i < config->table_length; i++) {
      arts_free(config->table[i].ip_address);
      config->table[i].ip_address = arts_config_make_new_var("127.0.0.1");
      config->table[i].rank = i;
    }

    config->master_rank = 0;
    config->master_node = arts_config_make_new_var("127.0.0.1");

    config->launcher_data =
        arts_launcher_create(0, NULL, config, config->kill_mode,
                             arts_launcher_local_startup_processes,
                             arts_launcher_local_cleanup_processes);

    ARTS_INFO("Local multi-node: %u nodes on 127.0.0.1", node_count);
  } else {
    /* Single-node local (original behavior). */
    config->master_boot = false;
    config->nodes = 1;
    config->table_length = 1;
    config->master_rank = 0;
  }
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
    config_setup_local(config, vars);
  } else {
    ARTS_ERROR("Invalid launcher: %s", config->launcher);
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

  /* Single-node: reclaim the progress thread as a worker (no peer to poll
   * the fabric for). */
  if (config->table_length <= 1) {
    if (config->progress_thread_count) {
      ARTS_WARN("Single-node: reclaiming progress_threads=%u as workers",
                config->progress_thread_count);
      config->worker_thread_count += config->progress_thread_count;
      config->progress_thread_count = 0;
    }
  }

  /* Networking conditional defaults (any launcher with multiple nodes). */
  if (config->table_length > 1) {
    if (!config->progress_thread_count) {
      config->progress_thread_count = 1;
      ARTS_WARN("Multi-node: defaulting progress_threads to 1");
    }

    /* Who fixes the listen ports.  A launcher that places ranks on other
       machines cannot leave it to the runtime: each of those ranks resolves
       its peers' ports from its own copy of the config, and a probe on the
       launching machine says nothing about a remote host.  A local run is the
       opposite — every rank is here, so the runtime always picks, and naming
       ports by hand would only invite the collisions it exists to avoid. */
    const bool local_run =
        config->launcher != NULL && strcmp(config->launcher, "local") == 0;

    if (!config->port_count) {
      config->port_count = config->ports_count > 0 ? config->ports_count : 1;
    }
    if (config->ports_count > 0 && config->ports_count != config->port_count) {
      ARTS_ERROR("ports names %u port(s) but port_count=%u — the ports key must "
                 "name exactly port_count ports",
                 config->ports_count, config->port_count);
    }

    if (!local_run) {
      if (config->ports_count == 0) {
        ARTS_ERROR("ports is required with launcher=%s: every rank resolves its "
                   "peers' ports from its own copy of this config, so nothing "
                   "else can supply them",
                   config->launcher ? config->launcher : "(unset)");
      }
    } else {
      if (config->ports_count > 0) {
        ARTS_ERROR("ports cannot be set with launcher=local: all ranks share "
                   "one machine, so the runtime claims a free block itself and "
                   "hands it to the ranks it spawns.  Remove the key");
      }
      /* A spawned rank is told what the run settled on; it must not choose
         again, or the ranks would disagree about each other's ports. */
      const char *resolved = getenv(ARTS_RESOLVED_PORTS_ENV);
      if (resolved != NULL) {
        config->ports = parse_port_spec(resolved, &config->ports_count);
        if (config->ports == NULL || config->ports_count != config->port_count) {
          ARTS_ERROR("%s from the spawning rank is unusable ('%s') — expected "
                     "%u port(s)",
                     ARTS_RESOLVED_PORTS_ENV, resolved, config->port_count);
        }
      } else {
        /* Seed the search somewhere this process is unlikely to share with
           another run starting at the same moment: two runs seeded alike would
           both probe a free block, both claim it, and one would lose the bind
           with no way left to move (its peers already hold the answer).  The
           pid is the only per-run entropy available before anything is bound.
           The transport slides this seed past whatever actually holds it. */
        const unsigned int span = config->table_length * config->port_count;
        unsigned int seed = ARTS_PORT_WINDOW_LO;
        if (ARTS_PORT_WINDOW_HI > ARTS_PORT_WINDOW_LO + span) {
          unsigned int room = ARTS_PORT_WINDOW_HI - ARTS_PORT_WINDOW_LO - span;
          seed += ((unsigned int)getpid() % (room / config->port_count + 1)) *
                  config->port_count;
        }
        config->ports_count = config->port_count;
        config->ports = (unsigned int *)arts_malloc(config->port_count *
                                                    sizeof(unsigned int));
        for (unsigned int i = 0; i < config->port_count; i++) {
          config->ports[i] = seed + i;
        }
      }
    }

    /* Every node's ports come from the one base list. */
    if (config->table != NULL) {
      for (unsigned int i = 0; i < config->table_length; i++) {
        if (config->table[i].ports == NULL) {
          config->table[i].ports = (unsigned int *)arts_malloc(
              config->port_count * sizeof(unsigned int));
          if (config->shared_pu_pool) {
            /* Every rank shares 127.0.0.1, so each takes its own disjoint
               block: node i gets ports[j] + i * port_count. */
            for (unsigned int j = 0; j < config->port_count; j++) {
              config->table[i].ports[j] =
                  config->ports[j] + (i * config->port_count);
            }
            ARTS_INFO("Local multi-node: node %u takes port(s) starting at %u",
                      i, config->table[i].ports[0]);
          } else {
            memcpy(config->table[i].ports, config->ports,
                   config->port_count * sizeof(unsigned int));
          }
        }
      }
    }
  }

  /* Compute total thread count.
     If thread_count was set directly (SLURM/env), derive worker count from it.
     Otherwise compute total from worker + progress. */
  if (config->thread_count > 0) {
    config->worker_thread_count =
        config->thread_count - config->progress_thread_count;
  }
  config->thread_count =
      config->worker_thread_count + config->progress_thread_count;
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
  /* Priority 1: Compiler-embedded config data (self-contained binary). */
  if (arts_config_override_data && arts_config_override_data[0] != '\0') {
    size_t len = strlen(arts_config_override_data);
    FILE *f = fmemopen((void *)arts_config_override_data, len, "r");
    if (f) {
      return f;
    }
  }

  /* Priority 2: Compiler-injected config path. */
  if (arts_config_override_path && arts_config_override_path[0] != '\0') {
    FILE *f = fopen(arts_config_override_path, "r");
    if (f) {
      return f;
    }
    ARTS_ERROR("Config file not found: %s", arts_config_override_path);
    return NULL;
  }

  /* Priority 3: ARTS_CONFIG env var → ./arts.cfg fallback. */
  const char *location = getenv("ARTS_CONFIG");
  FILE *f = fopen(location ? location : "arts.cfg", "r");
  if (!f) {
    ARTS_ERROR("Config file not found: %s", location ? location : "arts.cfg");
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

/*--- Removed-Key Rejection --------------------------------------------------
 *
 * Keys that no longer name anything in the current config surface must fail
 * loudly with a message that names the replacement, rather than being
 * silently ignored (a cfg with a stale key would otherwise run with an
 * unintended default).
 *---------------------------------------------------------------------------*/

static void config_reject_removed_keys(struct arts_config_variable_s **vars) {
  if (config_lookup(vars, "sender_threads")) {
    ARTS_ERROR(
        "sender_threads no longer exists: the transport injects directly "
        "from workers; remove the key and add its count to worker_threads");
  }
  if (config_lookup(vars, "receiver_threads")) {
    ARTS_ERROR("receiver_threads was renamed to progress_threads: rename the "
               "key (same value/semantics -- one progress thread per node, "
               "default 1 on multi-node runs, 0 single-node)");
  }
  if (config_lookup(vars, "default_ports")) {
    ARTS_ERROR("default_ports was renamed to ports: rename the key.  Nothing "
               "defaults them any more -- the list must name exactly "
               "port_count ports, and may be omitted only on launcher=local "
               "with port_auto_select on");
  }
}

/*=============================================================================
 * arts_config_load — Phased config loading
 *
 * Open file, parse key=value pairs into linked list
 * Reject removed keys
 * Allocate config, set non-zero pre-defaults
 * Table-driven parse (replaces ~280 lines of if-else)
 * Launcher-specific setup (nodes, routing table, master)
 * Computed fields, warnings
 *===========================================================================*/

void arts_config_load(struct arts_config_s *config) {
  /* Open config file, parse key=value pairs. */
  FILE *fp = config_open_file();
  struct arts_config_variable_s *vars = arts_config_get_variables(fp);
  (void)fclose(fp);

  /* Fail loudly on keys the config surface no longer recognizes, before any
   * of their (now-absent) defaults could silently apply. */
  config_reject_removed_keys(&vars);

  /* Zero-init and set non-zero pre-defaults. */
  memset(config, 0, sizeof(*config));
  config_set_pre_defaults(config);

  /* Table-driven parse. */
  config_parse_table(config, &vars);

  /* Launcher-specific setup (nodes, routing table, master). */
  config_setup_launcher(config, &vars);

  /* Computed fields, warnings. */
  config_compute_derived(config);
  config_print_warnings(config);

  /* Cleanup variable linked list. */
  config_free_variables(vars);
}

void arts_config_destroy(struct arts_config_s *config) {
  arts_free(config->launcher);
  if (config->launcher_data) {
    arts_free(config->launcher_data);
  }
  if (config->table) {
    for (unsigned int i = 0; i < config->table_length; i++) {
      arts_free(config->table[i].ip_address);
      if (config->table[i].ports) {
        arts_free(config->table[i].ports);
      }
    }
    arts_free(config->table);
  }
  if (config->ports) {
    arts_free(config->ports);
  }
  if (config->master_node) {
    arts_free(config->master_node);
  }
  if (config->net_interface) {
    arts_free(config->net_interface);
  }
  if (config->provider) {
    arts_free(config->provider);
  }
  if (config->fabric_domain) {
    arts_free(config->fabric_domain);
  }
  if (config->counter_folder) {
    arts_free(config->counter_folder);
  }
}
