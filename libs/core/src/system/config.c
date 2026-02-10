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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <unistd.h>

#include "arts.h"
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

char *arts_config_find_variable_char(struct arts_config_variable_s *head,
                                 const char *string) {
  struct arts_config_variable_s *found = NULL;
  char *overide = getenv(string);

  if (overide) {
    return overide;
}

  while (head != NULL) {
    if (strcmp(string, head->variable) == 0) {
      found = head;
      break;
    }
    head = head->next;
  }

  if (found) {
    return found->value;
}

  return NULL;
}

unsigned int arts_config_get_variable(FILE *config, const char *look_for_me) {
  char *line;
  size_t len = 0;
  ssize_t read;
  char *var;
  char *val;
  int size;
  struct arts_config_variable_s *c_var;
  struct arts_config_variable_s *head;
  struct arts_config_variable_s *next = NULL;

  while ((read = getline(&line, &len, config)) != -1) {
    var = strtok(line, "=");
    val = strtok(NULL, "=");

    if (strcmp(look_for_me, var) == 0) {
      if (val == NULL) {
        free(line);
        return 4;
}
      size = (int)strlen(val);

      if (val[size - 1] == '\n') {
        val[size - 1] = '\0';
}

      {
        long result = strtol(val, NULL, 10);
        free(line);
        return (unsigned int)result;
      }
    }
  }
  if (line) {
    free(line);
}
  return 4;
}

void remove_white_spaces(char *str) {
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

unsigned int arts_config_get_value(char *start, char *stop) {
  int i;
  int value;
  int size = (int)(stop - start);
  for (i = 0; i < size; i++) {
    if (isdigit(start[i])) {
      break;
}
  }
  if (i == size) {
    // No digits found, raise an error
    arts_printf("arts_config_get_value: No digits found in %s\n", start);
    arts_debug_generate_seg_fault();
  }
  if (*stop == ':') {
    *stop = '\0';
    value = (int)strtol(start + i, NULL, 10);
    *stop = ':';
  } else {
    value = (int)strtol(start + i, NULL, 10);
  }
  return value;
}

char *arts_config_get_node_name(char *start, const char *stop) {
  int i;
  int value;
  int size = (int)(stop - start);
  char *name;

  for (i = 0; i < size; i++) {
    if (isdigit(start[i])) {
      break;
}
  }
  if (i == size) {
    // No digits found, return the original string
    arts_debug_generate_seg_fault();
  }
  name = (char *)arts_malloc(size);
  strncpy(name, start, size);
  return name;
}

char *arts_config_get_hostname(char *name, unsigned int value) {
  unsigned int length = strlen(name);
  unsigned int digits = 1;
  unsigned int temp = value;
  unsigned int stop;
  char *out_name = (char *)arts_malloc(length);

  while (temp > 9) {
    temp /= 10;
    digits++;
  }

  temp = value;
  for (unsigned int i = 0; i < length; i++) {
    if (isdigit(name[i])) {
      stop = i;
      while (stop < length) {
        if (!isdigit(name[stop])) {
          break;
        }
        stop++;
      }

      for (unsigned int j = stop - 1; j > (stop - 1) - digits; j--) {
        // name[j]= itoa( value%10 );
        // sprintf(name+j,"%d",value%10);
        name[j] = (char)('0' + (value % 10));
        value /= 10;
      }

      for (unsigned int j = (stop - 1) - digits; j >= i; j--) {
        name[j] = '0';
      }
      break;
    }
  }
  strncpy(out_name, name, length);
  return out_name;
}

char *arts_config_get_slurm_hostname(char *name, char *digit_sample,
                                 unsigned int value, bool ib, char *prefix,
                                 char *suffix) {
  (void)ib;
  unsigned int length = strlen(name);
  unsigned int digit_length = strlen(digit_sample);
  unsigned int suffix_length = 0;
  unsigned int prefix_length = 0;
  unsigned int name_length;

  if (suffix != NULL) {
    suffix_length = strlen(suffix);
}

  if (prefix != NULL) {
    prefix_length = strlen(prefix);
}

  name_length = length + digit_length + 1 + prefix_length + suffix_length;
  char *out_name = (char *)arts_malloc(name_length);

  if (prefix != NULL) {
    strncpy(out_name, prefix, prefix_length);
    strncpy(out_name + prefix_length, name, length);
  } else {
    strncpy(out_name, name, length);
}

  for (unsigned int i = digit_length; i > 0; i--) {
    out_name[prefix_length + length + i - 1] = (char)('0' + (value % 10));
    value /= 10;
  }

  if (suffix != NULL) {
    strncpy(out_name + prefix_length + digit_length + length, suffix,
            suffix_length);
    strncpy(out_name + prefix_length + digit_length + length + suffix_length, "\0",
            1);
  } else {
    strncpy(out_name + prefix_length + digit_length + length, "\0", 1);
}
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

  unsigned int suffix_length = 0;
  unsigned int prefix_length = 0;
  unsigned int total_length = 0;

  char *prefix = (*config)->prefix;
  char *suffix = (*config)->suffix;

  if (suffix != NULL) {
    suffix_length = strlen(suffix);
}
  if (prefix != NULL) {
    prefix_length = strlen(prefix);
}

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
                table[current_node].ip_address = arts_config_get_slurm_hostname(
                    name, node_begin, start, (*config)->ib_names,
                    (*config)->prefix, (*config)->suffix);
                start += direction;
                current_node++;
              }
            } else {
              if (node_begin[strlen(node_begin) - 1] == ']') {
                node_begin[strlen(node_begin) - 1] = '\0';
                done = true;
              }

              unsigned int name_length = strlen(name);
              str_length = strlen(node_begin) + name_length;
              total_length = str_length + 1 + prefix_length + suffix_length;
              temp = (char *)arts_malloc(total_length);

              if (prefix != NULL) {
                strncpy(temp, prefix, prefix_length);
}
              strncpy(temp + prefix_length, name, name_length);

              strncpy(temp + prefix_length + name_length, node_begin,
                      strlen(node_begin));

              if (suffix != NULL) {
                strncpy(temp + prefix_length + str_length, suffix, suffix_length);
}
              strncpy(temp + total_length - 1, "\0", 1);

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
        total_length = str_length + 1 + prefix_length + suffix_length;
        temp = (char *)arts_malloc(total_length);

        if (prefix != NULL) {
          strncpy(temp, prefix, prefix_length);
}
        strncpy(temp + prefix_length, node_begin, str_length);

        if (suffix != NULL) {
          strncpy(temp + prefix_length + str_length, suffix, suffix_length);
}
        strncpy(temp + total_length - 1, "\0", 1);

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

unsigned int arts_config_get_number_of_threads(char *location) {
  FILE *config_file = NULL;
  if (location == NULL) {
    config_file = fopen("arts.cfg", "r");
  } else {
    config_file = fopen(location, "r");
}

  if (config_file == NULL) {
    return 4;
  }

  unsigned int result = arts_config_get_variable(config_file, "threads");
  (void)fclose(config_file);
  return result;
}

struct arts_config_s *arts_config_load() {
  FILE *config_file = NULL;
  struct arts_config_s *config;
  struct arts_config_variable_s *config_variables;
  struct arts_config_variable_s *found_variable;
  char *found_variable_char;

  char *end = NULL;

  config = (struct arts_config_s *)arts_calloc(1, sizeof(struct arts_config_s));

  char *location = getenv("ARTS_CONFIG");
  if (location) {
    config_file = fopen(location, "r");
  } else {
    config_file = fopen("arts.cfg", "r");
}

  if (config_file == NULL) {
    ARTS_INFO("No Config file found (./arts.cfg).");
    config_variables = NULL;
    arts_debug_generate_seg_fault();
  } else {
    config_variables = arts_config_get_variables(config_file);
    (void)fclose(config_file);
}

  found_variable = arts_config_find_variable(&config_variables, "launcher");
  if (found_variable == NULL) {
    config->launcher = arts_config_make_new_var("ssh");
  } else if (strncmp(found_variable->value, "slurm", 5) == 0) {
    config->launcher = arts_config_make_new_var("slurm");
  } else if (strncmp(found_variable->value, "lsf", 3) == 0) {
    config->launcher = arts_config_make_new_var("lsf");
  } else if (strncmp(found_variable->value, "local", 5) == 0) {
    config->launcher = arts_config_make_new_var("local");
  } else {
    config->launcher = arts_config_make_new_var("ssh");
}

  char *kill_set = getenv("kill_mode");
  if (kill_set == NULL) {
    if ((found_variable =
             arts_config_find_variable(&config_variables, "kill_mode")) != NULL) {
      config->kill_mode = strtol(found_variable->value, &end, 10);
    } else {
      config->kill_mode = 0;
    }
  } else {
    config->kill_mode = strtol(kill_set, &end, 10);
  }

  if ((found_variable = arts_config_find_variable(&config_variables, "core_dump")) !=
      NULL) {
    config->core_dump = strtol(found_variable->value, &end, 10);
  } else {
    config->core_dump = 0;
  }

  if ((found_variable = arts_config_find_variable(&config_variables, "pin_stride")) !=
      NULL) {
    config->pin_stride = strtol(found_variable->value, &end, 10);
  } else {
    config->pin_stride = 1;
  }

  if ((found_variable =
           arts_config_find_variable(&config_variables, "print_topology")) != NULL) {
    config->print_topology = strtol(found_variable->value, &end, 10);
  } else {
    config->print_topology = 0;
  }

  if ((found_variable = arts_config_find_variable(&config_variables, "threads")) !=
      NULL) {
    config->thread_count = strtol(found_variable->value, &end, 10);
  } else {
    config->thread_count = 4;
  }

  if ((found_variable = arts_config_find_variable(&config_variables, "os_threads")) !=
      NULL) {
    config->os_thread_count = strtol(found_variable->value, &end, 10);
  } else {
    config->os_thread_count = 0;
  }

  if ((found_variable = arts_config_find_variable(&config_variables,
                                              "cores_per_network_thread")) != NULL) {
    config->cores_per_network_thread = strtol(found_variable->value, &end, 10);
  } else {
    config->cores_per_network_thread = 1;
  }

  if ((found_variable = arts_config_find_variable(&config_variables, "ports")) !=
      NULL) {
    config->ports = strtol(found_variable->value, &end, 10);
  } else if (strncmp(config->launcher, "local", 5) != 0) {
    config->ports = 1;
  }

  if ((found_variable = arts_config_find_variable(&config_variables, "outgoing")) !=
      NULL) {
    config->sender_count = strtol(found_variable->value, &end, 10);
  } else if (strncmp(config->launcher, "local", 5) != 0) {
    config->sender_count = 1;
  }

  if ((found_variable = arts_config_find_variable(&config_variables, "incoming")) !=
      NULL) {
    config->receiver_count = strtol(found_variable->value, &end, 10);
  } else if (strncmp(config->launcher, "local", 5) != 0) {
    config->receiver_count = 1;
  }

  if ((found_variable =
           arts_config_find_variable(&config_variables, "net_interface")) != NULL) {
    config->net_interface = arts_config_make_new_var(found_variable->value);

    if (config->net_interface[0] == 'i') {
      config->ib_names = true;
    } else {
      config->ib_names = false;
}
  } else {
    config->net_interface = NULL;
    config->ib_names = false;
  }

  if ((found_variable =
           arts_config_find_variable(&config_variables, "master_node")) != NULL) {
    if (config->master_node) {
      arts_free(config->master_node);
    }
    config->master_node = arts_config_make_new_var(found_variable->value);
  } else if (strncmp(config->launcher, "local", 5) != 0) {
    config->master_node = NULL;
  }

  if ((found_variable = arts_config_find_variable(&config_variables, "prefix")) !=
      NULL) {
    config->prefix = arts_config_make_new_var(found_variable->value);
  } else {
    config->prefix = NULL;
}

  if ((found_variable = arts_config_find_variable(&config_variables, "suffix")) !=
      NULL) {
    config->suffix = arts_config_make_new_var(found_variable->value);
  } else {
    config->suffix = NULL;
}

  if ((found_variable =
           arts_config_find_variable(&config_variables, "counter_folder")) != NULL) {
    config->counter_folder = arts_config_make_new_var(found_variable->value);
  } else {
    config->counter_folder = arts_config_make_new_var("./counters");
}

  if ((found_variable = arts_config_find_variable(
           &config_variables, "counter_capture_interval")) != NULL) {
    config->counter_capture_interval = strtol(found_variable->value, &end, 10);
  } else {
    ARTS_DEBUG_ONCE("Defaulting the counter capture interval to 100 ms");
    config->counter_capture_interval = 100;
  }

  if ((found_variable_char = arts_config_find_variable_char(
           config_variables, "print_node_stats")) != NULL) {
    config->print_node_stats = strtol(found_variable_char, &end, 10);
  } else {
    config->print_node_stats = 0;
}

  if ((found_variable_char =
           arts_config_find_variable_char(config_variables, "scheduler")) != NULL) {
    config->scheduler = strtol(found_variable_char, &end, 10);
  } else {
    config->scheduler = 0;
}

  if ((found_variable_char = arts_config_find_variable_char(config_variables,
                                                      "shutdown_epoch")) != NULL) {
    config->shutdown_epoch = strtol(found_variable_char, &end, 10);
  } else {
    config->shutdown_epoch = 0;
}

  if ((found_variable_char = arts_config_find_variable_char(
           config_variables, "shad_loop_stride")) != NULL) {
    config->shad_loop_stride = strtol(found_variable_char, &end, 10);
  } else {
    config->shad_loop_stride = 32;
}

  // @awmm tmt
  if ((found_variable = arts_config_find_variable(&config_variables, "tmt")) != NULL) {
    config->tmt = strtol(found_variable->value, &end, 10);
  } else {
    config->tmt = 0;
}

  if ((found_variable = arts_config_find_variable(&config_variables, "core_count")) !=
      NULL) {
    config->core_count = strtol(found_variable->value, &end, 10);
  } else {
    config->core_count = 0;
}

  if ((found_variable = arts_config_find_variable(&config_variables, "gpu")) != NULL) {
    config->gpu = strtol(found_variable->value, &end, 10);
  } else {
    config->gpu = 0;
}

  if ((found_variable =
           arts_config_find_variable(&config_variables, "gpu_locality")) != NULL) {
    config->gpu_locality = strtol(found_variable->value, &end, 10);
  } else {
    config->gpu_locality = 0;
}

  if ((found_variable = arts_config_find_variable(&config_variables, "gpu_fit")) !=
      NULL) {
    config->gpu_fit = strtol(found_variable->value, &end, 10);
  } else {
    config->gpu_fit = 0;
}

  if ((found_variable = arts_config_find_variable(&config_variables, "gpu_lc_sync")) !=
      NULL) {
    config->gpu_lc_sync = strtol(found_variable->value, &end, 10);
  } else {
    config->gpu_lc_sync = 0;
}

  if ((found_variable =
           arts_config_find_variable(&config_variables, "gpu_max_edts")) != NULL) {
    config->gpu_max_edts = strtol(found_variable->value, &end, 10);
  } else {
    config->gpu_max_edts = (unsigned int)-1;
}

  if ((found_variable =
           arts_config_find_variable(&config_variables, "gpu_max_memory")) != NULL) {
    config->gpu_max_memory = strtol(found_variable->value, &end, 10);
  } else {
    config->gpu_max_memory = (uint64_t)-1;
}

  if ((found_variable = arts_config_find_variable(&config_variables, "gpu_p2p")) !=
      NULL) {
    config->gpu_p2p = strtol(found_variable->value, &end, 10) > 0;
  } else {
    config->gpu_p2p = false;
}

  if ((found_variable = arts_config_find_variable(&config_variables,
                                              "gpu_route_table_size")) != NULL) {
    config->gpu_route_table_size = strtol(found_variable->value, &end, 10);
  } else {
    config->gpu_route_table_size = 12; // 2^12
}

  if ((found_variable = arts_config_find_variable(&config_variables,
                                              "free_db_after_gpu_run")) != NULL) {
    config->free_db_after_gpu_run = strtol(found_variable->value, &end, 10) > 0;
  } else {
    config->free_db_after_gpu_run = false;
}

  if (config->free_db_after_gpu_run) {
    ARTS_INFO("FreeDbAfterGpuRun is turned on... This mode is intended for "
              "testing not performance.");
  }

  if ((found_variable =
           arts_config_find_variable(&config_variables, "run_gpu_gc_idle")) != NULL) {
    config->run_gpu_gc_idle = strtol(found_variable->value, &end, 10) > 0;
  } else {
    config->run_gpu_gc_idle = true;
}

  if ((found_variable =
           arts_config_find_variable(&config_variables, "run_gpu_gc_pre_edt")) != NULL) {
    config->run_gpu_gc_pre_edt = strtol(found_variable->value, &end, 10) > 0;
  } else {
    config->run_gpu_gc_pre_edt = false;
}
  if (config->run_gpu_gc_pre_edt) {
    ARTS_INFO(
        "RunGpuGcPreEdt is turned on... This mode is intended for testing "
        "not performance.");
  }

  if ((found_variable = arts_config_find_variable(&config_variables,
                                              "delete_zeros_gpu_gc")) != NULL) {
    config->delete_zeros_gpu_gc = strtol(found_variable->value, &end, 10) > 0;
  } else {
    config->delete_zeros_gpu_gc = true;
}

  if ((found_variable = arts_config_find_variable(&config_variables, "gpu_buff_on")) !=
      NULL) {
    config->gpu_buff_on = strtol(found_variable->value, &end, 10) > 0;
  } else {
    config->gpu_buff_on = false;
}

  // WARNING: Slurm Launcher Set!
  if (strncmp(config->launcher, "slurm", 5) == 0) {
    config->master_boot = false;

    char *threads_temp = getenv("SLURM_CPUS_PER_TASK");
    if (threads_temp != NULL) {
      config->thread_count = strtol(threads_temp, &end, 10);
}

    char *slurm_nodes;
    slurm_nodes = getenv("SLURM_NNODES");
    if (slurm_nodes != NULL) {
      config->nodes = strtol(slurm_nodes, &end, 10);
    } else {
      config->nodes = 1;
    }

    char *node_list = getenv("SLURM_STEP_NODELIST");
    arts_config_create_routing_table(&config, node_list);

    unsigned int length = strlen(config->table[0].ip_address) + 1;
    if (config->master_node) {
      arts_free(config->master_node);
    }
    config->master_node = (char *)arts_malloc(sizeof(char) * length);
    strncpy(config->master_node, config->table[0].ip_address, length);

    for (int i = 0; i < config->table_length; i++) {
      config->table[i].rank = i;
      if (strcmp(config->master_node, config->table[i].ip_address) == 0) {
        config->master_rank = i;
      }
    }
  } else if (strncmp(config->launcher, "lsf", 3) == 0) {
    config->master_boot = false;
    unsigned int count = 0;
    char *node_list = extract_nodelist_lsf("LSB_HOSTS", 1, &count);
    if (!node_list) {
      node_list = extract_nodelist_lsf("LSB_MCPU_HOSTS", 2, &count);
    }
    config->nodes = count;

    arts_config_create_routing_table(&config, node_list);

    unsigned int length = strlen(config->table[0].ip_address) + 1;
    if (config->master_node) {
      arts_free(config->master_node);
    }
    config->master_node = (char *)arts_malloc(sizeof(char) * length);

    strncpy(config->master_node, config->table[0].ip_address, length);

    for (int i = 0; i < config->table_length; i++) {
      config->table[i].rank = i;
      if (strcmp(config->master_node, config->table[i].ip_address) == 0) {
        config->master_rank = i;
      }
    }
  } else if (strncmp(config->launcher, "ssh", 3) == 0) {
    config->launcher_data =
        arts_remote_launcher_create(0, NULL, config, config->kill_mode,
                                 arts_remote_launcher_ssh_startup_processes,
                                 arts_remote_launcher_ssh_cleanup_processes);
    config->master_boot = true;

    char *node_list = 0;
    if ((found_variable = arts_config_find_variable(&config_variables, "nodes")) !=
        NULL) {
      node_list = found_variable->value;

      if ((found_variable =
               arts_config_find_variable(&config_variables, "node_count")) != NULL) {
        config->nodes = strtol(found_variable->value, &end, 10);
      } else {
        config->nodes = arts_config_count_nodes(node_list);
}
    } else {
      node_list = (char *)arts_malloc(sizeof(char) * strlen("localhost\0"));
      strncpy(node_list, "localhost\0", strlen("localhost\0") + 1);
      config->nodes = 1;
    }

    arts_config_create_routing_table(&config, node_list);

    if (config->master_node == NULL) {
      unsigned int length = strlen(config->table[0].ip_address) + 1;
      config->master_node = (char *)arts_malloc(sizeof(char) * length);
      strncpy(config->master_node, config->table[0].ip_address, length);
    }

    for (int i = 0; i < config->table_length; i++) {
      config->table[i].rank = i;
      if (strcmp(config->master_node, config->table[i].ip_address) == 0) {
        config->master_rank = i;
      }
    }
  } else if (strncmp(config->launcher, "local", 5) == 0) {
    config->master_boot = false;
    config->master_node = NULL;
    // OS Threads
    char *threads_os = getenv("OS_THREAD_COUNT");
    if (threads_os != NULL) {
      config->os_thread_count = strtol(threads_os, &end, 10);
    } else if (!config->os_thread_count) {
      config->os_thread_count = 0; // Default to single thread.
}
    // OS Threads
    char *threads_user = getenv("USER_THREAD_COUNT");
    if (threads_user != NULL) {
      config->thread_count = strtol(threads_user, &end, 10);
    } else if (!config->thread_count) {
      config->thread_count = 4; // Default to single thread.
}
    config->nodes = 1;
    config->table_length = 1; // for GUID
    config->master_rank = 0;
  } else {
    exit(1);
  }

  if ((found_variable = arts_config_find_variable(&config_variables, "stack_size")) !=
      NULL) {
    config->stack_size = strtoull(found_variable->value, &end, 10);
  } else {
    config->stack_size = 0;
  }

  if ((found_variable = arts_config_find_variable(&config_variables,
                                              "worker_init_deque_size")) != NULL) {
    config->deque_size = strtol(found_variable->value, &end, 10);
  } else {
    config->deque_size = 4096;
  }

  if ((found_variable = arts_config_find_variable(&config_variables, "port")) !=
      NULL) {
    if (found_variable->value[0] == '[') {
      char *ptr = found_variable->value + 1; // skip '['
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
        } else {
          config->port_range = false;
          config->port = 75563;
        }
      } else {
        config->port_range = false;
        config->port = 75563;
      }
    } else {
      config->port_range = false;
      config->port = strtol(found_variable->value, &end, 10);
    }
  } else if (strncmp(config->launcher, "local", 5) != 0) {
    config->port_range = false;
    config->port = 75563;
  }

  // Assign per-node ports from port range to routing table
  // Each node gets config->ports consecutive ports, non-overlapping
  // e.g., ports=2, port=[10001-10004]: node0=10001,10002 node1=10003,10004
  if (config->port_range && config->table != NULL) {
    for (int i = 0; i < config->table_length; i++) {
      config->table[i].port = config->port_start + (i * config->ports);
    }
  }

  if ((found_variable =
           arts_config_find_variable(&config_variables, "route_table_size")) != NULL) {
    config->route_table_size = strtol(found_variable->value, &end, 10);
  } else {
    config->route_table_size = 20;
  }

  int route_table_entries = 1;
  for (int i = 0; i < config->route_table_size; i++) {
    route_table_entries *= 2;
}
  config->route_table_entries = route_table_entries;

  int gpu_route_table_entries = 1;
  for (int i = 0; i < config->gpu_route_table_size; i++) {
    gpu_route_table_entries *= 2;
}
  config->gpu_route_table_entries = gpu_route_table_entries;

  if ((found_variable = arts_config_find_variable(&config_variables, "pin")) != NULL) {
    config->pin_threads = strtol(found_variable->value, &end, 10);
  } else {
    config->pin_threads = 1;
  }

  while (config_variables != NULL) {
    struct arts_config_variable_s *next_var = config_variables->next;
    arts_free(config_variables);
    config_variables = next_var;
  }

  return config;
}

void arts_config_destroy(struct arts_config_s *config) {
  arts_free(config->launcher);
  if (config->launcher_data) {
    arts_free(config->launcher_data);
  }
  if (config->table) {
    for (int i = 0; i < config->table_length; i++) {
      arts_free(config->table[i].ip_address);
}
    arts_free(config->table);
  }
  if (config->master_node) {
    arts_free(config->master_node);
  }
  arts_free(config);
}
