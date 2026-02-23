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
#define GNU_SOURCE // (unused — getaddrinfo_a() is not called; getaddrinfo() is
                   // POSIX)
#include "arts/network/socket_server.h"

#include <errno.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netdb.h>
#include <netinet/in.h>
#include <poll.h>
#include <unistd.h>

#include "arts.h"
#include "arts/network/connection.h"
#include "arts/network/remote_protocol.h"
#include "arts/network/server.h"
#include "arts/runtime/globals.h"
#include "arts/runtime/runtime.h"
#include "arts/system/config.h"
#include "arts/system/print.h"
#include "arts/utils/malloc.h"

struct arts_config_s *arts_global_message_table;
unsigned int ports;
// SOCKETS!
int *remote_socket_send_list;
volatile unsigned int *volatile remote_socket_send_lock_list;
struct sockaddr_in *remote_server_send_list;
bool *remote_connection_alive;

int *local_socket_recieve;
int *remote_socket_recieve_list;
fd_set read_set;
int max_fd;
struct sockaddr_in *remote_server_recieve_list;
struct pollfd *poll_incoming;

#define EDT_MUG_SIZE 32
#define PACKET_SIZE 4194304
#define INITIAL_OUT_SIZE 80000000

char *ip_list;

void arts_remote_set_message_table(struct arts_config_s *table) {
  arts_global_message_table = table;
  ports = table->port_count;
}
bool hostname_to_ip(char *host_name, char *ip) {
  int j;
  struct hostent *he;
  struct in_addr **addr_list;
  struct addrinfo *result;
  struct addrinfo hints;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_INET; // Force IPv4 - inet_addr() doesn't handle IPv6
  int error = getaddrinfo(host_name, NULL, &hints, &result);
  if (error == 0) {
    if (result->ai_addr->sa_family == AF_INET) {
      struct sockaddr_in *res = (struct sockaddr_in *)result->ai_addr;
      inet_ntop(AF_INET, &res->sin_addr, ip, 100);
    } else if (result->ai_addr->sa_family == AF_INET6) {
      struct sockaddr_in6 *res = (struct sockaddr_in6 *)result->ai_addr;
      inet_ntop(AF_INET6, &res->sin6_addr, ip, 100);
    }
    freeaddrinfo(result);
    return true;
  }
  ARTS_INFO("%s", gai_strerror(error));

  return false;
}

bool arts_server_set_ip(struct arts_config_s *config) {
  // Always initialize ip_list - it's used by arts_remote_setup_outgoing()
  ip_list = (char *)arts_malloc(100 * sizeof(char) * config->table_length);
  bool result;
  for (int i = 0; i < config->table_length; i++) {
    result = hostname_to_ip(config->table[i].ip_address,
                            ip_list + ((ptrdiff_t)100 * i));
    // result = hostname_to_ip("www.google.com", ip_list+100*i);

    if (!result) {
      ARTS_ERROR("Cannot get ip address for '%s'", config->table[i].ip_address);
    }
  }

  // When net_interface is set (e.g., "ib0"), remap resolved IPs to the
  // target interface's subnet.  Each node resolves hostnames to the default
  // interface (e.g., eno1 172.16.x.x).  We detect the local subnet
  // difference between the default and target interfaces, then apply the
  // same transformation to all resolved IPs so that traffic flows over the
  // target interface (e.g., ib0 172.17.x.x).
  if (config->net_interface) {
    char local_hostname[256];
    char local_default_ip[100];
    gethostname(local_hostname, sizeof(local_hostname));
    if (!hostname_to_ip(local_hostname, local_default_ip)) {
      ARTS_INFO("net_interface=%s: cannot resolve local hostname '%s', "
                "skipping IP remap",
                config->net_interface, local_hostname);
    } else {
      // Find the target interface's IP via getifaddrs
      struct in_addr default_addr;
      struct in_addr iface_addr;
      bool found_iface = false;
      struct ifaddrs *ifap;
      struct ifaddrs *ifa;
      getifaddrs(&ifap);
      for (ifa = ifap; ifa; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr && ifa->ifa_addr->sa_family == AF_INET &&
            strcmp(ifa->ifa_name, config->net_interface) == 0) {
          struct sockaddr_in *sa = (struct sockaddr_in *)ifa->ifa_addr;
          iface_addr = sa->sin_addr;
          found_iface = true;
          break;
        }
      }
      freeifaddrs(ifap);

      if (!found_iface) {
        ARTS_INFO("net_interface=%s: interface not found, using default IPs",
                  config->net_interface);
      } else {
        inet_pton(AF_INET, local_default_ip, &default_addr);
        uint32_t offset = ntohl(iface_addr.s_addr) - ntohl(default_addr.s_addr);

        if (offset != 0) {
          char iface_ip[100];
          inet_ntop(AF_INET, &iface_addr, iface_ip, sizeof(iface_ip));
          ARTS_INFO("net_interface=%s: remapping IPs (%s -> %s)",
                    config->net_interface, local_default_ip, iface_ip);
          for (int i = 0; i < config->table_length; i++) {
            struct in_addr addr;
            char old_ip[100];
            inet_pton(AF_INET, ip_list + ((ptrdiff_t)100 * i), &addr);
            inet_ntop(AF_INET, &addr, old_ip, sizeof(old_ip));
            addr.s_addr = htonl(ntohl(addr.s_addr) + offset);
            inet_ntop(AF_INET, &addr, ip_list + ((ptrdiff_t)100 * i), 100);
            ARTS_INFO("  node %d: %s -> %s", i, old_ip,
                      ip_list + ((ptrdiff_t)100 * i));
          }
        } else {
          ARTS_INFO("net_interface=%s: already on target subnet (%s)",
                    config->net_interface, local_default_ip);
        }
      }
    }
  }

  // Check if rank was passed via environment (SSH-launched child process)
  // This prevents recursive spawning when multiple nodes resolve to the same IP
  char *arts_rank_env = getenv("ARTS_RANK");
  if (arts_rank_env) {
    arts_global_rank_id = (unsigned int)strtol(arts_rank_env, NULL, 10);
    config->my_rank = arts_global_rank_id;
    arts_global_rank_count = config->table_length;
    return true;
  }

  // SLURM: use SLURM_PROCID for rank (srun sets this per task)
  // IP matching fails when all nodes resolve to the same address (e.g., WSL2)
  char *slurm_proc_id = getenv("SLURM_PROCID");
  if (slurm_proc_id) {
    arts_global_rank_id = (unsigned int)strtol(slurm_proc_id, NULL, 10);
    config->my_rank = arts_global_rank_id;
    arts_global_rank_count = config->table_length;
    return true;
  }

  int fd;
  struct ifreq ifr;
  char *connection = NULL;
  ifr.ifr_addr.sa_family = AF_INET;

  bool found = false;
  // if(config->net_interface == NULL)
  {
    struct ifaddrs *ifap;
    struct ifaddrs *ifa;
    struct sockaddr_in *sa;
    struct sockaddr_in6 *sa6;
    char addr[100];

    getifaddrs(&ifap);
    for (ifa = ifap; ifa && !found; ifa = ifa->ifa_next) {
      if (ifa->ifa_addr->sa_family == AF_INET) {
        sa = (struct sockaddr_in *)ifa->ifa_addr;
        inet_ntop(AF_INET, &sa->sin_addr, addr, 100);

        for (int i = 0; i < config->table_length && !found; i++) {
          if (strcmp(addr, ip_list + ((ptrdiff_t)100 * i)) == 0) {
            found = true;
            config->my_rank = i;
            arts_global_rank_id = i;
            arts_global_rank_count = arts_global_message_table->table_length;
          }
        }
      } else if (ifa->ifa_addr->sa_family == AF_INET6) {
        sa6 = (struct sockaddr_in6 *)ifa->ifa_addr;
        inet_ntop(AF_INET6, &sa6->sin6_addr, addr, 100);
        ;

        for (int i = 0; i < config->table_length && !found; i++) {
          if (strcmp(addr, ip_list + ((ptrdiff_t)100 * i)) == 0) {
            found = true;
            config->my_rank = i;
            arts_global_rank_id = i;
            arts_global_rank_count = arts_global_message_table->table_length;
          }
        }
      }
    }
    freeifaddrs(ifap);
  }
  return found;
}

void arts_ll_server_setup(struct arts_config_s *config) {
  arts_remote_set_message_table(config);

  if (!arts_server_set_ip(config) && config->nodes > 1) {
    // ARTS_INFO("[%d]Could not connect to %s", arts_global_rank_id,
    // config->net_interface);
    ARTS_ERROR("Could not resolve ip to any device");
  }
}

void arts_ll_server_shutdown() {
  int count = (int)arts_global_message_table->table_length;
  for (int i = 0; i < (count - 1) * ports; i++) {
    RSHUTDOWN(remote_socket_recieve_list[i], SHUT_RDWR);
    // RCLOSE(remote_socket_recieve_list[i]);
  }

  for (int i = 0; i < count * ports; i++) {
    if (i / ports != arts_global_rank_id) {
      RSHUTDOWN(remote_socket_send_list[i], SHUT_RDWR);
      //            RCLOSE(remote_socket_send_list[i]);
    }
  }
}

void arts_ll_server_cleanup() {
  arts_free(ip_list);
  arts_free(remote_socket_send_list);
  arts_free((void *)remote_socket_send_lock_list);
  arts_free(remote_server_send_list);
  arts_free(remote_connection_alive);
  arts_free(remote_socket_recieve_list);
  arts_free(remote_server_recieve_list);
  arts_free(poll_incoming);
  arts_free(local_socket_recieve);
}

unsigned int arts_remote_get_my_rank() {
  return arts_global_message_table->my_rank;
}

static inline bool arts_remote_connect(int rank, unsigned int port) {

  if (!remote_connection_alive[(rank * ports) + port]) {
    int res = RCONNECT(remote_socket_send_list[(rank * ports) + port],
                       (struct sockaddr *)(remote_server_send_list +
                                           ((size_t)rank * ports) + port),
                       sizeof(struct sockaddr_in));
    if (res < 0) {
      remote_connection_alive[(rank * ports) + port] = false;

      RCLOSE(remote_socket_send_list[(rank * ports) + port]);
      remote_socket_send_list[(rank * ports) + port] = arts_get_new_socket();

      // Retry with delay to handle SLURM startup skew (srun starts all
      // processes simultaneously, so the remote may not be listening yet)
      int max_retries = 300;
      int retry_count = 0;
      while (RCONNECT(remote_socket_send_list[(rank * ports) + port],
                      (struct sockaddr *)(remote_server_send_list +
                                          ((size_t)rank * ports) + port),
                      sizeof(struct sockaddr_in)) < 0) {
        if (++retry_count >= max_retries) {
          struct sockaddr_in *addr =
              remote_server_send_list + ((size_t)rank * ports) + port;
          ARTS_INFO("arts_remote_connect: Failed to connect to rank %d port %d "
                    "after %d retries (target %s:%d, errno=%d: %s)",
                    rank, port, max_retries, inet_ntoa(addr->sin_addr),
                    ntohs(addr->sin_port), errno, strerror(errno));
          return false;
        }
        RCLOSE(remote_socket_send_list[(rank * ports) + port]);
        remote_socket_send_list[(rank * ports) + port] = arts_get_new_socket();
        usleep(100000);
      }

      remote_connection_alive[(rank * ports) + port] = true;

      return true;
    }

    remote_connection_alive[(rank * ports) + port] = true;
  }

  return true;
}

// inline int arts_actual_send(char * message, unsigned int length, int rank,
// int port)
uint64_t arts_actual_send(char *message, uint64_t length, int rank, int port) {
  int res = 0;
  uint64_t total = 0;
  int iterations = 0;
  while (length != 0 && res >= 0) {
    res = RSEND(remote_socket_send_list[(rank * ports) + port], message + total,
                length, MSG_DONTWAIT);
    if (res >= 0) {
      total += res;
      length -= res;
    }
    iterations++;
    if (iterations > 1000000) {
      ARTS_INFO("arts_actual_send: stuck in loop, res=%d, length=%lu, "
                "total=%lu, errno=%d",
                res, length, total, errno);
      break;
    }
  }

  if (res < 0) {
    if (errno != EAGAIN) {
      struct arts_remote_packet_s *pk = (struct arts_remote_packet_s *)message;
      ARTS_INFO(
          "arts_remote_send_request %u Socket appears to be closed to rank %d: "
          " %s",
          pk->message_type, rank, strerror(errno));
      arts_runtime_stop();
      return -1;
    }
  }
  INCREMENT_BYTES_REMOTE_SENT_BY(total);
  INCREMENT_NUM_REMOTE_SEND_BY(1);
  return length;
}

uint64_t arts_remote_send_request(int rank, unsigned int queue, char *message,
                                  uint64_t length) {
  int port = (int)(queue % ports);
  if (arts_remote_connect(rank, port)) {
    return arts_actual_send(message, length, rank, port);
  }
  return length;
}

uint64_t arts_remote_send_payload_request(int rank, unsigned int queue,
                                          char *message, unsigned int length,
                                          char *payload, uint64_t length2) {
  int port = (int)(queue % ports);
  if (arts_remote_connect(rank, port)) {
    uint64_t temp_length = arts_actual_send(message, length, rank, port);
    if (temp_length) {
      return temp_length + length2;
    }

    return arts_actual_send(payload, length2, rank, port);
  }
  return length + length2;
}

bool arts_remote_setup_incoming() {
  // ARTS_INFO("%d", FD_SETSIZE);
  int i;
  int j;
  int k;
  int pos;
  unsigned int *my_ports =
      arts_global_message_table->table[arts_global_message_table->my_rank]
          .ports;
  socklen_t s_length = sizeof(struct sockaddr);
  int count = (int)(arts_global_message_table->table_length - 1);

  remote_socket_recieve_list =
      (int *)arts_malloc(sizeof(int) * (size_t)(count + 1) * ports);
  remote_server_recieve_list = (struct sockaddr_in *)arts_calloc(
      (size_t)(count + 1) * ports, sizeof(struct sockaddr_in));
  poll_incoming =
      (struct pollfd *)arts_malloc(sizeof(struct pollfd) * (count + 1) * ports);

  struct sockaddr_in test;

  struct sockaddr_in *local_server_addr =
      (struct sockaddr_in *)arts_calloc(ports, sizeof(struct sockaddr_in));
  local_socket_recieve = (int *)arts_calloc(ports, sizeof(int));

  int i_set_option;
  for (i = 0; i < (int)arts_global_message_table->port_count; i++) {
    local_socket_recieve[i] =
        arts_get_socket_listening(&local_server_addr[i], my_ports[i]);

    i_set_option = 1;
    setsockopt(local_socket_recieve[i], SOL_SOCKET, SO_REUSEADDR,
               (char *)&i_set_option, sizeof(i_set_option));

    int res =
        RBIND(local_socket_recieve[i], (struct sockaddr *)&local_server_addr[i],
              sizeof(local_server_addr[i]));

    if (res < 0) {
      ARTS_INFO("Bind Failed");
      ARTS_INFO("error %s", strerror(errno));
      return false;
    }

    res = RLISTEN(local_socket_recieve[i], 2 * count);

    if (res < 0) {
      ARTS_INFO("Listening Failed");
      ARTS_INFO("error %s", strerror(errno));
      return false;
    }
  }

  arts_free(local_server_addr);

  FD_ZERO(&read_set);
  for (i = 0; i < arts_global_message_table->table_length; i++) {
    if (arts_global_message_table->my_rank ==
        arts_global_message_table->table[i].rank) {
      for (j = 0; j < count; j++) {
        for (int z = 0; z < ports; z++) {
          s_length = sizeof(struct sockaddr_in);
          remote_socket_recieve_list[z + (j * ports)] = RACCEPT(
              local_socket_recieve[z], (struct sockaddr *)&test, &s_length);

          if (remote_socket_recieve_list[z + (j * ports)] < 0) {
            int retry = 0;
            int retry_limit = 3;
            while (remote_socket_recieve_list[z + (j * ports)] < 0) {
              if (retry == retry_limit) {
                ARTS_ERROR("Socket accept failed after %d retries",
                           retry_limit);
              }
              remote_socket_recieve_list[z + (j * ports)] = RACCEPT(
                  local_socket_recieve[z], (struct sockaddr *)&test, &s_length);
              retry++;
              if (remote_socket_recieve_list[z + (j * ports)] < 0) {
                ARTS_INFO("Accept Failed");
                ARTS_INFO("error %s", strerror(errno));
              }
            }
          }
          // FD_SET(remote_socket_recieve_list[j] , &read_set  );
          poll_incoming[z + (j * ports)].fd =
              remote_socket_recieve_list[z + (j * ports)];
          poll_incoming[z + (j * ports)].events = POLLIN;
        }
      }
    } else {
      for (int z = 0; z < ports; z++) {
        if (!arts_remote_connect(i, z)) {
          ARTS_INFO("Could not create initial connection");
          return false;
        }
      }
    }
  }

  return true;
}

void arts_remote_setup_outgoing() {
  int i;
  int j;
  int k;
  struct sockaddr_in server_address;
  struct sockaddr_in client_address;
  int count = (int)arts_global_message_table->table_length;
  struct hostent *he;
  struct in_addr **addr_list;
  char ip[100];
  int pos;

  remote_socket_send_list =
      (int *)arts_malloc(sizeof(int) * (size_t)count * ports);
  remote_socket_send_lock_list =
      (volatile unsigned int *)arts_calloc((size_t)count * ports, sizeof(int));
  remote_server_send_list = (struct sockaddr_in *)arts_calloc(
      (size_t)count * ports, sizeof(struct sockaddr_in));
  remote_connection_alive =
      (bool *)arts_calloc((size_t)count * ports, sizeof(bool));

  for (i = 0; i < count; i++) {
    unsigned int *target_ports = arts_global_message_table->table[i].ports;

    ARTS_INFO("arts_remote_setup_outgoing: node %d ip_list='%s' port=%u", i,
              ip_list + ((ptrdiff_t)100 * i), target_ports[0]);

    for (j = 0; j < ports; j++) {
      remote_socket_send_list[(i * ports) + j] = arts_get_socket_outgoing(
          remote_server_send_list + ((size_t)i * ports) + j, target_ports[j],
          inet_addr(ip_list + ((ptrdiff_t)100 * i)));
    }
  }
}

static ARTS_THREAD_LOCAL unsigned int thread_start;
static ARTS_THREAD_LOCAL unsigned int thread_stop;
static ARTS_THREAD_LOCAL char **bypass_buf;
static ARTS_THREAD_LOCAL uint64_t *bypass_packet_size;
static ARTS_THREAD_LOCAL int64_t *re_recieve_res;
static ARTS_THREAD_LOCAL void **re_recieve_packet;
static ARTS_THREAD_LOCAL bool *max_incoming;
static ARTS_THREAD_LOCAL bool max_out_working;

void arts_remote_set_thread_inbound_queues(unsigned int start,
                                           unsigned int stop) {
  thread_start = start;
  thread_stop = stop;
  // ARTS_INFO_MASTER("%d %d", start, stop);
  unsigned int size = stop - start;
  bypass_buf = (char **)arts_malloc(sizeof(char *) * size);
  bypass_packet_size = (uint64_t *)arts_malloc(sizeof(uint64_t) * size);
  re_recieve_res = (int64_t *)arts_calloc(size, sizeof(int64_t));
  re_recieve_packet = (void **)arts_calloc(size, sizeof(void *));
  max_incoming = (bool *)arts_calloc(size, sizeof(bool));
  for (int i = 0; i < size; i++) {
    bypass_buf[i] = (char *)arts_malloc(PACKET_SIZE);
    bypass_packet_size[i] = PACKET_SIZE;
  }
}

void arts_remote_thread_inbound_queues_cleanup() {
  unsigned int size = thread_stop - thread_start;
  for (int i = 0; i < size; i++) {
    arts_free(bypass_buf[i]);
  }
  arts_free(bypass_buf);
  arts_free(bypass_packet_size);
  arts_free(re_recieve_res);
  arts_free(re_recieve_packet);
  arts_free(max_incoming);
}

bool max_out_buffs(unsigned int ignore) {
  int time_out = 1;
  int64_t res;
  int64_t res2;
  struct arts_remote_packet_s *packet;
  // ARTS_INFO("MAX");
  res =
      RPOLL(poll_incoming + thread_start, thread_stop - thread_start, time_out);
  unsigned int pos;

  if (res == -1) {
    arts_shutdown();
    arts_runtime_stop();
  }
  if (res > 0) {
    // ARTS_INFO("MAX LOOP");
    time_out = 1;
    for (int i = (int)thread_start; i < (int)thread_stop; i++) {
      pos = i - (int)thread_start;
      if (i != ignore && poll_incoming[i].revents & POLLIN) {
        max_out_working = true;
        if (re_recieve_res[pos] == 0) {
          packet = (struct arts_remote_packet_s *)bypass_buf[pos];
          res = RRECV(remote_socket_recieve_list[i], bypass_buf[pos],
                      bypass_packet_size[pos], 0);
        } else {
          // packet = re_recieve_packet[pos];
          packet = (struct arts_remote_packet_s *)bypass_buf[pos];
          res = re_recieve_res[pos];
          re_recieve_res[pos] = 0;
        }
        if (res > 0) {
          while (res < bypass_packet_size[pos]) {
            if (bypass_buf[pos] != (char *)packet) {
              memmove(bypass_buf[pos], packet, res);
              packet = (struct arts_remote_packet_s *)bypass_buf[pos];
            }
            res2 = RRECV(remote_socket_recieve_list[i], bypass_buf[pos] + res,
                         bypass_packet_size[pos] - res, MSG_DONTWAIT);

            if (res2 < 0) {
              if (errno != EAGAIN) {
                ARTS_INFO("Error on recv return 0 %d %d", errno, EAGAIN);
                arts_shutdown();
                arts_runtime_stop();
              }

              re_recieve_res[pos] = res;
              max_incoming[pos] = true;
              break;
            }
            res += res2;
          }
          max_incoming[pos] = true;
          re_recieve_res[pos] = res;
        } else if (res == -1) {
          ARTS_INFO("Error on recv socket return 0");
          ARTS_INFO("error %s", strerror(errno));
          arts_shutdown();
          arts_runtime_stop();
          return false;
        } else if (res == 0) {
          arts_shutdown();
          arts_runtime_stop();
          return false;
        }
      }
    }
  }
  return true;
}

bool arts_server_try_to_receive(
    char **in_buffer, const int *in_packet_size,
    const volatile unsigned int *remote_steal_lock) {
  (void)in_buffer;
  (void)in_packet_size;
  (void)remote_steal_lock;
  int i;
  int steal_handler_thread = 0;
  int64_t res;
  int64_t res2;
  struct arts_remote_packet_s *packet;
  int count = (int)(arts_global_message_table->table_length - 1);
  fd_set temp_set;
  int time_out = 300000;
  struct timeval sel_timeout;
  unsigned int pos;
  res =
      RPOLL(poll_incoming + thread_start, thread_stop - thread_start, time_out);

  if (res == -1) {
    arts_shutdown();
    arts_runtime_stop();
  }

  unsigned int space_left;
  bool packet_incoming_on_a_socket = false;
  bool goto_next = false;
  if (res > 0) {
    // ARTS_INFO("POLL");
    time_out = 1;
    max_out_working = true;
    while (max_out_working) {
      max_out_working = false;
      for (i = (int)thread_start; i < (int)thread_stop; i++) {
        pos = i - thread_start;
        goto_next = false;
        // if( poll_incoming[i].revents & POLLIN )
        // if(!max_out_buffs(-1))
        //     return false;
        // if( max_incoming[pos] )
        if (poll_incoming[i].revents & POLLIN) {
          // ARTS_INFO("Here2");
          max_incoming[pos] = false;
          if (re_recieve_res[pos] == 0) {
            // ARTS_INFO("Here3a");
            packet = (struct arts_remote_packet_s *)bypass_buf[pos];
            res = RRECV(remote_socket_recieve_list[i], bypass_buf[pos],
                        bypass_packet_size[pos], MSG_DONTWAIT);
            if (res > 0) {
              INCREMENT_BYTES_REMOTE_RECEIVED_BY(res);
            }
          } else {
            // packet = re_recieve_packet[pos];
            packet = (struct arts_remote_packet_s *)bypass_buf[pos];
            res = re_recieve_res[pos];
            re_recieve_res[pos] = 0;
          }
          if (res > 0) {
            packet_incoming_on_a_socket = true;
            while (res > 0) {
              while (res < sizeof(struct arts_remote_packet_s)) {
                if (bypass_buf[pos] != (char *)packet) {
                  memmove(bypass_buf[pos], packet, res);
                  packet = (struct arts_remote_packet_s *)bypass_buf[pos];
                }
                res2 =
                    RRECV(remote_socket_recieve_list[i], bypass_buf[pos] + res,
                          bypass_packet_size[pos] - res, MSG_DONTWAIT);
                if (res2 > 0) {
                  INCREMENT_BYTES_REMOTE_RECEIVED_BY(res2);
                }

                if (res2 < 0) {
                  if (errno != EAGAIN) {
                    ARTS_INFO("Error on recv return 0 %d %d", errno, EAGAIN);
                    arts_shutdown();
                    arts_runtime_stop();
                  }

                  re_recieve_res[pos] = res;
                  goto_next = true;
                  break;
                }
                // space_left-=res2;
                res += res2;
              }
              if (goto_next) {
                break;
              }

              if (bypass_packet_size[pos] < packet->size) {
                // For large packets (>256MB), avoid 4x over-allocation
                uint64_t new_buf_size = (packet->size > (1ULL << 28))
                                            ? packet->size
                                            : packet->size * 4;
                char *next_buf = (char *)arts_malloc(new_buf_size);

                memcpy(next_buf, bypass_buf[pos], bypass_packet_size[pos]);

                arts_free(bypass_buf[pos]);

                packet = (struct arts_remote_packet_s *)(next_buf +
                                                         (((char *)packet) -
                                                          (bypass_buf[pos])));
                bypass_buf[pos] = next_buf;
                bypass_packet_size[pos] = new_buf_size;
              }

              while (res < packet->size) {
                if (bypass_buf[pos] != (char *)packet) {
                  memmove(bypass_buf[pos], packet, res);
                  packet = (struct arts_remote_packet_s *)bypass_buf[pos];
                }
                res2 =
                    RRECV(remote_socket_recieve_list[i], bypass_buf[pos] + res,
                          bypass_packet_size[pos] - res, MSG_DONTWAIT);
                if (res2 > 0) {
                  INCREMENT_BYTES_REMOTE_RECEIVED_BY(res2);
                }
                if (res2 < 0) {
                  if (errno != EAGAIN) {
                    ARTS_INFO("Error on recv return 0 %d %d", errno, EAGAIN);
                    ARTS_INFO("error %s", strerror(errno));
                    arts_shutdown();
                    arts_runtime_stop();
                  }
                  re_recieve_res[pos] = res;
                  goto_next = true;
                  break;
                }
                res += res2;
              }
              if (goto_next) {
                break;
              }
              INCREMENT_NUM_REMOTE_RECEIVE_BY(1);
              arts_server_process_packet(packet);

              res -= (int64_t)packet->size;
              packet = (struct arts_remote_packet_s *)(((char *)packet) +
                                                       packet->size);
            }
          } else if (res == -1) {
            arts_shutdown();
            arts_runtime_stop();
            return false;
          } else if (res == 0) {
            arts_shutdown();
            arts_runtime_stop();
            return false;
          }
        }
      }
    }
    return packet_incoming_on_a_socket;
  }
  return false;
}

void arts_server_ping_pong_test_recieve(char *in_buffer, int in_packet_size) {
  int packet_size = in_packet_size;
  char *buf = in_buffer;
  int i;
  int res;
  int res2;
  int steal_handler_thread = 0;
  int pos;
  struct arts_remote_packet_s *packet = (struct arts_remote_packet_s *)buf;
  int count = (int)(arts_global_message_table->table_length - 1);
  fd_set temp_set;
  int time_out = 100;
  struct timeval sel_timeout;
  temp_set = read_set;
  sel_timeout.tv_sec = 10;
  sel_timeout.tv_usec = time_out;
  bool recieved = false;

  while (!recieved) {
    res = RPOLL(poll_incoming, count, time_out);
    time_out = 1;
    // if(res)
    for (i = 0; i < count; i++) {
      if (poll_incoming[i].revents & POLLIN) {
        packet = (struct arts_remote_packet_s *)buf;
        res = RRECV(remote_socket_recieve_list[i], buf, packet_size, 0);
        if (res > 0) {
          while (res > 0) {
            while (res < sizeof(struct arts_remote_packet_s)) {
              if (buf != (char *)packet) {
                memmove(buf, packet, res);
                packet = (struct arts_remote_packet_s *)buf;
              }
              res2 = RRECV(remote_socket_recieve_list[i], buf + res,
                           packet_size - res, 0);
              res += res2;
              if (res2 == -1) {
                ARTS_INFO("Error on recv return 0");
                ARTS_INFO("error %s", strerror(errno));
                arts_shutdown();
                return;
              }
            }

            while (res < packet->size) {
              if (buf != (char *)packet) {
                memmove(buf, packet, res);
                packet = (struct arts_remote_packet_s *)buf;
              }
              res2 = RRECV(remote_socket_recieve_list[i], buf + res,
                           packet_size - res, 0);
              res += res2;
              if (res2 == -1) {
                ARTS_INFO("Error on recv return 0");
                ARTS_INFO("error %s", strerror(errno));
                arts_shutdown();
                return;
              }
            }
            if (packet->message_type == ARTS_REMOTE_PINGPONG_TEST_MSG) {
              recieved = true;
            } else {
              ARTS_INFO("Shit Packet %d %d %d", packet->message_type,
                        packet->size, packet->rank);
            }
            res -= (int)packet->size;
            packet = (struct arts_remote_packet_s *)(((char *)packet) +
                                                     packet->size);
          }
        } else if (res == -1) {
          ARTS_INFO("Error on recv socket return 0");
          ARTS_INFO("error %s", strerror(errno));
          arts_shutdown();
          return;
        } else if (res == 0) {
          ARTS_INFO("Hmm socket close?");
          arts_shutdown();
          return;
        }
      }
    }
  }
}

int arts_get_new_socket() {
  int socket_out = RSOCKET(PF_INET, SOCK_STREAM, 0);
  if (socket_out < 0) {
    ARTS_ERROR("socket() failed: %s", strerror(errno));
  }
  return socket_out;
}

int arts_get_socket_listening(struct sockaddr_in *listening_socket,
                              unsigned int port) {
  memset((char *)listening_socket, 0, sizeof(*listening_socket));
  int socket_out = RSOCKET(PF_INET, SOCK_STREAM, 0);
  if (socket_out < 0) {
    ARTS_ERROR("socket() failed: %s", strerror(errno));
  }
  listening_socket->sin_family = AF_INET;
  listening_socket->sin_addr.s_addr = htonl(INADDR_ANY);
  listening_socket->sin_port = htons(port);
  return socket_out;
}

int arts_get_socket_outgoing(struct sockaddr_in *outgoing_socket,
                             unsigned int port, in_addr_t s_addr) {
  memset((char *)outgoing_socket, 0, sizeof(*outgoing_socket));
  int socket_out = RSOCKET(PF_INET, SOCK_STREAM, 0);
  if (socket_out < 0) {
    ARTS_ERROR("socket() failed: %s", strerror(errno));
  }
  outgoing_socket->sin_family = AF_INET;
  outgoing_socket->sin_addr.s_addr = s_addr;
  outgoing_socket->sin_port = htons(port);
  return socket_out;
}
