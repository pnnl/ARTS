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
#include "arts/transport/net.h"

#ifdef ARTS_TRANSPORT_OFI

#include <errno.h>
#include <stdint.h>
#include <string.h>
#include <sys/socket.h>

#include "arts/system/identity.h" /* arts_global_rank_id / arts_global_rank_count */
#include "arts/system/print.h"
#include "arts/utils/malloc.h"

/* Data-mesh sockets owned by socket.c.  The exchange rides them once, while
 * the runtime is otherwise quiescent (this runs before any worker / sender /
 * receiver thread is spawned, so no live traffic contends for these fds), and
 * consumes exactly one address frame per peer connection — leaving each socket
 * byte-clean for the live TCP transport that keeps using them afterward.
 *   remote_socket_send_list[peer*ports + port] — this rank's connection to peer
 *   remote_socket_receive_list[port + conn*ports] — accepted peer connections,
 *     in ACCEPTANCE order (not rank order), hence the embedded-rank demux. */
extern int *remote_socket_send_list;
extern int *remote_socket_receive_list;
extern unsigned int ports;

/* One length-prefixed address frame: the sender's rank (so the receiver can
 * demux acceptance-ordered connections back to rank order), then the fabric
 * address blob length, then the blob.  The rank/len fields are host-endian on
 * the wire: peers share one ABI/endianness (the launcher never spans
 * heterogeneous nodes), so no byte-order conversion is applied. */
struct net_addr_frame_s {
  uint32_t rank;
  uint32_t len;
  uint8_t addr[ARTS_NET_ADDR_MAX];
};

static bool xfer_write_all(int fd, const void *buf, size_t len) {
  const uint8_t *p = (const uint8_t *)buf;
  size_t done = 0;
  while (done < len) {
    ssize_t n = send(fd, p + done, len - done, 0);
    if (n > 0) {
      done += (size_t)n;
    } else if (n < 0 && (errno == EINTR || errno == EAGAIN)) {
      continue;
    } else {
      return false;
    }
  }
  return true;
}

static bool xfer_read_all(int fd, void *buf, size_t len) {
  uint8_t *p = (uint8_t *)buf;
  size_t done = 0;
  while (done < len) {
    ssize_t n = recv(fd, p + done, len - done, 0);
    if (n > 0) {
      done += (size_t)n;
    } else if (n < 0 && (errno == EINTR || errno == EAGAIN)) {
      continue;
    } else {
      return false; /* n == 0 is a peer FIN mid-exchange — fatal */
    }
  }
  return true;
}

void arts_net_exchange_addresses(void) {
  unsigned n = arts_global_rank_count;
  unsigned self = arts_global_rank_id;

  /* Own address blob (fixed size within a homogeneous provider run). */
  uint8_t own[ARTS_NET_ADDR_MAX];
  unsigned own_len = arts_net_own_address(own, sizeof(own));
  if (own_len == 0 || own_len > ARTS_NET_ADDR_MAX) {
    ARTS_ERROR("arts_net_exchange: own address length %u out of range", own_len);
  }

  /* Rank-indexed contiguous address table (stride = own_len). */
  uint8_t *table = (uint8_t *)arts_calloc(n, own_len);
  memcpy(table + (size_t)self * own_len, own, own_len);

  /* Send our frame to every peer over that peer's port-0 send socket. */
  struct net_addr_frame_s out;
  out.rank = self;
  out.len = own_len;
  memcpy(out.addr, own, own_len);
  size_t frame_bytes = offsetof(struct net_addr_frame_s, addr) + own_len;
  for (unsigned r = 0; r < n; r++) {
    if (r == self) {
      continue;
    }
    int fd = remote_socket_send_list[(size_t)r * ports + 0];
    if (!xfer_write_all(fd, &out, frame_bytes)) {
      ARTS_ERROR("arts_net_exchange: send of address to rank %u failed: %s", r,
                 strerror(errno));
    }
  }

  /* Read one frame from each accepted connection's port-0 socket and demux by
   * the embedded rank (accept order != rank order). */
  unsigned peers = n - 1;
  for (unsigned c = 0; c < peers; c++) {
    int fd = remote_socket_receive_list[(size_t)c * ports + 0];
    struct net_addr_frame_s in;
    if (!xfer_read_all(fd, &in, offsetof(struct net_addr_frame_s, addr))) {
      ARTS_ERROR("arts_net_exchange: address frame header recv failed: %s",
                 strerror(errno));
    }
    if (!arts_net_addr_frame_ok(in.rank, in.len, n, own_len)) {
      ARTS_ERROR("arts_net_exchange: bad frame (rank=%u len=%u; expected "
                 "rank<%u len=%u)",
                 in.rank, in.len, n, own_len);
    }
    if (!xfer_read_all(fd, in.addr, in.len)) {
      ARTS_ERROR("arts_net_exchange: address blob recv from rank %u failed: %s",
                 in.rank, strerror(errno));
    }
    memcpy(table + (size_t)in.rank * own_len, in.addr, own_len);
  }

  /* Insert the whole table in rank order so fi_addr_t == rank. */
  arts_net_av_insert_table(table, own_len, n);
  arts_free(table);
  ARTS_INFO("arts_net: address exchange complete (%u ranks, %u-byte addrs)", n,
            own_len);
}

#endif /* ARTS_TRANSPORT_OFI */
