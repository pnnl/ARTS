/* SPDX-License-Identifier: Apache-2.0
 *
 * dispatcher_default_fatal — the dispatcher `default` (unknown message_type)
 * arm and the self-loopback dispatch_body entry (census 19-dispatcher §1 entry
 * points, §2 per-case table `default` row; dispatcher.c ~631).
 *
 * The dispatcher has two entry points: arts_transport_dispatch_packet (wire RX,
 * runs the optional SEQUENCENUMBERS ordering check first) and
 * arts_transport_dispatch_body (the bare switch), entered DIRECTLY by the
 * self-loopback drain because a self-send carries no per-sender sequence
 * number. This test drives the self-loopback, non-seq entry: it calls
 * arts_transport_dispatch_body with a header-only packet whose message_type is
 * MSG_COUNT — the sentinel ordinal, which is NOT a live case in ANY build's
 * switch — so routing falls into `default`:
 *     ARTS_INFO("Unknown Packet ...") + arts_shutdown() + arts_runtime_stop().
 *
 * The contract under test: an unknown/foreign-ordinal packet does not misparse
 * or crash — it routes to the single defensive default arm that idempotently
 * shuts the runtime down (arts_shutdown is a CAS-gated idempotent stop, and the
 * test's main_edt itself ran via the live runtime, so the runtime was up when
 * the body was invoked).  dispatch_body must RETURN to its caller after the
 * default arm (the switch breaks out of the function), which is what lets this
 * test print its PASS token and end cleanly rather than the process aborting.
 *
 * White-box: like the tests/unit dispatcher/route-table tests, this includes
 * the internal transport headers to call dispatch_body and to fill a wire
 * header. MSG_COUNT is the gapless enum's last sentinel (== live-member count),
 * so it is a stable "unknown" ordinal in every config (the live ordinals are
 * 0..MSG_COUNT-1).
 *
 * Config-agnostic: the default arm is unconditional in every protocol build, so
 * this runs unchanged under VAL (HOME/OWNER), WRF_VAL, EXCL.
 * Single-node (runtime_single): no wire, no peers — the self-loopback body call
 * is purely local.
 */

#include <stdint.h>
#include <stdio.h>

#include "arts.h"

/* White-box: dispatch_body + arts_msg_header_s + arts_fill_packet_header. */
#include "arts/transport/dispatcher.h"
#include "arts/transport/protocol.h"

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== dispatcher_default_fatal ===\n");

  /* Build a header-only packet with an unknown message_type.  MSG_COUNT is the
   * sentinel ordinal == (live member count); no switch case matches it, so the
   * body routes to `default`.  arts_fill_packet_header sets size + type + rank;
   * size == sizeof(header) (header-only, no trailing payload). */
  struct arts_msg_header_s pkt;
  arts_fill_packet_header(&pkt, (uint64_t)sizeof(pkt), (unsigned int)MSG_COUNT);

  /* Self-loopback, non-seq entry: the default arm calls arts_shutdown() +
   * arts_runtime_stop() then breaks out, returning here.  No crash, no
   * misparse, no infinite loop — the runtime is told to stop. */
  arts_transport_dispatch_body(&pkt);

  /* Reached here ⇒ dispatch_body routed the unknown ordinal to the default arm
   * and returned cleanly (the runtime stop was already requested inside it). */
  arts_printf(
      "PASS: dispatcher_default_fatal unknown message_type %u routed to "
      "default (shutdown+runtime_stop)\n",
      (unsigned int)MSG_COUNT);

  /* arts_shutdown was already invoked by the default arm; calling it again is
   * the idempotent CAS-gated no-op (safe, matches the runtime's own re-entry).
   */
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
