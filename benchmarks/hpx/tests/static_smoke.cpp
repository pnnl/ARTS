// Two localities, the exact static build: every runtime facility the ports
// depend on, exercised once — collectives (registered components), a plain
// action delivering into a receive buffer on the other locality, the
// parcelport counter registry, and the hpx_main shape in which no locality
// leaves before the global completion edge.
#include "common/e2e.hpp"
#include "common/hub.hpp"
#include "common/localities.hpp"
#include "common/rendezvous.hpp"
#include "common/runtime_defaults.hpp"

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/performance_counters.hpp>
#include <hpx/modules/collectives.hpp>
#include <hpx/runtime_local/config_entry.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

ARTS_HPX_DEFINE_HUB(std::uint64_t, smoke)

namespace {
int failures = 0;
void check(bool ok, char const* what)
{
    if (!ok) { std::fprintf(stderr, "SMOKE FAIL: %s\n", what); ++failures; }
}
}

int hpx_main(hpx::program_options::variables_map&)
{
    using namespace arts_hpx;
    std::uint32_t const me = hpx::get_locality_id();
    std::uint32_t const n = locality_count();
    check(n == 2, "two localities");
    check(localities()[me] == hpx::find_here(), "locality index equals id");

    bool const structural = struct_enabled();
    check(structural, "struct marker broadcast reached this locality");
    hpx::distributed::barrier::synchronize();

    // A delivery into the OTHER locality's hub, keyed by the receiver's own
    // numbering; the receiver waits in hpx_main (allowed), never in a task.
    std::uint64_t const key = pack_key(me, next_sequence(), 0);
    hpx::future<std::uint64_t> arrival = hub<std::uint64_t>::instance().receive(key);
    std::uint32_t const peer = (me + 1) % n;
    std::uint64_t const peer_key = pack_key(peer, 0, 0);   // peer's first sequence
    hpx::post<smoke_deliver_action>(localities()[peer], peer_key, std::uint64_t{100 + me});
    check(arrival.get() == 100 + peer, "delivery arrived with the peer's value");

    auto comm = hpx::collectives::create_communicator("/arts/smoke/sum",
        hpx::collectives::num_sites_arg(n), hpx::collectives::this_site_arg(me));
    std::uint64_t const sum = hpx::collectives::all_reduce(comm, std::uint64_t{me + 1},
        std::plus<std::uint64_t>()).get();
    check(sum == 3, "all_reduce over two sites");
    std::vector<std::uint64_t> const gathered =
        hpx::collectives::all_gather(comm, std::uint64_t{me}).get();
    check(gathered.size() == 2 && gathered[0] == 0 && gathered[1] == 1, "all_gather");
    int const flag = me == 0 ? hpx::collectives::broadcast_to(comm, 7).get()
                             : hpx::collectives::broadcast_from<int>(comm).get();
    check(flag == 7, "broadcast");

    hpx::performance_counters::performance_counter sent("/parcels/count/mpi/sent");
    check(sent.get_value<std::int64_t>(hpx::launch::sync) > 0, "parcelport counter readable and non-zero");

    // The scheduling policy the run actually got.  The default the ports
    // install is a non-forcing configuration entry, so the command line must
    // still win over it; which of the two answers is expected travels in the
    // environment, because --hpx:queuing is consumed by the runtime and never
    // reaches the program.  A policy that failed to compile into the build
    // does not silently fall back here: the runtime reports it as not
    // configured and the process then dies, so this also stands as the
    // check that the LIFO backends exist.
    char const* const want = std::getenv("ARTS_HPX_EXPECT_SCHEDULER");
    std::string const expected = want != nullptr ? want : "local-priority-lifo";
    std::string const scheduler = hpx::get_config_entry("hpx.scheduler", "");
    std::string const verdict = "scheduling policy is '" + scheduler +
        "', expected '" + expected + "'";
    check(scheduler == expected, verdict.c_str());

    hpx::distributed::barrier::synchronize();          // the completion edge
    print_geometry();
    if (structural)
    {
        print_struct({{"tasks", 1}, {"edges", 1}});
        print_parcels();
    }
    hub<std::uint64_t>::drain_and_report();
    if (me == 0)
        std::printf("SMOKE %s\n", failures == 0 ? "PASS" : "FAIL");
    // The verdict travels out as an exit status, on every locality: a
    // printed line alone leaves a crash after the print indistinguishable
    // from a pass, and only a rank's status reaches the launcher.
    int const rc = hpx::finalize();
    return failures != 0 ? 1 : rc;
}

int main(int argc, char* argv[])
{
    hpx::init_params params;
    params.cfg = arts_hpx::runtime_defaults();
    return hpx::init(argc, argv, params);
}
