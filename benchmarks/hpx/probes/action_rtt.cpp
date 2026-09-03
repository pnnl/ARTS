// One message, timed: the round trip of a plain action to another locality
// and its result back.  It is the unit a message count is denominated in — a
// count of messages only means something next to the cost of one — and the
// same quantity another runtime's own one-message probe reports, so the two
// are comparable directly.
#include "common/localities.hpp"
#include "common/runtime_defaults.hpp"

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/modules/collectives.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>

std::uint64_t ping(std::uint64_t x)
{
    return x + 1;
}
HPX_PLAIN_ACTION(ping, ping_action)

int hpx_main()
{
    if (hpx::get_locality_id() == 0)
    {
        // With one locality the peer is this locality: the same code path
        // measured without the wire, which is the control the ratio needs.
        hpx::id_type const& other =
            arts_hpx::localities()[arts_hpx::locality_count() > 1 ? 1 : 0];
        constexpr std::uint64_t warm = 1000, n = 100000;
        std::uint64_t v = 0;
        for (std::uint64_t i = 0; i != warm; ++i)
            v = hpx::async<ping_action>(other, v).get();
        auto const t0 = std::chrono::steady_clock::now();
        for (std::uint64_t i = 0; i != n; ++i)
            v = hpx::async<ping_action>(other, v).get();
        double const us = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - t0).count() / n;
        std::printf("action round trip: %.3f us over %llu iterations (%llu)\n",
            us, (unsigned long long) n, (unsigned long long) v);
    }
    hpx::distributed::barrier::synchronize();
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    hpx::init_params params;
    params.cfg = arts_hpx::runtime_defaults();
    return hpx::init(argc, argv, params);
}
