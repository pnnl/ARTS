#pragma once

#include "rendezvous.hpp"

#include <hpx/hpx.hpp>

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

namespace arts_hpx {

// Locality ids are dense and equal to the MPI rank under the MPI parcelport,
// and a bare locality id resolves arithmetically — no AGAS round trip, which
// a call inside a task must never make.  The one-time construction is also
// where the world size is held to what a rendezvous key can name: the bound
// is enforced, not assumed.
inline std::vector<hpx::id_type> const& localities()
{
    static std::vector<hpx::id_type> const ids = [] {
        std::vector<hpx::id_type> v;
        std::uint32_t const n = hpx::get_initial_num_localities();
        if (n > 256u)
            key_bound_violated(std::to_string(n) +
                " localities exceed the 256 a rendezvous key can name");
        v.reserve(n);
        for (std::uint32_t i = 0; i != n; ++i)
            v.push_back(hpx::naming::get_id_from_locality_id(i));
        return v;
    }();
    return ids;
}

inline std::uint32_t locality_count()
{
    return static_cast<std::uint32_t>(localities().size());
}

// The runtime's no-preference placement carried in the program: one counter
// per locality shared by all its workers, modulo the locality count.  Every
// locality starts at zero, so the k-th blind spawn everywhere aims at the
// same target; the policy being mirrored has the same herd.
inline hpx::id_type const& blind_target()
{
    static std::atomic<std::uint32_t> counter{0};
    return localities()[counter.fetch_add(1, std::memory_order_relaxed) %
        locality_count()];
}

}    // namespace arts_hpx
