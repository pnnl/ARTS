#include "tempest_core.hpp"

#include <cstdint>
#include <iostream>

namespace tp = arts_hpx::tempest;

namespace {
int failures = 0;
template <typename A, typename E>
void expect_equal(A const& a, E const& e, char const* what)
{
    if (a == e) return;
    std::cerr << "FAIL: " << what << ": expected " << e << ", got " << a << '\n';
    ++failures;
}
}    // namespace

int main()
{
    // symmetry: if Q is P's neighbour in some direction, P is Q's in exactly one
    for (std::uint64_t k : {2ull, 3ull, 4ull, 5ull, 8ull})
    {
        std::uint64_t corners = 0;
        for (std::uint64_t P = 0; P < 6 * k * k; ++P)
        {
            std::uint32_t n = 0;
            for (std::uint32_t d = 0; d < 8; ++d)
            {
                std::int64_t const Q = tp::find_neighbor_patch(P, k, d);
                if (Q < 0) continue;
                ++n;
                expect_equal(Q < static_cast<std::int64_t>(6 * k * k), true, "neighbour on the board");
                std::uint32_t const rel = tp::neighbor_relation(P, static_cast<std::uint64_t>(Q), k);
                expect_equal(rel < 8u, true, "relation exists");
                expect_equal(tp::find_neighbor_patch(static_cast<std::uint64_t>(Q), k, rel), static_cast<std::int64_t>(P), "symmetry");
            }
            if (n == 7) ++corners;
            expect_equal(n == 7 || n == 8, true, "seven or eight neighbours");
        }
        expect_equal(corners, 24ull, "24 corner patches");
        expect_equal(tp::edge_count(k), 48 * k * k - 24, "edge count");
    }
    expect_equal(tp::find_neighbor_patch(0, 48, tp::SE), 11521ll, "the calibrated pin: patch 0's SE neighbour");
    std::cout << "k 4: SE of patch 0 = " << tp::find_neighbor_patch(0, 4, tp::SE) << '\n';   // the gate answer

    // the hinted map: bands below six places, a cut above
    expect_equal(tp::home_rank(0, 4, 2), 0ull, "band first");
    expect_equal(tp::home_rank(95, 4, 2), 1ull, "band last");
    expect_equal(tp::home_rank(0, 48, 8) < 8, true, "cut in range");
    {
        std::uint64_t seen[8] = {};
        for (std::uint64_t P = 0; P < 6 * 48 * 48; ++P) ++seen[tp::home_rank(P, 48, 8)];
        for (std::uint64_t r = 0; r < 8; ++r) expect_equal(seen[r] > 0, true, "every place gets patches");
    }
    tp::patch_state const s = tp::make_state(0, 4, 20);
    expect_equal(s.nbr[tp::N], 1ll, "patch 0 north");
    expect_equal(s.nbr[tp::E], 4ll, "patch 0 east");
    expect_equal(s.nbr[tp::SW], -1ll, "patch 0 south-west off the board");

    if (failures != 0) std::cerr << failures << " failure(s)\n";
    return failures == 0 ? 0 : 1;
}
