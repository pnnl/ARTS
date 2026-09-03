#include "p2p_core.hpp"

#include <cstdint>
#include <iostream>

namespace p2p = arts_hpx::p2p;

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
    // geometry: m = 7 over p = 3 -> k = 2,3,2 ; myfirst = 0,2,5 (p2p.c:352-356)
    p2p::params a;
    a.p = 3; a.m = 7; a.n = 5; a.t = 2; a.gf = 1;
    expect_equal(p2p::geometry(a, 0).k, 2u, "k rank 0");
    expect_equal(p2p::geometry(a, 1).k, 3u, "k rank 1");
    expect_equal(p2p::geometry(a, 2).k, 2u, "k rank 2");
    expect_equal(p2p::geometry(a, 0).myfirst, 0u, "myfirst rank 0");
    expect_equal(p2p::geometry(a, 1).myfirst, 2u, "myfirst rank 1");
    expect_equal(p2p::geometry(a, 2).myfirst, 5u, "myfirst rank 2");
    expect_equal(p2p::geometry(a, 0).w, 4u, "w");

    // the pipeline reproduces the closed-form checksum, wrap-around included
    expect_equal(p2p::simulate(a), p2p::expected_checksum(a), "checksum 3x7x5x2");
    p2p::params b; b.p = 8; b.m = 256; b.n = 100; b.t = 10; b.gf = 1;
    expect_equal(p2p::simulate(b), 3894.0, "checksum gate arguments");
    p2p::params c; c.p = 4; c.m = 9; c.n = 7; c.t = 3; c.gf = 2;
    expect_equal(p2p::simulate(c), p2p::expected_checksum(c), "checksum gf=2");
    // gf does not divide n - 1: the last phase of a timestep is short
    p2p::params d; d.p = 3; d.m = 9; d.n = 7; d.t = 2; d.gf = 4;
    expect_equal(p2p::simulate(d), p2p::expected_checksum(d), "checksum short phase");
    expect_equal(p2p::geometry(d, 0).w, 2u, "w short phase");
    expect_equal(p2p::expected_tasks(d), 18ull, "tasks short phase");
    expect_equal(p2p::expected_sends(d), 14ull, "sends short phase");
    expect_equal(p2p::expected_bytes(d), 400ull, "bytes short phase");

    // the block map: 8 ranks over 2 places -> 4 per place; over 3 -> 3,3,2
    expect_equal(p2p::rank_locality(3, 8, 2), 0u, "block 3/2");
    expect_equal(p2p::rank_locality(4, 8, 2), 1u, "block 4/2");
    expect_equal(p2p::rank_locality(7, 8, 3), 2u, "block 7/3");

    // structural closed forms at the gate and calibrated arguments
    expect_equal(p2p::expected_tasks(b), 8712ull, "tasks gate");
    expect_equal(p2p::expected_sends(b), 7633ull, "sends gate");
    expect_equal(p2p::expected_bytes(b), 122048ull, "bytes gate");
    p2p::params cal; cal.p = 6912; cal.m = 1347840; cal.n = 6913; cal.t = 32; cal.gf = 1;
    expect_equal(p2p::expected_tasks(cal), 1576599552ull, "tasks calibrated");
    expect_equal(p2p::expected_sends(cal), 1576371488ull, "sends calibrated");
    expect_equal(p2p::expected_bytes(cal), 25221943552ull, "bytes calibrated");

    if (failures != 0)
        std::cerr << failures << " failure(s)\n";
    return failures == 0 ? 0 : 1;
}
