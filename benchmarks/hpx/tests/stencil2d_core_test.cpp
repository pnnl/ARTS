#include "stencil2d_core.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>

namespace st = arts_hpx::stencil2d;

namespace {
int failures = 0;
template <typename A, typename E>
void expect_equal(A const& a, E const& e, char const* what)
{
    if (a == e) return;
    std::cerr << "FAIL: " << what << ": expected " << e << ", got " << a << '\n';
    ++failures;
}
void expect_close(double a, double e, char const* what)
{
    if (std::abs(a - e) <= st::EPSILON) return;
    std::cerr << "FAIL: " << what << ": expected " << e << ", got " << a << '\n';
    ++failures;
}
}    // namespace

int main()
{
    std::int64_t nx, ny;
    st::split_cart2d(13824, nx, ny); expect_equal(nx, 108, "13824 nx"); expect_equal(ny, 128, "13824 ny");
    st::split_cart2d(16, nx, ny);    expect_equal(nx, 4, "16 nx");      expect_equal(ny, 4, "16 ny");
    st::split_cart2d(6, nx, ny);     expect_equal(nx, 2, "6 nx");       expect_equal(ny, 3, "6 ny");
    st::split_cart2d(1, nx, ny);     expect_equal(nx, 1, "1 nx");       expect_equal(ny, 1, "1 ny");

    st::grid cal = st::make_grid(31104, 13824, 400);
    st::tile t0 = st::make_tile(cal, 0);
    expect_equal(t0.np_x, 288, "calibrated np_x"); expect_equal(t0.np_y, 243, "calibrated np_y");
    expect_equal(st::expected_tasks(cal), 5543424ull, "tasks calibrated");
    expect_equal(st::expected_strips(cal), 21984424ull, "strips calibrated");
    expect_equal(st::expected_bytes(cal), 93395607552ull, "bytes calibrated");

    st::grid gate = st::make_grid(1000, 16, 10);
    expect_equal(st::expected_tasks(gate), 176ull, "tasks gate");
    expect_equal(st::expected_strips(gate), 528ull, "strips gate");
    expect_equal(st::expected_bytes(gate), 2112000ull, "bytes gate");

    // the place map: 16 tiles (4x4) over 4 places (2x2) -> tile (id_x, id_y) on place (id_x/2) + 2*(id_y/2)
    expect_equal(st::place_of(gate, 0, 2, 2), 0, "place 0");
    expect_equal(st::place_of(gate, 3, 2, 2), 1, "place 3");
    expect_equal(st::place_of(gate, 12, 2, 2), 2, "place 12");
    expect_equal(st::place_of(gate, 15, 2, 2), 3, "place 15");
    expect_equal(st::place_of(gate, 5, 1, 1), 0, "one place");

    // the norm is analytic: uneven tiles included
    expect_close(st::simulate(st::make_grid(40, 6, 3)), 8.0, "norm 40/6/3");
    expect_close(st::simulate(st::make_grid(37, 5, 2)), 6.0, "norm 37/5/2");
    expect_close(st::simulate(st::make_grid(12, 1, 4)), 10.0, "norm single tile");

    if (failures != 0) std::cerr << failures << " failure(s)\n";
    return failures == 0 ? 0 : 1;
}
