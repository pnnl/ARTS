#include "smithwaterman_core.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <tuple>

namespace sw = arts_hpx::smithwaterman;

namespace {
int failures = 0;
template <typename A, typename E>
void expect_equal(A const& a, E const& e, char const* what)
{
    if (a == e) return;
    std::cerr << "FAIL: " << what << ": expected " << e << ", got " << a << '\n';
    ++failures;
}
std::vector<std::int8_t> seq(std::string const& text) { return sw::map_sequence(text); }
}    // namespace

int main(int argc, char** argv)
{
    // tiled == untiled on lengths that do and do not divide the tile
    for (auto const& [a, b, tw, th] : {std::tuple{"ACGTACGTAC", "ACGTTCGAAC", 3, 4},
             std::tuple{"GATTACA", "GCATGCU", 2, 2}, std::tuple{"AAAAAAAA", "TTTT", 8, 4},
             std::tuple{"ACGT\nACGT\n", "TGCA TGCA", 5, 3}})
    {
        auto s1 = seq(a), s2 = seq(b);
        sw::params p = sw::make_params(tw, th, static_cast<std::int32_t>(s1.size()),
            static_cast<std::int32_t>(s2.size()), 0);
        expect_equal(sw::simulate_tiles(p, s1, s2), sw::align_score(s1, s2), a);
    }
    // only the four nucleotides are kept, as the reference reader does
    expect_equal(seq("A C\nG_T\rX").size(), std::size_t{4}, "non-nucleotides dropped");

    // the closed forms at the calibrated and gate sizes
    sw::params cal = sw::make_params(100, 100, 140000, 140400, 86360);
    expect_equal(cal.W, 1400, "W cal"); expect_equal(cal.H, 1404, "H cal");
    expect_equal(sw::expected_strips(cal), 5891193ull, "strips cal");
    expect_equal(sw::expected_bytes(cal), 1579209588ull, "bytes cal");
    sw::params gate = sw::make_params(10, 10, 2000, 2000, 0);
    expect_equal(sw::expected_strips(gate), 119201ull, "strips gate");
    expect_equal(sw::expected_bytes(gate), 3342404ull, "bytes gate");
    expect_equal(sw::effective_width(sw::make_params(100, 100, 140050, 100, 0), 1401), 50, "partial last column");

    // with two file arguments: print the untiled score (used once to pin the gate fixture)
    if (argc == 3)
    {
        std::ifstream f1(argv[1]), f2(argv[2]);
        std::string t1((std::istreambuf_iterator<char>(f1)), {}), t2((std::istreambuf_iterator<char>(f2)), {});
        std::cout << "score " << sw::align_score(seq(t1), seq(t2)) << '\n';
    }
    if (failures != 0) std::cerr << failures << " failure(s)\n";
    return failures == 0 ? 0 : 1;
}
