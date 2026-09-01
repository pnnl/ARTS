#include "nqueens_core.hpp"

#include <cstdint>
#include <iostream>
#include <utility>

namespace nq = arts_hpx::nqueens;

namespace {

int failures = 0;

template <typename Actual, typename Expected>
void expect_equal(
    Actual const& actual, Expected const& expected, char const* description)
{
    if (actual == expected)
        return;

    std::cerr << "FAIL: " << description << ": expected " << expected
              << ", got " << actual << '\n';
    ++failures;
}

void test_known_solution_counts()
{
    constexpr std::pair<std::uint32_t, std::uint64_t> cases[] = {
        {1, 1}, {4, 2}, {5, 10}, {6, 4}, {8, 92}};

    for (auto const& [board_size, expected] : cases)
    {
        nq::search_state const root =
            nq::initial_state(board_size, board_size - 1, 3);
        expect_equal(nq::count_solutions_sequential(root), expected,
            "known N-Queens solution count");
    }
}

void test_cutoff_uses_strict_boundary()
{
    nq::search_state state = nq::initial_state(8, 3, 3);

    state.cols = 0b0001'1111;
    expect_equal(nq::should_search_sequential(state), false,
        "cutoff does not trigger at max_set queens");

    state.cols = 0b0011'1111;
    expect_equal(nq::should_search_sequential(state), true,
        "cutoff triggers above max_set queens");
}

}    // namespace

int main()
{
    test_known_solution_counts();
    test_cutoff_uses_strict_boundary();

    if (failures != 0)
        return 1;

    std::cout << "nqueens_core_test PASS\n";
    return 0;
}
