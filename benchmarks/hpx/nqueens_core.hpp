#pragma once

#include <cstdint>

namespace arts_hpx::nqueens {

struct search_state
{
    std::uint32_t all = 0;
    std::uint32_t ldiag = 0;
    std::uint32_t cols = 0;
    std::uint32_t rdiag = 0;
    std::uint32_t max_set = 0;
    std::uint32_t scatter_levels = 0;

    template <typename Archive>
    void serialize(Archive& archive, unsigned)
    {
        archive & all & ldiag & cols & rdiag & max_set & scatter_levels;
    }
};

inline std::uint32_t count_set_bits(std::uint32_t value)
{
    std::uint32_t count = 0;
    while (value != 0)
    {
        value &= value - 1;
        ++count;
    }
    return count;
}

inline search_state initial_state(std::uint32_t board_size,
    std::uint32_t cutoff, std::uint32_t scatter_levels)
{
    return search_state{(std::uint32_t{1} << board_size) - 1, 0, 0, 0,
        board_size - cutoff, scatter_levels};
}

inline bool should_search_sequential(search_state const& state)
{
    return count_set_bits(state.cols) > state.max_set;
}

inline std::uint64_t count_solutions_sequential(search_state const& state)
{
    if (state.cols == state.all)
        return 1;

    std::uint64_t solutions = 0;
    std::uint32_t available =
        state.all & ~(state.ldiag | state.cols | state.rdiag);

    while (available != 0)
    {
        std::uint32_t const position = available & -available;
        available -= position;

        search_state const child{state.all,
            (state.ldiag | position) << 1, state.cols | position,
            (state.rdiag | position) >> 1, state.max_set,
            state.scatter_levels};
        solutions += count_solutions_sequential(child);
    }

    return solutions;
}

inline std::uint64_t mix_key(std::uint64_t value)
{
    value ^= value >> 33;
    value *= 0xff51afd7ed558ccdULL;
    value ^= value >> 33;
    value *= 0xc4ceb9fe1a85ec53ULL;
    value ^= value >> 33;
    return value;
}

}    // namespace arts_hpx::nqueens
