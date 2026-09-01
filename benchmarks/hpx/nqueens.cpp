#include "nqueens_core.hpp"

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/modules/program_options.hpp>

#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>

namespace nq = arts_hpx::nqueens;

std::uint64_t search_nqueens(nq::search_state state);
HPX_PLAIN_ACTION(search_nqueens, search_nqueens_action)

namespace {

hpx::id_type target_for_state(nq::search_state const& state,
    std::vector<hpx::id_type> const& localities, hpx::id_type const& here)
{
    if (nq::count_set_bits(state.cols) < state.scatter_levels)
    {
        return localities[nq::mix_key(state.cols) % localities.size()];
    }
    return here;
}

bool parse_u32(std::string const& text, std::uint32_t& value)
{
    if (text.empty())
        return false;

    char const* const begin = text.data();
    char const* const end = begin + text.size();
    auto const result = std::from_chars(begin, end, value);
    return result.ec == std::errc{} && result.ptr == end;
}

int fail_with_usage(char const* message)
{
    std::cerr << message << '\n'
              << "Usage: nqueens_hpx size cutoff [rounds [scatter-levels]]\n";
    hpx::finalize();
    return 1;
}

}    // namespace

std::uint64_t search_nqueens(nq::search_state state)
{
    if (nq::should_search_sequential(state))
        return nq::count_solutions_sequential(state);
    if (state.cols == state.all)
        return 1;

    std::uint32_t available =
        state.all & ~(state.ldiag | state.cols | state.rdiag);
    if (available == 0)
        return 0;

    hpx::id_type const here = hpx::find_here();
    std::vector<hpx::id_type> localities;
    if (nq::count_set_bits(state.cols) + 1 < state.scatter_levels)
        localities = hpx::find_all_localities();

    std::vector<hpx::future<std::uint64_t>> children;
    children.reserve(nq::count_set_bits(available));

    while (available != 0)
    {
        std::uint32_t const position = available & -available;
        available -= position;

        nq::search_state const child{state.all,
            (state.ldiag | position) << 1, state.cols | position,
            (state.rdiag | position) >> 1, state.max_set,
            state.scatter_levels};

        hpx::id_type const target = localities.empty() ? here :
            target_for_state(child, localities, here);
        children.push_back(
            hpx::async<search_nqueens_action>(target, child));
    }

    std::uint64_t solutions = 0;
    for (auto& child : children)
        solutions += child.get();
    return solutions;
}

int hpx_main(hpx::program_options::variables_map& variables)
{
    auto const started = std::chrono::steady_clock::now();

    if (!variables.count("hpx:positional"))
        return fail_with_usage("missing N-Queens arguments");

    std::vector<std::string> const arguments =
        variables["hpx:positional"].as<std::vector<std::string>>();
    if (arguments.size() < 2 || arguments.size() > 4)
        return fail_with_usage("invalid number of N-Queens arguments");

    std::uint32_t board_size = 0;
    std::uint32_t cutoff = 0;
    std::uint32_t rounds = 1;
    std::uint32_t scatter_levels = 3;
    if (!parse_u32(arguments[0], board_size) ||
        !parse_u32(arguments[1], cutoff) ||
        (arguments.size() >= 3 && !parse_u32(arguments[2], rounds)) ||
        (arguments.size() == 4 &&
            !parse_u32(arguments[3], scatter_levels)))
    {
        return fail_with_usage("N-Queens arguments must be unsigned integers");
    }
    if (board_size == 0 || board_size >= 31 || cutoff >= board_size)
        return fail_with_usage("size must be in [1, 30] and cutoff < size");
    if (rounds == 0)
        rounds = 1;

    nq::search_state const root =
        nq::initial_state(board_size, cutoff, scatter_levels);
    std::vector<hpx::id_type> const localities = hpx::find_all_localities();

    std::uint64_t solutions = 0;
    for (std::uint32_t round = 0; round != rounds; ++round)
    {
        hpx::id_type const target =
            target_for_state(root, localities, hpx::find_here());
        solutions = hpx::async<search_nqueens_action>(target, root).get();
    }

    std::cout << board_size << "-queens; " << board_size << 'x' << board_size
              << "; sols: " << solutions << '\n';

    if (std::getenv("ARTS_E2E_MARKER") != nullptr)
    {
        auto const elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started);
        std::cerr << "[E2E] " << elapsed.count() << '\n';
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    hpx::init_params params;
    // An externally installed affinity mask is authoritative: size and place
    // the worker set from it instead of the machine topology, defaulting to
    // one worker per physical core inside the mask so SMT siblings never
    // carry a second worker.  An explicit --hpx:threads still overrides.
    params.cfg = {"hpx.use_process_mask!=1", "hpx.os_threads!=cores"};
    return hpx::init(argc, argv, params);
}
