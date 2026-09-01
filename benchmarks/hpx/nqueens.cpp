#include "nqueens_core.hpp"

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/modules/program_options.hpp>

#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nq = arts_hpx::nqueens;

// Structural mirror of the event-driven original: a child never returns its
// count through the call that spawned it — it pushes the count to an arrival
// event the parent composed, and the parent's reduction is a continuation on
// those arrivals.  No dependency is ever a suspended thread, and no arrival
// event has a global name: a local edge is a plain future, a remote edge is
// addressed as (locality, slot) into the parent's own arrival table — so a
// pending remote dependency costs a table entry, never a globally registered
// id (whose per-object resolution round-trip parks the resolving thread) and
// never a stack.
void search_nqueens(
    nq::search_state state, std::uint32_t parent_locality, std::uint64_t slot);
HPX_PLAIN_ACTION(search_nqueens, search_nqueens_action)

void deliver_count(std::uint64_t slot, std::uint64_t count);
HPX_PLAIN_ACTION(deliver_count, deliver_count_action)

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

// The arrival events remote children deliver into, addressed by a slot
// number of this locality's own choosing.  The promise is moved out before
// set_value runs: delivery can fire an arbitrarily deep reduction inline,
// which must not happen under the table's lock.
class arrival_table
{
public:
    std::pair<std::uint64_t, hpx::future<std::uint64_t>> open()
    {
        std::lock_guard<hpx::spinlock> lock(mutex_);
        std::uint64_t const slot = next_++;
        return {slot, slots_[slot].get_future()};
    }

    void deliver(std::uint64_t slot, std::uint64_t count)
    {
        hpx::promise<std::uint64_t> arrival;
        {
            std::lock_guard<hpx::spinlock> lock(mutex_);
            auto const it = slots_.find(slot);
            arrival = std::move(it->second);
            slots_.erase(it);
        }
        arrival.set_value(count);
    }

private:
    hpx::spinlock mutex_;
    std::unordered_map<std::uint64_t, hpx::promise<std::uint64_t>> slots_;
    std::uint64_t next_ = 0;
};

arrival_table& arrivals()
{
    static arrival_table table;
    return table;
}

}    // namespace

void deliver_count(std::uint64_t slot, std::uint64_t count)
{
    arrivals().deliver(slot, count);
}

hpx::future<std::uint64_t> search_tree(nq::search_state state)
{
    if (nq::should_search_sequential(state))
        return hpx::make_ready_future(nq::count_solutions_sequential(state));
    if (state.cols == state.all)
        return hpx::make_ready_future(std::uint64_t{1});

    std::uint32_t available =
        state.all & ~(state.ldiag | state.cols | state.rdiag);
    if (available == 0)
        return hpx::make_ready_future(std::uint64_t{0});

    hpx::id_type const here = hpx::find_here();
    std::vector<hpx::id_type> localities;
    if (nq::count_set_bits(state.cols) + 1 < state.scatter_levels)
        localities = hpx::find_all_localities();

    std::vector<hpx::future<std::uint64_t>> pending;
    pending.reserve(nq::count_set_bits(available));

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
        if (target == here)
        {
            // A spawned task's future is the local form of the arrival
            // event (the constructor unwraps the task's own future).
            pending.push_back(hpx::future<std::uint64_t>(
                hpx::async(search_tree, child)));
        }
        else
        {
            // The remote form: an arrival slot of this locality's own
            // numbering.  The dispatch carries no reply channel, so the
            // callee never keeps a thread waiting on this subtree, and
            // delivery needs no globally registered name to resolve.
            auto [slot, arrival] = arrivals().open();
            pending.push_back(std::move(arrival));
            hpx::post<search_nqueens_action>(
                target, child, hpx::get_locality_id(), slot);
        }
    }

    return hpx::when_all(std::move(pending))
        .then(hpx::launch::sync,
            [](hpx::future<std::vector<hpx::future<std::uint64_t>>> ready) {
                std::uint64_t solutions = 0;
                for (auto& child : ready.get())
                    solutions += child.get();
                return solutions;
            });
}

void search_nqueens(
    nq::search_state state, std::uint32_t parent_locality, std::uint64_t slot)
{
    search_tree(std::move(state))
        .then(hpx::launch::sync,
            [parent_locality, slot](hpx::future<std::uint64_t> total) {
                hpx::post<deliver_count_action>(
                    hpx::naming::get_id_from_locality_id(parent_locality),
                    slot, total.get());
            });
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
        auto [slot, result] = arrivals().open();
        hpx::post<search_nqueens_action>(
            target, root, hpx::get_locality_id(), slot);
        solutions = result.get();
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
    // Sends go out immediately rather than through the cached-connection
    // window: a fire-and-forget delivery that waits for a connection slot
    // parks its thread, and a completion burst parks thousands at once.
    params.cfg = {"hpx.use_process_mask!=1", "hpx.os_threads!=cores",
        "hpx.parcel.mpi.sendimm!=1"};
    return hpx::init(argc, argv, params);
}
