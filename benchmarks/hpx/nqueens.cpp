#include "nqueens_core.hpp"

#include "common/arguments.hpp"
#include "common/e2e.hpp"
#include "common/hub.hpp"
#include "common/localities.hpp"
#include "common/rendezvous.hpp"
#include "common/runtime_defaults.hpp"

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/modules/program_options.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace nq = arts_hpx::nqueens;
using arts_hpx::hub;
using arts_hpx::localities;
using arts_hpx::pack_key;

// Structural mirror of the event-driven original: a child never returns its
// count through the call that spawned it — it delivers the count to a keyed
// slot its parent named, and the parent's reduction fires once every arrival
// is in.  No dependency is ever a suspended thread, and the reduction is a
// task in its own right, not a continuation riding the thread of whichever
// child delivered last.
ARTS_HPX_DEFINE_HUB(std::uint64_t, nq)

namespace {

struct counters_t
{
    std::atomic<std::uint64_t> tasks{0}, edges{0}, parcels{0}, reducers{0};
};
counters_t& counters()
{
    static counters_t c;
    return c;
}

// The reply address of a subtree: which locality's hub, which key.
struct reply_t
{
    std::uint32_t locality;
    std::uint64_t key;

    template <typename Archive>
    void serialize(Archive& ar, unsigned) { ar & locality & key; }
};

void deliver(reply_t const& to, std::uint64_t count)
{
    if (arts_hpx::struct_enabled())
        counters().edges.fetch_add(1, std::memory_order_relaxed);
    if (to.locality == hpx::get_locality_id())
    {
        hub<std::uint64_t>::instance().deliver(to.key, count);
        return;
    }
    if (arts_hpx::struct_enabled())
        counters().parcels.fetch_add(1, std::memory_order_relaxed);
    hpx::post<nq_deliver_action>(localities()[to.locality], to.key, count);
}

#if defined(HPX_APP_HINTED_PLACEMENT)
// The application's own key: subtrees above the scatter depth spread by a
// mixed column mask, deeper ones stay where they are.
hpx::id_type const& target_for(nq::search_state const& child)
{
    if (nq::count_set_bits(child.cols) < child.scatter_levels)
        return localities()[nq::mix_key(child.cols) % localities().size()];
    // This locality's own id, as a reference into the table: the runtime's
    // accessor for it returns by value, and a reference return may not bind
    // a temporary.
    return localities()[hpx::get_locality_id()];
}
#else
// The runtime's no-preference policy, carried in the program.
hpx::id_type const& target_for(nq::search_state const&)
{
    return arts_hpx::blind_target();
}
#endif

}    // namespace

void search_nqueens(nq::search_state state, reply_t reply);
HPX_PLAIN_ACTION(search_nqueens, search_action)

// The reducer: a spawn of its own under either placement, collecting its
// children's counts through its own hub and forwarding the sum to the reply
// address its creator gave it.  An inline continuation instead would run on
// the delivering child's thread and saturate the scheduler's thread-object
// ceiling; a spawn is demand-limited.
void reduce_nqueens(std::uint32_t creator, std::uint64_t sequence,
    std::uint32_t children, reply_t reply)
{
    if (arts_hpx::struct_enabled())
        counters().reducers.fetch_add(1, std::memory_order_relaxed);
    std::vector<hpx::future<std::uint64_t>> pending;
    pending.reserve(children);
    for (std::uint32_t i = 0; i != children; ++i)
        pending.push_back(hub<std::uint64_t>::instance().receive(
            pack_key(creator, sequence, i)));
    hpx::when_all(std::move(pending))
        .then(hpx::launch::sync,
            [reply](hpx::future<std::vector<hpx::future<std::uint64_t>>> ready) {
                std::uint64_t sum = 0;
                for (auto& f : ready.get())
                    sum += f.get();
                deliver(reply, sum);
            });
}
HPX_PLAIN_ACTION(reduce_nqueens, reduce_action)

void search_nqueens(nq::search_state state, reply_t reply)
{
    if (arts_hpx::struct_enabled())
        counters().tasks.fetch_add(1, std::memory_order_relaxed);
    if (nq::should_search_sequential(state))
    {
        deliver(reply, nq::count_solutions_sequential(state));
        return;
    }
    if (state.cols == state.all)
    {
        deliver(reply, 1);
        return;
    }
    std::uint32_t available = state.all & ~(state.ldiag | state.cols | state.rdiag);
    if (available == 0)
    {
        deliver(reply, 0);
        return;
    }

    std::uint32_t const children = nq::count_set_bits(available);
    std::uint32_t const here = hpx::get_locality_id();
    std::uint64_t const sequence = arts_hpx::next_sequence();

    // The reduction is a spawn like any other, placed by the tier's rule —
    // on the creating locality under the hinted layer, blindly otherwise —
    // and its inputs go to it.
#if defined(HPX_APP_HINTED_PLACEMENT)
    hpx::id_type const& reducer = localities()[here];
#else
    hpx::id_type const& reducer = target_for(state);
#endif
    std::uint32_t const reducer_locality =
        hpx::naming::get_locality_id_from_id(reducer);
    hpx::post<reduce_action>(reducer, here, sequence, children, reply);

    std::uint32_t index = 0;
    while (available != 0)
    {
        std::uint32_t const position = available & -available;
        available -= position;
        nq::search_state const child{state.all,
            (state.ldiag | position) << 1, state.cols | position,
            (state.rdiag | position) >> 1, state.max_set, state.scatter_levels};
        reply_t const child_reply{reducer_locality, pack_key(here, sequence, index++)};
        hpx::post<search_action>(target_for(child), child, child_reply);
    }
}

namespace {

constexpr char const* usage = "size cutoff [rounds [scatter-levels]]";

}    // namespace

int hpx_main(hpx::program_options::variables_map& variables)
{
    if (!variables.count("hpx:positional"))
        return arts_hpx::fail_with_usage("missing N-Queens arguments", usage);
    std::vector<std::string> const arguments =
        variables["hpx:positional"].as<std::vector<std::string>>();
    if (arguments.size() < 2 || arguments.size() > 4)
        return arts_hpx::fail_with_usage(
            "invalid number of N-Queens arguments", usage);

    std::uint32_t board_size = 0, cutoff = 0, rounds = 1, scatter_levels = 3;
    if (!arts_hpx::parse_int(arguments[0], board_size) ||
        !arts_hpx::parse_int(arguments[1], cutoff) ||
        (arguments.size() >= 3 && !arts_hpx::parse_int(arguments[2], rounds)) ||
        (arguments.size() == 4 && !arts_hpx::parse_int(arguments[3], scatter_levels)))
        return arts_hpx::fail_with_usage(
            "N-Queens arguments must be unsigned integers", usage);
    if (board_size == 0 || board_size >= 31 || cutoff >= board_size)
        return arts_hpx::fail_with_usage("size must be in [1, 30] and cutoff < size", usage);
    if (rounds == 0)
        rounds = 1;

    // Setup on every locality, then the barrier the start stamp follows.
    (void) localities();
    bool const structural = arts_hpx::struct_enabled();
    arts_hpx::print_geometry();
    arts_hpx::run_clock clock{std::chrono::steady_clock::now()};
    hpx::distributed::barrier::synchronize();

    std::uint64_t solutions = 0;
    if (hpx::get_locality_id() == 0)
    {
        nq::search_state const root =
            nq::initial_state(board_size, cutoff, scatter_levels);
        for (std::uint32_t round = 0; round != rounds; ++round)
        {
            std::uint64_t const key = pack_key(0, arts_hpx::next_sequence(), 0);
            hpx::future<std::uint64_t> result = hub<std::uint64_t>::instance().receive(key);
            hpx::post<search_action>(target_for(root), root, reply_t{0, key});
            solutions = result.get();    // hpx_main may wait; no task does
        }
    }

    // The global completion edge: locality 0 arrives with the root count in
    // hand, every other locality arrives when it has nothing left to do —
    // which is now, since all of its work was posted by others and the root
    // future transitively covers it.
    hpx::distributed::barrier::synchronize();

    if (hpx::get_locality_id() == 0)
    {
        arts_hpx::print_e2e(clock);
        arts_hpx::write_stdout_line(std::to_string(board_size) + "-queens; " +
            std::to_string(board_size) + "x" + std::to_string(board_size) +
            "; sols: " + std::to_string(solutions) + "\n");
    }
    if (structural)
    {
        auto const& c = counters();
        arts_hpx::print_struct({{"tasks", c.tasks.load()}, {"edges", c.edges.load()},
            {"parcels", c.parcels.load()}, {"bytes", c.parcels.load() * 8},
            {"reducers", c.reducers.load()}});
        arts_hpx::print_parcels();
    }
    hub<std::uint64_t>::drain_and_report();
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    hpx::init_params params;
    params.cfg = arts_hpx::runtime_defaults();
    return hpx::init(argc, argv, params);
}
