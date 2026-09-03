#include "triangle_core.hpp"

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
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace tri = arts_hpx::triangle;
using arts_hpx::hub;
using arts_hpx::localities;
using arts_hpx::pack_key;

// A subtree's count goes to the keyed slot its parent's summer named; the
// summer is a spawn of its own under either placement.
ARTS_HPX_DEFINE_HUB(std::uint64_t, tri)

namespace {

struct counters_t
{
    std::atomic<std::uint64_t> nodes{0}, summers{0}, boards{0}, board_bytes{0}, parcels{0}, bytes{0};
};
counters_t& counters()
{
    static counters_t c;
    return c;
}

struct run_t
{
    std::uint64_t rows = 0;      // 0 = the author's board (5 rows)
    std::uint64_t erows = tri::ROWS_DEFAULT;
    std::uint64_t depth = 0, rounds = 1;
};
run_t& run()
{
    static run_t r;
    return r;
}
tri::moves_t const& moves()
{
    static tri::moves_t const m = tri::gen_moves(run().erows);
    return m;
}

struct reply_t
{
    std::uint32_t locality;
    std::uint64_t key;
    template <typename Archive>
    void serialize(Archive& ar, unsigned) { ar & locality & key; }
};

struct node_t
{
    std::uint64_t nummoves;
    std::int64_t oldmove;    // -1 at the root
    tri::board_t board;      // the parent's post-move board
    reply_t reply;
    template <typename Archive>
    void serialize(Archive& ar, unsigned) { ar & nummoves & oldmove & board & reply; }
};

void deliver(reply_t const& to, std::uint64_t count)
{
    if (to.locality == hpx::get_locality_id())
    {
        hub<std::uint64_t>::instance().deliver(to.key, count);
        return;
    }
    if (arts_hpx::struct_enabled())
    {
        counters().parcels.fetch_add(1, std::memory_order_relaxed);
        counters().bytes.fetch_add(8, std::memory_order_relaxed);
    }
    hpx::post<tri_deliver_action>(localities()[to.locality], to.key, count);
}

#if defined(HPX_APP_HINTED_PLACEMENT)
hpx::id_type const& child_target(std::uint64_t level, std::uint64_t child_bits)
{
    if (level <= tri::SCATTER_LEVELS)
        return localities()[tri::mix_key(child_bits) % localities().size()];
    return localities()[hpx::get_locality_id()];
}
#else
hpx::id_type const& child_target(std::uint64_t, std::uint64_t)
{
    return arts_hpx::blind_target();
}
#endif

}    // namespace

void search_triangle(node_t node);
HPX_PLAIN_ACTION(search_triangle, search_action)

void reduce_triangle(std::uint32_t creator, std::uint64_t sequence, std::uint32_t children, reply_t reply)
{
    if (arts_hpx::struct_enabled())
        counters().summers.fetch_add(1, std::memory_order_relaxed);
    std::vector<hpx::future<std::uint64_t>> pending;
    pending.reserve(children);
    for (std::uint32_t i = 0; i != children; ++i)
        pending.push_back(hub<std::uint64_t>::instance().receive(pack_key(creator, sequence, i)));
    hpx::when_all(std::move(pending))
        .then(hpx::launch::sync, [reply](hpx::future<std::vector<hpx::future<std::uint64_t>>> ready) {
            std::uint64_t sum = 0;
            for (auto& f : ready.get())
                sum += f.get();
            deliver(reply, sum);
        });
}
HPX_PLAIN_ACTION(reduce_triangle, reduce_action)

void search_triangle(node_t node)
{
    if (arts_hpx::struct_enabled())
        counters().nodes.fetch_add(1, std::memory_order_relaxed);
    tri::moves_t const& m = moves();
    std::uint64_t nummoves = node.nummoves;
    tri::board_t board = std::move(node.board);
    if (node.oldmove != -1)
    {
        ++nummoves;
        tri::apply(board, m, static_cast<std::uint64_t>(node.oldmove));
    }
    if (nummoves == run().depth)
    {
        deliver(node.reply, 1);
        return;
    }
    std::uint64_t const nlegal = tri::legal_count(board, m);
    if (nlegal == 0)
    {
        deliver(node.reply, 0);
        return;
    }

    std::uint32_t const here = hpx::get_locality_id();
    std::uint64_t const sequence = arts_hpx::next_sequence();
#if defined(HPX_APP_HINTED_PLACEMENT)
    hpx::id_type const& summer = localities()[here];
#else
    hpx::id_type const& summer = arts_hpx::blind_target();
#endif
    std::uint32_t const summer_locality = hpx::naming::get_locality_id_from_id(summer);
    if (summer_locality != here && arts_hpx::struct_enabled())
        counters().parcels.fetch_add(1, std::memory_order_relaxed);
    hpx::post<reduce_action>(summer, here, sequence, static_cast<std::uint32_t>(nlegal), node.reply);

#if defined(HPX_APP_HINTED_PLACEMENT)
    std::uint64_t const bits = tri::bits_of(board);
#endif
    std::uint64_t const holes = board.size();
    std::uint32_t index = 0;
    for (std::uint64_t i = 0; i < m.size() / 3; ++i)
    {
        if (!tri::is_legal(board, m, i))
            continue;
#if defined(HPX_APP_HINTED_PLACEMENT)
        std::uint64_t const child_bits =
            (bits & ~(1ull << m[3 * i]) & ~(1ull << m[3 * i + 1])) | (1ull << m[3 * i + 2]);
        hpx::id_type const& target = child_target(nummoves + 1, child_bits);
#else
        hpx::id_type const& target = child_target(nummoves + 1, 0);
#endif
        if (arts_hpx::struct_enabled())
        {
            counters().boards.fetch_add(1, std::memory_order_relaxed);
            counters().board_bytes.fetch_add(8 * holes, std::memory_order_relaxed);
            if (hpx::naming::get_locality_id_from_id(target) != here)
            {
                counters().parcels.fetch_add(1, std::memory_order_relaxed);
                counters().bytes.fetch_add(8 * holes, std::memory_order_relaxed);
            }
        }
        hpx::post<search_action>(target,
            node_t{nummoves, static_cast<std::int64_t>(i), board,
                reply_t{summer_locality, pack_key(here, sequence, index++)}});
    }
}

int hpx_main(hpx::program_options::variables_map& variables)
{
    std::vector<std::string> arguments;
    if (variables.count("hpx:positional"))
        arguments = variables["hpx:positional"].as<std::vector<std::string>>();
    run_t& r = run();
    std::uint64_t value = 0;
    if (arguments.size() > 2 && arts_hpx::parse_int(arguments[2], value) && value >= 3 && value <= tri::ROWS_MAX)
        r.rows = value;
    r.erows = r.rows ? r.rows : tri::ROWS_DEFAULT;
    std::uint64_t const maxdepth = tri::holes(r.erows) - 2;
    r.depth = maxdepth;
    if (arguments.size() > 0 && arts_hpx::parse_int(arguments[0], value))
        r.depth = value;
    if (r.depth < 1 || r.depth > maxdepth)
        r.depth = maxdepth;
    if (arguments.size() > 1 && arts_hpx::parse_int(arguments[1], value) && value >= 1)
        r.rounds = value;
    if (r.erows == tri::ROWS_DEFAULT && !tri::generator_matches_author())
    {
        std::cerr << "triangle: move generator disagrees with the author's table\n";
        std::abort();
    }
    (void) moves();

    (void) localities();
    bool const structural = arts_hpx::struct_enabled();
    arts_hpx::print_geometry();
    if (hpx::get_locality_id() == 0)
    {
        char line[120];
        if (r.rows == 0)
            std::snprintf(line, sizeof line, "triangle puzzle depth %d \n", static_cast<int>(r.depth));
        else
            std::snprintf(line, sizeof line, "triangle puzzle depth %d rows %d \n",
                static_cast<int>(r.depth), static_cast<int>(r.rows));
        arts_hpx::write_stdout_line(line);
    }
    arts_hpx::run_clock clock{std::chrono::steady_clock::now()};
    hpx::distributed::barrier::synchronize();

    std::uint64_t count = 0;
    if (hpx::get_locality_id() == 0)
    {
        for (std::uint64_t round = 0; round != r.rounds; ++round)
        {
            std::uint64_t const key = pack_key(0, arts_hpx::next_sequence(), 0);
            hpx::future<std::uint64_t> result = hub<std::uint64_t>::instance().receive(key);
            hpx::post<search_action>(arts_hpx::blind_target(),
                node_t{0, -1, tri::initial_board(r.erows), reply_t{0, key}});
            count = result.get();
        }
    }
    hpx::distributed::barrier::synchronize();

    if (hpx::get_locality_id() == 0)
    {
        arts_hpx::print_e2e(clock);
        char line[120];
        if ((r.rows == 0 || r.rows == tri::ROWS_DEFAULT) && r.depth == tri::BOTTOM)
        {
            if (count == 29760)
                std::snprintf(line, sizeof line, "PASS  final count %d \n", static_cast<int>(count));
            else
                std::snprintf(line, sizeof line, "FAIL final count %d should be 29760 \n", static_cast<int>(count));
        }
        else
            std::snprintf(line, sizeof line, "final count %d at depth %d rows %d \n",
                static_cast<int>(count), static_cast<int>(r.depth), static_cast<int>(r.erows));
        arts_hpx::write_stdout_line(line);
    }
    if (structural)
    {
        auto const& c = counters();
        arts_hpx::print_struct({{"nodes", c.nodes.load()}, {"summers", c.summers.load()},
            {"tree", c.nodes.load() + c.summers.load()}, {"boards", c.boards.load()},
            {"board_bytes", c.board_bytes.load()}, {"parcels", c.parcels.load()},
            {"bytes", c.bytes.load()}});
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
