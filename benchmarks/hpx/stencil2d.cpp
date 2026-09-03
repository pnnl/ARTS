#include "stencil2d_core.hpp"

#include "common/arguments.hpp"
#include "common/e2e.hpp"
#include "common/hub.hpp"
#include "common/localities.hpp"
#include "common/rendezvous.hpp"
#include "common/runtime_defaults.hpp"

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/latch.hpp>
#include <hpx/modules/collectives.hpp>
#include <hpx/modules/program_options.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace st = arts_hpx::stencil2d;
using arts_hpx::hub;
using arts_hpx::localities;

// A halo strip lands in its consumer's locality keyed by (tile, side, round).
ARTS_HPX_DEFINE_HUB(std::vector<double>, st_strip)

namespace {

struct counters_t
{
    std::atomic<std::uint64_t> tasks{0}, strips{0}, parcels{0}, bytes{0}, inits{0};
};
counters_t& counters()
{
    static counters_t c;
    return c;
}

st::grid& grid()
{
    static st::grid g;
    return g;
}

enum side : std::uint64_t { LEFT = 0, RIGHT = 1, BELOW = 2, ABOVE = 3 };

// The tiles this locality holds under the program's map, resolved once at
// setup; a slot is written by that tile's init task alone.
struct registry_t
{
    std::int64_t PD_X = 1, PD_Y = 1;
    std::unordered_map<std::int64_t, std::size_t> index;
    std::vector<std::unique_ptr<st::tile>> tiles;
    std::mutex m;
    double norm_sum = 0.0, time_max = 0.0;
    std::unique_ptr<hpx::latch> done;
};
registry_t& registry()
{
    static registry_t r;
    return r;
}
st::tile& tile_of(std::int64_t id)
{
    return *registry().tiles[registry().index.at(id)];
}
std::uint32_t place_of(std::int64_t id)
{
    return static_cast<std::uint32_t>(st::place_of(grid(), id, registry().PD_X, registry().PD_Y));
}

std::uint64_t key(std::int64_t tile, side s, std::int64_t round)
{
    if (tile < 0 || tile >= (1ll << 24))
        arts_hpx::key_bound_violated("tile " + std::to_string(tile) + " exceeds 2^24");
    if (round < 0 || round >= (1ll << 36))
        arts_hpx::key_bound_violated("round " + std::to_string(round) + " exceeds 2^36");
    return (static_cast<std::uint64_t>(tile) << 40) | (static_cast<std::uint64_t>(s) << 36) |
        static_cast<std::uint64_t>(round);
}

void deliver_strip(std::int64_t to, std::uint64_t k, std::vector<double> strip)
{
    if (arts_hpx::struct_enabled())
    {
        counters().strips.fetch_add(1, std::memory_order_relaxed);
        counters().bytes.fetch_add(8ull * strip.size(), std::memory_order_relaxed);
    }
    std::uint32_t const loc = place_of(to);
    if (loc == hpx::get_locality_id())
    {
        hub<std::vector<double>>::instance().deliver(k, std::move(strip));
        return;
    }
    if (arts_hpx::struct_enabled())
        counters().parcels.fetch_add(1, std::memory_order_relaxed);
    hpx::post<st_strip_deliver_action>(localities()[loc], k, std::move(strip));
}

hpx::future<std::vector<double>> input(std::int64_t id, side s, std::int64_t round, bool present)
{
    if (!present)
        return hpx::make_ready_future(std::vector<double>{});
    return hub<std::vector<double>>::instance().receive(key(id, s, round));
}

}    // namespace

void tile_round(std::int64_t id);
HPX_PLAIN_ACTION(tile_round, round_action)

namespace {

void finish_round(std::int64_t id, std::vector<double> const& l, std::vector<double> const& r,
    std::vector<double> const& b, std::vector<double> const& a)
{
    st::grid const& g = grid();
    st::tile& t = tile_of(id);
    if (t.id_x != 0)          st::apply_from_left(t, l);
    if (t.id_x != g.NR_X - 1) st::apply_from_right(t, r);
    if (t.id_y != 0)          st::apply_from_below(t, b);
    if (t.id_y != g.NR_Y - 1) st::apply_from_above(t, a);
    if (t.round == 1)
        t.start = std::chrono::steady_clock::now();
    st::update(t, g);
    if (arts_hpx::struct_enabled())
        counters().tasks.fetch_add(1, std::memory_order_relaxed);
    if (t.round == g.NT)
    {
        t.elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t.start).count();
        double const norm = st::final_norm(t, g);
        registry_t& reg = registry();
        {
            std::lock_guard<std::mutex> lock(reg.m);
            reg.norm_sum += norm;
            reg.time_max = std::max(reg.time_max, t.elapsed);
        }
        reg.done->count_down(1);
        return;
    }
    ++t.round;
    hpx::post<round_action>(localities()[hpx::get_locality_id()], id);
}

}    // namespace

// One round of one tile: the strips go out first, read from IN as it stands;
// the update runs once every neighbour's strip of the same round is in.
void tile_round(std::int64_t id)
{
    st::grid const& g = grid();
    st::tile& t = tile_of(id);
    std::int64_t const r = t.round;
    if (t.id_x != 0)          deliver_strip(id - 1,      key(id - 1,      RIGHT, r), st::left_strip(t));
    if (t.id_x != g.NR_X - 1) deliver_strip(id + 1,      key(id + 1,      LEFT,  r), st::right_strip(t));
    if (t.id_y != 0)          deliver_strip(id - g.NR_X, key(id - g.NR_X, ABOVE, r), st::bottom_strip(t));
    if (t.id_y != g.NR_Y - 1) deliver_strip(id + g.NR_X, key(id + g.NR_X, BELOW, r), st::top_strip(t));
    hpx::dataflow(hpx::launch::async,
        [id](hpx::future<std::vector<double>> l, hpx::future<std::vector<double>> rr,
            hpx::future<std::vector<double>> b, hpx::future<std::vector<double>> a) {
            finish_round(id, l.get(), rr.get(), b.get(), a.get());
        },
        input(id, LEFT, r, t.id_x != 0), input(id, RIGHT, r, t.id_x != g.NR_X - 1),
        input(id, BELOW, r, t.id_y != 0), input(id, ABOVE, r, t.id_y != g.NR_Y - 1));
}

void init_tile(std::int64_t id)
{
    if (arts_hpx::struct_enabled())
        counters().inits.fetch_add(1, std::memory_order_relaxed);
    auto t = std::make_unique<st::tile>(st::make_tile(grid(), id));
    st::initialize(*t);
    registry().tiles[registry().index.at(id)] = std::move(t);
    tile_round(id);
}
HPX_PLAIN_ACTION(init_tile, init_action)

namespace {

struct max_of
{
    double operator()(double a, double b) const { return a > b ? a : b; }
};

}    // namespace

int hpx_main(hpx::program_options::variables_map& variables)
{
    std::vector<std::string> arguments;
    if (variables.count("hpx:positional"))
        arguments = variables["hpx:positional"].as<std::vector<std::string>>();
    std::int64_t NP = 1000, NR = 16, NT = 10;    // the program's defaults when the count is not 3
    if (arguments.size() == 3)
    {
        bool ok = arts_hpx::parse_int(arguments[0], NP) && arts_hpx::parse_int(arguments[1], NR) &&
            arts_hpx::parse_int(arguments[2], NT) && NP >= 2 * st::R + 1 && NR >= 1 &&
            NT >= 1 && NR <= NP;
        if (ok)
        {
            // A tile thinner than the halo has its own strip read past its far
            // edge, so every tile must be at least R wide in both dimensions.
            std::int64_t nx = 0, ny = 0;
            st::split_cart2d(NR, nx, ny);
            ok = NP / nx >= st::R && NP / ny >= st::R;
        }
        if (!ok)
        {
            if (hpx::get_locality_id() == 0)
                std::cerr << "Usage: NP NR NT (positive integers, NR <= NP, NP > 2*radius, "
                             "every tile at least the halo radius wide)\n";
            hpx::finalize();
            return 1;
        }
    }
    grid() = st::make_grid(NP, NR, NT);

    (void) localities();
    bool const structural = arts_hpx::struct_enabled();
    arts_hpx::print_geometry();
    registry_t& reg = registry();
    st::split_cart2d(static_cast<std::int64_t>(arts_hpx::locality_count()), reg.PD_X, reg.PD_Y);
    for (std::int64_t id = 0; id != NR; ++id)
        if (place_of(id) == hpx::get_locality_id())
            reg.index.emplace(id, reg.tiles.size()), reg.tiles.emplace_back();
    reg.done = std::make_unique<hpx::latch>(static_cast<std::ptrdiff_t>(reg.tiles.size()));
    arts_hpx::run_clock clock{std::chrono::steady_clock::now()};
    hpx::distributed::barrier::synchronize();

    if (hpx::get_locality_id() == 0)
        for (std::int64_t id = 0; id != NR; ++id)
            hpx::post<init_action>(localities()[place_of(id)], id);
    reg.done->wait();    // hpx_main may wait; no task does

    // The norm reduction is the completion edge; the timer's follows it.
    auto norm_comm = hpx::collectives::create_communicator("/arts/stencil2d/norm",
        hpx::collectives::num_sites_arg(hpx::get_initial_num_localities()),
        hpx::collectives::this_site_arg(hpx::get_locality_id()));
    double const norm_total = hpx::collectives::all_reduce(norm_comm, reg.norm_sum, std::plus<double>{}).get();
    auto time_comm = hpx::collectives::create_communicator("/arts/stencil2d/time",
        hpx::collectives::num_sites_arg(hpx::get_initial_num_localities()),
        hpx::collectives::this_site_arg(hpx::get_locality_id()));
    double const time_total = hpx::collectives::all_reduce(time_comm, reg.time_max, max_of{}).get();
    hpx::distributed::barrier::synchronize();

    if (hpx::get_locality_id() == 0)
    {
        arts_hpx::print_e2e(clock);
        st::grid const& g = grid();
        double const norm = norm_total / st::active_points(g);
        char line[200];
        std::snprintf(line, sizeof line, "Computed L1 norm = %.12f\n", norm);
        arts_hpx::write_stdout_line(line);
        if (std::abs(norm - st::reference_norm(g)) > st::EPSILON)
            std::snprintf(line, sizeof line, "ERROR: L1 norm = %f, Reference L1 norm = %f\n", norm, st::reference_norm(g));
        else
            std::snprintf(line, sizeof line, "Solution validates\n");
        arts_hpx::write_stdout_line(line);
        double const avgtime = time_total / static_cast<double>(g.NT);
        int const stencil_size = 4 * st::R + 1;
        double const flops = static_cast<double>(2 * stencil_size + 1) * st::active_points(g);
        std::snprintf(line, sizeof line, "Rate (MFlops/s): %f  Avg time (s): %f\n", 1.0E-06 * flops / avgtime, avgtime);
        arts_hpx::write_stdout_line(line);
    }
    if (structural)
    {
        auto const& c = counters();
        arts_hpx::print_struct({{"tasks", c.tasks.load()}, {"strips", c.strips.load()},
            {"parcels", c.parcels.load()}, {"bytes", c.bytes.load()}, {"inits", c.inits.load()}});
        arts_hpx::print_parcels();
    }
    hub<std::vector<double>>::drain_and_report();
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    hpx::init_params params;
    params.cfg = arts_hpx::runtime_defaults();
    return hpx::init(argc, argv, params);
}
