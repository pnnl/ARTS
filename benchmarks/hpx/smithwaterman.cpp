#include "smithwaterman_core.hpp"

#include "common/arguments.hpp"
#include "common/e2e.hpp"
#include "common/hub.hpp"
#include "common/localities.hpp"
#include "common/rendezvous.hpp"
#include "common/runtime_defaults.hpp"

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/modules/collectives.hpp>
#include <hpx/modules/program_options.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

namespace sw = arts_hpx::smithwaterman;
using arts_hpx::hub;
using arts_hpx::localities;

// A tile's three inputs arrive by key at its locality: two strips and a
// corner; the final score lands in locality 0's hub.
ARTS_HPX_DEFINE_HUB(std::vector<std::int32_t>, sw_strip)
ARTS_HPX_DEFINE_HUB(std::int32_t, sw_scalar)

namespace {

struct counters_t
{
    std::atomic<std::uint64_t> tasks{0}, posts_from_locality0{0}, strips{0}, parcels{0}, bytes{0};
};
counters_t& counters()
{
    static counters_t c;
    return c;
}

// The write-once inputs, resident on every locality after one broadcast.
struct inputs_t
{
    std::vector<std::int8_t> s1, s2;
    std::uint32_t check_score = 0;
    template <typename Archive>
    void serialize(Archive& ar, unsigned) { ar & s1 & s2 & check_score; }
};
inputs_t& inputs()
{
    static inputs_t in;
    return in;
}
sw::params& args()
{
    static sw::params p;
    return p;
}

enum input_kind : std::uint64_t { WEST = 0, NORTH = 1, CORNER = 2 };

std::uint64_t key(std::int32_t i, std::int32_t j, input_kind kind)
{
    if (i < 0 || i >= (1 << 24) || j < 0 || j >= (1 << 24))
        arts_hpx::key_bound_violated("tile (" + std::to_string(i) + "," + std::to_string(j) + ") exceeds 2^24");
    return (static_cast<std::uint64_t>(i) << 26) | (static_cast<std::uint64_t>(j) << 2) |
        static_cast<std::uint64_t>(kind);
}
constexpr std::uint64_t score_key = 1ull << 60;

#if defined(HPX_APP_HINTED_PLACEMENT)
// The application's own key: a band of tile rows per locality.
std::uint32_t target_locality(std::int32_t i, std::int32_t)
{
    std::uint32_t const n = arts_hpx::locality_count();
    if (n <= 1 || args().H == 0)
        return 0;
    std::uint64_t band = (static_cast<std::uint64_t>(i - 1) * n) / static_cast<std::uint64_t>(args().H);
    if (band >= n)
        band = n - 1;
    return static_cast<std::uint32_t>(band);
}
#else
// The runtime's no-preference policy as the creating loop sees it: the n-th
// spawn of one serial loop on locality 0 lands on locality n mod L.
std::uint32_t target_locality(std::int32_t i, std::int32_t j)
{
    std::uint64_t const n = static_cast<std::uint64_t>(i - 1) * args().W + static_cast<std::uint64_t>(j - 1);
    return static_cast<std::uint32_t>(n % arts_hpx::locality_count());
}
#endif

void deliver_strip(std::int32_t i, std::int32_t j, input_kind kind, std::vector<std::int32_t> strip)
{
    if (arts_hpx::struct_enabled())
    {
        counters().strips.fetch_add(1, std::memory_order_relaxed);
        counters().bytes.fetch_add(4ull * strip.size(), std::memory_order_relaxed);
    }
    std::uint32_t const loc = target_locality(i, j);
    if (loc == hpx::get_locality_id())
    {
        hub<std::vector<std::int32_t>>::instance().deliver(key(i, j, kind), std::move(strip));
        return;
    }
    if (arts_hpx::struct_enabled())
        counters().parcels.fetch_add(1, std::memory_order_relaxed);
    hpx::post<sw_strip_deliver_action>(localities()[loc], key(i, j, kind), std::move(strip));
}

void deliver_scalar(std::uint32_t loc, std::uint64_t k, std::int32_t value, bool count_as_strip)
{
    if (count_as_strip && arts_hpx::struct_enabled())
    {
        counters().strips.fetch_add(1, std::memory_order_relaxed);
        counters().bytes.fetch_add(4, std::memory_order_relaxed);
    }
    if (loc == hpx::get_locality_id())
    {
        hub<std::int32_t>::instance().deliver(k, value);
        return;
    }
    if (arts_hpx::struct_enabled())
        counters().parcels.fetch_add(1, std::memory_order_relaxed);
    hpx::post<sw_scalar_deliver_action>(localities()[loc], k, value);
}

void compute(std::int32_t i, std::int32_t j, std::vector<std::int32_t> const& west,
    std::vector<std::int32_t> const& north, std::int32_t corner)
{
    sw::params const& p = args();
    sw::tile_out o = sw::compute_tile(p, inputs().s1, inputs().s2, i, j, west, north, corner);
    if (arts_hpx::struct_enabled())
        counters().tasks.fetch_add(1, std::memory_order_relaxed);
    if (j < p.W)
        deliver_strip(i, j + 1, WEST, std::move(o.right_column));
    if (i < p.H)
        deliver_strip(i + 1, j, NORTH, std::move(o.bottom_row));
    if (i < p.H && j < p.W)
        deliver_scalar(target_locality(i + 1, j + 1), key(i + 1, j + 1, CORNER), o.bottom_right, true);
    if (i == p.H && j == p.W)
        deliver_scalar(0, score_key, o.bottom_row[sw::effective_width(p, j) - 1], false);
}

}    // namespace

// A tile: the continuation on its three inputs, seeded borders where the
// input would come from outside the matrix.
void create_tile(std::int32_t i, std::int32_t j)
{
    sw::params const& p = args();
    hpx::future<std::vector<std::int32_t>> west = j == 1 ?
        hpx::make_ready_future(sw::border_right_column(p, i)) :
        hub<std::vector<std::int32_t>>::instance().receive(key(i, j, WEST));
    hpx::future<std::vector<std::int32_t>> north = i == 1 ?
        hpx::make_ready_future(sw::border_bottom_row(p, j)) :
        hub<std::vector<std::int32_t>>::instance().receive(key(i, j, NORTH));
    hpx::future<std::int32_t> corner = (i == 1 || j == 1) ?
        hpx::make_ready_future(sw::border_corner(p, i, j)) :
        hub<std::int32_t>::instance().receive(key(i, j, CORNER));
    hpx::dataflow(hpx::launch::async,
        [i, j](hpx::future<std::vector<std::int32_t>> w, hpx::future<std::vector<std::int32_t>> n,
            hpx::future<std::int32_t> c) { compute(i, j, w.get(), n.get(), c.get()); },
        std::move(west), std::move(north), std::move(corner));
}
HPX_PLAIN_ACTION(create_tile, create_tile_action)

namespace {

constexpr char const* usage =
    "tileWidth tileHeight fileName1 fileName2 scoreFile";

std::string read_text(std::string const& path, bool& ok)
{
    std::ifstream f(path);
    ok = static_cast<bool>(f);
    return ok ? std::string((std::istreambuf_iterator<char>(f)), {}) : std::string{};
}

}    // namespace

int hpx_main(hpx::program_options::variables_map& variables)
{
    std::vector<std::string> arguments;
    if (variables.count("hpx:positional"))
        arguments = variables["hpx:positional"].as<std::vector<std::string>>();
    if (arguments.size() != 5)
        return arts_hpx::fail_with_usage("invalid number of smithwaterman arguments", usage);
    std::int32_t tile_w = 0, tile_h = 0;
    if (!arts_hpx::parse_int(arguments[0], tile_w) || !arts_hpx::parse_int(arguments[1], tile_h) || tile_w < 1 || tile_h < 1)
        return arts_hpx::fail_with_usage("tile sizes must be positive integers", usage);

    (void) localities();
    bool const structural = arts_hpx::struct_enabled();
    arts_hpx::print_geometry();
    arts_hpx::run_clock clock{std::chrono::steady_clock::now()};
    hpx::distributed::barrier::synchronize();

    // Locality 0 reads once; every locality holds the inputs for the run.
    // The read is inside the measured span because it is the program's own
    // input, not runtime setup, and a file-open failure is a collective
    // outcome: every locality decides on the broadcast result, never alone.
    auto comm = hpx::collectives::create_communicator("/arts/smithwaterman/inputs",
        hpx::collectives::num_sites_arg(hpx::get_initial_num_localities()),
        hpx::collectives::this_site_arg(hpx::get_locality_id()));
    if (hpx::get_locality_id() == 0)
    {
        bool ok1 = false, ok2 = false, ok3 = false;
        std::string const t1 = read_text(arguments[2], ok1), t2 = read_text(arguments[3], ok2),
                          t3 = read_text(arguments[4], ok3);
        inputs_t in;
        if (ok1 && ok2 && ok3)
        {
            in.s1 = sw::map_sequence(t1);
            in.s2 = sw::map_sequence(t2);
            in.check_score = static_cast<std::uint32_t>(std::atoi(t3.c_str()));
        }
        else
            std::cerr << "could not open an input file\n";
        inputs() = hpx::collectives::broadcast_to(comm, in).get();
    }
    else
        inputs() = hpx::collectives::broadcast_from<inputs_t>(comm).get();
    if (inputs().s1.empty() || inputs().s2.empty())
    {
        hpx::finalize();
        return 1;
    }
    args() = sw::make_params(tile_w, tile_h, static_cast<std::int32_t>(inputs().s1.size()),
        static_cast<std::int32_t>(inputs().s2.size()), inputs().check_score);
    sw::params const& p = args();
    if (hpx::get_locality_id() == 0)
    {
        char line[160];
        std::snprintf(line, sizeof line, "Size of input string 1 is %d\n", p.len1); arts_hpx::write_stdout_line(line);
        std::snprintf(line, sizeof line, "Size of input string 2 is %d\n", p.len2); arts_hpx::write_stdout_line(line);
        std::snprintf(line, sizeof line, "Score to get it %u\n", p.check_score); arts_hpx::write_stdout_line(line);
        std::snprintf(line, sizeof line, "Tile width is %d\n", p.tile_w); arts_hpx::write_stdout_line(line);
        std::snprintf(line, sizeof line, "Tile height is %d\n", p.tile_h); arts_hpx::write_stdout_line(line);
        std::snprintf(line, sizeof line, "Imported %d x %d tiles.\n", p.W, p.H); arts_hpx::write_stdout_line(line);
    }

    std::int32_t score = 0;
    if (hpx::get_locality_id() == 0)
    {
        hpx::future<std::int32_t> result = hub<std::int32_t>::instance().receive(score_key);
        for (std::int32_t i = 1; i <= p.H; ++i)
            for (std::int32_t j = 1; j <= p.W; ++j)
            {
#if defined(HPX_APP_HINTED_PLACEMENT)
                hpx::id_type const& target = localities()[target_locality(i, j)];
#else
                hpx::id_type const& target = arts_hpx::blind_target();
                if (hpx::naming::get_locality_id_from_id(target) != target_locality(i, j))
                    arts_hpx::key_bound_violated("the blind spawn and its closed form disagree");
#endif
                if (structural)
                    counters().posts_from_locality0.fetch_add(1, std::memory_order_relaxed);
                hpx::post<create_tile_action>(target, i, j);
            }
        score = result.get();
    }
    hpx::distributed::barrier::synchronize();

    if (hpx::get_locality_id() == 0)
    {
        arts_hpx::print_e2e(clock);
        char line[160];
        std::snprintf(line, sizeof line, "score: %d\n", score);
        arts_hpx::write_stdout_line(line);
        std::snprintf(line, sizeof line, "%s Expected score: %u\n",
            static_cast<std::uint32_t>(score) == p.check_score ? "PASSED" : "FAILURE", p.check_score);
        arts_hpx::write_stdout_line(line);
    }
    if (structural)
    {
        auto const& c = counters();
        arts_hpx::print_struct({{"tasks", c.tasks.load()},
            {"posts_from_locality0", c.posts_from_locality0.load()}, {"strips", c.strips.load()},
            {"parcels", c.parcels.load()}, {"bytes", c.bytes.load()}});
        arts_hpx::print_parcels();
    }
    hub<std::vector<std::int32_t>>::drain_and_report();
    hub<std::int32_t>::drain_and_report();
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    hpx::init_params params;
    params.cfg = arts_hpx::runtime_defaults();
    return hpx::init(argc, argv, params);
}
