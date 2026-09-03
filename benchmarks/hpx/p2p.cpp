#include "p2p_core.hpp"

#include "common/arguments.hpp"
#include "common/e2e.hpp"
#include "common/hub.hpp"
#include "common/localities.hpp"
#include "common/rendezvous.hpp"
#include "common/runtime_defaults.hpp"

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/modules/program_options.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace p2p = arts_hpx::p2p;
using arts_hpx::hub;
using arts_hpx::localities;

// A rank's boundary lands in the next rank's locality by key, one entry per
// consumer generation; the checksum lands in locality 0's hub.
ARTS_HPX_DEFINE_HUB(p2p::boundary, p2p_boundary)
ARTS_HPX_DEFINE_HUB(double, p2p_result)

namespace {

struct counters_t
{
    std::atomic<std::uint64_t> tasks{0}, sends{0}, parcels{0}, bytes{0}, inits{0};
};
counters_t& counters()
{
    static counters_t c;
    return c;
}

p2p::params& args()
{
    static p2p::params a;
    return a;
}

struct rank_state
{
    p2p::rank_geometry g;
    std::uint32_t timestep = 0, row = 0, phase = 0;
    std::vector<double> data;
};

// The ranks this locality holds under the block map; a slot is written by
// that rank's own init task and read by that rank's generations alone.
std::vector<std::unique_ptr<rank_state>>& ranks_here()
{
    static std::vector<std::unique_ptr<rank_state>> v;
    return v;
}
std::uint32_t first_rank_here()
{
    return hpx::get_locality_id() * p2p::block_size(args().p, arts_hpx::locality_count());
}
rank_state& state_of(std::uint32_t rank)
{
    return *ranks_here()[rank - first_rank_here()];
}

// The consumer's generation index names the edge.
std::uint64_t key(std::uint32_t rank, std::uint64_t generation)
{
    if (rank >= (1u << 24))
        arts_hpx::key_bound_violated("rank " + std::to_string(rank) + " exceeds 2^24");
    if (generation >= (1ull << 40))
        arts_hpx::key_bound_violated("generation " + std::to_string(generation) + " exceeds 2^40");
    return (static_cast<std::uint64_t>(rank) << 40) | generation;
}
constexpr std::uint64_t result_key = 0;

void deliver_boundary(std::uint32_t rank, std::uint64_t k, p2p::boundary const& b)
{
    if (arts_hpx::struct_enabled())
    {
        counters().sends.fetch_add(1, std::memory_order_relaxed);
        counters().bytes.fetch_add(8ull * b.count, std::memory_order_relaxed);
    }
    std::uint32_t const loc = p2p::rank_locality(rank, args().p, arts_hpx::locality_count());
    if (loc == hpx::get_locality_id())
    {
        hub<p2p::boundary>::instance().deliver(k, b);
        return;
    }
    if (arts_hpx::struct_enabled())
        counters().parcels.fetch_add(1, std::memory_order_relaxed);
    hpx::post<p2p_boundary_deliver_action>(localities()[loc], k, b);
}

void deliver_result(double checksum)
{
    if (hpx::get_locality_id() == 0)
    {
        hub<double>::instance().deliver(result_key, checksum);
        return;
    }
    if (arts_hpx::struct_enabled())
        counters().parcels.fetch_add(1, std::memory_order_relaxed);
    hpx::post<p2p_result_deliver_action>(localities()[0], result_key, checksum);
}

}    // namespace

void step(std::uint32_t rank);
HPX_PLAIN_ACTION(step, step_action)

namespace {

// One generation: consume the left boundary, compute this phase's rows, hand
// the right column on, then the next generation of the same rank, on the same
// locality — the chain never moves.
void run_generation(std::uint32_t rank, p2p::boundary const& in)
{
    if (arts_hpx::struct_enabled())
        counters().tasks.fetch_add(1, std::memory_order_relaxed);
    rank_state& s = state_of(rank);
    p2p::params const& a = args();
    std::uint32_t const k = s.g.k, w = s.g.w;
    std::uint64_t const g = static_cast<std::uint64_t>(s.timestep) * w + s.phase;
    auto A = [&](std::uint32_t i, std::uint32_t j) -> double& {
        return s.data[static_cast<std::size_t>(i) * k + j];
    };

    if (rank == 0 && s.row == 0 && s.timestep != 0)
        A(0, 0) = -in.v[0];
    std::uint32_t const numrows = p2p::rows_this_phase(a, s.row);
    p2p::compute(s.data, k, rank, s.row, numrows, in.v);

    if (rank != a.p - 1)
    {
        p2p::boundary out;
        out.count = static_cast<std::uint8_t>(numrows + 1);
        for (std::uint32_t i = 0; i != numrows + 1; ++i)
            out.v[i] = A(s.row + i, k - 1);
        deliver_boundary(rank + 1, key(rank + 1, g), out);
    }

    if (s.timestep == a.t && s.phase == w - 1)
    {
        if (rank == a.p - 1)
            deliver_result(A(a.n - 1, k - 1));
        return;
    }

    if (rank == a.p - 1 && s.phase == w - 1)
    {
        p2p::boundary wrap;
        wrap.count = 1;
        wrap.v[0] = A(a.n - 1, k - 1);
        deliver_boundary(0, key(0, g + 1), wrap);
    }

    if (s.phase == w - 1)
    {
        s.row = 0;
        ++s.timestep;
        s.phase = 0;
    }
    else
    {
        s.row += a.gf;
        ++s.phase;
    }
    hpx::post<step_action>(localities()[hpx::get_locality_id()], rank);
}

}    // namespace

void step(std::uint32_t rank)
{
    rank_state& s = state_of(rank);
    std::uint64_t const g = static_cast<std::uint64_t>(s.timestep) * s.g.w + s.phase;
    bool const has_input = rank != 0 || (s.phase == 0 && s.timestep != 0);
    hpx::future<p2p::boundary> in = has_input ?
        hub<p2p::boundary>::instance().receive(key(rank, g)) :
        hpx::make_ready_future(p2p::boundary{});
    in.then(hpx::launch::async, [rank](hpx::future<p2p::boundary> f) {
        run_generation(rank, f.get());
    });
}

void init_rank(std::uint32_t rank)
{
    if (arts_hpx::struct_enabled())
        counters().inits.fetch_add(1, std::memory_order_relaxed);
    auto s = std::make_unique<rank_state>();
    s->g = p2p::geometry(args(), rank);
    p2p::initialize(s->data, args(), s->g, rank);
    ranks_here()[rank - first_rank_here()] = std::move(s);
    step(rank);
}
HPX_PLAIN_ACTION(init_rank, init_action)

namespace {

constexpr char const* usage = "p m n t [gf]";

}    // namespace

int hpx_main(hpx::program_options::variables_map& variables)
{
    std::vector<std::string> arguments;
    if (variables.count("hpx:positional"))
        arguments = variables["hpx:positional"].as<std::vector<std::string>>();
    p2p::params& a = args();
    if (arguments.size() == 4 || arguments.size() == 5)
    {
        if (!arts_hpx::parse_int(arguments[0], a.p) || !arts_hpx::parse_int(arguments[1], a.m) ||
            !arts_hpx::parse_int(arguments[2], a.n) || !arts_hpx::parse_int(arguments[3], a.t) ||
            (arguments.size() == 5 && !arts_hpx::parse_int(arguments[4], a.gf)))
            return arts_hpx::fail_with_usage("p2p arguments must be unsigned integers", usage);
    }
    else if (!arguments.empty())
        return arts_hpx::fail_with_usage("invalid number of p2p arguments", usage);
    if (a.p == 0 || a.m < a.p || a.n < 2 || a.t == 0 || a.gf == 0 ||
        a.gf >= p2p::boundary::capacity)
        return arts_hpx::fail_with_usage("p2p needs p >= 1, m >= p, n >= 2, t >= 1, 1 <= gf <= 7", usage);

    (void) localities();
    bool const structural = arts_hpx::struct_enabled();
    arts_hpx::print_geometry();
    {
        std::uint32_t const first = first_rank_here();
        std::uint32_t const block = p2p::block_size(a.p, arts_hpx::locality_count());
        std::uint32_t const count = first >= a.p ? 0 : std::min(block, a.p - first);
        ranks_here().resize(count);
    }
    arts_hpx::run_clock clock{std::chrono::steady_clock::now()};
    hpx::distributed::barrier::synchronize();

    double checksum = 0.0;
    if (hpx::get_locality_id() == 0)
    {
        hpx::future<double> result = hub<double>::instance().receive(result_key);
        for (std::uint32_t rank = 0; rank != a.p; ++rank)
            hpx::post<init_action>(
                localities()[p2p::rank_locality(rank, a.p, arts_hpx::locality_count())], rank);
        checksum = result.get();
    }
    hpx::distributed::barrier::synchronize();

    if (hpx::get_locality_id() == 0)
    {
        double const seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - clock.start).count();
        arts_hpx::print_e2e(clock);
        char line[160];
        if (checksum == p2p::expected_checksum(a))
            std::snprintf(line, sizeof line, "PASS checksum = %f   \n", checksum);
        else
            std::snprintf(line, sizeof line, "FAIL  checksum = %f  should be %d \n",
                checksum, static_cast<int>(p2p::expected_checksum(a)));
        arts_hpx::write_stdout_line(line);
        double const avgtime = seconds / a.t;
        double flops = 1.0e-06 * 2 * static_cast<double>(a.m - 1) * (a.n - 1) / avgtime;
        if (a.gf > 1)
            flops = -flops;
        std::snprintf(line, sizeof line, "Rate (MFlops/s): %f Avg time (s): %f\n", flops, avgtime);
        arts_hpx::write_stdout_line(line);
    }
    if (structural)
    {
        auto const& c = counters();
        // Every field is summed over localities, so the rank count is what
        // this locality holds, not the program-wide total each would report.
        arts_hpx::print_struct({{"tasks", c.tasks.load()}, {"sends", c.sends.load()},
            {"parcels", c.parcels.load()}, {"bytes", c.bytes.load()},
            {"ranks", static_cast<std::uint64_t>(ranks_here().size())},
            {"inits", c.inits.load()}});
        arts_hpx::print_parcels();
    }
    hub<p2p::boundary>::drain_and_report();
    hub<double>::drain_and_report();
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    hpx::init_params params;
    params.cfg = arts_hpx::runtime_defaults();
    return hpx::init(argc, argv, params);
}
