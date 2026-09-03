#include "tempest_core.hpp"

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

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace tp = arts_hpx::tempest;
using arts_hpx::hub;
using arts_hpx::localities;

// A neighbour's value lands at the patch's creating locality keyed by
// (patch, side, generation); under the base tier the generation that runs
// elsewhere asks for its inputs and receives them in one batched return.
ARTS_HPX_DEFINE_HUB(std::int64_t, tp_value)
ARTS_HPX_DEFINE_HUB(std::vector<std::int64_t>, tp_batch)

namespace {

struct counters_t
{
    std::atomic<std::uint64_t> generations{0}, deliveries{0}, parcels{0}, bytes{0}, interests{0},
        returns{0}, state_posts{0}, state_bytes{0}, map_exchange{0}, inits{0};
};
counters_t& counters()
{
    static counters_t c;
    return c;
}

struct run_t
{
    std::uint64_t k = tp::PATCH_RANGE_DEFAULT, duration = tp::DURATION_DEFAULT;
};
run_t& run()
{
    static run_t r;
    return r;
}

// Patches created here and, after the exchange, every patch's creator.
struct registry_t
{
    std::mutex m;
    std::vector<std::uint64_t> created;
    std::vector<std::uint32_t> creator;
    std::vector<tp::patch_state> states;    // hinted: the resident state, by slot
    std::vector<std::size_t> slot;           // hinted: patch -> slot
    std::size_t next_slot = 0;               // hinted: guarded by m
    std::unique_ptr<hpx::latch> done;
};
registry_t& registry()
{
    static registry_t r;
    return r;
}

std::uint32_t creator_of(std::uint64_t P)
{
#if defined(HPX_APP_HINTED_PLACEMENT)
    return static_cast<std::uint32_t>(tp::home_rank(P, run().k, arts_hpx::locality_count()));
#else
    return registry().creator[P];
#endif
}

// Bit 63 is the result key and bit 62 the panel-ready keys, so a patch
// number shifted into bit 40 must stay inside 22 bits to name only itself.
std::uint64_t value_key(std::uint64_t P, std::uint32_t side, std::uint64_t g)
{
    if (P >= (1ull << 22))
        arts_hpx::key_bound_violated("patch " + std::to_string(P) + " exceeds 2^22");
    if (side >= 8)
        arts_hpx::key_bound_violated("side " + std::to_string(side) + " exceeds 2^3");
    if (g >= (1ull << 37))
        arts_hpx::key_bound_violated("generation " + std::to_string(g) + " exceeds 2^37");
    return (P << 40) | (static_cast<std::uint64_t>(side) << 37) | g;
}
std::uint64_t batch_key(std::uint64_t P, std::uint64_t g)
{
    if (P >= (1ull << 22))
        arts_hpx::key_bound_violated("patch " + std::to_string(P) + " exceeds 2^22");
    if (g >= (1ull << 40))
        arts_hpx::key_bound_violated("generation " + std::to_string(g) + " exceeds 2^40");
    return (P << 40) | g;
}
constexpr std::uint64_t result_key = 1ull << 63;
std::uint64_t panel_ready_key(std::uint64_t panel) { return (1ull << 62) | panel; }

void deliver_value(std::uint64_t P, std::uint32_t side, std::uint64_t g, std::int64_t value)
{
    if (arts_hpx::struct_enabled())
        counters().deliveries.fetch_add(1, std::memory_order_relaxed);
    std::uint32_t const loc = creator_of(P);
    if (loc == hpx::get_locality_id())
    {
        hub<std::int64_t>::instance().deliver(value_key(P, side, g), value);
        return;
    }
    if (arts_hpx::struct_enabled())
    {
        counters().parcels.fetch_add(1, std::memory_order_relaxed);
        counters().bytes.fetch_add(8, std::memory_order_relaxed);
    }
    hpx::post<tp_value_deliver_action>(localities()[loc], value_key(P, side, g), value);
}

void deliver_batch(std::uint32_t loc, std::uint64_t k, std::vector<std::int64_t> values)
{
    if (loc == hpx::get_locality_id())
    {
        hub<std::vector<std::int64_t>>::instance().deliver(k, std::move(values));
        return;
    }
    if (arts_hpx::struct_enabled())
    {
        counters().parcels.fetch_add(1, std::memory_order_relaxed);
        counters().bytes.fetch_add(8ull * values.size(), std::memory_order_relaxed);
    }
    hpx::post<tp_batch_deliver_action>(localities()[loc], k, std::move(values));
}

}    // namespace

void patch_done(std::uint64_t)
{
    registry().done->count_down(1);
}
HPX_PLAIN_ACTION(patch_done, done_action)

namespace {

void signal_done(std::uint64_t P)
{
    std::uint32_t const loc = creator_of(P);
    if (loc == hpx::get_locality_id())
    {
        patch_done(P);
        return;
    }
    if (arts_hpx::struct_enabled())
        counters().parcels.fetch_add(1, std::memory_order_relaxed);
    hpx::post<done_action>(localities()[loc], P);
}

void launch(tp::patch_state const& s);

// One generation: the received values are this patch's inputs; forwarding
// its own number to every neighbour is the exchange; the terminal
// generation only reports.
void generation(tp::patch_state& s, std::vector<std::int64_t> const& v)
{
    if (arts_hpx::struct_enabled())
        counters().generations.fetch_add(1, std::memory_order_relaxed);
    std::uint64_t const g = s.timestep, P = s.patchNum;
    if (P == tp::TEST_PATCH)
        arts_hpx::write_stdout_line("timestep: " + std::to_string(g) + "\n");
    if (g == s.duration - 1)
    {
        if (P == tp::TEST_PATCH)
        {
            std::vector<std::int64_t> received(8, -1);
            for (std::uint32_t i = 0; i < 8; ++i)
                if (s.nbr[i] > -1)
                    received[i] = v[i];
            deliver_batch(0, result_key, std::move(received));
        }
        signal_done(P);
        return;
    }
    for (std::uint32_t i = 0; i < 8; ++i)
        if (s.nbr[i] > -1)
        {
            std::uint64_t const Q = static_cast<std::uint64_t>(s.nbr[i]);
            deliver_value(Q, tp::neighbor_relation(P, Q, s.k), g + 1, static_cast<std::int64_t>(P));
        }
    ++s.timestep;
    launch(s);
}

}    // namespace

#if defined(HPX_APP_HINTED_PLACEMENT)

// The state stays on the home; a generation is the continuation on its
// locally keyed inputs.
void step_patch(std::uint64_t P)
{
    registry_t& reg = registry();
    tp::patch_state& s = reg.states[reg.slot[P]];
    std::uint64_t const g = s.timestep;
    std::vector<hpx::future<std::int64_t>> inputs;
    inputs.reserve(8);
    for (std::uint32_t i = 0; i < 8; ++i)
        inputs.push_back(s.nbr[i] > -1 ? hub<std::int64_t>::instance().receive(value_key(P, i, g)) :
                                          hpx::make_ready_future(std::int64_t{-1}));
    hpx::when_all(std::move(inputs))
        .then(hpx::launch::async, [P](hpx::future<std::vector<hpx::future<std::int64_t>>> ready) {
            std::vector<std::int64_t> v;
            for (auto& f : ready.get())
                v.push_back(f.get());
            registry_t& r = registry();
            generation(r.states[r.slot[P]], v);
        });
}
HPX_PLAIN_ACTION(step_patch, step_action)

namespace {
void launch(tp::patch_state const& s)
{
    hpx::post<step_action>(localities()[hpx::get_locality_id()], s.patchNum);
}
}    // namespace

#else

// At the creating locality: gather the generation's inputs and return them
// to wherever the generation runs, in one parcel.
void serve_interest(std::uint64_t P, std::uint64_t g, std::uint32_t requester, std::uint64_t key)
{
    if (arts_hpx::struct_enabled())
        counters().interests.fetch_add(1, std::memory_order_relaxed);
    std::vector<hpx::future<std::int64_t>> inputs;
    inputs.reserve(8);
    for (std::uint32_t i = 0; i < 8; ++i)
        inputs.push_back(tp::find_neighbor_patch(P, run().k, i) > -1 ?
                hub<std::int64_t>::instance().receive(value_key(P, i, g)) :
                hpx::make_ready_future(std::int64_t{-1}));
    hpx::when_all(std::move(inputs))
        .then(hpx::launch::sync, [requester, key](hpx::future<std::vector<hpx::future<std::int64_t>>> ready) {
            std::vector<std::int64_t> v;
            for (auto& f : ready.get())
                v.push_back(f.get());
            if (arts_hpx::struct_enabled())
                counters().returns.fetch_add(1, std::memory_order_relaxed);
            deliver_batch(requester, key, std::move(v));
        });
}
HPX_PLAIN_ACTION(serve_interest, interest_action)

// The generation runs wherever the blind spawn put it, carrying the state;
// it registers for its batched inputs and asks the creator for them.
void step_state(tp::patch_state s)
{
    std::uint64_t const P = s.patchNum, g = s.timestep;
    std::uint32_t const here = hpx::get_locality_id();
    std::uint64_t const key = batch_key(P, g);
    hub<std::vector<std::int64_t>>::instance().receive(key).then(hpx::launch::async,
        [s](hpx::future<std::vector<std::int64_t>> f) mutable { generation(s, f.get()); });
    std::uint32_t const c = creator_of(P);
    if (c == here)
    {
        serve_interest(P, g, here, key);
        return;
    }
    if (arts_hpx::struct_enabled())
    {
        counters().parcels.fetch_add(1, std::memory_order_relaxed);
        counters().bytes.fetch_add(32, std::memory_order_relaxed);
    }
    hpx::post<interest_action>(localities()[c], P, g, here, key);
}
HPX_PLAIN_ACTION(step_state, step_action)

namespace {
void launch(tp::patch_state const& s)
{
    hpx::id_type const& target = arts_hpx::blind_target();
    if (arts_hpx::struct_enabled())
    {
        counters().state_posts.fetch_add(1, std::memory_order_relaxed);
        counters().state_bytes.fetch_add(sizeof(tp::patch_state), std::memory_order_relaxed);
        if (hpx::naming::get_locality_id_from_id(target) != hpx::get_locality_id())
        {
            counters().parcels.fetch_add(1, std::memory_order_relaxed);
            counters().bytes.fetch_add(sizeof(tp::patch_state), std::memory_order_relaxed);
        }
    }
    hpx::post<step_action>(target, s);
}
}    // namespace

#endif

// Creation registers the patch here, seeds generation 0's inputs, and —
// where the state is resident — starts the patch.
void patch_init(std::uint64_t P)
{
    if (arts_hpx::struct_enabled())
        counters().inits.fetch_add(1, std::memory_order_relaxed);
    run_t const& r = run();
    tp::patch_state const s = tp::make_state(P, r.k, r.duration);
    for (std::uint32_t i = 0; i < 8; ++i)
        if (s.nbr[i] > -1)
            hub<std::int64_t>::instance().deliver(value_key(P, i, 0), -1);
    registry_t& reg = registry();
    {
        std::lock_guard<std::mutex> lock(reg.m);
        reg.created.push_back(P);
#if defined(HPX_APP_HINTED_PLACEMENT)
        reg.slot[P] = reg.next_slot;
        reg.states[reg.next_slot++] = s;
#endif
    }
#if defined(HPX_APP_HINTED_PLACEMENT)
    launch(s);
#endif
}
HPX_PLAIN_ACTION(patch_init, patch_init_action)

void panel_init(std::uint64_t panel)
{
    if (arts_hpx::struct_enabled())
        counters().inits.fetch_add(1, std::memory_order_relaxed);
    std::uint64_t const k = run().k;
#if defined(HPX_APP_HINTED_PLACEMENT)
    for (std::uint64_t i = 0; i < k * k; ++i)
    {
        std::uint64_t const P = k * k * panel + i;
        hpx::post<patch_init_action>(localities()[creator_of(P)], P);
    }
#else
    std::vector<hpx::future<void>> pending;
    pending.reserve(k * k);
    for (std::uint64_t i = 0; i < k * k; ++i)
        pending.push_back(hpx::async<patch_init_action>(arts_hpx::blind_target(), k * k * panel + i));
    hpx::when_all(std::move(pending)).then(hpx::launch::sync, [panel](hpx::future<std::vector<hpx::future<void>>> f) {
        f.get();
        if (hpx::get_locality_id() == 0)
            hub<std::int64_t>::instance().deliver(panel_ready_key(panel), 1);
        else
            hpx::post<tp_value_deliver_action>(localities()[0], panel_ready_key(panel), std::int64_t{1});
    });
#endif
}
HPX_PLAIN_ACTION(panel_init, panel_init_action)

namespace {

void append_grid(std::string& out, std::int64_t const* g)
{
    char line[128];
    std::snprintf(line, sizeof line, "%lld\t%lld\t%lld\n", (long long) g[tp::NW], (long long) g[tp::N], (long long) g[tp::NE]);
    out += line;
    std::snprintf(line, sizeof line, "%lld\t%lld\t%lld\n", (long long) g[tp::W], (long long) g[8], (long long) g[tp::E]);
    out += line;
    std::snprintf(line, sizeof line, "%lld\t%lld\t%lld\n", (long long) g[tp::SW], (long long) g[tp::S], (long long) g[tp::SE]);
    out += line;
}

}    // namespace

int hpx_main(hpx::program_options::variables_map& variables)
{
    std::vector<std::string> arguments;
    if (variables.count("hpx:positional"))
        arguments = variables["hpx:positional"].as<std::vector<std::string>>();
    run_t& r = run();
    if (arguments.size() > 2)
    {
        if (hpx::get_locality_id() == 0)
            std::cerr << "USAGE: tempest [patchRange [duration]]  expected at most 2 arguments, got "
                      << arguments.size() << '\n';
        hpx::finalize();
        return 1;
    }
    if (arguments.empty())
    {
        if (hpx::get_locality_id() == 0)
            arts_hpx::write_stdout_line("NO PATCHRANGE ARG GIVEN. USING DEFAULT PARAMS (patchRange=2).\n");
    }
    else
    {
        std::uint64_t* out[2] = {&r.k, &r.duration};
        for (std::size_t i = 0; i < arguments.size(); ++i)
            if (!arts_hpx::parse_int(arguments[i], *out[i]) || *out[i] == 0)
            {
                if (hpx::get_locality_id() == 0)
                    std::cerr << "USAGE: tempest [patchRange [duration]]  arguments must be positive integers\n";
                hpx::finalize();
                return 1;
            }
    }
    std::uint64_t const patches = 6 * r.k * r.k;

    (void) localities();
    bool const structural = arts_hpx::struct_enabled();
    arts_hpx::print_geometry();
    registry_t& reg = registry();
#if defined(HPX_APP_HINTED_PLACEMENT)
    reg.slot.assign(patches, 0);
    std::size_t mine = 0;
    for (std::uint64_t P = 0; P < patches; ++P)
        if (creator_of(P) == hpx::get_locality_id())
            ++mine;
    // The slot a running generation holds a reference to must not move, so
    // the vector reaches its final length here, before any generation exists,
    // and creation only fills a slot.
    reg.states.resize(mine);
    reg.done = std::make_unique<hpx::latch>(static_cast<std::ptrdiff_t>(mine));
#else
    reg.creator.assign(patches, 0);
#endif
    arts_hpx::run_clock clock{std::chrono::steady_clock::now()};
    hpx::distributed::barrier::synchronize();

    hpx::future<std::vector<std::int64_t>> result;
    if (hpx::get_locality_id() == 0)
        result = hub<std::vector<std::int64_t>>::instance().receive(result_key);
#if defined(HPX_APP_HINTED_PLACEMENT)
    if (hpx::get_locality_id() == 0)
        for (std::uint64_t panel = 0; panel < 6; ++panel)
            hpx::post<panel_init_action>(localities()[creator_of(panel * r.k * r.k)], panel);
#else
    if (hpx::get_locality_id() == 0)
    {
        std::vector<hpx::future<std::int64_t>> ready;
        for (std::uint64_t panel = 0; panel < 6; ++panel)
            ready.push_back(hub<std::int64_t>::instance().receive(panel_ready_key(panel)));
        for (std::uint64_t panel = 0; panel < 6; ++panel)
            hpx::post<panel_init_action>(arts_hpx::blind_target(), panel);
        hpx::wait_all(ready);    // hpx_main may wait
    }
    // Every patch exists somewhere: publish who created what, once.
    hpx::distributed::barrier::synchronize();
    auto comm = hpx::collectives::create_communicator("/arts/tempest/map",
        hpx::collectives::num_sites_arg(hpx::get_initial_num_localities()),
        hpx::collectives::this_site_arg(hpx::get_locality_id()));
    std::vector<std::vector<std::uint64_t>> const lists =
        hpx::collectives::all_gather(comm, reg.created).get();
    for (std::uint32_t loc = 0; loc < lists.size(); ++loc)
        for (std::uint64_t P : lists[loc])
            reg.creator[P] = loc;
    if (hpx::get_locality_id() == 0)
        counters().map_exchange.store(1);
    reg.done = std::make_unique<hpx::latch>(static_cast<std::ptrdiff_t>(reg.created.size()));
    // A generation is placed blindly, so the first one may run on any
    // locality and it reads both the map and the completion latch there.
    // Gathering the lists only makes each locality's own copy available;
    // every locality must have installed it before any generation exists,
    // or one that arrives early reads the pre-exchange default and asks the
    // wrong locality for inputs that are never delivered to it.
    hpx::distributed::barrier::synchronize();
    for (std::uint64_t P : reg.created)
        launch(tp::make_state(P, r.k, r.duration));
#endif
    reg.done->wait();
    hpx::distributed::barrier::synchronize();

    if (hpx::get_locality_id() == 0)
    {
        arts_hpx::print_e2e(clock);
        std::vector<std::int64_t> const received = result.get();
        tp::patch_state const s0 = tp::make_state(tp::TEST_PATCH, r.k, r.duration);
        std::int64_t grid[9];
        for (std::uint32_t i = 0; i < 8; ++i) grid[i] = s0.nbr[i];
        grid[8] = static_cast<std::int64_t>(tp::TEST_PATCH);
        // A reader of the merged stream parses the marker and the grid that
        // follows it as one record, and the launcher interleaves the streams
        // it merges at write granularity: a record split across writes can
        // take another descriptor's line in the middle of it.  So the whole
        // record is built first and issued as one write, which is why the
        // result is resolved before any of it is formatted.
        std::string out;
        append_grid(out, grid);
        out += "\n*CROSS-CHECKING NEIGHBOR DATA EXCHANGE*\n\n";
        for (std::uint32_t i = 0; i < 8; ++i) grid[i] = received[i];
        append_grid(out, grid);
        out += "DONE.\n";
        arts_hpx::write_stdout_line(out);
    }
    if (structural)
    {
        auto const& c = counters();
        arts_hpx::print_struct({{"generations", c.generations.load()}, {"deliveries", c.deliveries.load()},
            {"parcels", c.parcels.load()}, {"bytes", c.bytes.load()}, {"interests", c.interests.load()},
            {"returns", c.returns.load()}, {"state_posts", c.state_posts.load()},
            {"state_bytes", c.state_bytes.load()}, {"map_exchange", c.map_exchange.load()},
            {"inits", c.inits.load()}});
        arts_hpx::print_parcels();
    }
    hub<std::int64_t>::drain_and_report();
    hub<std::vector<std::int64_t>>::drain_and_report();
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    hpx::init_params params;
    params.cfg = arts_hpx::runtime_defaults();
    return hpx::init(argc, argv, params);
}
