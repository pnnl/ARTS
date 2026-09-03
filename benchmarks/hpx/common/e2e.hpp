#pragma once

#include <hpx/hpx.hpp>
#include <hpx/include/performance_counters.hpp>
#include <hpx/modules/collectives.hpp>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace arts_hpx {

// The measured span opens BEFORE the start barrier, not after it: no task can
// run until every locality has arrived, so that barrier is part of what the
// program costs, and including it is what makes the span comparable with a
// runtime whose own stamp precedes its start barriers.
struct run_clock
{
    std::chrono::steady_clock::time_point start =
        std::chrono::steady_clock::now();
};

inline bool e2e_enabled()
{
    static bool const on = std::getenv("ARTS_E2E_MARKER") != nullptr;
    return on;
}

// The structural pass is decided by locality 0 and broadcast, so a variable
// that did not reach every rank fails as a mismatch instead of desynchronizing
// the collectives that follow.  The decision is collective, so it must be
// made once on every locality during setup, before any task exists; every
// later call — including the ones a task makes to decide whether to count —
// only reads the cached answer.
inline bool struct_enabled()
{
    static bool const on = [] {
        auto comm = hpx::collectives::create_communicator("/arts/common/struct",
            hpx::collectives::num_sites_arg(hpx::get_initial_num_localities()),
            hpx::collectives::this_site_arg(hpx::get_locality_id()));
        int flag = std::getenv("ARTS_STRUCT_MARKER") != nullptr ? 1 : 0;
        if (hpx::get_locality_id() == 0)
            return hpx::collectives::broadcast_to(comm, flag).get() == 1;
        return hpx::collectives::broadcast_from<int>(comm).get() == 1;
    }();
    return on;
}

// One write per line: the launcher merges every locality's stream into one
// and the driver parses whole lines, so a line split across two writes can
// be interleaved with another stream's line.  A short or interrupted write
// is resumed rather than dropped — a truncated line is a lost record, not a
// cosmetic defect.
inline void write_fd_line(int fd, std::string const& line)
{
    char const* cursor = line.data();
    std::size_t left = line.size();
    while (left != 0)
    {
        ssize_t const n = ::write(fd, cursor, left);
        if (n > 0)
        {
            cursor += n;
            left -= static_cast<std::size_t>(n);
        }
        else if (n < 0 && errno == EINTR)
        {
            continue;
        }
        else
        {
            return;    // the stream is gone; there is nowhere left to report
        }
    }
}

inline void write_line(std::string const& line)
{
    write_fd_line(2, line);
}

inline void write_stdout_line(std::string const& line)
{
    write_fd_line(1, line);
}

inline void print_geometry()
{
    if (!e2e_enabled())
        return;
    write_line("[HPX] locality=" + std::to_string(hpx::get_locality_id()) +
        " localities=" + std::to_string(hpx::get_initial_num_localities()) +
        " threads=" + std::to_string(hpx::get_os_thread_count()) + "\n");
}

inline void print_e2e(run_clock const& clock)
{
    if (!e2e_enabled() || hpx::get_locality_id() != 0)
        return;
    auto const ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - clock.start).count();
    write_line("[E2E] " + std::to_string(ns) + "\n");
}

// A reduction operator travels to the other sites as an action argument, so
// it must be default-constructible: the receiving side default-constructs
// the argument before loading it.  A closure type is not, whatever it
// captures, so the operator is a named empty struct.
struct add_elementwise
{
    std::vector<std::uint64_t> operator()(
        std::vector<std::uint64_t> a, std::vector<std::uint64_t> const& b) const
    {
        std::size_t const n = std::min(a.size(), b.size());
        for (std::size_t i = 0; i != n; ++i)
            a[i] += b[i];
        return a;
    }
};

// Sums the port's own counters over localities and prints them from
// locality 0.  Every locality calls it (it is a collective), after the end
// stamp, only when struct_enabled().  It registers its communicator's
// basename as it goes, and a basename is registered once, so one program
// calls it at most once.
inline void print_struct(std::vector<std::pair<char const*, std::uint64_t>> const& fields)
{
    std::vector<std::uint64_t> mine;
    for (auto const& f : fields)
        mine.push_back(f.second);
    auto comm = hpx::collectives::create_communicator("/arts/common/struct_sum",
        hpx::collectives::num_sites_arg(hpx::get_initial_num_localities()),
        hpx::collectives::this_site_arg(hpx::get_locality_id()));
    std::vector<std::uint64_t> total =
        hpx::collectives::all_reduce(comm, mine, add_elementwise{}).get();
    if (hpx::get_locality_id() != 0)
        return;
    std::string line = "[STRUCT]";
    for (std::size_t i = 0; i != fields.size(); ++i)
        line += std::string(" ") + fields[i].first + "=" + std::to_string(total[i]);
    write_line(line + "\n");
}

// The runtime's own parcel counters, summed over localities: an upper bound
// on the wire traffic (no coalescing is configured), printed once, after the
// end stamp, when the structural pass is on.  `bytes` is the argument data a
// parcel carried, `wire` the serialized parcel including its headers — the
// second is what a message census on another runtime's transport counts.
// A single locality never instantiates a parcelport, so its counters are not
// registered and the traffic they would measure is zero.
inline void print_parcels()
{
    std::vector<std::uint64_t> mine{0, 0, 0};
    if (hpx::get_initial_num_localities() > 1)
    {
        hpx::performance_counters::performance_counter sent(
            "/parcels/count/mpi/sent");
        hpx::performance_counters::performance_counter data(
            "/data/count/mpi/sent");
        hpx::performance_counters::performance_counter wire(
            "/serialize/count/mpi/sent");
        mine = {static_cast<std::uint64_t>(
                    sent.get_value<std::int64_t>(hpx::launch::sync)),
            static_cast<std::uint64_t>(
                data.get_value<std::int64_t>(hpx::launch::sync)),
            static_cast<std::uint64_t>(
                wire.get_value<std::int64_t>(hpx::launch::sync))};
    }
    auto comm = hpx::collectives::create_communicator("/arts/common/parcels",
        hpx::collectives::num_sites_arg(hpx::get_initial_num_localities()),
        hpx::collectives::this_site_arg(hpx::get_locality_id()));
    std::vector<std::uint64_t> const total =
        hpx::collectives::all_reduce(comm, mine, add_elementwise{}).get();
    if (hpx::get_locality_id() != 0)
        return;
    write_line("[PARCELS] sent=" + std::to_string(total[0]) +
        " bytes=" + std::to_string(total[1]) +
        " wire=" + std::to_string(total[2]) + "\n");
}

}    // namespace arts_hpx
