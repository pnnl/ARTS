#pragma once

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <unistd.h>

namespace arts_hpx {

// A key names exactly one edge, so a field that does not fit is fatal rather
// than truncated: two edges sharing a key deliver one subtree's value to the
// other and the program silently computes a wrong answer.  The check is a
// branch and a write, not an assertion macro — the optimised build, the only
// one that runs long enough to exhaust a field, is where the macro is gone.
[[noreturn]] inline void key_bound_violated(std::string const& what)
{
    std::string const line = "arts_hpx: " + what + "\n";
    ssize_t const written = ::write(2, line.data(), line.size());
    (void) written;
    std::abort();
}

// A key for an edge whose consumer is a task the producer's creator placed:
// the creator's locality and its own sequence number make the key unique
// program-wide, the child index distinguishes siblings.  The widths — 256
// localities, 2^48 creations per locality, 256 children per node — are
// enforced here and at the locality table, never assumed.
inline std::uint64_t pack_key(
    std::uint32_t locality, std::uint64_t sequence, std::uint32_t index)
{
    if (index > 0xffu)
        key_bound_violated("child index " + std::to_string(index) +
            " exceeds the 256 a rendezvous key can name");
    if ((sequence >> 48) != 0)
        key_bound_violated("sequence " + std::to_string(sequence) +
            " exceeds the 2^48 a rendezvous key can name");
    return (static_cast<std::uint64_t>(locality & 0xffu) << 56) |
        ((sequence & 0xffffffffffffull) << 8) |
        static_cast<std::uint64_t>(index & 0xffu);
}

inline std::uint64_t next_sequence()
{
    static std::atomic<std::uint64_t> counter{0};
    return counter.fetch_add(1, std::memory_order_relaxed);
}

}    // namespace arts_hpx
