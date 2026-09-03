#pragma once

#include <hpx/hpx.hpp>
#include <hpx/include/lcos_local.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>

namespace arts_hpx {

// The per-locality mailbox every remote edge lands in: K independent
// receive buffers, a key's shard chosen by a mix of all its fields so no
// lock is global — the low field alone (a child index, usually 0) would
// fold every single-consumer edge onto a handful of shards.  An entry is
// created by whichever of receive/deliver arrives first and erased once
// both have touched it, so live state tracks outstanding rendezvous only.
// The singleton is leaked on purpose: a static destructor would run after
// the runtime is gone, and the buffers assert emptiness at destruction —
// drain() from hpx_main is the teardown.
template <typename T>
class hub
{
public:
    static constexpr std::size_t shard_count = 1024;

    static hub& instance()
    {
        static hub* const h = new hub;
        return *h;
    }

    hpx::future<T> receive(std::uint64_t key)
    {
        return shard(key).receive(static_cast<std::size_t>(key));
    }

    void deliver(std::uint64_t key, T value)
    {
        shard(key).store_received(static_cast<std::size_t>(key), std::move(value));
    }

    std::size_t drain()
    {
        std::size_t dropped = 0;
        auto const why = std::make_exception_ptr(
            std::runtime_error("hub drained at shutdown"));
        for (auto& s : shards_)
            dropped += s.cancel_waiting(why, true);
        return dropped;
    }

    // An entry still waiting at shutdown is an edge nobody satisfied — a
    // defect in the program's dependency graph, so it is reported rather
    // than swallowed by the teardown that discards it.
    static std::size_t drain_and_report()
    {
        std::size_t const dropped = instance().drain();
        if (dropped != 0)
        {
            std::string const line = "hub: " + std::to_string(dropped) +
                " rendezvous entries dropped at shutdown\n";
            ssize_t const written = ::write(2, line.data(), line.size());
            (void) written;
        }
        return dropped;
    }

private:
    hub() = default;

    hpx::lcos::local::receive_buffer<T>& shard(std::uint64_t key)
    {
        return shards_[(key ^ (key >> 8) ^ (key >> 34) ^ (key >> 56)) %
            shard_count];
    }

    std::array<hpx::lcos::local::receive_buffer<T>, shard_count> shards_;
};

}    // namespace arts_hpx

// A plain action registers non-inline symbols, so the delivery action is
// defined once per program, in the port's translation unit, per payload
// type.  Action names need be unique only within one program.
#define ARTS_HPX_DEFINE_HUB(T, tag)                                            \
    void tag##_deliver(std::uint64_t key, T value)                             \
    {                                                                          \
        ::arts_hpx::hub<T>::instance().deliver(key, std::move(value));         \
    }                                                                          \
    HPX_PLAIN_ACTION(tag##_deliver, tag##_deliver_action)
