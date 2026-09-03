#pragma once

#include <cstdint>

namespace arts_hpx::tempest {

enum dir : std::uint32_t { N = 0, E = 1, S = 2, W = 3, NE = 4, SE = 5, SW = 6, NW = 7 };

constexpr std::uint64_t TEST_PATCH = 0;
constexpr std::uint64_t DURATION_DEFAULT = 100;
constexpr std::uint64_t PATCH_RANGE_DEFAULT = 2;

inline std::uint64_t local_num(std::uint64_t P, std::uint64_t k) { return P % (k * k); }
inline std::uint64_t panel_num(std::uint64_t P, std::uint64_t k) { return (P - local_num(P, k)) / (k * k); }
inline std::uint64_t find_x(std::uint64_t local, std::uint64_t k) { return local / k; }
inline std::uint64_t find_y(std::uint64_t local, std::uint64_t k) { return local % k; }

//  N, E, S, W neighbour panel of each of the six panels
constexpr std::uint32_t panel_neighbors[6][4] = {
    {4, 1, 5, 3}, {4, 2, 5, 0}, {4, 3, 5, 1}, {4, 0, 5, 2}, {3, 2, 1, 0}, {1, 2, 3, 0}};

inline std::uint64_t neighbor_panel(std::uint64_t panel, std::uint32_t d) { return panel_neighbors[panel][d]; }

// The side of neighPanel that faces myPanel.
inline std::uint32_t neighbor_dir(std::uint64_t myPanel, std::uint64_t neighPanel)
{
    std::uint32_t d = 0;
    for (; d < 4; ++d)
        if (panel_neighbors[neighPanel][d] == myPanel)
            break;
    return d;
}

// The neighbour of P in direction d, or -1 off the board (the eight
// panel-corner directions).  A verbatim transcription of the original's
// case analysis; the core test holds it to symmetry and to the edge count.
inline std::int64_t find_neighbor_patch(std::uint64_t P, std::uint64_t k, std::uint32_t d)
{
    std::uint64_t const panel = panel_num(P, k), local = local_num(P, k);
    std::uint64_t const x = find_x(local, k), y = find_y(local, k);
    auto across = [&](std::uint32_t side, std::uint64_t nx, std::uint64_t ny) -> std::int64_t {
        std::uint64_t const np = neighbor_panel(panel, side);
        return static_cast<std::int64_t>(np * (k * k) + k * nx + ny);
    };
    auto face = [&](std::uint32_t side) { return neighbor_dir(panel, neighbor_panel(panel, side)); };
    switch (d)
    {
    case N:
        if (y == k - 1)
            switch (face(N))
            {
            case N: return across(N, (k - 1) - x, k - 1);
            case E: return across(N, k - 1, x);
            case S: return across(N, x, 0);
            case W: return across(N, 0, (k - 1) - x);
            default: return -1;
            }
        return static_cast<std::int64_t>(P + 1);
    case E:
        if (x == k - 1)
            switch (face(E))
            {
            case N: return across(E, y, k - 1);
            case S: return across(E, (k - 1) - y, 0);
            case W: return across(E, 0, y);
            default: return -1;
            }
        return static_cast<std::int64_t>(P + k);
    case S:
        if (y == 0)
            switch (face(S))
            {
            case N: return across(S, x, k - 1);
            case E: return across(S, k - 1, (k - 1) - x);
            case S: return across(S, (k - 1) - x, 0);
            case W: return across(S, 0, x);
            default: return -1;
            }
        return static_cast<std::int64_t>(P - 1);
    case W:
        if (x == 0)
            switch (face(W))
            {
            case N: return across(W, (k - 1) - y, k - 1);
            case E: return across(W, k - 1, y);
            case S: return across(W, y, 0);
            default: return -1;
            }
        return static_cast<std::int64_t>(P - k);
    case NE:
        if (x == k - 1 && y == k - 1) return -1;
        if (x == k - 1)
            switch (face(E))
            {
            case N: return across(E, y + 1, k - 1);
            case S: return across(E, ((k - 1) - y) - 1, 0);
            case W: return across(E, 0, y + 1);
            default: return -1;
            }
        if (y == k - 1)
            switch (face(N))
            {
            case N: return across(N, ((k - 1) - x) - 1, k - 1);
            case E: return across(N, k - 1, x + 1);
            case S: return across(N, x + 1, 0);
            case W: return across(N, 0, ((k - 1) - x) - 1);
            default: return -1;
            }
        return static_cast<std::int64_t>(P + k + 1);
    case SE:
        if (x == k - 1 && y == 0) return -1;
        if (x == k - 1)
            switch (face(E))
            {
            case N: return across(E, y - 1, k - 1);
            case S: return across(E, ((k - 1) - y) + 1, 0);
            case W: return across(E, 0, y - 1);
            default: return -1;
            }
        if (y == 0)
            switch (face(S))
            {
            case N: return across(S, x + 1, k - 1);
            case E: return across(S, k - 1, ((k - 1) - x) - 1);
            case S: return across(S, ((k - 1) - x) - 1, 0);
            case W: return across(S, 0, x + 1);
            default: return -1;
            }
        return static_cast<std::int64_t>(P + k - 1);
    case SW:
        if (x == 0 && y == 0) return -1;
        if (x == 0)
            switch (face(W))
            {
            case N: return across(W, k - y, k - 1);
            case E: return across(W, k - 1, y - 1);
            case S: return across(W, y - 1, 0);
            default: return -1;
            }
        if (y == 0)
            switch (face(S))
            {
            case N: return across(S, x - 1, k - 1);
            case E: return across(S, k - 1, k - x);
            case S: return across(S, k - x, 0);
            case W: return across(S, 0, x - 1);
            default: return -1;
            }
        return static_cast<std::int64_t>(P - k - 1);
    case NW:
        if (x == 0 && y == k - 1) return -1;
        if (x == 0)
            switch (face(W))
            {
            case N: return across(W, k - (y + 2), k - 1);
            case E: return across(W, k - 1, y + 1);
            case S: return across(W, y + 1, 0);
            default: return -1;
            }
        if (y == k - 1)
            switch (face(N))
            {
            case N: return across(N, k - x, k - 1);
            case E: return across(N, k - 1, x - 1);
            case S: return across(N, x - 1, 0);
            case W: return across(N, 0, k - x);
            default: return -1;
            }
        return static_cast<std::int64_t>(P - k + 1);
    default:
        return -1;
    }
}

// The direction from Q in which P lies: the slot P's deliveries fill at Q.
inline std::uint32_t neighbor_relation(std::uint64_t P, std::uint64_t Q, std::uint64_t k)
{
    std::uint32_t rel = 0;
    for (; rel < 8; ++rel)
        if (find_neighbor_patch(Q, k, rel) == static_cast<std::int64_t>(P))
            break;
    return rel;
}

// Below six places a patch-number band; from six up a P x Q cut of the
// k x 6k strip minimising the total cut length.
inline std::uint64_t home_rank(std::uint64_t P, std::uint64_t k, std::uint64_t nranks)
{
    if (nranks < 6)
        return (P * nranks) / (6 * k * k);
    std::uint64_t Pn = 1, best = static_cast<std::uint64_t>(-1);
    for (std::uint64_t p = 1; p <= nranks; ++p)
    {
        if (nranks % p)
            continue;
        std::uint64_t const q = nranks / p;
        std::uint64_t const cut = (p - 1) * 6 * k + (q - 1) * k;
        if (cut < best)
        {
            best = cut;
            Pn = p;
        }
    }
    std::uint64_t const Q = nranks / Pn;
    std::uint64_t const face = P / (k * k), idx = P % (k * k);
    std::uint64_t const row = idx / k, gcol = face * k + (idx % k);
    std::uint64_t const br = (row * Pn) / k, bc = (gcol * Q) / (6 * k);
    return br * Q + bc;
}

inline std::uint64_t edge_count(std::uint64_t k)
{
    std::uint64_t n = 0;
    for (std::uint64_t P = 0; P < 6 * k * k; ++P)
        for (std::uint32_t d = 0; d < 8; ++d)
            if (find_neighbor_patch(P, k, d) >= 0)
                ++n;
    return n;
}

// The application's own per-patch data; what a base-tier generation carries.
struct patch_state
{
    std::uint64_t patchNum = 0, k = 0, duration = 0, timestep = 0;
    std::int64_t nbr[8] = {-1, -1, -1, -1, -1, -1, -1, -1};

    template <typename Archive>
    void serialize(Archive& ar, unsigned)
    {
        ar & patchNum & k & duration & timestep;
        for (std::uint32_t i = 0; i != 8; ++i)
            ar & nbr[i];
    }
};

inline patch_state make_state(std::uint64_t P, std::uint64_t k, std::uint64_t duration)
{
    patch_state s;
    s.patchNum = P; s.k = k; s.duration = duration; s.timestep = 0;
    for (std::uint32_t d = 0; d < 8; ++d)
        s.nbr[d] = find_neighbor_patch(P, k, d);
    return s;
}

}    // namespace arts_hpx::tempest
