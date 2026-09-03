#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace arts_hpx::stencil2d {

constexpr std::int64_t R = 2;    // halo radius; the weights below are radius-specific
constexpr double EPSILON = 1e-8;

struct grid
{
    std::int64_t NP = 0, NR = 0, NT = 0, NR_X = 0, NR_Y = 0;
};

// The largest divisor of n not above floor(sqrt(n + 1)), and its cofactor.
inline void split_cart2d(std::int64_t n, std::int64_t& nx, std::int64_t& ny)
{
    nx = static_cast<std::int64_t>(std::sqrt(static_cast<double>(n + 1)));
    for (; nx > 0; --nx)
        if (n % nx == 0)
            break;
    ny = n / nx;
}

inline grid make_grid(std::int64_t NP, std::int64_t NR, std::int64_t NT)
{
    grid g;
    g.NP = NP; g.NR = NR; g.NT = NT;
    split_cart2d(NR, g.NR_X, g.NR_Y);
    return g;
}

inline void partition_bounds(std::int64_t id, std::int64_t lb, std::int64_t ub,
    std::int64_t parts, std::int64_t& s, std::int64_t& e)
{
    std::int64_t const N = ub - lb + 1;
    s = id * N / parts + lb;
    e = (id + 1) * N / parts + lb - 1;
}

inline std::int64_t partition_id(std::int64_t i, std::int64_t lb, std::int64_t ub, std::int64_t parts)
{
    std::int64_t const N = ub - lb + 1;
    for (std::int64_t r = 0; r < parts; ++r)
    {
        std::int64_t const s = r * N / parts + lb;
        std::int64_t const e = (r + 1) * N / parts + lb - 1;
        if (s <= i && i <= e)
            return r;
    }
    return parts;
}

struct tile
{
    std::int64_t id = 0, id_x = 0, id_y = 0, ib = 0, ie = 0, jb = 0, je = 0, np_x = 0, np_y = 0;
    std::int64_t round = 0;
    std::vector<double> in, out;
    double weight[(2 * R + 1) * (2 * R + 1)] = {};
    std::chrono::steady_clock::time_point start{};
    double elapsed = 0.0;

    double& IN(std::int64_t i, std::int64_t j)
    {
        return in[static_cast<std::size_t>((i - ib + R) + (j - jb + R) * (np_x + 2 * R))];
    }
    double& OUT(std::int64_t i, std::int64_t j)
    {
        return out[static_cast<std::size_t>((i - ib) + (j - jb) * np_x)];
    }
    double& W(std::int64_t ii, std::int64_t jj)
    {
        return weight[jj + R + (ii + R) * (2 * R + 1)];
    }
};

inline tile make_tile(grid const& g, std::int64_t id)
{
    tile t;
    t.id = id;
    t.id_x = id % g.NR_X;
    t.id_y = id / g.NR_X;
    partition_bounds(t.id_x, 0, g.NP - 1, g.NR_X, t.ib, t.ie);
    partition_bounds(t.id_y, 0, g.NP - 1, g.NR_Y, t.jb, t.je);
    t.np_x = t.ie - t.ib + 1;
    t.np_y = t.je - t.jb + 1;
    t.in.assign(static_cast<std::size_t>((t.np_x + 2 * R) * (t.np_y + 2 * R)), 0.0);
    t.out.assign(static_cast<std::size_t>(t.np_x * t.np_y), 0.0);
    return t;
}

// The program's own map: the tile grid is cut into a PD_X x PD_Y grid of
// places by the same partition rule as the points.
inline std::int64_t place_of(grid const& g, std::int64_t id, std::int64_t PD_X, std::int64_t PD_Y)
{
    std::int64_t const id_x = id % g.NR_X, id_y = id / g.NR_X;
    std::int64_t const pd_x = partition_id(id_x, 0, g.NR_X - 1, PD_X);
    std::int64_t const pd_y = partition_id(id_y, 0, g.NR_Y - 1, PD_Y);
    return PD_X * pd_y + pd_x;
}

inline void initialize(tile& t)
{
    for (std::int64_t j = t.jb; j <= t.je; ++j)
        for (std::int64_t i = t.ib; i <= t.ie; ++i)
        {
            t.IN(i, j) = static_cast<double>(i) + static_cast<double>(j);
            t.OUT(i, j) = 0.0;
        }
    for (std::int64_t jj = -R; jj <= R; ++jj)
        for (std::int64_t ii = -R; ii <= R; ++ii)
            t.W(ii, jj) = 0.0;
    for (std::int64_t ii = 1; ii <= R; ++ii)
    {
        t.W(0, ii) = t.W(ii, 0) = 1.0 / (2.0 * ii * R);
        t.W(0, -ii) = t.W(-ii, 0) = -1.0 / (2.0 * ii * R);
    }
}

// Strips are read from IN as it stands before the round's update, in the
// (j outer, i inner) order the receiving side writes them back.
inline std::vector<double> left_strip(tile& t)
{
    std::vector<double> s;
    s.reserve(static_cast<std::size_t>(R * t.np_y));
    for (std::int64_t j = t.jb; j <= t.je; ++j)
        for (std::int64_t i = t.ib; i < t.ib + R; ++i)
            s.push_back(t.IN(i, j));
    return s;
}
inline std::vector<double> right_strip(tile& t)
{
    std::vector<double> s;
    s.reserve(static_cast<std::size_t>(R * t.np_y));
    for (std::int64_t j = t.jb; j <= t.je; ++j)
        for (std::int64_t i = t.ie - R + 1; i <= t.ie; ++i)
            s.push_back(t.IN(i, j));
    return s;
}
inline std::vector<double> bottom_strip(tile& t)
{
    std::vector<double> s;
    s.reserve(static_cast<std::size_t>(R * t.np_x));
    for (std::int64_t j = t.jb; j < t.jb + R; ++j)
        for (std::int64_t i = t.ib; i <= t.ie; ++i)
            s.push_back(t.IN(i, j));
    return s;
}
inline std::vector<double> top_strip(tile& t)
{
    std::vector<double> s;
    s.reserve(static_cast<std::size_t>(R * t.np_x));
    for (std::int64_t j = t.je - R + 1; j <= t.je; ++j)
        for (std::int64_t i = t.ib; i <= t.ie; ++i)
            s.push_back(t.IN(i, j));
    return s;
}
// What the LEFT neighbour sent (its right strip) fills the halo left of ib.
inline void apply_from_left(tile& t, std::vector<double> const& s)
{
    std::size_t k = 0;
    for (std::int64_t j = t.jb; j <= t.je; ++j)
        for (std::int64_t i = t.ib - R; i < t.ib; ++i)
            t.IN(i, j) = s[k++];
}
inline void apply_from_right(tile& t, std::vector<double> const& s)
{
    std::size_t k = 0;
    for (std::int64_t j = t.jb; j <= t.je; ++j)
        for (std::int64_t i = t.ie + 1; i <= t.ie + R; ++i)
            t.IN(i, j) = s[k++];
}
inline void apply_from_below(tile& t, std::vector<double> const& s)
{
    std::size_t k = 0;
    for (std::int64_t j = t.jb - R; j < t.jb; ++j)
        for (std::int64_t i = t.ib; i <= t.ie; ++i)
            t.IN(i, j) = s[k++];
}
inline void apply_from_above(tile& t, std::vector<double> const& s)
{
    std::size_t k = 0;
    for (std::int64_t j = t.je + 1; j <= t.je + R; ++j)
        for (std::int64_t i = t.ib; i <= t.ie; ++i)
            t.IN(i, j) = s[k++];
}

inline void update(tile& t, grid const& g)
{
    for (std::int64_t j = std::max(t.jb, R); j <= std::min(g.NP - R - 1, t.je); ++j)
        for (std::int64_t i = std::max(t.ib, R); i <= std::min(g.NP - R - 1, t.ie); ++i)
        {
            for (std::int64_t jj = -R; jj <= R; ++jj)
                t.OUT(i, j) += t.W(0, jj) * t.IN(i, j + jj);
            for (std::int64_t ii = -R; ii < 0; ++ii)
                t.OUT(i, j) += t.W(ii, 0) * t.IN(i + ii, j);
            for (std::int64_t ii = 1; ii <= R; ++ii)
                t.OUT(i, j) += t.W(ii, 0) * t.IN(i + ii, j);
        }
    for (std::int64_t j = t.jb; j <= t.je; ++j)
        for (std::int64_t i = t.ib; i <= t.ie; ++i)
            t.IN(i, j) += 1.0;
}

inline double final_norm(tile& t, grid const& g)
{
    double norm = 0.0;
    for (std::int64_t j = std::max(t.jb, R); j <= std::min(g.NP - R - 1, t.je); ++j)
        for (std::int64_t i = std::max(t.ib, R); i <= std::min(g.NP - R - 1, t.ie); ++i)
            norm += std::abs(t.OUT(i, j));
    return norm;
}

inline double reference_norm(grid const& g) { return static_cast<double>(g.NT + 1) * 2; }
inline double active_points(grid const& g)
{
    return static_cast<double>((g.NP - 2 * R) * (g.NP - 2 * R));
}

// Structural closed forms: one task per tile per round; strips per round are
// the directed interior edges; bytes assume the uniform tiles of a grid the
// tile counts divide.
inline std::uint64_t expected_tasks(grid const& g) { return g.NR * (g.NT + 1); }
inline std::uint64_t expected_strips(grid const& g)
{
    return static_cast<std::uint64_t>(4 * g.NR - 2 * (g.NR_X + g.NR_Y)) * (g.NT + 1);
}
inline std::uint64_t expected_bytes(grid const& g)
{
    std::uint64_t const lr = 8 * R * (g.NP / g.NR_Y), bt = 8 * R * (g.NP / g.NR_X);
    return (2 * (g.NR_X - 1) * g.NR_Y * lr + 2 * g.NR_X * (g.NR_Y - 1) * bt) * (g.NT + 1);
}

// The whole grid in one thread: every tile posts, every tile receives, every
// tile updates, round after round.  Returns the normalised L1 norm.
inline double simulate(grid const& g)
{
    std::vector<tile> tiles;
    for (std::int64_t id = 0; id != g.NR; ++id)
    {
        tiles.push_back(make_tile(g, id));
        initialize(tiles.back());
    }
    for (std::int64_t r = 0; r <= g.NT; ++r)
    {
        std::vector<std::vector<double>> fl(g.NR), fr(g.NR), fb(g.NR), ft(g.NR);
        for (auto& t : tiles)
        {
            if (t.id_x != 0)          fr[t.id - 1] = left_strip(t);      // my left strip -> left neighbour's "from right" slot
            if (t.id_x != g.NR_X - 1) fl[t.id + 1] = right_strip(t);     // my right strip -> right neighbour's "from left" slot
            if (t.id_y != 0)          ft[t.id - g.NR_X] = bottom_strip(t); // my bottom strip -> below neighbour's "from above" slot
            if (t.id_y != g.NR_Y - 1) fb[t.id + g.NR_X] = top_strip(t);    // my top strip -> above neighbour's "from below" slot
        }
        for (auto& t : tiles)
        {
            if (t.id_x != 0)          apply_from_left(t, fl[t.id]);
            if (t.id_x != g.NR_X - 1) apply_from_right(t, fr[t.id]);
            if (t.id_y != 0)          apply_from_below(t, fb[t.id]);
            if (t.id_y != g.NR_Y - 1) apply_from_above(t, ft[t.id]);
            update(t, g);
        }
    }
    double norm = 0.0;
    for (auto& t : tiles)
        norm += final_norm(t, g);
    return norm / active_points(g);
}

}    // namespace arts_hpx::stencil2d
