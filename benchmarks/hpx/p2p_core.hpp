#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace arts_hpx::p2p {

struct params
{
    std::uint32_t p = 10, m = 100, n = 1000, t = 100, gf = 1;
};

struct rank_geometry
{
    std::uint32_t k = 0;         // columns this rank owns
    std::uint32_t myfirst = 0;   // global index of its first column
    std::uint32_t w = 0;         // phases per timestep
};

// The right-column strip one phase hands to the next rank: numrows + 1
// values.  The wire carries count values, not the capacity.
struct boundary
{
    static constexpr std::uint32_t capacity = 8;
    std::uint8_t count = 0;
    double v[capacity] = {};

    template <typename Archive>
    void serialize(Archive& ar, unsigned)
    {
        ar & count;
        for (std::uint32_t i = 0; i != count; ++i)
            ar & v[i];
    }
};

inline rank_geometry geometry(params const& a, std::uint32_t rank)
{
    rank_geometry g;
    g.k = a.m / a.p;
    g.myfirst = g.k * rank;
    if (rank != 0 && rank <= a.m % a.p)
        ++g.k;
    for (std::uint32_t i = 1; i < rank; ++i)
        if (i - 1 < a.m % a.p)
            ++g.myfirst;
    g.w = (a.n - 1 + a.gf - 1) / a.gf;
    return g;
}

inline std::uint32_t block_size(std::uint32_t p, std::uint32_t places)
{
    return (p + places - 1) / places;
}

inline std::uint32_t rank_locality(std::uint32_t rank, std::uint32_t p, std::uint32_t places)
{
    return rank / block_size(p, places);
}

inline std::uint32_t rows_this_phase(params const& a, std::uint32_t row)
{
    std::uint32_t numrows = a.gf;
    if (a.n - row <= a.gf)
        numrows = a.n - row - 1;
    return numrows;
}

// data is n rows by k columns, row-major.  in has numrows + 1 entries: the
// left neighbour's right column for rows row .. row + numrows.
inline void compute(std::vector<double>& data, std::uint32_t k, std::uint32_t rank,
    std::uint32_t row, std::uint32_t numrows, double const* in)
{
    auto A = [&](std::uint32_t i, std::uint32_t j) -> double& {
        return data[static_cast<std::size_t>(i) * k + j];
    };
    if (rank != 0)
        for (std::uint32_t i = row + 1; i <= row + numrows; ++i)
            A(i, 0) = A(i - 1, 0) + in[i - row] - in[i - row - 1];
    for (std::uint32_t j = 1; j < k; ++j)
        for (std::uint32_t i = row + 1; i <= row + numrows; ++i)
            A(i, j) = A(i - 1, j) + A(i, j - 1) - A(i - 1, j - 1);
}

inline void initialize(std::vector<double>& data, params const& a,
    rank_geometry const& g, std::uint32_t rank)
{
    data.assign(static_cast<std::size_t>(a.n) * g.k, 0.0);
    for (std::uint32_t i = 0; i < g.k; ++i)
        data[i] = static_cast<double>(g.myfirst + i);
    if (rank == 0)
        for (std::uint32_t j = 1; j < a.n; ++j)
            data[static_cast<std::size_t>(j) * g.k] = static_cast<double>(j);
}

inline double expected_checksum(params const& a)
{
    return static_cast<double>((a.t + 1) * (a.n + a.m - 2));
}

// Closed forms of the port's structural counters.
inline std::uint64_t generations_per_rank(params const& a)
{
    return static_cast<std::uint64_t>(a.t + 1) * geometry(a, 0).w;
}
inline std::uint64_t expected_tasks(params const& a) { return a.p * generations_per_rank(a); }
inline std::uint64_t expected_sends(params const& a) { return (a.p - 1) * generations_per_rank(a) + a.t; }
// A boundary carries numrows + 1 values and a timestep's phases advance
// exactly n - 1 rows however gf divides them, so one rank sends
// (n - 1) + w values per timestep whether or not the last phase is short.
inline std::uint64_t expected_bytes(params const& a)
{
    std::uint64_t const per_timestep = (a.n - 1) + geometry(a, 0).w;
    return 8ull * (a.p - 1) * (a.t + 1) * per_timestep + 8ull * a.t;
}

// The whole pipeline in one thread, generation-major then rank-major: rank r's
// generation g needs rank r-1's generation g (produced earlier in the same
// sweep) and rank p-1's previous-timestep wrap (produced in the previous
// sweep).  Returns rank p-1's checksum.
inline double simulate(params const& a)
{
    std::vector<rank_geometry> geo(a.p);
    std::vector<std::vector<double>> data(a.p);
    std::vector<std::uint32_t> timestep(a.p, 0), row(a.p, 0), phase(a.p, 0);
    for (std::uint32_t r = 0; r != a.p; ++r)
    {
        geo[r] = geometry(a, r);
        initialize(data[r], a, geo[r], r);
    }
    std::uint32_t const w = geo[0].w;
    std::uint64_t const G = generations_per_rank(a);
    boundary wrap;    // rank p-1 -> rank 0, consumed at the next timestep's phase 0
    double checksum = 0.0;
    for (std::uint64_t g = 0; g != G; ++g)
    {
        boundary left;    // rank r-1 -> rank r within this sweep
        for (std::uint32_t r = 0; r != a.p; ++r)
        {
            std::uint32_t const k = geo[r].k;
            auto A = [&](std::uint32_t i, std::uint32_t j) -> double& {
                return data[r][static_cast<std::size_t>(i) * k + j];
            };
            boundary in = (r == 0) ? wrap : left;
            if (r == 0 && row[r] == 0 && timestep[r] != 0)
                A(0, 0) = -in.v[0];
            std::uint32_t const numrows = rows_this_phase(a, row[r]);
            compute(data[r], k, r, row[r], numrows, in.v);
            if (r != a.p - 1)
            {
                left.count = static_cast<std::uint8_t>(numrows + 1);
                for (std::uint32_t i = 0; i != numrows + 1; ++i)
                    left.v[i] = A(row[r] + i, k - 1);
            }
            if (timestep[r] == a.t && phase[r] == w - 1)
            {
                if (r == a.p - 1)
                    checksum = A(a.n - 1, k - 1);
                continue;
            }
            if (r == a.p - 1 && phase[r] == w - 1)
            {
                wrap.count = 1;
                wrap.v[0] = A(a.n - 1, k - 1);
            }
            if (phase[r] == w - 1)
            {
                row[r] = 0;
                ++timestep[r];
                phase[r] = 0;
            }
            else
            {
                row[r] += a.gf;
                ++phase[r];
            }
        }
    }
    return checksum;
}

}    // namespace arts_hpx::p2p
