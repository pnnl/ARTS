#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace arts_hpx::smithwaterman {

constexpr std::int32_t GAP_PENALTY = -1;
constexpr std::int32_t TRANSITION_PENALTY = -2;
constexpr std::int32_t TRANSVERSION_PENALTY = -4;
constexpr std::int32_t MATCH = 2;

enum nucleotide : std::int8_t { GAP = 0, ADENINE, CYTOSINE, GUANINE, THYMINE };

inline std::int8_t char_mapping(char c)
{
    switch (c)
    {
    case '_': return GAP;
    case 'A': return ADENINE;
    case 'C': return CYTOSINE;
    case 'G': return GUANINE;
    case 'T': return THYMINE;
    default: return -1;
    }
}

inline std::int32_t score_matrix(std::int8_t a, std::int8_t b)
{
    static constexpr std::int32_t m[5][5] = {
        {GAP_PENALTY, GAP_PENALTY, GAP_PENALTY, GAP_PENALTY, GAP_PENALTY},
        {GAP_PENALTY, MATCH, TRANSVERSION_PENALTY, TRANSITION_PENALTY, TRANSVERSION_PENALTY},
        {GAP_PENALTY, TRANSVERSION_PENALTY, MATCH, TRANSVERSION_PENALTY, TRANSITION_PENALTY},
        {GAP_PENALTY, TRANSITION_PENALTY, TRANSVERSION_PENALTY, MATCH, TRANSVERSION_PENALTY},
        {GAP_PENALTY, TRANSVERSION_PENALTY, TRANSITION_PENALTY, TRANSVERSION_PENALTY, MATCH}};
    return m[a][b];
}

// Only the four nucleotide characters survive the read: everything else —
// the newlines the sample inputs carry, and any byte the scoring matrix has
// no row for — is dropped rather than mapped, so the mapped length is the
// sequence length the tiling is derived from and no index into the matrix
// can be out of range.
inline std::vector<std::int8_t> map_sequence(std::string const& text)
{
    std::vector<std::int8_t> s;
    s.reserve(text.size());
    for (char c : text)
    {
        std::int8_t const mapped = char_mapping(c);
        if (mapped <= GAP)
            continue;
        s.push_back(mapped);
    }
    return s;
}

struct params
{
    std::int32_t tile_w = 0, tile_h = 0, W = 0, H = 0, len1 = 0, len2 = 0;
    std::uint32_t check_score = 0;
};

inline params make_params(std::int32_t tile_w, std::int32_t tile_h, std::int32_t len1,
    std::int32_t len2, std::uint32_t check_score)
{
    params p;
    p.tile_w = tile_w; p.tile_h = tile_h; p.len1 = len1; p.len2 = len2; p.check_score = check_score;
    p.W = (len1 + tile_w - 1) / tile_w;
    p.H = (len2 + tile_h - 1) / tile_h;
    return p;
}

inline std::int32_t effective_width(params const& p, std::int32_t j)
{
    std::int32_t w = p.tile_w;
    if (j == p.W)
    {
        std::int32_t const remaining = p.len1 - (j - 1) * p.tile_w;
        if (remaining < p.tile_w && remaining > 0)
            w = remaining;
    }
    return w;
}
inline std::int32_t effective_height(params const& p, std::int32_t i)
{
    std::int32_t h = p.tile_h;
    if (i == p.H)
    {
        std::int32_t const remaining = p.len2 - (i - 1) * p.tile_h;
        if (remaining < p.tile_h && remaining > 0)
            h = remaining;
    }
    return h;
}

// The seeded borders: what tile (i,1) sees to its west, tile (1,j) to its
// north, and the corner north-west of a first-row or first-column tile.
inline std::vector<std::int32_t> border_right_column(params const& p, std::int32_t i)
{
    std::vector<std::int32_t> col(static_cast<std::size_t>(p.tile_h), 0);
    for (std::int32_t k = 0; k < effective_height(p, i); ++k)
        col[k] = GAP_PENALTY * ((i - 1) * p.tile_h + k + 1);
    return col;
}
inline std::vector<std::int32_t> border_bottom_row(params const& p, std::int32_t j)
{
    std::vector<std::int32_t> row(static_cast<std::size_t>(p.tile_w), 0);
    for (std::int32_t k = 0; k < effective_width(p, j); ++k)
        row[k] = GAP_PENALTY * ((j - 1) * p.tile_w + k + 1);
    return row;
}
inline std::int32_t border_corner(params const& p, std::int32_t i, std::int32_t j)
{
    if (i == 1 && j == 1)
        return 0;
    if (i == 1)
        return GAP_PENALTY * ((j - 2) * p.tile_w + effective_width(p, j - 1));
    return GAP_PENALTY * ((i - 2) * p.tile_h + effective_height(p, i - 1));
}

struct tile_out
{
    std::int32_t bottom_right = 0;
    std::vector<std::int32_t> right_column, bottom_row;
};

inline tile_out compute_tile(params const& p, std::vector<std::int8_t> const& s1,
    std::vector<std::int8_t> const& s2, std::int32_t i, std::int32_t j,
    std::vector<std::int32_t> const& left, std::vector<std::int32_t> const& above,
    std::int32_t diagonal)
{
    std::int32_t const eff_w = effective_width(p, j), eff_h = effective_height(p, i);
    std::vector<std::int32_t> cur(static_cast<std::size_t>((1 + p.tile_w) * (1 + p.tile_h)), 0);
    auto C = [&](std::int32_t ii, std::int32_t jj) -> std::int32_t& {
        return cur[static_cast<std::size_t>(ii) * (1 + p.tile_w) + jj];
    };
    C(0, 0) = diagonal;
    for (std::int32_t k = 1; k < eff_h + 1; ++k)
        C(k, 0) = left[k - 1];
    for (std::int32_t k = 1; k < eff_w + 1; ++k)
        C(0, k) = above[k - 1];
    for (std::int32_t ii = 1; ii < eff_h + 1; ++ii)
        for (std::int32_t jj = 1; jj < eff_w + 1; ++jj)
        {
            std::int8_t const c1 = s1[(j - 1) * p.tile_w + (jj - 1)];
            std::int8_t const c2 = s2[(i - 1) * p.tile_h + (ii - 1)];
            std::int32_t const diag = C(ii - 1, jj - 1) + score_matrix(c2, c1);
            std::int32_t const lft = C(ii, jj - 1) + score_matrix(c1, GAP);
            std::int32_t const top = C(ii - 1, jj) + score_matrix(GAP, c2);
            std::int32_t const bigger = lft > top ? lft : top;
            C(ii, jj) = bigger > diag ? bigger : diag;
        }
    tile_out o;
    o.bottom_right = C(eff_h, eff_w);
    o.right_column.assign(static_cast<std::size_t>(p.tile_h), 0);
    for (std::int32_t k = 0; k < eff_h; ++k)
        o.right_column[k] = C(k + 1, eff_w);
    o.bottom_row.assign(static_cast<std::size_t>(p.tile_w), 0);
    for (std::int32_t k = 0; k < eff_w; ++k)
        o.bottom_row[k] = C(eff_h, k + 1);
    return o;
}

// Untiled reference of the same recurrence and borders.
inline std::int32_t align_score(std::vector<std::int8_t> const& s1, std::vector<std::int8_t> const& s2)
{
    std::size_t const n1 = s1.size(), n2 = s2.size();
    std::vector<std::int32_t> prev(n1 + 1), cur(n1 + 1);
    for (std::size_t x = 0; x <= n1; ++x)
        prev[x] = GAP_PENALTY * static_cast<std::int32_t>(x);
    for (std::size_t y = 1; y <= n2; ++y)
    {
        cur[0] = GAP_PENALTY * static_cast<std::int32_t>(y);
        for (std::size_t x = 1; x <= n1; ++x)
        {
            std::int32_t const diag = prev[x - 1] + score_matrix(s2[y - 1], s1[x - 1]);
            std::int32_t const lft = cur[x - 1] + score_matrix(s1[x - 1], GAP);
            std::int32_t const top = prev[x] + score_matrix(GAP, s2[y - 1]);
            cur[x] = std::max(diag, std::max(lft, top));
        }
        std::swap(prev, cur);
    }
    return prev[n1];
}

// Every tile in wavefront-compatible (i, j) order through compute_tile with
// the seeded borders; returns what the bottom-right tile prints.
inline std::int32_t simulate_tiles(params const& p, std::vector<std::int8_t> const& s1,
    std::vector<std::int8_t> const& s2)
{
    std::vector<std::vector<std::int32_t>> right((p.H + 1) * (p.W + 1)), bottom((p.H + 1) * (p.W + 1));
    std::vector<std::int32_t> corner((p.H + 1) * (p.W + 1), 0);
    auto at = [&](std::int32_t i, std::int32_t j) { return static_cast<std::size_t>(i) * (p.W + 1) + j; };
    std::int32_t score = 0;
    for (std::int32_t i = 1; i <= p.H; ++i)
        for (std::int32_t j = 1; j <= p.W; ++j)
        {
            std::vector<std::int32_t> const left = j == 1 ? border_right_column(p, i) : right[at(i, j - 1)];
            std::vector<std::int32_t> const above = i == 1 ? border_bottom_row(p, j) : bottom[at(i - 1, j)];
            std::int32_t const diag = (i == 1 || j == 1) ? border_corner(p, i, j) : corner[at(i - 1, j - 1)];
            tile_out o = compute_tile(p, s1, s2, i, j, left, above, diag);
            right[at(i, j)] = o.right_column;
            bottom[at(i, j)] = o.bottom_row;
            corner[at(i, j)] = o.bottom_right;
            if (i == p.H && j == p.W)
                score = o.bottom_row[effective_width(p, j) - 1];
        }
    return score;
}

inline std::uint64_t expected_strips(params const& p)
{
    std::uint64_t const W = p.W, H = p.H;
    return H * (W - 1) + (H - 1) * W + (H - 1) * (W - 1);
}
inline std::uint64_t expected_bytes(params const& p)
{
    std::uint64_t const W = p.W, H = p.H;
    return 4ull * p.tile_h * H * (W - 1) + 4ull * p.tile_w * (H - 1) * W + 4ull * (H - 1) * (W - 1);
}

}    // namespace arts_hpx::smithwaterman
