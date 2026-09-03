#pragma once

#include <cstdint>
#include <vector>

namespace arts_hpx::triangle {

constexpr std::uint64_t ROWS_DEFAULT = 5;
constexpr std::uint64_t ROWS_MAX = 10;
constexpr std::uint64_t MOVESIZE = 36;
constexpr std::uint64_t BOTTOM = 13;
constexpr std::uint32_t SCATTER_LEVELS = 3;

using board_t = std::vector<std::uint64_t>;    // holes entries, 1 = peg
using moves_t = std::vector<std::uint64_t>;    // 3 entries per jump: from, over, to

inline std::uint64_t holes(std::uint64_t rows) { return rows * (rows + 1) / 2; }
inline std::uint64_t hole_index(std::uint64_t i, std::uint64_t j) { return i * (i + 1) / 2 + j; }

// Jumps along the triangle's three axes, both directions; legal iff over
// and landing are on the board.
inline moves_t gen_moves(std::uint64_t rows)
{
    static constexpr std::int64_t dir[6][2] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}, {1, 1}, {-1, -1}};
    moves_t moves;
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(rows); ++i)
        for (std::int64_t j = 0; j <= i; ++j)
            for (std::uint64_t d = 0; d < 6; ++d)
            {
                std::int64_t const oi = i + dir[d][0], oj = j + dir[d][1];
                std::int64_t const ti = i + 2 * dir[d][0], tj = j + 2 * dir[d][1];
                if (ti < 0 || ti >= static_cast<std::int64_t>(rows) || tj < 0 || tj > ti)
                    continue;
                if (oi < 0 || oi >= static_cast<std::int64_t>(rows) || oj < 0 || oj > oi)
                    continue;
                moves.push_back(hole_index(i, j));
                moves.push_back(hole_index(oi, oj));
                moves.push_back(hole_index(ti, tj));
            }
    return moves;
}

inline moves_t const& author_moves()
{
    static moves_t const table = {0,1,3, 3,1,0, 0,2,5, 5,2,0, 3,4,5, 5,4,3, 1,3,6, 6,3,1, 1,4,8, 8,4,1,
        2,4,7, 7,4,2, 2,5,9, 9,5,2, 6,7,8, 8,7,6, 7,8,9, 9,8,7, 3,6,10, 10,6,3, 3,7,12, 12,7,3,
        4,7,11, 11,7,4, 4,8,13, 13,8,4, 5,8,12, 12,8,5, 5,9,14, 14,9,5, 10,11,12, 12,11,10,
        11,12,13, 13,12,11, 12,13,14, 14,13,12};
    return table;
}

// The generator must reproduce the author's table as a set — the one
// independent oracle of the jump geometry.
inline bool generator_matches_author()
{
    moves_t const gen = gen_moves(ROWS_DEFAULT);
    if (gen.size() != 3 * MOVESIZE)
        return false;
    moves_t const& a = author_moves();
    std::uint64_t found = 0;
    for (std::uint64_t i = 0; i < MOVESIZE; ++i)
        for (std::uint64_t k = 0; k < MOVESIZE; ++k)
            if (gen[3 * k] == a[3 * i] && gen[3 * k + 1] == a[3 * i + 1] && gen[3 * k + 2] == a[3 * i + 2])
            {
                ++found;
                break;
            }
    return found == MOVESIZE;
}

// The apex empty, every other hole pegged.
inline board_t initial_board(std::uint64_t rows)
{
    board_t b(holes(rows), 1);
    b[0] = 0;
    return b;
}

inline bool is_legal(board_t const& b, moves_t const& m, std::uint64_t i)
{
    return b[m[3 * i]] && b[m[3 * i + 1]] && !b[m[3 * i + 2]];
}
inline std::uint64_t legal_count(board_t const& b, moves_t const& m)
{
    std::uint64_t n = 0;
    for (std::uint64_t i = 0; i < m.size() / 3; ++i)
        if (is_legal(b, m, i))
            ++n;
    return n;
}
inline void apply(board_t& b, moves_t const& m, std::uint64_t i)
{
    b[m[3 * i]] = 0;
    b[m[3 * i + 1]] = 0;
    b[m[3 * i + 2]] = 1;
}

// Finalizer-style bit mix: a board bitmask carries semantically fixed low
// bits, so a raw modulus is parity-biased toward one place.
inline std::uint64_t mix_key(std::uint64_t x)
{
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}
inline std::uint64_t bits_of(board_t const& b)
{
    std::uint64_t bits = 0;
    for (std::uint64_t i = 0; i < b.size(); ++i)
        if (b[i])
            bits |= (1ull << i);
    return bits;
}

struct walk_stats
{
    std::uint64_t leaves = 0, nodes = 0, internal = 0;
};

// One node of the search as the port runs it: a node with children is an
// internal node (one summer), a node at depth is a leaf worth 1.
inline void walk(board_t const& board, std::uint64_t nummoves, std::uint64_t depth,
    moves_t const& m, walk_stats& s)
{
    ++s.nodes;
    if (nummoves == depth)
    {
        ++s.leaves;
        return;
    }
    if (legal_count(board, m) == 0)
        return;
    ++s.internal;
    for (std::uint64_t i = 0; i < m.size() / 3; ++i)
        if (is_legal(board, m, i))
        {
            board_t child = board;
            apply(child, m, i);
            walk(child, nummoves + 1, depth, m, s);
        }
}

inline walk_stats count_solutions(std::uint64_t rows, std::uint64_t depth)
{
    walk_stats s;
    walk(initial_board(rows), 0, depth, gen_moves(rows), s);
    return s;
}

}    // namespace arts_hpx::triangle
