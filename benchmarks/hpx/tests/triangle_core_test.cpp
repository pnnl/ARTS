#include "triangle_core.hpp"

#include <cstdint>
#include <iostream>

namespace tri = arts_hpx::triangle;

namespace {
int failures = 0;
template <typename A, typename E>
void expect_equal(A const& a, E const& e, char const* what)
{
    if (a == e) return;
    std::cerr << "FAIL: " << what << ": expected " << e << ", got " << a << '\n';
    ++failures;
}
}    // namespace

int main()
{
    expect_equal(tri::holes(5), 15ull, "holes 5");
    expect_equal(tri::holes(8), 36ull, "holes 8");
    expect_equal(tri::gen_moves(5).size() / 3, 36ull, "moves 5");
    expect_equal(tri::gen_moves(8).size() / 3, 126ull, "moves 8");
    expect_equal(tri::generator_matches_author(), true, "generator vs author table");

    tri::walk_stats const s = tri::count_solutions(5, 13);
    expect_equal(s.leaves, 29760ull, "author's puzzle");
    // Printed for the structural gate at the small arguments: nodes,
    // summers (= internal), boards (= nodes - 1), tree (= nodes + summers).
    std::cout << "rows 5 depth 13: nodes " << s.nodes << " summers " << s.internal
              << " boards " << (s.nodes - 1) << " tree " << (s.nodes + s.internal)
              << " board_bytes " << (s.nodes - 1) * 8 * tri::holes(5) << '\n';

    // from the initial board the only legal first moves are the two jumps
    // into the empty apex, so depth 1 has two leaves under one internal node
    tri::walk_stats const t = tri::count_solutions(5, 1);
    expect_equal(t.leaves, 2ull, "depth-1 leaves");
    expect_equal(t.nodes, 3ull, "depth-1 nodes");
    expect_equal(t.internal, 1ull, "depth-1 internal");

    if (failures != 0) std::cerr << failures << " failure(s)\n";
    return failures == 0 ? 0 : 1;
}
