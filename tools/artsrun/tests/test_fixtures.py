"""Fixtures the tool synthesizes: generated when missing, reported when not."""

from __future__ import annotations

from artsrun.fixtures import stage


def test_a_known_fixture_is_generated_and_an_unknown_one_reported(tmp_path):
    known = tmp_path / "cholesky_input.mat"
    unknown = tmp_path / "mystery.bin"
    left = stage([str(known), str(unknown)])
    assert left == [str(unknown)]
    rows = known.read_text().splitlines()
    assert len(rows) == 50
    assert rows[0].split()[0] == "1.0" and rows[1].split()[1] == "1.0"
    assert not list(tmp_path.glob("*.staging"))


def test_an_existing_file_is_never_rewritten(tmp_path):
    fixture = tmp_path / "basicIO_test.dat"
    fixture.write_text("sentinel")
    assert stage([str(fixture)]) == []
    assert fixture.read_text() == "sentinel"


def test_tile_stream_bytes_are_pinned():
    # 2x2 identity at ts=1: tiles (0,0),(1,0),(1,1) -> doubles 1.0, 0.0, 1.0.
    # Pins the layout (lower-triangular tile order, host-order doubles) so a
    # writer change that would silently re-lay the stream fails here.
    import struct
    from pathlib import Path
    from artsrun.fixtures import _identity_tiles_bin
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.bin"
        _identity_tiles_bin(p, 2, 1)
        assert p.read_bytes() == struct.pack("=3d", 1.0, 0.0, 1.0)


def test_tile_stream_size_matches_the_reader_contract():
    from pathlib import Path
    from artsrun.fixtures import _identity_tiles_bin
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.bin"
        n, ts = 50, 10
        _identity_tiles_bin(p, n, ts)
        tiles = n // ts
        assert p.stat().st_size == tiles * (tiles + 1) // 2 * ts * ts * 8
