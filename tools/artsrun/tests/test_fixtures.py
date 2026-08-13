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
