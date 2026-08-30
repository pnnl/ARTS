"""The rendered configurations must match the ones the runtimes have been
running, key for key."""

from __future__ import annotations

import re

import pytest

from artsrun.paths import repo_root
from artsrun.render import render_arts, render_ocr
from artsrun.store import load_profile


def _keys(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line and not line.startswith("["):
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out


@pytest.mark.parametrize("nodes,name", [(1, "1n_sc"), (2, "2n_sc"), (4, "4n_sc"),
                                        (8, "8n_sc")])
def test_arts_config_matches_the_committed_one(nodes, name):
    reference = repo_root() / "configs" / "local" / "ferrari" / f"{name}.cfg"
    if not reference.is_file():
        pytest.skip(f"{reference} not present")
    profile = load_profile("ferrari")
    rendered = _keys(render_arts(profile, nodes))
    expected = _keys(reference.read_text())
    for key, value in expected.items():
        assert rendered.get(key) == value, f"{name}: {key}"


def test_arts_config_omits_ports_for_a_local_run():
    profile = load_profile("ferrari")
    assert "ports" not in render_arts(profile, 4)


def test_arts_config_names_ports_for_a_cluster_run():
    profile = load_profile("junction")
    text = render_arts(profile, 8)
    assert "ports=25000" in text
    assert "node_count=8" in text
    assert "provider=verbs" in text


@pytest.mark.parametrize("nodes,name", [(1, "1n_sc"), (2, "2n_sc"), (4, "4n_sc"),
                                        (8, "8n_sc")])
def test_reference_config_matches_the_committed_one(nodes, name):
    reference = repo_root() / "configs" / "mpi" / "ferrari" / f"{name}.cfg"
    if not reference.is_file():
        pytest.skip(f"{reference} not present")
    profile = load_profile("ferrari")
    rendered = render_ocr(profile, nodes)

    def normalize(text: str) -> list[str]:
        lines = []
        for line in text.splitlines():
            line = line.split("#", 1)[0]
            line = re.sub(r"\s+", " ", line).strip()
            if line:
                lines.append(line)
        return lines

    assert normalize(rendered) == normalize(reference.read_text())


def test_reference_config_binds_on_every_launcher_when_pinning():
    # Local colocated ranks additionally get the BLOCK policy, which shifts
    # each rank's absolute block by its own MPI rank inside the runtime —
    # one file serves every rank.  A remote rank owns its host, so the plain
    # 0..W-1 range stands alone there.
    ferrari = load_profile("ferrari")          # local, 15+1
    assert "binding\t=\t0-15" in render_ocr(ferrari, 1)
    assert "numa" not in render_ocr(ferrari, 1)
    four = render_ocr(ferrari, 4)
    assert "binding\t=\t0-15" in four
    assert "numa\t=\tBLOCK:4:16" in four

    junction = load_profile("junction")        # slurm, 63+1
    two = render_ocr(junction, 2)
    assert "binding\t=\t0-63" in two
    assert "numa" not in two


def test_reference_config_binding_follows_the_pin_flag():
    profile = load_profile("ferrari").model_copy(update={"pin": False})
    text = render_ocr(profile, 4)
    assert "binding" not in text
    assert "numa" not in text


def test_reference_config_width_follows_the_profile():
    # The reference gets the same per-node thread budget as the runtime under
    # test, so the two are measured on the same width.
    profile = load_profile("junction")
    text = render_ocr(profile, 2)
    assert "0-63" in text
