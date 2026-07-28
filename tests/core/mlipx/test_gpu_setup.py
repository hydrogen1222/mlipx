"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest
from mlipx.gpu_setup import (
    GpuInfo,
    cc_arch_name,
    detect_gpus,
    format_setup_report,
    recommend_torch,
    setup_report_json,
)

# --- cc_arch_name -----------------------------------------------------------


@pytest.mark.parametrize(
    "cc,expected",
    [
        ((3, 7), "Kepler"),
        ((5, 0), "Maxwell"),
        ((5, 2), "Maxwell"),
        ((6, 0), "Pascal"),
        ((6, 1), "Pascal"),
        ((7, 0), "Volta"),
        ((7, 5), "Turing"),
        ((8, 0), "Ampere"),
        ((8, 6), "Ampere"),
        ((8, 9), "Ada Lovelace"),
        ((9, 0), "Hopper"),
        ((10, 0), "Blackwell"),
        ((12, 0), "Blackwell"),
    ],
)
def test_cc_arch_name(cc, expected):
    assert cc_arch_name(*cc) == expected


# --- recommend_torch --------------------------------------------------------


@pytest.mark.parametrize(
    "cc,version,cu",
    [
        ((5, 0), "2.6.0+cu124", "cu124"),  # GTX 750 Ti
        ((5, 2), "2.6.0+cu124", "cu124"),  # GTX 960
        ((6, 0), "2.6.0+cu124", "cu124"),  # Tesla P100
        ((6, 1), "2.6.0+cu124", "cu124"),  # P104-100
        ((7, 0), "2.6.0+cu124", "cu124"),
        ((7, 5), "2.6.0+cu124", "cu124"),
        ((8, 6), "2.6.0+cu124", "cu124"),
        ((8, 9), "2.6.0+cu124", "cu124"),  # RTX 40 (binary-compat via sm_86)
        ((9, 0), "2.6.0+cu124", "cu124"),
    ],
)
def test_recommend_torch_modern(cc, version, cu):
    rec = recommend_torch(*cc)
    assert rec.supported
    assert rec.version == version
    assert rec.cu_tag == cu
    assert rec.install_commands
    assert "pyproject" in rec.pyproject_snippet
    assert rec.index_url.endswith(cu)


@pytest.mark.parametrize("cc", [(10, 0), (12, 0)])
def test_recommend_torch_blackwell(cc):
    rec = recommend_torch(*cc)
    assert rec.supported
    assert rec.version == "2.8.0+cu128"
    assert rec.cu_tag == "cu128"
    assert rec.index_url == "https://download.pytorch.org/whl/cu128"


@pytest.mark.parametrize("cc", [(3, 0), (3, 7)])
def test_recommend_torch_kepler_unsupported(cc):
    rec = recommend_torch(*cc)
    assert not rec.supported
    assert rec.version == ""
    assert not rec.install_commands


def test_recommend_torch_floor_is_maxwell():
    """The documented support floor (GTX 960, sm_52) must be supported."""
    rec = recommend_torch(5, 2)
    assert rec.supported
    assert rec.version == "2.6.0+cu124"


# --- detect_gpus (monkeypatched subprocess) --------------------------------


def _fake_completed(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess:
    return SimpleNamespace(stdout=stdout, stderr="", returncode=returncode)


def test_detect_gpus_single(monkeypatch):
    csv = "NVIDIA P104-100, 6.1, 580.167.08, 8192\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _fake_completed(csv))
    gpus = detect_gpus()
    assert gpus is not None
    assert len(gpus) == 1
    g = gpus[0]
    assert g.name == "NVIDIA P104-100"
    assert (g.cc_major, g.cc_minor) == (6, 1)
    assert g.driver_version == "580.167.08"
    assert g.vram_mib == 8192
    assert g.sm == "sm_61"
    assert g.compute_capability == "6.1"


def test_detect_gpus_multi(monkeypatch):
    csv = "NVIDIA GeForce RTX 4090, 8.9, 550.0, 24576\nTesla P100, 6.0, 470.0, 16384\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _fake_completed(csv))
    gpus = detect_gpus()
    assert gpus is not None
    assert len(gpus) == 2
    assert gpus[0].sm == "sm_89"
    assert gpus[1].sm == "sm_60"


def test_detect_gpus_no_nvidia_smi(monkeypatch):
    def raise_fnf(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", raise_fnf)
    assert detect_gpus() is None


def test_detect_gpus_bad_returncode(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _fake_completed("", returncode=127)
    )
    assert detect_gpus() is None


def test_detect_gpus_empty_output(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _fake_completed(""))
    assert detect_gpus() is None


# --- reports ----------------------------------------------------------------


def test_format_setup_report_no_gpu():
    out = format_setup_report(None)
    assert "No NVIDIA GPU" in out
    assert "--device cpu" in out


def test_format_setup_report_pascal():
    gpu = GpuInfo("NVIDIA P104-100", 6, 1, "580.167.08", 8192)
    out = format_setup_report([gpu])
    assert "P104-100" in out
    assert "2.6.0+cu124" in out
    assert "download.pytorch.org/whl/cu124" in out
    assert "uv pip install" in out
    assert "clashctl on" in out


def test_format_setup_report_blackwell():
    gpu = GpuInfo("NVIDIA GeForce RTX 5090", 12, 0, "570.0", 32768)
    out = format_setup_report([gpu])
    assert "Blackwell" in out
    assert "2.8.0+cu128" in out


def test_format_setup_report_kepler_unsupported():
    gpu = GpuInfo("GeForce GTX 780", 3, 5, "390.0", 3072)
    out = format_setup_report([gpu])
    assert "NOT SUPPORTED" in out


def test_format_setup_report_low_vram_warning():
    gpu = GpuInfo("GeForce GTX 960", 5, 2, "470.0", 1024)
    out = format_setup_report([gpu])
    assert "VRAM below" in out


def test_setup_report_json_structure():
    gpu = GpuInfo("NVIDIA P104-100", 6, 1, "580.167.08", 8192)
    data = setup_report_json([gpu])
    assert data["has_gpu"] is True
    assert len(data["gpus"]) == 1
    assert data["gpus"][0]["sm"] == "sm_61"
    assert data["recommended"]["version"] == "2.6.0+cu124"
    assert data["recommended"]["supported"] is True


def test_setup_report_json_no_gpu():
    data = setup_report_json(None)
    assert data["has_gpu"] is False
    assert data["recommended"] is None
