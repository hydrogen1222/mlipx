"""Tests for mlipx.gpu_setup (nvidia-smi parsing and recommendations)."""

from __future__ import annotations

from unittest import mock

from mlipx.gpu_setup import cc_arch_name, detect_gpus, recommend_torch


def test_detect_gpus_parses_plain_csv() -> None:
    fake = mock.Mock(
        returncode=0,
        stdout="NVIDIA GeForce RTX 4090,8.9,535.104.05,24564\n",
    )
    with mock.patch("mlipx.gpu_setup.subprocess.run", return_value=fake):
        gpus = detect_gpus()
    assert gpus is not None and len(gpus) == 1
    g = gpus[0]
    assert g.name == "NVIDIA GeForce RTX 4090"
    assert (g.cc_major, g.cc_minor) == (8, 9)
    assert g.vram_mib == 24564
    assert g.driver_version == "535.104.05"


def test_detect_gpus_handles_quoted_commas_in_name() -> None:
    """Regression: a GPU name containing a comma is quoted by nvidia-smi;
    a naive split(',') corrupted it into extra fields."""
    fake = mock.Mock(
        returncode=0,
        stdout='"Tesla V100-SXM2-16GB, Special",7.0,535.104.05,16384\n',
    )
    with mock.patch("mlipx.gpu_setup.subprocess.run", return_value=fake):
        gpus = detect_gpus()
    assert gpus is not None and len(gpus) == 1
    assert gpus[0].name == "Tesla V100-SXM2-16GB, Special"
    assert (gpus[0].cc_major, gpus[0].cc_minor) == (7, 0)


def test_detect_gpus_skips_malformed_rows() -> None:
    fake = mock.Mock(
        returncode=0,
        stdout="bad row without enough fields\n"
        'NVIDIA A100,8.0,535.104.05,40960\n',
    )
    with mock.patch("mlipx.gpu_setup.subprocess.run", return_value=fake):
        gpus = detect_gpus()
    assert gpus is not None and len(gpus) == 1
    assert gpus[0].name == "NVIDIA A100"


def test_detect_gpus_missing_nvidia_smi() -> None:
    with mock.patch(
        "mlipx.gpu_setup.subprocess.run",
        side_effect=FileNotFoundError("nvidia-smi"),
    ):
        assert detect_gpus() is None


def test_detect_gpus_nonzero_returncode() -> None:
    fake = mock.Mock(returncode=1, stdout="")
    with mock.patch("mlipx.gpu_setup.subprocess.run", return_value=fake):
        assert detect_gpus() is None


def test_detect_gpus_empty_output() -> None:
    fake = mock.Mock(returncode=0, stdout="")
    with mock.patch("mlipx.gpu_setup.subprocess.run", return_value=fake):
        assert detect_gpus() is None


def test_cc_arch_name_families() -> None:
    assert cc_arch_name(3, 7) == "Kepler"
    assert cc_arch_name(5, 2) == "Maxwell"
    assert cc_arch_name(6, 1) == "Pascal"
    assert cc_arch_name(7, 0) == "Volta"
    assert cc_arch_name(7, 5) == "Turing"
    assert cc_arch_name(8, 0) == "Ampere"
    assert cc_arch_name(8, 9) == "Ada Lovelace"
    assert cc_arch_name(9, 0) == "Hopper"
    assert cc_arch_name(10, 0) == "Blackwell"
    assert cc_arch_name(12, 0) == "Blackwell"


def test_recommend_torch_kepler_unsupported() -> None:
    rec = recommend_torch(3, 7)
    assert rec.supported is False


def test_recommend_torch_pascal_uses_2_6() -> None:
    rec = recommend_torch(6, 1)
    assert rec.supported is True
    assert rec.version == "2.6.0+cu124"


def test_recommend_torch_blackwell_uses_2_8() -> None:
    rec = recommend_torch(10, 0)
    assert rec.supported is True
    assert rec.version == "2.8.0+cu128"
