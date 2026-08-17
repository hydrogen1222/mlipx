"""Shared pytest fixtures/configuration for mlipx.

Deliberately imports nothing from torch/fairchem/ray so that test discovery
does not pull in heavy ML backends.
"""

from __future__ import annotations

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: mark test to run only on GPU workers")
    config.addinivalue_line(
        "markers", "cpu_and_gpu: mark test to run on both GPU and CPU workers"
    )


def pytest_runtest_setup(item):
    import torch

    if (
        "gpu" in item.keywords
        and "cpu_and_gpu" not in item.keywords
        and not torch.cuda.is_available()
    ):
        pytest.skip("CUDA not available, skipping GPU test")


@pytest.fixture(scope="session")
def water_xyz_file(tmp_path_factory):
    """Provide a reusable minimal water molecule XYZ file path."""
    contents = (
        "3\n"
        "water\n"
        "O 0.000000 0.000000 0.000000\n"
        "H 0.758602 0.000000 0.504284\n"
        "H -0.758602 0.000000 0.504284\n"
    )
    d = tmp_path_factory.mktemp("xyz_inputs")
    fpath = d / "water.xyz"
    fpath.write_text(contents)
    return str(fpath)
