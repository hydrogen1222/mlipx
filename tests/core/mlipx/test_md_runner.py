"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from mlipx.runners.md import MDRunner


class _CalculatorStub:
    inference_mode = "turbo"


@pytest.mark.parametrize("ensemble", ["NVT", "NVE"])
def test_md_initializes_velocities_at_target_temperature(tmp_path, ensemble):
    atoms = Atoms(
        "Ar4",
        positions=[
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ],
    )
    runner = MDRunner(
        _CalculatorStub(),
        ensemble=ensemble,
        temperature=900.0,
        output_dir=tmp_path,
        pre_relax=False,
        verbose=False,
    )

    runner._initialize_velocities(atoms)

    assert atoms.get_temperature() == pytest.approx(900.0)
    assert np.sum(atoms.get_momenta(), axis=0) == pytest.approx(np.zeros(3), abs=1e-12)
