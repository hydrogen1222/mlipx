from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from mlipx.analysis import TrajectoryDataset
from mlipx.analysis.arrhenius import fit_arrhenius
from mlipx.analysis.msd import calculate_msd
from mlipx.analysis.spectral import (
    calculate_vacf,
    direct_vacf,
    fft_vacf,
    one_sided_cosine_taper,
    velocity_spectrum,
)
from mlipx.analysis.transport import nernst_einstein_tracer_conductivity
from mlipx.analysis.units import BOLTZMANN_J_K, ELEMENTARY_CHARGE_C


def test_nernst_einstein_units_and_charge_squared() -> None:
    inputs = {
        "particle_density_m3": 1.2e28,
        "tracer_diffusion_m2_s": 2.3e-10,
        "temperature_K": 700.0,
    }
    monovalent = nernst_einstein_tracer_conductivity(**inputs, ionic_charge_e=1)
    divalent = nernst_einstein_tracer_conductivity(**inputs, ionic_charge_e=2)
    expected = (
        inputs["particle_density_m3"]
        * ELEMENTARY_CHARGE_C**2
        * inputs["tracer_diffusion_m2_s"]
        / (BOLTZMANN_J_K * inputs["temperature_K"])
    )
    assert monovalent["sigma_NE_tracer_S_m"] == pytest.approx(expected)
    assert monovalent["sigma_NE_tracer_S_cm"] == pytest.approx(expected * 0.01)
    assert monovalent["sigma_NE_tracer_mS_cm"] == pytest.approx(expected * 10)
    assert divalent["sigma_NE_tracer_S_m"] == pytest.approx(4 * expected)
    with pytest.raises(ValueError, match="ionic_charge_e is required"):
        nernst_einstein_tracer_conductivity(**inputs, ionic_charge_e=None)


def test_arrhenius_known_activation_energy() -> None:
    temperatures = np.asarray([500.0, 600.0, 750.0, 900.0])
    activation_energy_eV = 0.25
    preexponential = 2.0e-7
    kB_eV_K = 8.617333262145e-5
    diffusion = preexponential * np.exp(
        -activation_energy_eV / (kB_eV_K * temperatures)
    )
    result = fit_arrhenius(
        temperatures,
        diffusion,
        extrapolate_temperatures_K=[300.0, 700.0],
        source_run_ids=["T500", "T600", "T750", "T900"],
    )
    assert result["activation_energy_eV"] == pytest.approx(
        activation_energy_eV, rel=1e-10
    )
    assert result["preexponential_factor_m2_s"] == pytest.approx(
        preexponential, rel=1e-10
    )
    assert result["extrapolations"][0]["extrapolated"] is True
    assert result["extrapolations"][1]["extrapolated"] is False


def test_two_point_arrhenius_warns_and_weighted_fit_reports_uncertainty() -> None:
    two_point = fit_arrhenius([500, 700], [1e-12, 3e-11])
    assert "Only two temperatures" in two_point["warnings"][0]
    assert two_point["activation_energy_std_eV"] is None

    weighted = fit_arrhenius(
        [500, 600, 750, 900],
        [1.0e-12, 5.0e-12, 2.0e-11, 5.0e-11],
        diffusivity_std_m2_s=[1e-13, 5e-13, 2e-12, 5e-12],
    )
    assert weighted["activation_energy_std_eV"] is not None
    assert weighted["uncertainty_method"].startswith("weighted")


def test_anisotropic_brownian_diffusion_recovery() -> None:
    rng = np.random.default_rng(42)
    n_frames = 1200
    n_particles = 128
    diffusion_A2_fs = np.asarray([0.01, 0.02, 0.04])
    increments = rng.normal(
        scale=np.sqrt(2.0 * diffusion_A2_fs)[None, None, :],
        size=(n_frames - 1, n_particles, 3),
    )
    positions = np.concatenate(
        (np.zeros((1, n_particles, 3)), np.cumsum(increments, axis=0)), axis=0
    )
    frames = [
        Atoms(
            f"Li{n_particles}",
            positions=frame,
            cell=[1000, 1000, 1000],
            pbc=True,
        )
        for frame in positions
    ]
    dataset = TrajectoryDataset.from_frames(
        frames,
        times_fs=np.arange(n_frames, dtype=float),
        positions_convention="unwrapped",
    )
    result = calculate_msd(
        dataset,
        mobile_species="Li",
        axes="x,y,z",
        fit_start_ps=0.1,
        fit_stop_ps=0.5,
    )
    estimates = np.asarray(
        [
            result["diagnostic_linear_diffusion_fits"][axis]["D_diagnostic_m2_s"]
            for axis in "xyz"
        ]
    )
    np.testing.assert_allclose(estimates / estimates[0], [1, 2, 4], rtol=0.2, atol=0.0)


def test_vacf_fft_direct_taper_and_harmonic_peak() -> None:
    n_frames = 512
    interval_fs = 1.0
    frequency_THz = 10.0
    time_s = np.arange(n_frames) * interval_fs * 1e-15
    velocity_x = np.cos(2 * np.pi * frequency_THz * 1e12 * time_s)
    velocities = np.zeros((n_frames, 2, 3))
    velocities[:, :, 0] = velocity_x[:, None]
    np.testing.assert_allclose(
        fft_vacf(velocities), direct_vacf(velocities), rtol=1e-11, atol=1e-11
    )
    frames = []
    for velocity in velocities:
        atoms = Atoms(
            "LiS",
            positions=[[1, 1, 1], [3, 3, 3]],
            cell=[5, 5, 5],
            pbc=True,
        )
        atoms.set_velocities(velocity)
        frames.append(atoms)
    dataset = TrajectoryDataset.from_frames(
        frames,
        times_fs=np.arange(n_frames) * interval_fs,
        positions_convention="unwrapped",
    )
    vacf = calculate_vacf(dataset)
    spectrum = velocity_spectrum(vacf)
    taper = one_sided_cosine_taper(n_frames)
    assert taper[0] == 1.0
    assert taper[-1] == 0.0
    assert vacf["vacf_normalized"][0] == pytest.approx(1.0)
    peak_index = int(np.argmax(spectrum["spectrum"][1:]) + 1)
    bin_width = spectrum["frequency_THz"][1]
    assert abs(spectrum["frequency_THz"][peak_index] - frequency_THz) <= bin_width
    assert np.trapezoid(
        spectrum["spectrum"], spectrum["frequency_THz"]
    ) == pytest.approx(1.0)


def test_velocity_spectrum_retains_negative_estimates() -> None:
    result = velocity_spectrum(
        {
            "vacf_normalized": np.asarray([1.0, -2.0, 1.0, -2.0]),
            "frame_interval_fs": 1.0,
        },
        taper="none",
        normalization="raw_spectrum",
    )
    assert result["negative_fraction"] > 0
    assert np.min(result["spectrum"]) < 0
