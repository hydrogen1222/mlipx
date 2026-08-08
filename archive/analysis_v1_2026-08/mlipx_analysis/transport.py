"""Optional kinisi-backed transport and uncertainty analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
from ase import Atoms

from mlipx.analysis.dataset import TrajectoryDataset


def _require_kinisi():
    try:
        import scipp as sc
        from kinisi.analyze import ConductivityAnalyzer, DiffusionAnalyzer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "The kinisi transport task requires kinisi and scipp; "
            "install mlipx[transport]."
        ) from exc
    return sc, DiffusionAnalyzer, ConductivityAnalyzer


def _ase_frames(
    dataset: TrajectoryDataset, framework: str | None = None
) -> list[Atoms]:
    frames: list[Atoms] = []
    positions_all, _ = dataset.corrected_positions(framework=framework)
    for positions, cell in zip(positions_all, dataset.cells, strict=True):
        fractional = positions @ np.linalg.inv(cell)
        frames.append(
            Atoms(
                symbols=dataset.symbols,
                scaled_positions=np.mod(fractional, 1.0),
                cell=cell,
                pbc=dataset.pbc,
            )
        )
    return frames


def _sample_summary(variable: Any, *, unit: str | None = None) -> dict[str, Any]:
    values = np.asarray(variable.values, dtype=float).reshape(-1)
    return {
        "unit": unit or str(variable.unit),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "median": float(np.median(values)),
        "credible_interval_95": [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ],
        "samples": int(len(values)),
    }


def kinisi_transport(
    dataset: TrajectoryDataset,
    *,
    species: str,
    framework: str | None = None,
    dimensions: str = "xyz",
    fit_start_ps: float | None = None,
    temperature_K: float | None = None,
    ionic_charge_e: float | None = 1.0,
    n_samples: int = 1000,
    n_walkers: int = 32,
    n_burn: int = 500,
    n_thin: int = 10,
    random_seed: int = 0,
) -> dict[str, Any]:
    """Run kinisi's covariance-aware tracer and collective transport fits."""
    if dataset.nframes < 4:
        raise ValueError("kinisi transport analysis requires at least four frames")
    if (
        not dimensions
        or any(axis not in "xyz" for axis in dimensions)
        or len(set(dimensions)) != len(dimensions)
    ):
        raise ValueError("dimensions must be a non-empty subset of 'xyz'")
    interval_fs = dataset.require_time()
    sc, DiffusionAnalyzer, ConductivityAnalyzer = _require_kinisi()
    frames = _ase_frames(dataset, framework=framework)
    fit_start_ps = (
        0.2 * (dataset.nframes - 1) * interval_fs / 1000
        if fit_start_ps is None
        else fit_start_ps
    )
    total_time_ps = (dataset.nframes - 1) * interval_fs / 1000
    if fit_start_ps < 0 or fit_start_ps >= total_time_ps:
        raise ValueError(
            f"fit_start_ps must be in [0, {total_time_ps:g}) for this trajectory"
        )
    if temperature_K is not None and temperature_K <= 0:
        raise ValueError("temperature_K must be positive")
    common = {
        "trajectory": frames,
        "specie": species,
        "time_step": sc.scalar(interval_fs, unit="fs"),
        "step_skip": sc.scalar(1, unit="dimensionless"),
        "dimension": dimensions,
        "progress": False,
    }
    mcmc = {
        "n_samples": n_samples,
        "n_walkers": n_walkers,
        "n_burn": n_burn,
        "n_thin": n_thin,
        "progress": False,
        "random_state": np.random.RandomState(random_seed),
    }

    diffusion = DiffusionAnalyzer.from_ase(**common)
    # kinisi 2.0.5 converts units in diffusion(), but conductivity() expects
    # start_dt to already use the parser's time unit. Supplying fs works for
    # both paths and avoids that upstream inconsistency.
    fit_start = sc.scalar(fit_start_ps * 1000, unit="fs")
    diffusion.diffusion(fit_start, **mcmc)
    d_samples_cm2_s = np.asarray(diffusion.D.values, dtype=float)
    diffusion_summary = _sample_summary(diffusion.D)
    diffusion_m2_summary = {
        "unit": "m^2/s",
        "mean": diffusion_summary["mean"] * 1e-4,
        "std": diffusion_summary["std"] * 1e-4,
        "median": diffusion_summary["median"] * 1e-4,
        "credible_interval_95": [
            value * 1e-4 for value in diffusion_summary["credible_interval_95"]
        ],
        "samples": diffusion_summary["samples"],
    }
    result: dict[str, Any] = {
        "lag_time_ps": np.asarray(diffusion.dt.to(unit="ps").values, dtype=float),
        "msd_A2": np.asarray(diffusion.msd.to(unit="angstrom^2").values, dtype=float),
        "msd_variance_A4": (
            np.asarray(diffusion.msd.to(unit="angstrom^2").variances, dtype=float)
            if diffusion.msd.variances is not None
            else np.full(diffusion.msd.shape, np.nan)
        ),
        "diffusivity_samples_cm2_s": d_samples_cm2_s,
        "summary": {
            "method": "kinisi covariance-aware Bayesian regression",
            "species": species,
            "dimensions": dimensions,
            "framework_drift": framework,
            "fit_start_ps": fit_start_ps,
            "diffusivity": diffusion_summary,
            "diffusivity_m2_s": diffusion_m2_summary,
        },
    }

    if ionic_charge_e is not None and temperature_K is not None:
        conductivity = ConductivityAnalyzer.from_ase(
            **common,
            ionic_charge=ionic_charge_e * sc.Unit("e"),
        )
        conductivity.conductivity(
            fit_start,
            temperature=sc.scalar(temperature_K, unit="K"),
            **mcmc,
        )
        result.update(
            {
                "mscd_lag_time_ps": np.asarray(
                    conductivity.dt.to(unit="ps").values, dtype=float
                ),
                "mscd": np.asarray(conductivity.mscd.values, dtype=float),
                "mscd_variance": (
                    np.asarray(conductivity.mscd.variances, dtype=float)
                    if conductivity.mscd.variances is not None
                    else np.full(conductivity.mscd.shape, np.nan)
                ),
                "conductivity_samples_mS_cm": np.asarray(
                    conductivity.sigma.values, dtype=float
                ),
            }
        )
        result["summary"]["temperature_K"] = temperature_K
        result["summary"]["ionic_charge_e"] = ionic_charge_e
        result["summary"]["conductivity"] = _sample_summary(
            conductivity.sigma, unit="mS/cm"
        )
    else:
        result["summary"]["conductivity"] = None
        result["summary"]["conductivity_note"] = (
            "Conductivity was not fitted because temperature or ionic charge is absent."
        )
    return result


def kinisi_arrhenius(
    temperatures_K: np.ndarray,
    diffusivities_cm2_s: np.ndarray,
    variances_cm4_s2: np.ndarray,
    *,
    n_samples: int = 1000,
    n_walkers: int = 32,
    n_burn: int = 500,
    n_thin: int = 10,
) -> dict[str, Any]:
    """Fit a Bayesian Arrhenius relation using kinisi's native implementation."""
    try:
        import scipp as sc
        from kinisi.arrhenius import Arrhenius
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Bayesian Arrhenius analysis requires mlipx[transport]."
        ) from exc
    temperature = sc.array(dims=["temperature"], values=temperatures_K, unit="K")
    diffusion = sc.array(
        dims=["temperature"],
        values=diffusivities_cm2_s,
        variances=variances_cm4_s2,
        unit="cm^2/s",
    )
    model = Arrhenius(sc.DataArray(diffusion, coords={"temperature": temperature}))
    model.mcmc(
        n_samples=n_samples,
        n_walkers=n_walkers,
        n_burn=n_burn,
        n_thin=n_thin,
    )
    return {
        "activation_energy_samples_eV": np.asarray(
            model.activation_energy.values, dtype=float
        ),
        "preexponential_samples_cm2_s": np.asarray(
            model.preexponential_factor.values, dtype=float
        ),
        "summary": {
            "activation_energy": _sample_summary(model.activation_energy),
            "preexponential_factor": _sample_summary(model.preexponential_factor),
        },
    }
