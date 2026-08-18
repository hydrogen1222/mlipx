"""Headless matplotlib views of already-computed Analysis v2 results."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Any


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional environment
        from mlipx.analysis.validation import OptionalDependencyError

        raise OptionalDependencyError(
            "Plotting requires matplotlib; install mlipx[analysis]."
        ) from exc
    return plt


def _save(fig, output_stem: str | Path) -> list[Path]:
    stem = Path(output_stem)
    paths = [stem.with_suffix(".png"), stem.with_suffix(".svg")]
    for path in paths:
        fig.savefig(path, dpi=180, bbox_inches="tight")
    _pyplot().close(fig)
    return paths


def plot_thermo(result: dict[str, Any], output_stem: str | Path) -> list[Path]:
    plt = _pyplot()
    columns = result["columns"]
    x = columns.get("time_ps", columns.get("step"))
    series = [
        (name, values)
        for name, values in columns.items()
        if name not in {"time_ps", "absolute_time_ps", "step", "phase"}
    ]
    if x is None or not series:
        return []
    fig, axes = plt.subplots(
        len(series), 1, figsize=(7, 2.5 * len(series)), sharex=True
    )
    if len(series) == 1:
        axes = [axes]
    for axis, (name, values) in zip(axes, series, strict=True):
        axis.plot(x, values, linewidth=0.8)
        axis.set_ylabel(name)
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Time (ps)" if "time_ps" in columns else "Step")
    return _save(fig, output_stem)


def plot_rdf(result: dict[str, Any], output_stem: str | Path) -> list[Path]:
    plt = _pyplot()
    fig, first = plt.subplots(figsize=(7, 4.5))
    first.plot(result["r_A"], result["g_center_neighbor"], label="g(r)")
    first.set_xlabel("r (A)")
    first.set_ylabel("g(r)")
    second = first.twinx()
    second.plot(
        result["r_A"],
        result["coordination_number_center_neighbor"],
        color="tab:orange",
        label="CN(r)",
    )
    second.set_ylabel("Coordination number")
    first.grid(alpha=0.25)
    return _save(fig, output_stem)


def plot_msd(result: dict[str, Any], output_stem: str | Path) -> list[Path]:
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for axes, values in result["msd_by_axes_A2"].items():
        axis.plot(result["lag_time_ps"], values, label=f"MSD {axes}")
    axis.set_xlabel("Lag time (ps)")
    axis.set_ylabel("MSD (A^2)")
    _apply_msd_fit_window(axis, result)
    axis.legend()
    axis.grid(alpha=0.25)
    return _save(fig, output_stem)


def _apply_msd_fit_window(axis, result: dict[str, Any]) -> None:
    window = result.get("fit_window_ps")
    if window is None:
        return
    start = float(window["start"])
    stop = float(window["stop"])
    if not np.isfinite([start, stop]).all() or start < 0 or stop <= start:
        raise ValueError("Invalid MSD fit window in plot result")
    axis.set_xlim(start, stop)


def plot_msd_alpha(result: dict[str, Any], output_stem: str | Path) -> list[Path]:
    """Plot the local log-log MSD exponent for each requested direction."""

    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for axes, values in result["log_log_alpha_by_axes"].items():
        axis.plot(result["lag_time_ps"], values, label=f"alpha {axes}")
    axis.axhline(
        1.0,
        color="black",
        linestyle="--",
        linewidth=1.0,
        alpha=0.6,
        label="normal diffusion (alpha = 1)",
    )
    axis.set_xlabel("Lag time (ps)")
    axis.set_ylabel("Local exponent alpha = d ln(MSD) / d ln(t)")
    _apply_msd_fit_window(axis, result)
    axis.set_ylim(0.0, 2.0)
    axis.set_yticks(np.arange(0.0, 2.1, 0.5))
    axis.legend()
    axis.grid(alpha=0.25)
    return _save(fig, output_stem)


def plot_transport(result: dict[str, Any], output_stem: str | Path) -> list[Path]:
    """Plot kinisi MSD with the diffusion regression fit window highlighted.

    The figure only visualizes the kinisi MSD data and the explicit fit window
    used by the covariance-aware Bayesian regression; it deliberately does not
    draw an OLS fit line, because transport does not use ordinary least squares.
    """

    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(7, 4.5))
    lag_ps = np.asarray(result["lag_time_ps"], dtype=float)
    msd = np.asarray(result["kinisi_msd_A2"], dtype=float)
    variance = np.asarray(result["kinisi_msd_variance_A4"], dtype=float)
    error = np.sqrt(np.abs(variance))
    axis.errorbar(
        lag_ps,
        msd,
        yerr=error,
        fmt=".",
        markersize=3,
        elinewidth=0.8,
        capsize=0,
        label="kinisi MSD",
    )
    tracer = result["tracer_diffusion"]
    fit_start = float(tracer["fit_start_ps"])
    fit_stop = float(tracer["fit_stop_ps"])
    axis.axvspan(
        fit_start,
        fit_stop,
        alpha=0.15,
        color="tab:orange",
        label="diffusion fit window",
    )
    axis.set_xlabel("Lag time (ps)")
    axis.set_ylabel("MSD (A^2)")
    axis.legend()
    axis.grid(alpha=0.25)
    return _save(fig, output_stem)


def _plot_transport_series(
    result: dict[str, Any],
    output_stem: str | Path,
    *,
    values_key: str,
    variance_key: str,
    ylabel: str,
    label: str,
) -> list[Path]:
    plt = _pyplot()
    values = np.asarray(result[values_key], dtype=float)
    variance = np.asarray(result.get(variance_key, np.full(values.shape, np.nan)))
    lag_ps = np.asarray(result["lag_time_ps"], dtype=float)[: len(values)]
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.errorbar(
        lag_ps,
        values,
        yerr=np.sqrt(np.abs(variance)),
        fmt=".",
        markersize=3,
        elinewidth=0.8,
        capsize=0,
        label=label,
    )
    tracer = result.get("tracer_diffusion", {})
    if tracer.get("fit_start_ps") is not None:
        axis.axvspan(
            float(tracer["fit_start_ps"]),
            float(tracer.get("fit_stop_ps", lag_ps[-1])),
            alpha=0.15,
            color="tab:orange",
            label="diffusion fit window",
        )
    axis.set_xlabel("Lag time (ps)")
    axis.set_ylabel(ylabel)
    axis.legend()
    axis.grid(alpha=0.25)
    return _save(fig, output_stem)


def plot_transport_mscd(result: dict[str, Any], output_stem: str | Path) -> list[Path]:
    """Plot the collective mean-squared charge displacement."""

    return _plot_transport_series(
        result,
        output_stem,
        values_key="kinisi_mscd",
        variance_key="kinisi_mscd_variance",
        ylabel=f"MSCD ({result.get('collective_conductivity', {}).get('mscd_unit', 'backend units')})",
        label="kinisi MSCD",
    )


def plot_transport_mstd(result: dict[str, Any], output_stem: str | Path) -> list[Path]:
    """Plot total/jump mean-squared displacement (a diagnostic)."""

    return _plot_transport_series(
        result,
        output_stem,
        values_key="kinisi_mstd",
        variance_key="kinisi_mstd_variance",
        ylabel=f"MSTD ({result.get('jump_diffusion', {}).get('mstd_unit', 'backend units')})",
        label="kinisi MSTD (diagnostic)",
    )


def plot_electrolyte_density(result: dict[str, Any], output_stem: str | Path) -> list[Path]:
    """Plot a compact density projection for GEMDAT mechanism analysis."""

    plt = _pyplot()
    density = np.asarray(result["density_counts"], dtype=float)
    if density.ndim != 3:
        return []
    fig, axis = plt.subplots(figsize=(5.5, 4.5))
    axis.imshow(np.mean(density, axis=2).T, origin="lower", aspect="auto")
    axis.set_xlabel("Voxel x")
    axis.set_ylabel("Voxel y")
    axis.set_title("Production mobile-ion density projection")
    return _save(fig, output_stem)


def plot_electrolyte_paths(result: dict[str, Any], output_stem: str | Path) -> list[Path]:
    """Plot finite-temperature occupancy free energy along percolation paths."""

    plt = _pyplot()
    paths = result.get("paths", {})
    if not paths:
        return []
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for axis_name, values in paths.items():
        energy = np.asarray(values.get("free_energy_eV", []), dtype=float)
        axis.plot(np.arange(len(energy)), energy, label=axis_name)
    axis.set_xlabel("Path step")
    axis.set_ylabel("Occupancy-derived free energy (eV)")
    axis.set_title("Finite-temperature free-energy paths (not NEB barriers)")
    axis.legend()
    axis.grid(alpha=0.25)
    return _save(fig, output_stem)


def plot_electrolyte_distribution(
    result: dict[str, Any], output_stem: str | Path, *, title: str, xlabel: str
) -> list[Path]:
    """Plot a numeric distribution from a GEMDAT table when available."""

    plt = _pyplot()
    table = result.get("table")
    if table is None:
        return []
    if hasattr(table, "select_dtypes"):
        numeric = table.select_dtypes(include=["number"])
        if numeric.empty:
            return []
        values = np.asarray(numeric.iloc[:, 0], dtype=float)
    else:
        values = np.asarray(table, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        return []
    fig, axis = plt.subplots(figsize=(6, 4.2))
    axis.hist(values, bins=min(30, max(5, int(np.sqrt(len(values))))), alpha=0.8)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Count")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    return _save(fig, output_stem)


def plot_arrhenius(result: dict[str, Any], output_stem: str | Path) -> list[Path]:
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(6, 4.5))
    axis.scatter(result["inverse_temperature_K^-1"], result["ln_diffusivity"])
    order = result["inverse_temperature_K^-1"].argsort()
    axis.plot(
        result["inverse_temperature_K^-1"][order],
        result["ln_diffusivity_fit"][order],
    )
    axis.set_xlabel("1/T (K^-1)")
    axis.set_ylabel("ln(D / m^2 s^-1)")
    axis.grid(alpha=0.25)
    return _save(fig, output_stem)


def plot_vacf(result: dict[str, Any], output_stem: str | Path) -> list[Path]:
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(result["lag_time_fs"], result["vacf_normalized"])
    axis.set_xlabel("Lag time (fs)")
    axis.set_ylabel("Normalized VACF")
    axis.grid(alpha=0.25)
    return _save(fig, output_stem)


def plot_spectrum(result: dict[str, Any], output_stem: str | Path) -> list[Path]:
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(result["frequency_THz"], result["spectrum"])
    axis.set_xlabel("Frequency (THz)")
    axis.set_ylabel("Normalized intensity / arbitrary units")
    axis.grid(alpha=0.25)
    return _save(fig, output_stem)
