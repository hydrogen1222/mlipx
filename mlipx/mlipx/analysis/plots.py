"""Headless matplotlib views of already-computed Analysis v2 results."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

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
    axis.legend()
    axis.grid(alpha=0.25)
    return _save(fig, output_stem)


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
    axis.legend()
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
