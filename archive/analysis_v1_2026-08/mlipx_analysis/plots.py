"""Small plotting helpers; matplotlib is imported only when requested."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("Plots require matplotlib; install mlipx[analysis].") from exc
    return plt


def line_plot(
    path: Path,
    x: np.ndarray,
    series: Iterable[tuple[str, np.ndarray]],
    *,
    xlabel: str,
    ylabel: str,
    title: str,
) -> Path:
    plt = _pyplot()
    series = list(series)
    figure, axis = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    for label, values in series:
        axis.plot(x, values, label=label)
    axis.set(xlabel=xlabel, ylabel=ylabel, title=title)
    if len(series) > 1:
        axis.legend()
    axis.grid(alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def density_projection(path: Path, probability: np.ndarray, *, title: str) -> Path:
    plt = _pyplot()
    figure, axes = plt.subplots(1, 3, figsize=(10, 3.2), constrained_layout=True)
    projections = (
        ("xy", probability.sum(axis=2).T),
        ("xz", probability.sum(axis=1).T),
        ("yz", probability.sum(axis=0).T),
    )
    for axis, (label, image) in zip(axes, projections, strict=True):
        axis.imshow(image, origin="lower", extent=(0, 1, 0, 1), aspect="equal")
        axis.set(title=label, xlabel="fractional", ylabel="fractional")
    figure.suptitle(title)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path
