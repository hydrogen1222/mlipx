from __future__ import annotations

import numpy as np
import pytest

from mlipx.analysis import plots


def test_plot_msd_alpha_has_all_axes_and_normal_diffusion_reference(
    tmp_path, monkeypatch
) -> None:
    pytest.importorskip("matplotlib")
    lag_time_ps = np.asarray([0.0, 0.1, 0.2, 0.3])
    requested_axes = ("x", "y", "z", "xy", "xyz")
    result = {
        "lag_time_ps": lag_time_ps,
        "log_log_alpha_by_axes": {
            axes: np.asarray([np.nan, 0.8, 1.0, 1.2])
            for axes in requested_axes
        },
    }
    saved = {}

    def capture_figure(fig, output_stem):
        saved["figure"] = fig
        saved["output_stem"] = output_stem
        return []

    monkeypatch.setattr(plots, "_save", capture_figure)
    assert plots.plot_msd_alpha(result, tmp_path / "alpha") == []

    lines = saved["figure"].axes[0].lines
    assert [line.get_label() for line in lines[:-1]] == [
        f"alpha {axes}" for axes in requested_axes
    ]
    assert lines[-1].get_label() == "normal diffusion (alpha = 1)"
    np.testing.assert_allclose(lines[-1].get_ydata(), [1.0, 1.0])
    assert saved["output_stem"] == tmp_path / "alpha"
    plots._pyplot().close(saved["figure"])
