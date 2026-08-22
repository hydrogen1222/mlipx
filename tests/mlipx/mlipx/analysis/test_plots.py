from __future__ import annotations

import numpy as np
import pytest

from mlipx.analysis import plots


class _NamedTable:
    def __init__(self, columns):
        self._columns = columns
        self.columns = tuple(columns)

    def __getitem__(self, key):
        return self._columns[key]


def test_plot_msd_alpha_has_all_axes_and_normal_diffusion_reference(
    tmp_path, monkeypatch
) -> None:
    pytest.importorskip("matplotlib")
    lag_time_ps = np.asarray([0.0, 20.0, 100.0, 180.0, 200.0])
    requested_axes = ("x", "y", "z", "xy", "xyz")
    result = {
        "lag_time_ps": lag_time_ps,
        "fit_window_ps": {"start": 20.0, "stop": 180.0},
        "msd_by_axes_A2": {
            axes: np.asarray([0.0, 1.0, 4.0, 7.0, 8.0]) for axes in requested_axes
        },
        "log_log_alpha_by_axes": {
            axes: np.asarray([np.nan, 0.8, 1.0, 1.2, 1.1]) for axes in requested_axes
        },
    }
    saved = {}

    def capture_figure(fig, output_stem):
        saved["figure"] = fig
        saved["output_stem"] = output_stem
        return []

    monkeypatch.setattr(plots, "_save", capture_figure)
    assert plots.plot_msd_alpha(result, tmp_path / "alpha") == []

    alpha_axis = saved["figure"].axes[0]
    lines = alpha_axis.lines
    assert [line.get_label() for line in lines[:-1]] == [
        f"alpha {axes}" for axes in requested_axes
    ]
    assert lines[-1].get_label() == "normal diffusion (alpha = 1)"
    np.testing.assert_allclose(lines[-1].get_ydata(), [1.0, 1.0])
    np.testing.assert_allclose(alpha_axis.get_xlim(), [20.0, 180.0])
    np.testing.assert_allclose(alpha_axis.get_ylim(), [0.0, 2.0])
    np.testing.assert_allclose(alpha_axis.get_yticks(), [0.0, 0.5, 1.0, 1.5, 2.0])
    assert saved["output_stem"] == tmp_path / "alpha"
    plots._pyplot().close(saved["figure"])

    assert plots.plot_msd(result, tmp_path / "msd") == []
    np.testing.assert_allclose(saved["figure"].axes[0].get_xlim(), [20.0, 180.0])
    assert saved["output_stem"] == tmp_path / "msd"
    plots._pyplot().close(saved["figure"])


def test_gemdat_distribution_uses_only_explicit_physical_field(
    tmp_path, monkeypatch
) -> None:
    pytest.importorskip("matplotlib")
    saved = {}

    def capture_figure(fig, output_stem):
        saved["figure"] = fig
        saved["output_stem"] = output_stem
        return []

    monkeypatch.setattr(plots, "_save", capture_figure)
    table = _NamedTable(
        {
            "atom index": np.asarray([100, 200]),
            "residence_time_ps": np.asarray([0.1, 0.2]),
        }
    )
    assert (
        plots.plot_electrolyte_distribution(
            {"table": table},
            tmp_path / "residence",
            value_field="residence_time_ps",
            title="Residence-time distribution",
            xlabel="Residence time (ps)",
        )
        == []
    )
    axis = saved["figure"].axes[0]
    right_edges = [patch.get_x() + patch.get_width() for patch in axis.patches]
    assert right_edges
    assert max(right_edges) == pytest.approx(0.2, abs=1.0e-12)
    assert axis.get_xlabel() == "Residence time (ps)"
    plots._pyplot().close(saved["figure"])


def test_gemdat_distribution_warns_and_skips_unlabeled_or_missing_fields(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        plots,
        "_pyplot",
        lambda: pytest.fail("matplotlib must not be opened for an unsafe table"),
    )
    warning_sink = []
    with pytest.warns(RuntimeWarning, match="no explicit 'jump_distance_A'"):
        assert (
            plots.plot_electrolyte_distribution(
                {"table": _NamedTable({"atom index": np.asarray([0, 1])})},
                tmp_path / "jump-distance",
                value_field="jump_distance_A",
                title="Jump-distance distribution",
                xlabel="Jump distance (A)",
                warning_sink=warning_sink,
            )
            == []
        )
    assert len(warning_sink) == 1

    with pytest.warns(RuntimeWarning, match="unlabeled numeric array"):
        assert (
            plots.plot_electrolyte_distribution(
                {"table": np.asarray([[1.0, 2.0]])},
                tmp_path / "jump-rate",
                value_field="jump_rate_s^-1",
                title="Jump-rate distribution",
                xlabel="Jump rate (s^-1)",
            )
            == []
        )
