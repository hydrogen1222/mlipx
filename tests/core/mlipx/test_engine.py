"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest


class TestEngineConfig:
    """Tests for EngineConfig dataclass."""

    def test_minimal_config(self):
        from mlipx.engine import EngineConfig  # noqa: PLC0415

        config = EngineConfig(
            calc_type="sp",
            model_path=Path("model.pt"),
            task="omat",
            device="cpu",
            inference_mode="default",
            output_dir=Path("./results"),
        )
        assert config.calc_type == "sp"
        assert config.options == {}
        assert config.detach is False

    def test_config_with_options(self):
        from mlipx.engine import EngineConfig  # noqa: PLC0415

        config = EngineConfig(
            calc_type="opt",
            model_path=Path("model.pt"),
            task="omat",
            device="cpu",
            inference_mode="default",
            output_dir=Path("./results"),
            options={"fmax": 0.02, "cell_opt": True},
        )
        assert config.options["fmax"] == 0.02
        assert config.options["cell_opt"] is True

    def test_config_detach(self):
        from mlipx.engine import EngineConfig  # noqa: PLC0415

        config = EngineConfig(
            calc_type="sp",
            model_path=Path("model.pt"),
            task="omat",
            device="cpu",
            inference_mode="default",
            output_dir=Path("./results"),
            detach=True,
        )
        assert config.detach is True


class TestCalculationEngineSetup:
    """Tests for CalculationEngine construction and config validation."""

    def test_engine_from_config_sp(self):
        from mlipx.engine import CalculationEngine, EngineConfig  # noqa: PLC0415

        config = EngineConfig(
            calc_type="sp",
            model_path=Path("model.pt"),
            task="omat",
            device="cpu",
            inference_mode="default",
            output_dir=Path("./results"),
        )
        with patch.object(Path, "exists", return_value=True):
            engine = CalculationEngine.from_config(config)
            assert engine.config.calc_type == "sp"

    def test_invalid_calc_type_raises(self):
        from mlipx.engine import CalculationEngine, EngineConfig  # noqa: PLC0415

        config = EngineConfig(
            calc_type="invalid",
            model_path=Path("model.pt"),
            task="omat",
            device="cpu",
            inference_mode="default",
            output_dir=Path("./results"),
        )
        with pytest.raises(ValueError, match="Unknown calc_type"):
            CalculationEngine.from_config(config)

    def test_engine_config_options_validation_unknown_key_warns(self):
        """Unknown options keys should not raise, just warn."""
        from mlipx.engine import CalculationEngine, EngineConfig  # noqa: PLC0415

        config = EngineConfig(
            calc_type="sp",
            model_path=Path("model.pt"),
            task="omat",
            device="cpu",
            inference_mode="default",
            output_dir=Path("./results"),
            options={"made_up_key": 42},
        )
        # Should not raise
        with patch.object(Path, "exists", return_value=True):
            engine = CalculationEngine.from_config(config)
            assert engine is not None


def test_engine_creates_live_log_and_tail_hint(tmp_path):
    from mlipx.engine import CalculationEngine, EngineConfig  # noqa: PLC0415

    config = EngineConfig(
        calc_type="sp",
        model_path=Path("model.pt"),
        output_dir=tmp_path,
        job_name="job-01",
    )
    engine = CalculationEngine.from_config(config)
    runner = Mock()
    runner.execute.return_value = {"energy": -1.0}
    messages = []

    with (
        patch.object(engine, "_create_calculator", return_value=object()),
        patch.object(engine, "_create_runner", return_value=runner),
    ):
        engine.run(
            atoms=object(),
            log_fn=lambda message, level: messages.append(message),
            started_at=123.0,
        )

    expected_log = (tmp_path / "job-01" / "run.log").resolve()
    assert engine.run_log_path == expected_log
    assert expected_log.exists()
    assert "tail -f" in expected_log.read_text(encoding="utf-8")
    assert any("Follow live output:" in message for message in messages)
    runner.execute.assert_called_once()
    assert runner.execute.call_args.kwargs["started_at"] == 123.0
