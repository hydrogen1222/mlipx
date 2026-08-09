"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Textual TUI interface for mlipx.

Provides an interactive terminal-based UI for configuring and running
calculations with a make menuconfig-like experience.
"""

from __future__ import annotations

from mlipx.tui.app import MlipxApp
from mlipx.tui.config_screen import ConfigScreen
from mlipx.tui.main_screen import MainScreen, TemplateScreen
from mlipx.tui.run_screen import RunScreen
from mlipx.tui.analysis_screen import AnalysisScreen

__all__ = [
    "AnalysisScreen",
    "ConfigScreen",
    "MainScreen",
    "MlipxApp",
    "RunScreen",
    "TemplateScreen",
]
