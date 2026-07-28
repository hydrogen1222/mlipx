"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Logging utilities for mlipx.

Provides structured logging for all calculations with support for
file output and console output.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from logging import Logger


def setup_logger(
    name: str = "mlipx",
    level: int = logging.INFO,
    log_file: Path | str | None = None,
    console: bool = True,
) -> Logger:
    """Setup a logger with file and/or console output.

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        console: Whether to output to console

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


class CalculationLogger:
    """Logger for calculation runs.

    Provides structured logging for a single calculation run
    with timing and progress tracking.
    """

    def __init__(
        self,
        name: str,
        output_dir: Path | str,
        verbose: bool = True,
    ):
        """Initialize calculation logger.

        Args:
            name: Calculation name
            output_dir: Output directory for logs
            verbose: Whether to print to console
        """
        self.name = name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        log_file = self.output_dir / "calculation.log"
        self.logger = setup_logger(
            name=f"mlipx.{name}",
            log_file=log_file,
            console=verbose,
        )

    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)

    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)
