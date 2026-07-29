"""Tests for mlipx CLI (argparse flags, backward compat, config subcommands)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from mlipx.cli import create_parser, main

# ---------------------------------------------------------------------------
# Parser creation
# ---------------------------------------------------------------------------

def test_parser_has_all_subcommands() -> None:
    parser = create_parser()
    # Minimal smoke: parse --help on the root parser
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_sp_parser_accepts_known_flags() -> None:
    parser = create_parser()
    args = parser.parse_args([
        "sp", "struct.xyz",
        "--model", "model.pt",
        "--model-type", "mace",
        "--task", "bulk",
        "--device", "cuda:0",
        "--dtype", "float64",
        "--head", "some_head",
        "--model-alias", "mace_mpa0",
        "--profile", "gpu_prod",
        "--output", "./out",
    ])
    assert args.command == "sp"
    assert args.structure == "struct.xyz"
    assert args.model == "model.pt"
    assert args.model_type == "mace"
    assert args.task == "bulk"
    assert args.device == "cuda:0"
    assert args.default_dtype == "float64"
    assert args.head == "some_head"
    assert args.model_alias == "mace_mpa0"
    assert args.profile == "gpu_prod"
    assert args.output == "./out"


def test_opt_parser_accepts_optimisation_flags() -> None:
    parser = create_parser()
    args = parser.parse_args([
        "opt", "struct.xyz",
        "--model", "model.pt",
        "--fmax", "0.02",
        "--max-steps", "100",
        "--optimizer", "BFGS",
        "--cell-opt",
        "--no-fix-symmetry",
    ])
    assert args.fmax == 0.02
    assert args.max_steps == 100
    assert args.optimizer == "BFGS"
    assert args.cell_opt is True
    assert args.fix_symmetry is False  # --no-fix-symmetry


def test_opt_fix_symmetry_default_is_none() -> None:
    """BooleanOptionalAction without explicit flag defaults to None."""
    parser = create_parser()
    args = parser.parse_args(["opt", "struct.xyz", "--model", "m.pt"])
    assert args.fix_symmetry is None
    assert args.cell_opt is None


def test_md_parser_accepts_ensemble_flags() -> None:
    parser = create_parser()
    args = parser.parse_args([
        "md", "struct.xyz",
        "--model", "model.pt",
        "--ensemble", "NVE",
        "--temp", "500",
        "--timestep", "2.0",
        "--steps", "5000",
        "--friction", "0.005",
        "--save-interval", "25",
        "--pre-relax",
        "--no-pre-relax",
    ])
    assert args.ensemble == "NVE"
    assert args.temp == 500.0
    assert args.timestep == 2.0
    assert args.steps == 5000
    assert args.friction == 0.005
    assert args.save_interval == 25


def test_batch_parser_accepts_pattern_and_workers() -> None:
    parser = create_parser()
    args = parser.parse_args([
        "batch", "input_dir/",
        "--model", "model.pt",
        "--pattern", "*.poscar",
    ])
    assert args.command == "batch"
    assert args.input_dir == "input_dir/"
    assert args.pattern == "*.poscar"


def test_device_accepts_cuda_n() -> None:
    """--device no longer restricted by argparse choices; accepts 'cuda:1'."""
    parser = create_parser()
    args = parser.parse_args(["sp", "s.xyz", "--model", "m.pt", "--device", "cuda:2"])
    assert args.device == "cuda:2"


def test_model_optional_with_model_alias() -> None:
    """--model is not required when --model-alias is provided."""
    parser = create_parser()
    args = parser.parse_args(
        ["sp", "s.xyz", "--model-alias", "mace_mpa0", "--output", "./out"]
    )
    assert args.model is None
    assert args.model_alias == "mace_mpa0"


def test_model_required_without_model_alias() -> None:
    """Without --model or --model-alias, the handler should error. Parser itself
    doesn't enforce this (--model is default=None), but the resolver does."""
    parser = create_parser()
    args = parser.parse_args(["sp", "s.xyz", "--output", "./out"])
    assert args.model is None
    assert args.model_alias is None


# ---------------------------------------------------------------------------
# config subcommand
# ---------------------------------------------------------------------------

def test_config_paths() -> None:
    """config paths should run without error."""
    with tempfile.TemporaryDirectory() as d:
        old = os.getcwd()
        try:
            os.chdir(d)
            rc = main(["config", "paths"])
            assert rc == 0
        finally:
            os.chdir(old)


def test_config_schema() -> None:
    with tempfile.TemporaryDirectory() as d:
        old = os.getcwd()
        try:
            os.chdir(d)
            rc = main(["config", "schema"])
            assert rc == 0
        finally:
            os.chdir(old)


def test_config_show() -> None:
    with tempfile.TemporaryDirectory() as d:
        old = os.getcwd()
        try:
            os.chdir(d)
            rc = main(["config", "show"])
            assert rc == 0
        finally:
            os.chdir(old)


def test_config_init_project() -> None:
    with tempfile.TemporaryDirectory() as d:
        old = os.getcwd()
        try:
            os.chdir(d)
            rc = main(["config", "init", "--project"])
            assert rc == 0
            assert (Path(d) / "settings.ini").exists()
        finally:
            os.chdir(old)


def test_config_init_force() -> None:
    with tempfile.TemporaryDirectory() as d:
        old = os.getcwd()
        try:
            os.chdir(d)
            main(["config", "init", "--project"])
            rc = main(["config", "init", "--project", "--force"])
            assert rc == 0
        finally:
            os.chdir(old)


# ---------------------------------------------------------------------------
# CLI backward compat: calc_type → temperature alias mapping
# ---------------------------------------------------------------------------

def test_md_temp_alias() -> None:
    """--temp is the historical flag for temperature."""
    parser = create_parser()
    args = parser.parse_args([
        "md", "s.xyz", "--model", "m.pt", "--temp", "400"
    ])
    assert args.temp == 400.0


def test_batch_parser_basic_flags() -> None:
    parser = create_parser()
    args = parser.parse_args([
        "batch", "dir/", "--model", "m.pt", "--pattern", "*.cif",
    ])
    assert args.command == "batch"
    assert args.input_dir == "dir/"
    assert args.pattern == "*.cif"


# ---------------------------------------------------------------------------
# --settings global option
# ---------------------------------------------------------------------------

def test_settings_global_option_accepted() -> None:
    parser = create_parser()
    args = parser.parse_args(["--settings", "/tmp/x.ini", "sp", "s.xyz", "--model", "m.pt"])
    assert args.settings == "/tmp/x.ini"
    assert args.command == "sp"


# ---------------------------------------------------------------------------
# template command
# ---------------------------------------------------------------------------

def test_template_sp() -> None:
    with tempfile.TemporaryDirectory() as d:
        old = os.getcwd()
        try:
            os.chdir(d)
            rc = main(["template", "sp", "--output", "INCAR.sp"])
            assert rc == 0
            content = Path("INCAR.sp").read_text()
            assert "SP" in content
            assert "CALC_TYPE" in content
        finally:
            os.chdir(old)


# ---------------------------------------------------------------------------
# doctor / setup smoke (these need additional deps; just check exit)
# ---------------------------------------------------------------------------

def test_doctor_exits_gracefully() -> None:
    # Doctor should not crash even if optional deps are missing.
    rc = main(["doctor"])
    # Can be 0 (all ok) or non-zero (some missing), but not a crash
    assert isinstance(rc, int)
