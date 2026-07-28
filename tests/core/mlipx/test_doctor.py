"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from mlipx.doctor import format_diagnostics, run_diagnostics


def test_doctor_runs_without_crashing():
    checks, failures = run_diagnostics()
    assert isinstance(checks, list)
    assert isinstance(failures, int)
    assert len(checks) > 0
    for c in checks:
        assert "name" in c
        assert "status" in c
        assert c["status"] in ("ok", "fail", "warn", "skip")


def test_doctor_format_output():
    checks, _ = run_diagnostics()
    output = format_diagnostics(checks)
    assert "mlipx Environment Diagnostic" in output
    assert isinstance(output, str)
    assert len(output) > 0


def test_doctor_with_nonexistent_model():
    checks, _ = run_diagnostics(model_path="/nonexistent/model.pt")
    model_check = [c for c in checks if c["name"] == "Model file"]
    assert len(model_check) == 1
    assert model_check[0]["status"] == "warn"


def test_doctor_pre_torch_gives_gpu_specific_guidance(monkeypatch):
    """When torch is missing but a GPU is detected, doctor must print the
    GPU-specific torch install command (not a generic 'install torch')."""
    import builtins  # noqa: PLC0415

    from mlipx.gpu_setup import GpuInfo  # noqa: PLC0415

    real_import = builtins.__import__

    def blocked(name, *a, **k):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("simulated")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", blocked)
    monkeypatch.setattr(
        "mlipx.doctor.detect_gpus",
        lambda: [GpuInfo("NVIDIA P104-100", 6, 1, "580.167.08", 8192)],
    )

    checks, failures = run_diagnostics()
    output = format_diagnostics(checks)

    pt_check = [c for c in checks if c["name"] == "PyTorch"][0]
    assert pt_check["status"] == "fail"
    assert "2.6.0+cu124" in pt_check["detail"]
    assert "uv pip install" in pt_check["detail"]
    assert "NVIDIA driver" in [c["name"] for c in checks]
    assert failures >= 1
    assert "2.6.0+cu124" in output


def test_doctor_no_gpu_cpu_guidance(monkeypatch):
    """No nvidia-smi / no GPU: doctor warns about driver and suggests CPU."""
    monkeypatch.setattr("mlipx.doctor.detect_gpus", lambda: None)
    checks, _ = run_diagnostics()
    names = {c["name"]: c for c in checks}
    assert names["NVIDIA driver"]["status"] == "warn"
