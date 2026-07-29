# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Modified for the mlipx project: MACE engine wrapper.

"""
MACE engine wrapper.

Wraps ``mace.calculators.MACECalculator`` behind the ``BaseMLIPCalculator``
contract. The ``mace-torch`` package is imported lazily so mlipx works without
it installed; users only need it when selecting ``--model-type mace``.

Phase 1 (plan section 11.1 / 12 / 17.3) propagates the full set of
backend-specific options that actually reach the underlying ASE calculator:

* ``default_dtype`` (float32 / float64) -- the historical single most-requested
  option that previously did not flow through from CLI/INCAR/API;
* ``head`` -- the model head for MACE foundation multi-head models.

The remaining acceleration options (``compile_mode``, ``fullgraph``,
``enable_cueq`` ...) are intentionally deferred to Phase 2 per the plan.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mlipx.base_calculator import BaseMLIPCalculator

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator
    from typing import Any


# Dtype values accepted by MACE's MACECalculator.
_VALID_DTYPES = {"float32", "float64"}


class MACECalculatorWrapper(BaseMLIPCalculator):
    """Wrapper for MACE ASE calculators."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
        default_dtype: str = "float64",
        task: str = "bulk",
        head: str | None = None,
    ):
        """
        Initialize MACE calculator wrapper.

        Args:
            model_path: Path to a MACE model file (``.model`` / ``.pt``).
            device: Device for calculation (``cpu``, ``cuda`` or ``cuda:N``).
            default_dtype: Model dtype (``float32`` or ``float64``).
            task: PBC hint (``bulk`` or ``molecule``); not consumed by MACE.
            head: Optional MACE foundation-model head name. Forwarded to
                ``MACECalculator`` as ``head=`` (a single string) when not
                ``None``; MACE ignores ``None`` and picks its default head.

        Raises:
            FileNotFoundError: If ``model_path`` does not exist.
            ValueError: If ``default_dtype`` is not a recognised dtype.
        """
        if str(default_dtype).lower() not in _VALID_DTYPES:
            raise ValueError(
                f"Invalid default_dtype {default_dtype!r}. "
                f"Must be one of: {', '.join(sorted(_VALID_DTYPES))}"
            )

        self.model_path = Path(model_path)
        self._device = device
        self._default_dtype = str(default_dtype).lower()
        self._task = task
        self._head = head
        self._calculator: Calculator | None = None

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

    def get_calculator(self) -> Calculator:
        """Return the cached MACE ASE calculator (lazy import)."""
        if self._calculator is None:
            from mlipx.doctor import (  # noqa: PLC0415
                _installed_dependency_conflicts,
            )

            conflicts = _installed_dependency_conflicts(("mace-torch",))
            if conflicts:
                details = "\n".join(f"  - {item}" for item in conflicts)
                raise RuntimeError(
                    "MACE environment is incompatible:\n"
                    f"{details}\n"
                    "MACE and UMA must not share the same virtual environment.\n"
                    "Create .venv-mace and launch the TUI with:\n"
                    "  .venv-mace/bin/mlipx tui\n"
                    "See uma/docs/README_CN.md or README_EN.md."
                )
            try:
                from mace.calculators import MACECalculator  # noqa: PLC0415
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "MACE support requires the 'mace-torch' package.\n"
                    "Install it in the dedicated .venv-mace environment; "
                    "see uma/docs/README_CN.md or README_EN.md."
                ) from e

            kwargs: dict[str, Any] = {
                "model_paths": str(self.model_path),
                "device": self._device,
                "default_dtype": self._default_dtype,
            }
            # MACE's MACECalculator selects the active head via the ``head``
            # keyword (singular, a string) -- NOT ``heads`` (plural). The
            # plural ``heads`` attribute lives on the trained *model* object;
            # passing ``heads=[...]`` to the calculator is silently swallowed
            # into ASE's parameter dict and never selects a head, so a user's
            # ``--head`` choice would be ignored (or fall back / error).
            if self._head is not None:
                kwargs["head"] = self._head
            self._calculator = MACECalculator(**kwargs)
        return self._calculator

    @property
    def task(self) -> str:
        """PBC hint (``bulk``/``molecule``); MACE itself is task-agnostic."""
        return self._task

    @property
    def has_stress(self) -> bool:
        """Whether stress is supported."""
        return "stress" in self.implemented_properties

    def info(self) -> dict:
        """Return model metadata.

        Enriched per plan section 17.3 with the elements/cutoff/head/dtype/
        device/model-count fields when the underlying calculator exposes them,
        so OUTCAR/JSON output records what actually ran.
        """
        base = {
            "model_type": "mace",
            "model_path": str(self.model_path),
            "device": self._device,
            "default_dtype": self._default_dtype,
            "head": self._head,
            "task": self._task,
            "implemented_properties": self.implemented_properties,
            "has_stress": self.has_stress,
        }
        # Best-effort enrichment from the real MACE model. These attributes
        # are not part of the ASE Calculator contract, so guard every access.
        # NOTE: MACE exposes ``r_max``/``atomic_numbers`` on each *model*, and
        # ``available_heads``/``head``/``z_table``/``num_models`` on the
        # calculator -- there is no ``elements``/``heads``/``num_elements``
        # attribute, so the previous lookups silently never resolved.
        calc = self._calculator
        if calc is not None:
            model = getattr(calc, "models", None)
            if isinstance(model, list) and model:
                first = model[0]
                r_max = getattr(first, "r_max", None)
                if r_max is not None:
                    base["cutoff"] = float(r_max)
                atomic_numbers = getattr(first, "atomic_numbers", None)
                if atomic_numbers is not None:
                    base["num_elements"] = len(atomic_numbers)
            # Cutoff may also be readable from the calculator.
            if "cutoff" not in base:
                r_max = getattr(calc, "r_max", None)
                if r_max is not None:
                    base["cutoff"] = float(r_max)
            available_heads = getattr(calc, "available_heads", None)
            if available_heads is not None:
                base["available_heads"] = list(available_heads)
            active_head = getattr(calc, "head", None)
            if active_head is not None:
                # Record the head MACE actually selected (may differ from the
                # requested ``self._head`` when it was None / not found).
                base["active_head"] = active_head
            z_table = getattr(calc, "z_table", None)
            if z_table is not None:
                try:
                    from ase.data import chemical_symbols  # noqa: PLC0415

                    zs = list(getattr(z_table, "zs", z_table))
                    base["elements"] = [chemical_symbols[int(z)] for z in zs]
                except Exception:
                    pass
            base["num_models"] = len(model) if isinstance(model, list) else 1
        return base
