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
            head: Optional MACE foundation-model head name. Only forwarded to
                ``MACECalculator`` when not ``None`` (older MACE builds do not
                accept ``heads=``).

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
            try:
                from mace.calculators import MACECalculator  # noqa: PLC0415
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "MACE support requires the 'mace-torch' package.\n"
                    "Install with: pip install mace-torch"
                ) from e

            kwargs: dict[str, Any] = {
                "model_paths": str(self.model_path),
                "device": self._device,
                "default_dtype": self._default_dtype,
            }
            # ``heads`` is only accepted by newer MACE builds and only when a
            # head is actually requested; pass it conditionally so older MACE
            # installs keep working.
            if self._head is not None:
                kwargs["heads"] = [self._head]
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
        # Best-effort enrichment from the real MACE model. These attributes are
        # not part of the ASE Calculator contract, so guard every access.
        calc = self._calculator
        if calc is not None:
            model = getattr(calc, "models", None)
            if isinstance(model, list) and model:
                first = model[0]
                for attr, key in (
                    ("r_max", "cutoff"),
                    ("num_elements", "num_elements"),
                ):
                    value = getattr(first, attr, None)
                    if value is not None:
                        base[key] = value
                # Element list / head list live on the MACECalculator itself.
                for attr, key in (
                    ("elements", "elements"),
                    ("heads", "heads"),
                ):
                    value = getattr(calc, attr, None)
                    if value is not None:
                        base[key] = value
            base["num_models"] = len(model) if isinstance(model, list) else 1
        return base
