"""INCAR-style configuration parser.

Moved here from the top-level :mod:`mlipx.config` module so the configuration
package owns all parsing code (plan section 17.7). The behaviour is unchanged;
the legacy module re-exports this class for backward compatibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


class IncarConfig(dict):
    """VASP-style INCAR configuration parser.

    Parses key-value pairs from INCAR files with support for:
    - Boolean values: .TRUE., .FALSE., T, F
    - Integer values
    - Float values
    - String values
    - Lists (space-separated values)

    Example:
        >>> config = IncarConfig.from_file("INCAR.uma")
        >>> print(config["CALC_TYPE"])  # "SP"
        >>> print(config.get("FMAX", 0.05))  # 0.05 with default
    """

    # Boolean mappings (VASP-style and common variants)
    TRUE_VALUES = {".true.", ".t.", "true", "t", "yes", "y", "1", ".TRUE.", ".T."}
    FALSE_VALUES = {".false.", ".f.", "false", "f", "no", "n", "0", ".FALSE.", ".F."}

    def __init__(self, *args, **kwargs):
        """Initialize configuration with optional initial values."""
        super().__init__(*args, **kwargs)
        self._comments: dict[str, str] = {}

    @classmethod
    def from_file(cls, filepath: str | Path) -> IncarConfig:
        """Parse INCAR file and return configuration object.

        Args:
            filepath: Path to INCAR file

        Returns:
            IncarConfig instance with parsed values

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file has invalid format
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        config = cls()

        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith(("#", "!")):
                    continue

                # Extract inline comment
                comment = ""
                if "#" in line:
                    line, comment = line.split("#", 1)
                    line = line.strip()

                # Parse key-value pair
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip().upper()
                    value = value.strip()

                    if not key:
                        continue

                    try:
                        parsed_value = cls._parse_value(value)
                        config[key] = parsed_value
                        if comment:
                            config._comments[key] = comment.strip()
                    except ValueError as e:
                        raise ValueError(
                            f"Error parsing line {line_num} in {filepath}: {e}"
                        ) from e

        return config

    @classmethod
    def from_string(cls, content: str) -> IncarConfig:
        """Parse configuration from string content.

        Args:
            content: String containing INCAR-style configuration

        Returns:
            IncarConfig instance with parsed values
        """
        config = cls()

        for line_num, line in enumerate(content.split("\n"), 1):
            line = line.strip()

            if not line or line.startswith(("#", "!")):
                continue

            comment = ""
            if "#" in line:
                line, comment = line.split("#", 1)
                line = line.strip()

            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip().upper()
                value = value.strip()

                if not key:
                    continue

                try:
                    parsed_value = cls._parse_value(value)
                    config[key] = parsed_value
                    if comment:
                        config._comments[key] = comment.strip()
                except ValueError as e:
                    raise ValueError(f"Error parsing line {line_num}: {e}") from e

        return config

    @classmethod
    def _parse_value(cls, value: str) -> Any:
        """Parse a value string into appropriate Python type.

        Args:
            value: Raw value string from INCAR file

        Returns:
            Parsed value (bool, int, float, or string)
        """
        value_lower = value.lower().strip()

        # Try boolean first
        if value_lower in cls.TRUE_VALUES:
            return True
        if value_lower in cls.FALSE_VALUES:
            return False

        # Try integer
        try:
            return int(value)
        except ValueError:
            pass

        # Try float
        try:
            return float(value)
        except ValueError:
            pass

        # Return as string (strip quotes if present)
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]

        return value

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean value with default."""
        value = self.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            value_lower = value.lower()
            if value_lower in self.TRUE_VALUES:
                return True
            if value_lower in self.FALSE_VALUES:
                return False
        raise ValueError(f"Cannot convert {key}={value!r} to boolean")

    def get_int(self, key: str, default: int = 0) -> int:
        """Get integer value with default."""
        value = self.get(key, default)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        return int(value)

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get float value with default."""
        value = self.get(key, default)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return float(value)

    def get_str(self, key: str, default: str = "") -> str:
        """Get string value with default."""
        value = self.get(key, default)
        return str(value)

    def write(self, filepath: str | Path) -> None:
        """Write configuration to file."""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_string())

    def to_string(self) -> str:
        """Convert configuration to INCAR-formatted string."""
        lines = []
        lines.append("# mlipx Calculation Settings")
        lines.append("")

        # Group by categories for readability
        categories = {
            "CALC_TYPE": "Calculation Type",
            "TASK": "Task Selection",
            "MODEL_TYPE": "Model Settings",
            "MODEL_PATH": "Model Settings",
            "MODEL": "Model Settings",
            "DEVICE": "Device Settings",
            "INFERENCE_MODE": "Inference Settings",
            "DEFAULT_DTYPE": "Inference Settings",
            "HEAD": "Inference Settings",
            "OPT_ALGO": "Optimization",
            "FMAX": "Optimization",
            "MAX_STEPS": "Optimization",
            "CELL_OPT": "Optimization",
            "FIX_SYMMETRY": "Optimization",
            "MD_ENSEMBLE": "Molecular Dynamics",
            "TEMPERATURE": "Molecular Dynamics",
            "TIMESTEP": "Molecular Dynamics",
            "STEPS": "Molecular Dynamics",
            "FRICTION": "Molecular Dynamics",
        }

        current_category = None

        for key in sorted(self.keys()):
            value = self[key]

            # Determine category
            category = categories.get(key, "Other")
            if category != current_category:
                lines.append(f"# {category}")
                current_category = category

            # Format value
            if isinstance(value, bool):
                formatted_value = ".TRUE." if value else ".FALSE."
            else:
                formatted_value = str(value)

            # Add comment if present
            comment = self._comments.get(key, "")
            if comment:
                lines.append(f"{key:20s} = {formatted_value:20s} # {comment}")
            else:
                lines.append(f"{key:20s} = {formatted_value}")

        return "\n".join(lines) + "\n"

    def validate(self, required_keys: list[str] | None = None) -> list[str]:
        """Validate configuration.

        Args:
            required_keys: List of required keys

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if required_keys:
            for key in required_keys:
                if key not in self:
                    errors.append(f"Required key '{key}' is missing")

        # Validate specific keys
        valid_model_types = {"uma", "fairchem", "mace", "dpa", "grace"}
        if "MODEL_TYPE" in self:
            mt = self.get_str("MODEL_TYPE").lower()
            if mt not in valid_model_types:
                errors.append(
                    f"Invalid MODEL_TYPE '{mt}'. "
                    f"Must be one of: {', '.join(valid_model_types - {'fairchem'})}"
                )

        # UMA tasks + generic bulk/molecule (non-UMA engines)
        valid_tasks = {
            "omat",
            "omol",
            "oc20",
            "oc25",
            "odac",
            "omc",
            "bulk",
            "molecule",
        }
        if "TASK" in self:
            task = self.get_str("TASK").lower()
            if task not in valid_tasks:
                errors.append(
                    f"Invalid TASK '{task}'. Must be one of: {', '.join(valid_tasks)}"
                )
        valid_calc_types = {"sp", "opt", "md", "batch", "phonon", "analyze"}
        if "CALC_TYPE" in self:
            calc_type = self.get_str("CALC_TYPE").lower()
            if calc_type not in valid_calc_types:
                errors.append(
                    f"Invalid CALC_TYPE '{calc_type}. "
                    f"Must be one of: {', '.join(valid_calc_types)}"
                )

        valid_devices = {"cpu", "cuda", "gpu"}
        if "DEVICE" in self:
            device = self.get_str("DEVICE").lower()
            # Allow cuda:N device strings (plan section 17.2).
            is_device = device in valid_devices or device.startswith("cuda:")
            if not is_device:
                errors.append(
                    f"Invalid DEVICE '{device}'. "
                    f"Must be one of: {', '.join(valid_devices)} or cuda:N"
                )

        valid_optimizers = {"fire", "bfgs", "lbfgs", "gpmin", "mdmin"}
        if "OPT_ALGO" in self:
            optimizer = self.get_str("OPT_ALGO").lower()
            if optimizer not in valid_optimizers:
                errors.append(
                    f"Invalid OPT_ALGO '{optimizer}. "
                    f"Must be one of: {', '.join(valid_optimizers)}"
                )

        valid_md_ensembles = {"nve", "nvt"}
        if "MD_ENSEMBLE" in self:
            ensemble = self.get_str("MD_ENSEMBLE").lower()
            if ensemble not in valid_md_ensembles:
                errors.append(
                    f"Invalid MD_ENSEMBLE '{ensemble}. "
                    f"Must be one of: {', '.join(valid_md_ensembles)}"
                )

        # Validate MACE dtype if present.
        if "DEFAULT_DTYPE" in self:
            dtype = self.get_str("DEFAULT_DTYPE").lower()
            if dtype not in {"float32", "float64"}:
                errors.append(
                    f"Invalid DEFAULT_DTYPE '{dtype}'. Must be float32 or float64"
                )

        return errors
