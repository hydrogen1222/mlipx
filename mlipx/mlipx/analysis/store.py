"""Versioned, reproducible storage for analysis results."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def package_versions(names: Iterable[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


@dataclass(slots=True)
class AnalysisStore:
    """Task directory whose name is derived from input and parameters."""

    run_dir: Path
    task: str
    source_path: Path
    source_sha256: str
    parameters: dict[str, Any]

    @property
    def signature(self) -> str:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "task": self.task,
            "source_sha256": self.source_sha256,
            "parameters": _json_value(self.parameters),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()[:12]

    @property
    def path(self) -> Path:
        return self.run_dir / "analysis" / self.task / self.signature

    @property
    def metadata_path(self) -> Path:
        return self.path / "metadata.json"

    def cached(self) -> bool:
        if not self.metadata_path.exists():
            return False
        try:
            with self.metadata_path.open(encoding="utf-8") as handle:
                metadata = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return False
        matches = (
            metadata.get("status") == "complete"
            and metadata.get("source", {}).get("sha256") == self.source_sha256
            and metadata.get("parameters") == _json_value(self.parameters)
        )
        if not matches:
            return False
        outputs = metadata.get("outputs", [])
        return bool(outputs) and all((self.path / name).exists() for name in outputs)

    def prepare(self) -> Path:
        self.path.mkdir(parents=True, exist_ok=True)
        return self.path

    def write_json(self, name: str, value: Any) -> Path:
        path = self.prepare() / name
        with path.open("w", encoding="utf-8") as handle:
            json.dump(_json_value(value), handle, indent=2, sort_keys=True)
            handle.write("\n")
        return path

    def write_csv(self, name: str, columns: dict[str, Any]) -> Path:
        arrays = {key: np.asarray(value) for key, value in columns.items()}
        lengths = {len(value) for value in arrays.values()}
        if len(lengths) > 1:
            raise ValueError(f"CSV columns have unequal lengths: {sorted(lengths)}")
        path = self.prepare() / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(arrays)
            for row in zip(*(arrays[key] for key in arrays), strict=True):
                writer.writerow([_json_value(item) for item in row])
        return path

    def write_npz(self, name: str, **arrays: Any) -> Path:
        path = self.prepare() / name
        np.savez_compressed(path, **arrays)
        return path

    def complete(
        self,
        *,
        outputs: Iterable[Path],
        summary: dict[str, Any] | None = None,
        warnings: Iterable[str] = (),
        packages: Iterable[str] = (),
    ) -> dict[str, Any]:
        output_paths = [Path(item) for item in outputs]
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "task": self.task,
            "signature": self.signature,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": {
                "path": str(self.source_path),
                "sha256": self.source_sha256,
            },
            "parameters": _json_value(self.parameters),
            "software": {
                "python": platform.python_version(),
                "packages": package_versions(("mlipx", "ase", "numpy", *packages)),
            },
            "warnings": list(warnings),
            "outputs": [str(path.relative_to(self.path)) for path in output_paths],
            "summary": _json_value(summary or {}),
        }
        self.write_json("metadata.json", metadata)
        return metadata
