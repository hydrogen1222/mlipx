# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Source profiles — where to download packages from.

PyPI packages (numpy, ase, …) and PyTorch CUDA wheels are handled
through *separate* channels.  The installer never touches the user's
``~/.config/uv/uv.toml``; it sets ``UV_NO_CONFIG=1`` and passes
index/find-links through environment variables or CLI flags.

Source profiles:

    ``auto``
        Resolves to ``official`` for now (lightweight connectivity test is
        future work).  Logs the choice so the user knows.
    ``official``
        PyPI: pypi.org.  PyTorch: download.pytorch.org.
    ``china``
        PyPI: mirrors.tuna.tsinghua.edu.cn.  PyTorch: mirrors.aliyun.com
        (via ``--find-links``, because the Aliyun PyTorch mirror is a flat
        HTML listing, not a PEP 503 /simple/ registry).
    ``offline``
        No network; use already-cached wheels only.
    ``custom``
        User supplies ``UV_INDEX_URL`` / ``UV_FIND_LINKS`` / ``UV_EXTRA_INDEX_URL``
        via environment variables.  The installer passes them through unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SourceProfile:
    """One package source configuration.

    Attributes:
        name: Machine-readable key (``"official"``, ``"china"`` …).
        label: Human-readable label.
        pypi_index: PyPI index URL (``None`` = default, i.e. pypi.org).
        pypi_extra_index: Optional extra PyPI index (e.g. a mirror).
        pytorch_index: PyTorch CUDA wheel index URL
            (``None`` = use ``download.pytorch.org/whl/{cuda_tag}``).
        pytorch_find_links: Flat HTML mirror for PyTorch wheels
            (``None`` = no find-links).  Used for Aliyun.
        env: Environment variables to set during installation.
    """

    name: str
    label: str
    pypi_index: str | None = None
    pypi_extra_index: str | None = None
    pytorch_index: str | None = None
    pytorch_find_links: str | None = None
    env: dict[str, str] = field(default_factory=dict)


SOURCE_PROFILES: dict[str, SourceProfile] = {
    "auto": SourceProfile(
        name="auto",
        label="auto (→ official)",
        # Currently resolves to official.  Future: lightweight connectivity
        # test to pick between official and china.
    ),
    "official": SourceProfile(
        name="official",
        label="Official (PyPI + pytorch.org)",
    ),
    "china": SourceProfile(
        name="china",
        label="China mirrors (tuna + aliyun)",
        pypi_index="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple",
        # Aliyun PyTorch mirror is a flat HTML listing, not a PEP 503
        # registry.  The installer passes it via --find-links, not --index-url.
        pytorch_find_links="https://mirrors.aliyun.com/pytorch-wheels/{cuda_tag}/",
    ),
    "offline": SourceProfile(
        name="offline",
        label="Offline (no network)",
        env={"UV_NO_BUILD": "1"},
    ),
    "custom": SourceProfile(
        name="custom",
        label="Custom (user-provided env vars)",
    ),
}


def resolve_source(name: str) -> SourceProfile:
    """Resolve a source profile name to its :class:`SourceProfile`.

    ``"auto"`` currently resolves to ``"official"``.  Future versions may
    add a lightweight connectivity test.

    Args:
        name: Source profile name (``"auto"``, ``"official"``, ``"china"``,
            ``"offline"``, ``"custom"``).

    Returns:
        The resolved :class:`SourceProfile`.

    Raises:
        ValueError: If the profile name is unknown.
    """
    if name == "auto":
        name = "official"
    if name not in SOURCE_PROFILES:
        raise ValueError(
            f"Unknown source profile '{name}'. "
            f"Choose from: {', '.join(SOURCE_PROFILES)}"
        )
    return SOURCE_PROFILES[name]