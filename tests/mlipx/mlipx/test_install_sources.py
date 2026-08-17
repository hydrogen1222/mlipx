"""Tests for mlipx.install.sources — source profiles and uv arg builders."""

from __future__ import annotations

import pytest

from mlipx.install.sources import (
    SOURCE_PROFILES,
    build_offline_args,
    build_package_source_args,
    build_torch_source_args,
    resolve_source,
)


def test_resolve_auto_is_official() -> None:
    assert resolve_source("auto").name == "official"


def test_resolve_unknown_raises() -> None:
    with pytest.raises(ValueError):
        resolve_source("nope")


def test_official_no_package_override() -> None:
    src = resolve_source("official")
    assert build_package_source_args(src) == []


def test_china_package_uses_tuna() -> None:
    src = resolve_source("china")
    args = build_package_source_args(src)
    assert args == [
        "--index-url",
        "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple",
    ]


def test_china_torch_uses_aliyun_find_links() -> None:
    src = resolve_source("china")
    args = build_torch_source_args(src, "cu126")
    assert args == ["--find-links", "https://mirrors.aliyun.com/pytorch-wheels/cu126/"]


def test_official_torch_uses_pytorch_index() -> None:
    src = resolve_source("official")
    args = build_torch_source_args(src, "cu128")
    assert args == ["--index-url", "https://download.pytorch.org/whl/cu128"]


def test_offline_no_urls() -> None:
    src = resolve_source("offline")
    assert src.offline is True
    assert build_offline_args(src) == ["--offline"]
    assert build_package_source_args(src) == []
    assert build_torch_source_args(src, "cu126") == []
    assert build_torch_source_args(src, "cu128") == []


def test_custom_no_overrides() -> None:
    src = resolve_source("custom")
    assert build_package_source_args(src) == []
    assert build_torch_source_args(src, "cu126") == []
    assert build_torch_source_args(src, "cu128") == []


def test_all_profiles_defined() -> None:
    for name in ("auto", "official", "china", "offline", "custom"):
        assert name in SOURCE_PROFILES
