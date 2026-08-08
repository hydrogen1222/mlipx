# Analysis v1 archive

Status: **ARCHIVED / UNSUPPORTED / DO NOT USE FOR SCIENTIFIC RESULTS**

This directory contains the historical mlipx trajectory-analysis
implementation frozen in August 2026. It is preserved only as a development
reference and intentionally does not participate in CI.

本目录仅保存历史实现，未完成当前科研正确性审计，不属于受支持功能，不应直接
用于论文数据。

Do not:

- import it from the live `mlipx` package;
- expose it through CLI, TUI, or API;
- use its numerical output directly for scientific publication;
- add dependencies solely to keep this archive runnable.

Known scientific issues are documented in `KNOWN_ISSUES.md`. The current
mlipx project focuses on reliable MLIP calculations and MLMD trajectory
generation. A future Analysis v2 must be redesigned after the underlying
transport and statistical-mechanics methodology is re-evaluated.
