#!/usr/bin/env bash
#
# mlipx — one-command installer (thin bootstrap wrapper)
# ======================================================
#
# This wrapper only:
#   1. verifies `uv` is available
#   2. locates the repository root
#   3. selects a Python 3.10–3.12 interpreter (via uv, never the old system
#      python3) so the modern mlipx.install code can run
#   4. delegates everything else to `python -m mlipx.install`
#
# All logic (GPU detection, compatibility matrix, plan generation, dry-run,
# execution) lives in Python under mlipx/mlipx/install/.
#
# Usage: ./scripts/install_mlipx.sh [--engines ...] [--device ...] [--source ...]
#        [--python ...] [--clean] [--skip-doctor] [--dry-run]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v uv >/dev/null 2>&1; then
    echo "[mlipx] ERROR: 'uv' not found on PATH." >&2
    echo "        Install it first:  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

cd "$REPO_ROOT"

# Pick a Python interpreter that uv manages (3.12 preferred, 3.10/3.11 ok).
# This avoids depending on an old system python3 that cannot parse the
# modern type annotations used by mlipx.install.
PY_SPEC="${MLIPX_INSTALL_PYTHON:-3.12}"
UV_PY="$(uv python find "$PY_SPEC" 2>/dev/null \
    || { uv python install "$PY_SPEC" >/dev/null 2>&1 && uv python find "$PY_SPEC"; } \
    || { echo "[mlipx] ERROR: could not obtain a Python $PY_SPEC interpreter via uv." >&2; exit 1; })"

# Make the mlipx package importable: PYTHONPATH -> <repo>/mlipx (project root),
# so `import mlipx` resolves to <repo>/mlipx/mlipx.
export PYTHONPATH="$REPO_ROOT/mlipx${PYTHONPATH:+:$PYTHONPATH}"

exec "$UV_PY" -m mlipx.install "$@"
