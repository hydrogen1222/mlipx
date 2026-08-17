#!/usr/bin/env bash
#
# mlipx — one-command installer for all four MLIP engines
# =========================================================
#
# Detects your NVIDIA GPU (or CPU-only), selects the matching CUDA channel
# and framework versions, and installs UMA/MACE/DPA/GRACE into isolated venvs.
#
# This script is a thin wrapper around the Python compatibility matrix in
# mlipx/mlipx/install/.  All logic lives there; the shell script only handles
# argument parsing, GPU detection, and step execution.
#
# Supported hardware:
#   Maxwell  GTX 960 / TITAN X (sm_50/52)  → cu126 Legacy, EXPERIMENTAL
#   Pascal   GTX 1080 Ti, P100 (sm_60/61)   → cu126 Legacy
#   Volta    V100 (sm_70)                    → cu126 Legacy
#   Turing   RTX 20xx (sm_75)               → cu128+ Modern
#   Ampere   RTX 30xx (sm_80/86)            → cu128+ Modern
#   Ada      RTX 40xx / 4090 (sm_89)        → cu128+ Modern
#   Hopper   H100 (sm_90)                    → cu128+ Modern
#   Blackwell RTX 50xx (sm_100/120)          → cu128+ Modern
#   none                                     → CPU wheels
#
# Usage:
#   ./scripts/install_mlipx.sh [options]
#
# Options:
#   --engines uma,mace,dpa,grace   Engines to install (default: all four)
#   --device auto|cuda|cpu         Target device (default: auto)
#   --python 3.12                  Python version for the isolated venvs
#   --source auto|official|china   Package source (default: auto → official)
#   --clean                        Recreate existing venvs from scratch
#   --dry-run                      Print the plan and exit without installing
#   --skip-doctor                  Do not run `mlipx doctor` after installing
#   -h, --help                     Show this help and exit
#
# Examples:
#   ./scripts/install_mlipx.sh                       # auto-detect, install all 4
#   ./scripts/install_mlipx.sh --source china        # use China mirrors
#   ./scripts/install_mlipx.sh --engines uma,mace    # only UMA + MACE
#   ./scripts/install_mlipx.sh --device cpu          # CPU-only machine
#   ./scripts/install_mlipx.sh --clean --dry-run     # preview a clean install
#
# Exit codes: 0 = success, 1 = error, 2 = usage error.
set -euo pipefail

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
say()  { printf '\033[1;32m[mlipx]\033[0m %s\n' "$*"; }
info() { printf '\033[1;34m[mlipx]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[mlipx]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[mlipx] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }
usage(){ sed -n '3,52p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

need_cmd() { command -v "$1" >/dev/null 2>&1 || die "'$1' not found on PATH"; }

# ---------------------------------------------------------------------------
# Option parsing
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINES="uma,mace,dpa,grace"
DEVICE="auto"
PYVER="3.12"
SOURCE="auto"
CLEAN=0
DRY_RUN=0
RUN_DOCTOR=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --engines)  ENGINES="$2"; shift 2 ;;
        --device)   DEVICE="$2"; shift 2 ;;
        --python)   PYVER="$2"; shift 2 ;;
        --source)   SOURCE="$2"; shift 2 ;;
        --clean)    CLEAN=1; shift ;;
        --dry-run)  DRY_RUN=1; RUN_DOCTOR=0; shift ;;
        --skip-doctor) RUN_DOCTOR=0; shift ;;
        -h|--help)  usage; exit 0 ;;
        *) die "unknown option '$1' (see --help)" ;;
    esac
done

# Validate
case "$DEVICE" in auto|cuda|cpu) ;; *) die "--device must be auto, cuda, or cpu" ;; esac
case "$SOURCE" in auto|official|china|offline|custom) ;; *) die "--source must be auto, official, china, offline, or custom" ;; esac

IFS=',' read -ra ENGINE_LIST <<< "$ENGINES"
for e in "${ENGINE_LIST[@]}"; do
    case "$e" in uma|mace|dpa|grace) ;; *) die "unknown engine '$e'" ;; esac
done

# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------
need_cmd uv
need_cmd git
need_cmd python3

cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------
GPU_JSON="null"
if have_gpu() { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; }; then
    if have_gpu; then
        info "Detecting GPU ..."
        CC_RAW="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo "")"
        GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || echo "")"
        GPU_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 || echo "")"
        GPU_VRAM="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo "0")"

        CC_MAJOR="${CC_RAW%%.*}"; CC_MINOR="${CC_RAW#*.}"; CC_MINOR="${CC_MINOR%%.*}"
        CC_MAJOR="${CC_MAJOR//[^0-9]/}"; CC_MINOR="${CC_MINOR//[^0-9]/}"
        GPU_VRAM="${GPU_VRAM//[^0-9]/}"

        if [[ -n "$CC_MAJOR" && -n "$CC_MINOR" ]]; then
            GPU_JSON="[{\"name\": \"$GPU_NAME\", \"cc_major\": $CC_MAJOR, \"cc_minor\": $CC_MINOR, \"driver_version\": \"$GPU_DRIVER\", \"vram_mib\": ${GPU_VRAM:-0}}]"
            info "GPU: $GPU_NAME (CC ${CC_MAJOR}.${CC_MINOR}, ${GPU_VRAM} MiB)"
        else
            warn "Could not parse nvidia-smi output. Falling back to CPU."
        fi
    fi
fi

if [[ "$DEVICE" == "cpu" ]]; then
    GPU_JSON="null"
    info "CPU-only mode (--device cpu)"
fi

# ---------------------------------------------------------------------------
# Generate plan via Python
# ---------------------------------------------------------------------------
info "Source profile: $SOURCE"
info "Engines: ${ENGINE_LIST[*]}"
info "Generating installation plan ..."

PLAN_SCRIPT=$(python3 -c "
import sys, json
sys.path.insert(0, '$REPO_ROOT/mlipx')
from mlipx.install.plan import generate_plan, plan_to_json
from mlipx.install.hardware import GpuInfo

gpus_raw = json.loads('''$GPU_JSON''')
gpus = None
if gpus_raw:
    gpus = [GpuInfo(**g) for g in gpus_raw]

plan = generate_plan(
    gpus=gpus,
    engines='${ENGINES}'.split(','),
    source='${SOURCE}',
    python_version='${PYVER}',
    device='${DEVICE}',
)
print(plan_to_json(plan))
")

# ---------------------------------------------------------------------------
# Show warnings
# ---------------------------------------------------------------------------
echo "$PLAN_SCRIPT" | python3 -c "
import json, sys
plan = json.load(sys.stdin)
for w in plan.get('warnings', []):
    print(f'\033[1;33m[mlipx] WARNING:\033[0m {w}', file=sys.stderr)
"

# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------
if (( DRY_RUN )); then
    echo
    info "DRY RUN — no changes were made.  Install plan:"
    echo
    echo "$PLAN_SCRIPT" | python3 -c "
import json, sys
plan = json.load(sys.stdin)
print(f\"  GPU arch : {plan['gpu_arch']}\")
print(f\"  Source   : {plan['source']}\")
print(f\"  Python   : {plan['python_version']}\")
print()
for i, s in enumerate(plan['steps']):
    print(f\"  [{s['stage']}] {s['description']}\")
    cmd = s['command']
    if len(cmd) > 110:
        cmd = cmd[:107] + '...'
    print(f\"    \$ {cmd}\")
"
    exit 0
fi

# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------
echo
echo "$PLAN_SCRIPT" | python3 -c "
import json, sys, subprocess, os

plan = json.load(sys.stdin)
failed = 0
for i, step in enumerate(plan['steps']):
    label = f'[{step[\"stage\"]}] {step[\"description\"]}'
    print(f'\033[1;34m[mlipx]\033[0m {label}')
    env = os.environ.copy()
    env.update(step.get('env', {}))
    # Ensure uv doesn't read user/project config
    env.setdefault('UV_NO_CONFIG', '1')
    r = subprocess.run(step['command'], shell=True, cwd='$REPO_ROOT', env=env)
    if r.returncode != 0:
        print(f'\033[1;31m[mlipx] FAILED:\033[0m {label} (exit {r.returncode})')
        failed += 1
        break
print()
if failed:
    print(f'\033[1;31m[mlipx] {failed} step(s) failed.\033[0m')
    sys.exit(1)
else:
    print(f'\033[1;32m[mlipx] All {len(plan[\"steps\"])} steps completed.\033[0m')
"

echo
info "Done. Quick reference:"
for engine in "${ENGINE_LIST[@]}"; do
    case "$engine" in
        uma)   echo "  UMA   -> uv run mlipx ..." ;;
        mace)  echo "  MACE  -> .venv-mace/bin/mlipx ..." ;;
        dpa)   echo "  DPA   -> .venv-dpa/bin/mlipx ..." ;;
        grace) echo "  GRACE -> .venv-grace/bin/mlipx ..." ;;
    esac
done
