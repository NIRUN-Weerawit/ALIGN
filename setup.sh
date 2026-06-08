#!/usr/bin/env bash
# =============================================================================
# ALIGN — Dependency Installation Script
#
# Supports:
#   1. Conda (recommended)  — ./setup.sh conda
#   2. pip / venv            — ./setup.sh pip
#
# Environment is created as 'align' for conda, or uses the active venv for pip.
#
# Usage:
#   ./setup.sh                     # auto-detect (conda preferred)
#   ./setup.sh conda               # force conda
#   ./setup.sh pip                 # force pip into current env
#   ./setup.sh -y                  # non-interactive (auto-confirm)
#   ./setup.sh --minimal           # training/inference only (no Isaac Sim)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Color helpers ──
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${BLUE}[ALIGN]${NC} $*"; }
ok()    { echo -e "${GREEN}[ALIGN]${NC} $*"; }
warn()  { echo -e "${YELLOW}[ALIGN]${NC} $*"; }
err()   { echo -e "${RED}[ALIGN] ERROR:${NC} $*" >&2; }

# ── Parse flags ──
AUTO_CONFIRM=false
MINIMAL=false
METHOD="${1:-auto}"

case "$METHOD" in
    -y|--yes) AUTO_CONFIRM=true; METHOD=auto ;;
    --minimal) MINIMAL=true; METHOD=auto ;;
    conda|pip|auto) ;;
    *) err "Unknown method: $METHOD. Use 'conda', 'pip', or 'auto'."; exit 1 ;;
esac

# Shift consumed args
if [ "$METHOD" = "auto" ] && [ "${1:-}" = "--minimal" ]; then MINIMAL=true; fi
if [ "$METHOD" = "auto" ] && ([ "${1:-}" = "-y" ] || [ "${1:-}" = "--yes" ]); then AUTO_CONFIRM=true; fi

# ── Detect available package managers ──
HAS_CONDA=false
if command -v conda &>/dev/null; then
    HAS_CONDA=true
fi

if [ "$METHOD" = "auto" ]; then
    if $HAS_CONDA; then
        METHOD="conda"
    else
        METHOD="pip"
    fi
fi

# ── Confirm ──
FULL_NOTE=""
MINIMAL_NOTE=""
if $MINIMAL; then
    MINIMAL_NOTE=" (training/inference only, no data collection tools)"
fi

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║            ALIGN — Dependency Installation             ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
info "Method:   ${METHOD}${FULL_NOTE}${MINIMAL_NOTE}"
info "Minimal:  ${MINIMAL} (excludes Isaac Sim and data collection deps)"
echo ""

if ! $AUTO_CONFIRM; then
    echo -n "Proceed? [Y/n] "
    read -r REPLY
    REPLY="${REPLY:-y}"
    if [[ ! "$REPLY" =~ ^[Yy] ]]; then
        info "Aborted."
        exit 0
    fi
fi

# =============================================================================
# Core training/inference dependencies (always installed)
# =============================================================================
CORE_CONDA=(
    "python=3.10"
    "pytorch=2.4.0"
    "torchvision=0.19.0"
    "numpy"
    "scipy"
    "h5py"
    "pillow"
    "wandb"
    "tqdm"
    "requests"
)

# pip-equivalent of core deps (used in conda for packages not on conda-forge)
CORE_PIP=(
    "open-clip-torch"
    "lerobot>=1.0"
    "xformers==0.0.28"
)

# =============================================================================
# Optional: data collection / Isaac Sim / VR deps
# =============================================================================
OPTIONAL_CONDA=(
    "paho-mqtt"
)
OPTIONAL_PIP=(
    "tensorflow-datasets"
)

# =============================================================================
# Install
# =============================================================================

install_conda() {
    local env_name="${1:-align}"

    info "Creating conda environment '${env_name}'..."
    conda create -y -n "$env_name" python=3.10 -c conda-forge
    eval "$(conda shell.bash hook)"
    conda activate "$env_name"

    info "Installing core deps via conda..."
    conda install -y -n "$env_name" "${CORE_CONDA[@]}" -c pytorch -c conda-forge || {
        warn "conda install had issues; trying pip fallback..."
    }

    info "Installing core deps via pip..."
    pip install "${CORE_PIP[@]}"

    if ! $MINIMAL; then
        info "Installing optional deps..."
        if [ ${#OPTIONAL_CONDA[@]} -gt 0 ]; then
            conda install -y -n "$env_name" "${OPTIONAL_CONDA[@]}" -c conda-forge 2>/dev/null || true
        fi
        pip install "${OPTIONAL_PIP[@]}"
    fi

    # Optional: install this package in dev mode
    pip install -e "$SCRIPT_DIR"

    ok "Conda environment '${env_name}' ready!"
    echo ""
    echo "  Activate:  conda activate ${env_name}"
    echo "  Deactivate later: conda deactivate"
    echo ""
    echo "  For Isaac Sim data collection (separate install required):"
    echo "    pip install isaacsim   # follow NVIDIA's install instructions"
}

install_pip() {
    info "Installing core deps via pip..."
    pip install --upgrade pip

    info "Installing PyTorch 2.4.0 + CUDA 12.1..."
    pip install torch==2.4.0 torchvision==0.19.0 \
        --index-url https://download.pytorch.org/whl/cu121
    pip install xformers==0.0.28 \
        --index-url https://download.pytorch.org/whl/cu121 --no-deps
    pip install numpy scipy h5py Pillow wandb tqdm requests
    pip install open-clip-torch lerobot

    if ! $MINIMAL; then
        info "Installing optional deps..."
        pip install paho-mqtt tensorflow-datasets
    fi

    pip install -e "$SCRIPT_DIR"

    ok "Dependencies installed in current Python environment!"
    echo ""
    echo "  For Isaac Sim data collection (separate install required):"
    echo "    pip install isaacsim   # follow NVIDIA's install instructions"
    echo ""
    echo "  If you want an isolated venv:"
    echo "    python3 -m venv align-env"
    echo "    source align-env/bin/activate"
    echo "    ./setup.sh pip"
}

# ── Execute ──
case "$METHOD" in
    conda)
        install_conda "align"
        ;;
    pip)
        install_pip
        ;;
esac

ok "Done! Run your first training with:"
echo ""
echo "    conda activate align  (or: source align-env/bin/activate)"
echo "    python training/pretrain_streaming.py --epochs 10"
echo ""
if $MINIMAL; then
    echo "NOTE: Ran with --minimal. Isaac Sim + data collection deps are not installed."
    echo "      If you need them later, re-run without --minimal."
fi