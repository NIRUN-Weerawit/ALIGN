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
echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║            ALIGN — Dependency Installation             ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
info "Method:   ${METHOD}"
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
# Dependencies — latest compatible versions (no dead pins)
# =============================================================================
# Conda core (installed first via conda, then pip for what conda can't provide)
CORE_CONDA=(
    "python=3.10"
    "numpy"
    "scipy"
    "h5py"
    "pillow"
    "wandb"
    "tqdm"
    "requests"
)

# Pip core — install these via pip even in conda, because conda-forge
# versions lag behind or are missing
CORE_PIP=(
    "open-clip-torch"
    "lerobot"
    "torchcodec"
    "xformers"
    "transformers"
    "PyAV"
)

# PyTorch is special — install via pip with CUDA index, not conda
# This avoids channel conflicts and version lag
TORCH_PIP=(
    "torch"
    "torchvision"
)

# Optional: data collection / Isaac Sim / VR deps
OPTIONAL_PIP=(
    "paho-mqtt"
    "tensorflow-datasets"
)

# =============================================================================
# Install functions
# =============================================================================

install_conda() {
    local env_name="${1:-align}"

    info "Creating conda environment '${env_name}'..."
    conda create -y -n "$env_name" python=3.10 -c conda-forge

    info "Installing system deps via conda..."
    conda install -y -n "$env_name" "${CORE_CONDA[@]}" -c conda-forge

    info "Installing PyTorch + CUDA (latest, via pip)..."
    conda run -n "$env_name" pip install "${TORCH_PIP[@]}" \
        --index-url https://download.pytorch.org/whl/cu124

    info "Installing core ML deps via pip..."
    conda run -n "$env_name" pip install "${CORE_PIP[@]}"

    if ! $MINIMAL; then
        info "Installing optional deps..."
        conda run -n "$env_name" pip install "${OPTIONAL_PIP[@]}"
    fi

    # Install ALIGN as dev package
    conda run -n "$env_name" pip install -e "$SCRIPT_DIR"

    ok "Conda environment '${env_name}' ready!"
    echo ""
    echo "  Activate:  conda activate ${env_name}"
    echo "  Verify:    conda run -n ${env_name} python -c \"import open_clip; import xformers; import lerobot; import torchcodec; print('All deps OK')\""
    echo ""
    echo "  Training:  conda run -n ${env_name} python training/pretrain_streaming.py --epochs 10"
    echo ""

    # ── Verify ──
    conda run -n "$env_name" python -c "
import open_clip; print(f'  open_clip:   {open_clip.__version__}')
import xformers; print(f'  xformers:    {xformers.__version__}')
import lerobot; print(f'  lerobot:     {lerobot.__version__}')
import av; print(f'  pyav:        {av.__version__}')
import torch; print(f'  torch:       {torch.__version__}  CUDA:{torch.cuda.is_available()}')
import transformers; print(f'  transformers:{transformers.__version__}')
import wandb; print(f'  wandb:       {wandb.__version__}')
print('All deps OK')
"
}

install_pip() {
    info "Installing PyTorch + CUDA (latest)..."
    pip install "${TORCH_PIP[@]}" \
        --index-url https://download.pytorch.org/whl/cu124

    info "Installing core deps..."
    pip install numpy scipy h5py Pillow wandb tqdm requests
    pip install "${CORE_PIP[@]}"

    if ! $MINIMAL; then
        info "Installing optional deps..."
        pip install "${OPTIONAL_PIP[@]}"
    fi

    pip install -e "$SCRIPT_DIR"

    ok "Dependencies installed in current Python environment!"
    echo ""
    echo "  To verify:"
    echo "    python -c \"import open_clip; import xformers; import lerobot; import torchcodec; print('All deps OK')\""
    echo ""
    echo "  For an isolated venv:"
    echo "    python3 -m venv align-env && source align-env/bin/activate && ./setup.sh pip"
    echo ""
}

# ── Execute ──
case "$METHOD" in
    conda) install_conda "align" ;;
    pip) install_pip ;;
esac

ok "Done!"
echo ""
echo "  Activate:  conda activate align  (or: source align-env/bin/activate)"
echo "  Run:       python training/pretrain_streaming.py --epochs 10"
echo ""
if $MINIMAL; then
    echo "NOTE: Ran with --minimal. Isaac Sim + data collection deps not installed."
    echo "      If needed later, re-run without --minimal."
fi