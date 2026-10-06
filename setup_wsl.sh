#!/usr/bin/env bash
# ============================================================================
# WSL2 Ubuntu setup script for LLVM IR SHAP research project
# Installs LLVM 17 prebuilt toolchain and Python dependencies.
# Run this ONCE inside WSL2 Ubuntu:  bash setup_wsl.sh
# ============================================================================
set -euo pipefail

echo "=== LLVM IR SHAP Project — WSL2 Setup ==="

# ---------- 1. System packages ----------
sudo apt-get update
sudo apt-get install -y --no-install-recommends wget gnupg software-properties-common python3 python3-pip python3-venv git lsb-release
sudo apt-get clean

# ---------- 2. LLVM 17 ----------
sudo apt-get update
sudo apt-get install -y --no-install-recommends clang-17 llvm-17 llvm-17-tools
sudo apt-get clean

# Symlink so scripts can call 'clang', 'opt', etc. without version suffix
sudo update-alternatives --install /usr/bin/clang   clang   /usr/bin/clang-17   100
sudo update-alternatives --install /usr/bin/opt     opt     /usr/bin/opt-17     100
sudo update-alternatives --install /usr/bin/llvm-extract llvm-extract /usr/bin/llvm-extract-17 100
sudo update-alternatives --install /usr/bin/llvm-dis llvm-dis /usr/bin/llvm-dis-17 100
sudo update-alternatives --install /usr/bin/llvm-as  llvm-as  /usr/bin/llvm-as-17  100

# ---------- 3. Python virtual environment ----------
VENV_DIR="$(cd "$(dirname "$0")" && pwd)/.venv"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip --no-cache-dir
pip install --no-cache-dir -r "$(cd "$(dirname "$0")" && pwd)/requirements.txt"

# ---------- 4. Verify ----------
echo ""
echo "=== Verification ==="
clang --version | head -1
opt --version 2>&1 | head -1
python3 -c "import xgboost, shap, sklearn; print('Python packages OK')"
echo ""
echo "Setup complete.  Activate the venv with:"
echo "  source $VENV_DIR/bin/activate"
