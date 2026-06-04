#!/bin/bash
# Run this on maya@10.20.4.33 to set up the CPM training environment.
set -e

# ── 1. Miniconda ────────────────────────────────────────────────
if [ ! -f "$HOME/miniconda3/bin/conda" ]; then
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -u -p "$HOME/miniconda3"
    rm /tmp/miniconda.sh
fi
export PATH="$HOME/miniconda3/bin:$PATH"
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

# ── 2. Conda env ────────────────────────────────────────────────
ENV=egcnoi_env
if ! conda env list | grep -q "^$ENV "; then
    echo "Creating conda env $ENV (CPU-only PyTorch)..."
    conda create -y -n $ENV python=3.10
fi
conda activate $ENV

# ── 3. Core packages ────────────────────────────────────────────
echo "Installing packages..."
# PyTorch CPU-only
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu -q

# PyG + PyG Temporal
pip install torch_geometric -q
pip install torch_geometric_temporal -q

# ML utils
pip install scikit-learn numpy tqdm hiddenlayer six -q

echo ""
echo "Verifying install..."
python -c "
import torch, torch_geometric, torch_geometric_temporal, sklearn
print(f'torch={torch.__version__}')
print(f'torch_geometric={torch_geometric.__version__}')
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
print('DynamicGraphTemporalSignal OK')
from torch_geometric_temporal.nn.recurrent import GConvGRU, TGCN
print('GConvGRU/TGCN OK')
print('ALL OK')
"
echo "Setup complete. Activate with: conda activate $ENV"
