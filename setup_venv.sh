#!/usr/bin/env bash
# Crée le venv Python 3.11, installe les dépendances et enregistre le kernel Jupyter.
# Usage : bash setup_venv.sh
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in python3.11 python3.12 "$HOME/.local/bin/python3.11"; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi
if [[ -z "$PYTHON_BIN" ]]; then
  echo "ERREUR: python3.11 ou 3.12 introuvable (requis pour PyTorch)." >&2
  echo "Installez-le puis relancez, ou: PYTHON_BIN=/chemin/vers/python bash setup_venv.sh" >&2
  exit 1
fi
echo "Python utilisé : $PYTHON_BIN ($("$PYTHON_BIN" --version))"

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

python -m ipykernel install --user --name cellid-llm --display-name "Python (cellid-llm)"

python - <<'EOF'
import torch
print(f"torch {torch.__version__}")
print(f"MPS disponible : {torch.backends.mps.is_available()}")
print(f"CUDA disponible : {torch.cuda.is_available()}")
EOF

echo
echo "OK. Lancez Jupyter avec :"
echo "  source .venv/bin/activate && jupyter lab"
echo "puis ouvrez train_cellid_llm.ipynb avec le kernel 'Python (cellid-llm)'."
