#!/usr/bin/env bash
# Sets up an isolated virtualenv, installs dependencies and seeds a .env file.
# Usage:  chmod +x setup.sh && ./setup.sh
set -euo pipefail

VENV_DIR="$HOME/.venvs/codebase_qna_rag"

echo "==> Creating virtual environment at $VENV_DIR"
python3 -m venv "$VENV_DIR"

echo "==> Activating virtual environment"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

echo "==> Installing dependencies"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [ ! -f .env ]; then
  if [ -f .env.example ]; then
    echo "==> Creating .env from .env.example"
    cp .env.example .env
  else
    echo "==> Creating .env"
    printf 'NEBIUS_API_KEY=\nGITHUB_TOKEN=\n' > .env
  fi
else
  echo "==> .env already exists, leaving it untouched"
fi

cat <<EOF

Setup complete.

1. Fill in your secrets:   nano .env
2. Activate the venv:      source $VENV_DIR/bin/activate
3. Run the app:            streamlit run main.py

(You can also skip .env entirely and paste both keys into the app sidebar.)
EOF
