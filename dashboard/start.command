#!/bin/bash
set -euo pipefail
dash_folder="$(cd "$(dirname "$0")" && pwd)"
cd "$dash_folder"
dash_port="${1:-8501}"
if ! [[ "$dash_port" =~ ^[0-9]+$ ]]; then
  echo "Choose a numeric local port, such as 8501."
  exit 1
fi
if [ ! -x .venv/bin/streamlit ]; then
  if command -v uv >/dev/null 2>&1; then
    dash_uv="$(command -v uv)"
  elif [ -x "$HOME/.local/bin/uv" ]; then
    dash_uv="$HOME/.local/bin/uv"
  else
    echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/"
    echo "Then open this launcher again. No CCTV model or raw images are required."
    exit 1
  fi
  "$dash_uv" sync --frozen --python 3.11
fi
echo "Dashboard: http://127.0.0.1:$dash_port"
exec .venv/bin/streamlit run app.py --server.address=127.0.0.1 --server.port="$dash_port"
