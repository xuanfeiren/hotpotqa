#!/usr/bin/env sh
set -eu

# Determine script directory
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

# Ensure uv is installed
if command -v uv >/dev/null 2>&1; then
  echo "uv is already installed: $(command -v uv)"
else
  echo "uv not found; installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Ensure uv is available on PATH for this shell session
if ! command -v uv >/dev/null 2>&1; then
  if [ -f "$HOME/.local/bin/env" ]; then
    . "$HOME/.local/bin/env"
  elif [ -x "$HOME/.local/bin/uv" ]; then
    PATH="$HOME/.local/bin:$PATH"
    export PATH
  fi
  hash -r 2>/dev/null || true
fi

# Locate uv binary
if command -v uv >/dev/null 2>&1; then
  UV_BIN=$(command -v uv)
elif [ -x "$HOME/.local/bin/uv" ]; then
  UV_BIN="$HOME/.local/bin/uv"
else
  echo "uv not found after installation; ensure \$HOME/.local/bin is on PATH"
  exit 1
fi

# Install Python dependencies with uv
echo "Installing Python dependencies with uv in $SCRIPT_DIR"
(cd "$SCRIPT_DIR" && "$UV_BIN" sync)
echo "Activating the Python environment at $SCRIPT_DIR/.venv"
. "$SCRIPT_DIR/.venv/bin/activate"

# Verify key imports
echo ""
echo "Verifying installation..."
python -c "import opto; print('  opto (Trace): OK')"
python -c "import dspy; print('  dspy: OK')"
python -c "import gepa; print('  gepa: OK')"
python -c "from openevolve.api import run_evolution; print('  openevolve: OK')"
python -c "from hotpotqa_eval import create_dataset; print('  hotpotqa_eval: OK')"

echo ""
echo "=== Installation Complete ==="
ACTIVATE_CMD="source \"$SCRIPT_DIR/.venv/bin/activate\""
echo "To activate this environment later, run: $ACTIVATE_CMD"
echo "Or simply use: uv run python <script.py>"
