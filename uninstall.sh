#!/usr/bin/env sh
set -eu

# Determine script directory
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

echo "=== HotpotQA Uninstall Script ==="
echo "This will remove:"
echo "  - Virtual environment (.venv)"
echo "  - Build artifacts (*.egg-info, __pycache__, etc.)"
echo ""
read -p "Continue? (y/N) " -n 1 -r
echo
if [ "$REPLY" != "y" ] && [ "$REPLY" != "Y" ]; then
  echo "Uninstall cancelled."
  exit 0
fi

# Remove virtual environment
if [ -d "$SCRIPT_DIR/.venv" ]; then
  echo "Removing virtual environment at $SCRIPT_DIR/.venv"
  rm -rf "$SCRIPT_DIR/.venv"
else
  echo "No virtual environment found at $SCRIPT_DIR/.venv"
fi

# Remove build artifacts
echo "Removing build artifacts in $SCRIPT_DIR"
find "$SCRIPT_DIR" -maxdepth 3 -depth -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
find "$SCRIPT_DIR" -maxdepth 3 -depth -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$SCRIPT_DIR" -maxdepth 3 -depth -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

# Remove uv lock file (optional)
if [ -f "$SCRIPT_DIR/uv.lock" ]; then
  echo "Removing uv.lock"
  rm -f "$SCRIPT_DIR/uv.lock"
fi

echo ""
echo "=== Uninstall Complete ==="
echo "HotpotQA environment has been cleaned up."
echo ""
echo "Note: This script does NOT uninstall uv itself."
echo "  To uninstall uv: rm -rf ~/.local/bin/uv ~/.cargo/bin/uv"
