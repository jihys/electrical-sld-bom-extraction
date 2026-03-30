#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv"
PYTHON="${PYTHON:-python3}"

echo "=== Electrical SLD BOM Extraction — Setup ==="

# 1. Python version check
echo "[1/5] Checking Python version..."
$PYTHON -c "
import sys
v = sys.version_info
assert v >= (3, 10), f'Python 3.10+ required, got {v.major}.{v.minor}'
print(f'  Python {v.major}.{v.minor}.{v.micro} ✓')
"

# 2. Create venv
echo "[2/5] Creating virtual environment..."
if [ -d "$VENV_DIR" ]; then
    echo "  venv already exists. Delete and recreate? (y/N)"
    read -r answer
    if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
        rm -rf "$VENV_DIR"
        $PYTHON -m venv "$VENV_DIR"
        echo "  Recreated ✓"
    else
        echo "  Keeping existing venv ✓"
    fi
else
    $PYTHON -m venv "$VENV_DIR"
    echo "  Created ✓"
fi

source "$VENV_DIR/bin/activate"

# 3. Install dependencies
echo "[3/5] Installing dependencies..."
pip install --upgrade pip -q
pip install -e ".[dev]" -q
echo "  Installed ✓"

# 4. .env file
echo "[4/5] Checking .env file..."
if [ ! -f "$PROJECT_DIR/.env" ]; then
    cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
    echo "  Created .env from .env.example — please fill in your keys"
else
    echo "  .env exists ✓"
fi

# 5. Create output directories
echo "[5/5] Creating directories..."
mkdir -p "$PROJECT_DIR/outputs" "$PROJECT_DIR/checkpoints"
echo "  Done ✓"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  source venv/bin/activate"
echo "  # Edit .env with your Azure keys"
echo "  streamlit run src/hitl/streamlit_app.py --server.port 8501"
echo ""
