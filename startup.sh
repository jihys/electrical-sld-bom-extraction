#!/usr/bin/env bash
# startup.sh - Azure App Service startup script

echo "=== SLD BOM Extraction - App Service Startup ==="

cd /home/site/wwwroot

# Persistent virtualenv (survives container restarts on /home)
VENV="/home/site/venv"
MARKER="$VENV/.req_hash"
REQ_HASH=$(md5sum requirements.txt 2>/dev/null | cut -d' ' -f1)

if [[ ! -f "$MARKER" ]] || [[ "$(cat "$MARKER" 2>/dev/null)" != "$REQ_HASH" ]]; then
  echo "Creating virtualenv and installing dependencies..."
  rm -rf "$VENV"
  python -m venv "$VENV"
  source "$VENV/bin/activate"
  pip install --no-cache-dir -r requirements.txt 2>&1
  echo "$REQ_HASH" > "$MARKER"
  echo "Dependencies installed."
else
  echo "Dependencies already installed (cached)."
  source "$VENV/bin/activate"
fi

# Ensure project code is importable
export PYTHONPATH="/home/site/wwwroot:${PYTHONPATH:-}"

echo "=== Starting Streamlit ==="
exec streamlit run src/hitl/streamlit_app.py \
  --server.port 8000 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false
