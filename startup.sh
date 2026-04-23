#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# startup.sh – Azure App Service startup script
# Runs before the Streamlit process starts
# ──────────────────────────────────────────────────────────────────
set -e

echo "=== SLD BOM Extraction – App Service Startup ==="

# Install system dependencies for OpenCV
apt-get update -qq && apt-get install -y -qq \
  libgl1-mesa-glx \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender1 \
  2>/dev/null || true

# Install Python dependencies
pip install --upgrade pip -q
pip install -r requirements.lock.txt -q 2>/dev/null || pip install -e . -q

echo "=== Starting Streamlit ==="
exec streamlit run src/hitl/streamlit_app.py \
  --server.port 8000 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false
