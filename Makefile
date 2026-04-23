.PHONY: setup run run-api test clean venv-fix mode-local mode-cloud

# === Setup ===
setup:
	bash setup.sh

# === Run ===
run:
	streamlit run src/hitl/streamlit_app.py --server.port 8501

run-bg:
	tmux new -d -s demo "cd $(PWD) && source venv/bin/activate && python -m streamlit run src/hitl/streamlit_app.py --server.port 8501"
	@echo "Server started in tmux session 'demo'"
	@echo "  Attach: tmux attach -t demo"
	@echo "  Stop:   tmux kill-session -t demo"

run-api:
	uvicorn src.api.app:app --reload --port 8000

# === Test ===
test:
	pytest tests/ -v -s

test-e2e:
	python tests/e2e_demo_test.py

# === Utilities ===
venv-fix:
	@echo "Recreating venv to fix broken shebang..."
	rm -rf venv
	python3 -m venv venv
	. venv/bin/activate && pip install --upgrade pip -q && pip install -e ".[dev]" -q
	@echo "Done. Run: source venv/bin/activate"

freeze:
	python -m pip freeze > requirements.lock.txt
	@echo "Saved to requirements.lock.txt"

clean:
	rm -rf outputs/ checkpoints/ __pycache__ src/__pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

status:
	@echo "=== Server ==="
	@ss -tlnp 2>/dev/null | grep "850" || echo "No server running"
	@echo "=== tmux ==="
	@tmux ls 2>/dev/null || echo "No tmux sessions"
	@echo "=== venv ==="
	@head -1 venv/bin/streamlit 2>/dev/null || echo "streamlit not found"
	@echo "=== Storage Mode ==="
	@grep -s "^ENABLE_PERSISTENT_STATE" .env 2>/dev/null || echo "ENABLE_PERSISTENT_STATE not set (default: local)"

# === Storage Mode Switching ===
mode-local:
	@if [ -f .env.local ]; then \
		cp .env.local .env; \
	elif [ -f .env.local.example ]; then \
		echo "WARN: .env.local not found, copying from .env.local.example"; \
		echo "      Edit .env to fill in your API keys"; \
		cp .env.local.example .env; \
	else \
		echo "ERROR: Neither .env.local nor .env.local.example found"; \
		exit 1; \
	fi
	@echo "Switched to LOCAL storage mode"
	@echo "  Checkpoints → ./checkpoints/"
	@echo "  Outputs     → ./outputs/"

mode-cloud:
	@if [ -f .env.cloud ]; then \
		cp .env.cloud .env; \
	elif [ -f .env.cloud.example ]; then \
		echo "WARN: .env.cloud not found. Run 'infra/deploy.sh' first to deploy Azure resources."; \
		echo "      Copying .env.cloud.example — edit .env to fill in endpoints"; \
		cp .env.cloud.example .env; \
	else \
		echo "ERROR: Neither .env.cloud nor .env.cloud.example found"; \
		exit 1; \
	fi
	@echo "Switched to CLOUD storage mode"
	@echo "  State     → Azure Cosmos DB"
	@echo "  Artifacts → Azure Blob Storage"
	@echo "  Local     → ./checkpoints/ (also kept)"

mode-info:
	@echo "── Storage Mode ──"
	@if grep -sq "ENABLE_PERSISTENT_STATE=true" .env 2>/dev/null; then \
		echo "  Current: CLOUD (Cosmos DB + Blob Storage)"; \
		grep "AZURE_COSMOS_ENDPOINT\|AZURE_STORAGE_BLOB_ENDPOINT" .env 2>/dev/null | sed 's/^/  /'; \
	else \
		echo "  Current: LOCAL (filesystem only)"; \
		grep "CHECKPOINT_DIR\|OUTPUT_DIR" .env 2>/dev/null | sed 's/^/  /' || echo "  Defaults: ./checkpoints/, ./outputs/"; \
	fi
	@echo ""
	@echo "  Switch:  make mode-local  or  make mode-cloud"
