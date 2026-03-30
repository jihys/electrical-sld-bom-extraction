.PHONY: setup run run-api test clean venv-fix

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
