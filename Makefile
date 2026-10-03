.PHONY: install install-dev install-models install-models-gpu test api ui worker up down logs smoke

install:
	python -m pip install -r requirements.txt

install-dev:
	python -m pip install -r requirements-dev.txt

install-models:
	python -m pip install -r requirements-paddle-cpu.txt -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
	python -m pip install -r requirements-torch-cpu.txt --index-url https://download.pytorch.org/whl/cpu
	python -m pip install -r requirements-models.txt

install-models-gpu:
	test -n "$$PADDLE_GPU_INDEX_URL" || (echo "Set PADDLE_GPU_INDEX_URL for the verified target CUDA runtime" >&2; exit 1)
	test -n "$$TORCH_GPU_INDEX_URL" || (echo "Set TORCH_GPU_INDEX_URL for the verified target CUDA runtime" >&2; exit 1)
	python -m pip install -r requirements-paddle-gpu.txt --index-url "$$PADDLE_GPU_INDEX_URL"
	python -m pip install -r requirements-torch-gpu.txt --index-url "$$TORCH_GPU_INDEX_URL"
	python -m pip install -r requirements-models.txt

api:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

ui:
	streamlit run ui/streamlit_app.py --server.port 8501

worker:
	python run.py --worker

test:
	python -m pytest -q

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f api worker ui

smoke:
	python scripts/smoke_test.py
