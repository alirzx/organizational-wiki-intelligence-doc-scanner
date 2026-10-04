.PHONY: install install-dev install-models test api ui worker up down logs smoke

install:
	python -m pip install -r requirements.txt

install-dev:
	python -m pip install -r requirements-dev.txt

# Production/model runtime is GPU-only and matches the verified CUDA baseline.
install-models:
	python -m pip install -r requirements-paddle-gpu.txt --index-url https://www.paddlepaddle.org.cn/packages/stable/cu129/
	python -m pip install -r requirements-torch-gpu.txt --index-url https://download.pytorch.org/whl/cu130
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
