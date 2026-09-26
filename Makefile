.PHONY: install install-dev install-models test api ui worker up down logs smoke

install:
	python -m pip install -r requirements.txt

install-dev:
	python -m pip install -r requirements-dev.txt

install-models:
	python -m pip install -r requirements-paddle-cpu.txt -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
	python -m pip install -r requirements-torch-cpu.txt --index-url https://download.pytorch.org/whl/cpu
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
