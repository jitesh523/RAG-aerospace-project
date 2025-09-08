PY=python

export PYTHONPATH := $(PWD)

install:
	pip install -r requirements.txt

run:
	uvicorn src.app.fastapi_app:app --host 0.0.0.0 --port $${PORT:-8000}

format:
	@echo "Add your formatter here (ruff/black) if you like"

test:
	pytest -q

ingest:
	$(PY) src/ingest/ingest.py --input $(INPUT_DIR) --batch_size 200

eval:
	$(PY) src/eval/evaluate.py --questions ./data/eval/questions.json

docker-build:
	docker build -t rag-aerospace:local -f docker/Dockerfile .

docker-run:
	docker run --env-file .env -p 8000:8000 rag-aerospace:local

helm-dryrun:
	helm template k8s/helm
