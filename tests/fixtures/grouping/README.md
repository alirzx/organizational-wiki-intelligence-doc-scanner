# Grouping Test Fixtures

Fixtures in this directory must be small, deterministic, and safe to commit.

- Record the source and license for any real document excerpt.
- Prefer synthetic Persian/English examples that exercise one behavior at a time.
- Never store OCR secrets, embeddings, production documents, or trained production models.
- The default pytest suite must not contact Ollama, download a model, or require LightGBM.
- Live-service and optional-dependency smoke tests must use explicit pytest markers and opt-in configuration.

Generated JSONL datasets, model packages, and benchmark reports belong under ignored
runtime paths (`data/grouping/`, `models/grouping/`, and `reports/grouping/`).
