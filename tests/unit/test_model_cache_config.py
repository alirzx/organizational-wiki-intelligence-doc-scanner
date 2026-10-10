from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_compose_persists_framework_and_ollama_caches():
    dockerfile = (ROOT / "Dockerfile").read_text()
    compose = (ROOT / "compose.yaml").read_text()

    assert "PADDLE_PDX_CACHE_HOME=/app/.cache/paddlex" in dockerfile
    assert "PADDLE_PDX_CACHE_HOME: /app/.cache/paddlex" in compose
    assert "wiki_hami_model_cache:/app/.cache" in compose
    assert "wiki_hami_ollama_models:/root/.ollama" in compose
    assert "OLLAMA_MODELS: /root/.ollama/models" in compose


def test_docker_image_supports_explicit_cpu_and_gpu_dependency_profiles():
    dockerfile = (ROOT / "Dockerfile").read_text()
    compose = (ROOT / "compose.yaml").read_text()

    assert "ARG MODEL_RUNTIME=gpu" in dockerfile
    assert "requirements-paddle-cpu.txt" in dockerfile
    assert "requirements-paddle-gpu.txt" in dockerfile
    assert "requirements-torch-cpu.txt" in dockerfile
    assert "requirements-torch-gpu.txt" in dockerfile
    assert 'MODEL_RUNTIME: "${WIKI_HAMI_MODEL_RUNTIME:-cpu}"' in compose
    assert 'runtime: "${WIKI_HAMI_CONTAINER_RUNTIME:-runc}"' in compose


def test_compose_owns_ollama_and_pulls_only_selected_model_when_needed():
    compose = (ROOT / "compose.yaml").read_text()

    assert "ollama:" in compose
    assert "ollama-model-init:" in compose
    assert "http://ollama:11434" in compose
    assert 'OLLAMA_MODEL: "${WIKI_HAMI_VLM_MODEL_ID:-deepseek-ocr:latest}"' in compose
    assert 'if ollama show "$$OLLAMA_MODEL"' in compose
    assert 'ollama pull "$$OLLAMA_MODEL"' in compose
    assert 'if [ "$$TEXT_EXTRACTION_MODE" != "vlm" ]' in compose
    assert "service_completed_successfully" in compose
