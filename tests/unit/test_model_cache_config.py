from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_container_profiles_persist_the_actual_paddlex_cache():
    dockerfile = (ROOT / "Dockerfile").read_text()
    local_compose = (ROOT / "compose.yaml").read_text()
    prod_compose = (ROOT / "deployment/compose.prod.yaml").read_text()

    assert "PADDLE_PDX_CACHE_HOME=/app/.cache/paddlex" in dockerfile
    assert "PADDLE_PDX_CACHE_HOME: /app/.cache/paddlex" in local_compose
    assert "wiki_hami_model_cache:/app/.cache" in local_compose
    assert "PADDLE_PDX_CACHE_HOME: /var/lib/wiki-hami/cache/paddlex" in prod_compose
    assert "/cache:/var/lib/wiki-hami/cache" in prod_compose


def test_gpu_image_profile_is_explicit_and_cpu_override_removes_reservations():
    dockerfile = (ROOT / "Dockerfile").read_text()
    local_compose = (ROOT / "compose.yaml").read_text()
    prod_compose = (ROOT / "deployment/compose.prod.yaml").read_text()
    cpu_override = (ROOT / "compose.cpu.yaml").read_text()

    assert "ARG MODEL_RUNTIME=gpu" in dockerfile
    assert "requirements-paddle-gpu.txt" in dockerfile
    assert "requirements-torch-gpu.txt" in dockerfile
    assert "PADDLE_GPU_INDEX_URL is required" in dockerfile
    assert local_compose.count("capabilities: [gpu]") == 2
    assert prod_compose.count("capabilities: [gpu]") == 2
    assert cpu_override.count("devices: !reset []") == 2
