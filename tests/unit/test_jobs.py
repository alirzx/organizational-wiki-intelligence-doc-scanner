from app.core.config import Settings
from app.jobs.callback import deliver_callback
from app.jobs.store import JobStore


def test_memory_job_store_tracks_terminal_state():
    store = JobStore(Settings(job_store_backend="memory"))
    created = store.create("job-1", "41")
    assert created["status"] == "queued"
    processing = store.update("job-1", status="processing")
    assert processing["status"] == "processing"
    completed = store.update("job-1", status="completed", outputs={"ocr": "documents/41/OCR.txt"})
    assert completed["status"] == "completed"
    assert store.get("job-1")["outputs"]["ocr"] == "documents/41/OCR.txt"


def test_callback_uses_bearer_auth(monkeypatch):
    calls = []

    class Response:
        def raise_for_status(self):
            return None

    def fake_post(url, json, headers, timeout):
        calls.append((url, json, headers, timeout))
        return Response()

    monkeypatch.setattr("app.jobs.callback.requests.post", fake_post)
    settings = Settings(
        callback_url="http://backend.test/callback",
        callback_token="secret-token",
        callback_timeout_seconds=5,
        callback_max_attempts=1,
    )
    deliver_callback(settings, {"job_id": "job-1", "status": "completed"})
    assert calls[0][0] == "http://backend.test/callback"
    assert calls[0][2]["Authorization"] == "Bearer secret-token"
