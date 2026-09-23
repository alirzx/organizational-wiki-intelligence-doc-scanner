from celery import Celery

from app.core.config import get_settings


settings = get_settings()
celery_app = Celery(
    "wiki_hami",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.jobs.tasks"],
)
celery_app.conf.update(
    task_default_queue=settings.celery_queue,
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)
