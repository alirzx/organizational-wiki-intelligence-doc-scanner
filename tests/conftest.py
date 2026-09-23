"""Keep the automated suite deterministic and free of model/network downloads."""

import os


os.environ["WIKI_HAMI_OCR_BACKEND"] = "mock"
os.environ["WIKI_HAMI_FIGURE_TABLE_BACKEND"] = "mock"
os.environ["WIKI_HAMI_STAMP_SIGNATURE_BACKEND"] = "mock"
os.environ["WIKI_HAMI_MINIO_ENABLED"] = "true"
os.environ["WIKI_HAMI_MINIO_ENDPOINT"] = "minio.test:9000"
os.environ["WIKI_HAMI_MINIO_PUBLIC_BASE_URL"] = "http://minio.test:9000"
os.environ["WIKI_HAMI_MINIO_BUCKET"] = "media"
os.environ["WIKI_HAMI_MINIO_ACCESS_KEY"] = "test-access"
os.environ["WIKI_HAMI_MINIO_SECRET_KEY"] = "test-secret"
