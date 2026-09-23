from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from threading import Lock
from urllib.parse import quote, unquote, urlparse

from minio import Minio
from minio.error import S3Error

from app.core.config import Settings


class MinioStorageError(RuntimeError):
    """Base error for MinIO acquisition and persistence failures."""


class MinioConfigurationError(MinioStorageError):
    pass


class MinioUrlError(MinioStorageError):
    pass


class MinioObjectNotFound(MinioStorageError):
    pass


@dataclass(frozen=True)
class MinioObjectRef:
    bucket: str
    object_key: str
    source_url: str


@dataclass(frozen=True)
class MinioObjectData:
    ref: MinioObjectRef
    data: bytes
    filename: str
    content_type: str | None
    etag: str | None
    size: int
    last_modified: datetime | None


class MinioStorageService:
    """Authenticated MinIO/S3 access with strict URL-to-object validation.

    Backend-provided URLs are never fetched directly over HTTP. The URL is used only
    to identify a configured bucket/object. Bytes are read and artifacts are written
    through the authenticated MinIO client using ``minio_endpoint``; this keeps
    container/internal S3 routing independent from the host/port in
    ``minio_public_base_url``.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client: Minio | None = None
        self._client_lock = Lock()

    def _require_enabled(self) -> None:
        if not self.settings.minio_enabled:
            raise MinioConfigurationError("MinIO integration is disabled")
        if not self.settings.minio_bucket.strip():
            raise MinioConfigurationError("WIKI_HAMI_MINIO_BUCKET is required")
        if not self.settings.minio_access_key or not self.settings.minio_secret_key:
            raise MinioConfigurationError(
                "WIKI_HAMI_MINIO_ACCESS_KEY and WIKI_HAMI_MINIO_SECRET_KEY are required"
            )

    def _client_endpoint(self) -> tuple[str, bool]:
        raw = self.settings.minio_endpoint.strip()
        if not raw:
            raise MinioConfigurationError("WIKI_HAMI_MINIO_ENDPOINT is required")
        if "://" in raw:
            parsed = urlparse(raw)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise MinioConfigurationError("invalid WIKI_HAMI_MINIO_ENDPOINT")
            if parsed.path not in {"", "/"}:
                raise MinioConfigurationError("MinIO client endpoint must not contain a path")
            return parsed.netloc, parsed.scheme == "https"
        return raw.strip("/"), self.settings.minio_secure

    def public_base_url(self) -> str:
        raw = self.settings.minio_public_base_url.strip()
        if not raw:
            endpoint, secure = self._client_endpoint()
            raw = f"{'https' if secure else 'http'}://{endpoint}"
        parsed = urlparse(raw)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise MinioConfigurationError("invalid WIKI_HAMI_MINIO_PUBLIC_BASE_URL")
        if parsed.username or parsed.password:
            raise MinioConfigurationError("MinIO public base URL must not contain credentials")
        return raw.rstrip("/")

    @staticmethod
    def _authority(parsed) -> tuple[str, int]:
        if not parsed.hostname:
            raise MinioUrlError("image_url has no hostname")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        return parsed.hostname.lower(), port

    @staticmethod
    def _normalize_object_key(object_key: str) -> str:
        key = object_key.lstrip("/")
        if not key:
            raise MinioUrlError("object_key is required")
        if any(segment in {"", ".", ".."} for segment in key.split("/")):
            raise MinioUrlError("object_key contains an invalid path segment")
        return key

    def parse_image_url(self, image_url: str) -> MinioObjectRef:
        self._require_enabled()
        supplied = urlparse(image_url)
        expected = urlparse(self.public_base_url())
        if supplied.scheme not in {"http", "https"}:
            raise MinioUrlError("image_url must use http or https")
        if supplied.username or supplied.password or supplied.fragment:
            raise MinioUrlError("image_url contains unsupported URL components")
        if self._authority(supplied) != self._authority(expected):
            raise MinioUrlError(
                "image_url host/port does not match WIKI_HAMI_MINIO_PUBLIC_BASE_URL"
            )

        supplied_path = unquote(supplied.path).strip("/")
        base_path = unquote(expected.path).strip("/")
        if base_path:
            prefix = f"{base_path}/"
            if not supplied_path.startswith(prefix):
                raise MinioUrlError("image_url path does not match the configured MinIO base path")
            supplied_path = supplied_path[len(prefix) :]

        parts = supplied_path.split("/", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise MinioUrlError("image_url must contain /<bucket>/<object_key>")
        bucket, object_key = parts
        if bucket != self.settings.minio_bucket:
            raise MinioUrlError(
                f"image_url bucket must be {self.settings.minio_bucket!r}"
            )
        key = self._normalize_object_key(object_key)
        # Rebuild a canonical URL without a query string so presigned tokens or
        # other transient credentials can never leak into response provenance.
        canonical_url = self.build_public_url(key)
        return MinioObjectRef(bucket=bucket, object_key=key, source_url=canonical_url)

    def build_public_url(self, object_key: str) -> str:
        key = self._normalize_object_key(object_key)
        return (
            f"{self.public_base_url()}/{quote(self.settings.minio_bucket, safe='')}/"
            f"{quote(key, safe='/')}"
        )

    def _get_client(self) -> Minio:
        self._require_enabled()
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    endpoint, secure = self._client_endpoint()
                    self._client = Minio(
                        endpoint,
                        access_key=self.settings.minio_access_key,
                        secret_key=self.settings.minio_secret_key,
                        secure=secure,
                    )
        return self._client

    @staticmethod
    def _translate_s3_error(exc: S3Error, *, object_key: str | None = None) -> MinioStorageError:
        if exc.code in {"NoSuchKey", "NoSuchObject", "NoSuchBucket", "XMinioInvalidObjectName"}:
            target = f" object {object_key!r}" if object_key else ""
            return MinioObjectNotFound(f"MinIO{target} was not found: {exc.code}")
        return MinioStorageError(f"MinIO error {exc.code}: {exc.message}")

    def check_connection(self) -> bool:
        client = self._get_client()
        try:
            return bool(client.bucket_exists(self.settings.minio_bucket))
        except S3Error as exc:
            raise self._translate_s3_error(exc) from exc
        except Exception as exc:
            raise MinioStorageError(f"could not reach MinIO: {exc}") from exc

    def fetch_url(self, image_url: str) -> MinioObjectData:
        ref = self.parse_image_url(image_url)
        return self.fetch_object(ref.object_key, source_url=ref.source_url)

    def fetch_object(self, object_key: str, *, source_url: str | None = None) -> MinioObjectData:
        self._require_enabled()
        key = self._normalize_object_key(object_key)
        client = self._get_client()
        try:
            stat = client.stat_object(self.settings.minio_bucket, key)
            size = int(stat.size)
            if size <= 0:
                raise MinioStorageError("MinIO object is empty")
            if size > self.settings.max_upload_bytes:
                raise MinioStorageError(
                    f"MinIO object exceeds max_upload_bytes={self.settings.max_upload_bytes}"
                )

            response = client.get_object(self.settings.minio_bucket, key)
            try:
                data = response.read(self.settings.max_upload_bytes + 1)
            finally:
                response.close()
                response.release_conn()
            if len(data) > self.settings.max_upload_bytes:
                raise MinioStorageError(
                    f"MinIO object exceeds max_upload_bytes={self.settings.max_upload_bytes}"
                )
        except S3Error as exc:
            raise self._translate_s3_error(exc, object_key=key) from exc
        except MinioStorageError:
            raise
        except Exception as exc:
            raise MinioStorageError(f"failed to read MinIO object {key!r}: {exc}") from exc

        ref = MinioObjectRef(
            bucket=self.settings.minio_bucket,
            object_key=key,
            source_url=source_url or self.build_public_url(key),
        )
        return MinioObjectData(
            ref=ref,
            data=data,
            filename=key.rsplit("/", 1)[-1] or "image",
            content_type=getattr(stat, "content_type", None),
            etag=getattr(stat, "etag", None),
            size=size,
            last_modified=getattr(stat, "last_modified", None),
        )

    def object_exists(self, object_key: str) -> bool:
        self._require_enabled()
        key = self._normalize_object_key(object_key)
        client = self._get_client()
        try:
            client.stat_object(self.settings.minio_bucket, key)
            return True
        except S3Error as exc:
            if exc.code in {"NoSuchKey", "NoSuchObject"}:
                return False
            raise self._translate_s3_error(exc, object_key=key) from exc
        except Exception as exc:
            raise MinioStorageError(f"failed to stat MinIO object {key!r}: {exc}") from exc

    def put_bytes(self, object_key: str, data: bytes, *, content_type: str) -> str:
        """Write one object and return its canonical configured URL.

        ``put_object`` replaces an existing object with the same key, which gives the
        artifact pipeline deterministic/idempotent overwrite behavior on Celery retries.
        """
        self._require_enabled()
        key = self._normalize_object_key(object_key)
        client = self._get_client()
        try:
            client.put_object(
                self.settings.minio_bucket,
                key,
                BytesIO(data),
                length=len(data),
                content_type=content_type,
            )
        except S3Error as exc:
            raise self._translate_s3_error(exc, object_key=key) from exc
        except Exception as exc:
            raise MinioStorageError(f"failed to write MinIO object {key!r}: {exc}") from exc
        return self.build_public_url(key)

    def put_text(
        self,
        object_key: str,
        text: str,
        *,
        content_type: str = "text/plain; charset=utf-8",
    ) -> str:
        return self.put_bytes(object_key, text.encode("utf-8"), content_type=content_type)

    def delete_prefix(self, prefix: str) -> int:
        """Delete every object under a non-empty prefix.

        Callers must pass an AI-owned prefix such as
        ``documents/<document_id>/OCR/``. This deliberately refuses an empty prefix so a
        retry-cleanup bug cannot turn into a bucket-wide deletion.
        """
        self._require_enabled()
        normalized = prefix.lstrip("/")
        if not normalized or normalized in {".", ".."}:
            raise MinioUrlError("a non-empty object prefix is required for deletion")
        if any(segment in {".", ".."} for segment in normalized.split("/") if segment):
            raise MinioUrlError("prefix contains an invalid path segment")

        client = self._get_client()
        deleted = 0
        try:
            for item in client.list_objects(
                self.settings.minio_bucket,
                prefix=normalized,
                recursive=True,
                include_user_meta=False,
            ):
                if getattr(item, "is_dir", False):
                    continue
                client.remove_object(self.settings.minio_bucket, item.object_name)
                deleted += 1
        except S3Error as exc:
            raise self._translate_s3_error(exc) from exc
        except Exception as exc:
            raise MinioStorageError(f"failed to delete MinIO prefix {normalized!r}: {exc}") from exc
        return deleted

    def list_objects(self, *, prefix: str = "", limit: int | None = None) -> tuple[list[dict], bool]:
        self._require_enabled()
        effective_limit = max(1, min(limit or self.settings.minio_list_limit, self.settings.minio_list_limit))
        client = self._get_client()
        objects: list[dict] = []
        truncated = False
        try:
            iterator = client.list_objects(
                self.settings.minio_bucket,
                prefix=prefix,
                recursive=True,
                include_user_meta=False,
            )
            for item in iterator:
                if getattr(item, "is_dir", False):
                    continue
                if len(objects) >= effective_limit:
                    truncated = True
                    break
                key = item.object_name
                objects.append(
                    {
                        "object_key": key,
                        "image_url": self.build_public_url(key),
                        "size": int(getattr(item, "size", 0) or 0),
                        "etag": getattr(item, "etag", None),
                        "last_modified": getattr(item, "last_modified", None),
                        "content_type": None,
                    }
                )
        except S3Error as exc:
            raise self._translate_s3_error(exc) from exc
        except Exception as exc:
            raise MinioStorageError(f"failed to list MinIO objects: {exc}") from exc
        return objects, truncated
