import pytest

from app.core.config import Settings
from app.storage.minio_service import MinioStorageService, MinioUrlError


def settings(**overrides):
    values = {
        "minio_enabled": True,
        "minio_endpoint": "minio:9000",
        "minio_public_base_url": "http://storage.example:9002",
        "minio_bucket": "media",
        "minio_access_key": "access",
        "minio_secret_key": "secret",
        "minio_secure": False,
    }
    values.update(overrides)
    return Settings(**values)


def test_public_and_internal_minio_endpoints_are_independent():
    service = MinioStorageService(settings())
    ref = service.parse_image_url(
        "http://storage.example:9002/media/documents/a/images/page%2001.jpg"
    )
    assert ref.bucket == "media"
    assert ref.object_key == "documents/a/images/page 01.jpg"
    assert service._client_endpoint() == ("minio:9000", False)


def test_minio_url_rejects_unconfigured_host_port():
    service = MinioStorageService(settings())
    with pytest.raises(MinioUrlError):
        service.parse_image_url(
            "http://storage.example:9003/media/documents/a/images/page.jpg"
        )


def test_minio_url_rejects_other_bucket():
    service = MinioStorageService(settings())
    with pytest.raises(MinioUrlError):
        service.parse_image_url(
            "http://storage.example:9002/other-bucket/documents/a/images/page.jpg"
        )


def test_build_public_url_uses_configured_external_base():
    service = MinioStorageService(settings())
    assert service.build_public_url("documents/a/images/page 01.jpg") == (
        "http://storage.example:9002/media/documents/a/images/page%2001.jpg"
    )


def test_presigned_query_is_not_retained_in_source_provenance():
    service = MinioStorageService(settings())
    ref = service.parse_image_url(
        "http://storage.example:9002/media/documents/a/images/page.jpg?X-Amz-Signature=secret"
    )
    assert ref.source_url == "http://storage.example:9002/media/documents/a/images/page.jpg"
    assert "secret" not in ref.source_url


def test_output_object_key_rejects_parent_segments():
    service = MinioStorageService(settings())
    with pytest.raises(MinioUrlError):
        service.build_public_url("documents/a/OCR/../images/page.jpg")
