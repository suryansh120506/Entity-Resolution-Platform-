"""S3-compatible object storage for uploads (Cloudflare R2, AWS S3, MinIO).

The browser PUTs the CSV straight to the bucket with a short-lived presigned
URL; the API never carries the bytes. That removes the API (and any
serverless body limit in front of it) from the upload path entirely, and it
keeps the API stateless: the signature *is* the authorisation, nothing is
stored to check it against.

Three guards on a presigned upload:

1. **The server chooses the key** (`uploads/<tenant>/<job>.csv`). A client
   cannot write outside its tenant's prefix or overwrite another job.
2. **Size and type are signed.** Content-Length and Content-Type are part of
   the SigV4 signature, so S3/R2 reject a PUT whose body differs from what
   was declared at presign time.
3. **Verified again on confirm.** `head` checks the stored object's size
   before any work is enqueued, because (2) depends on the provider
   enforcing it, and an emulator or a misconfigured proxy may not.
"""

from __future__ import annotations

import contextlib
import tempfile
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import get_settings

S3_SCHEME = "s3://"
CSV_CONTENT_TYPE = "text/csv"


class StorageNotConfiguredError(RuntimeError):
    pass


class UploadMissingError(LookupError):
    """The object the client claimed to have uploaded is not there."""


@dataclass(frozen=True, slots=True)
class PresignedUpload:
    url: str
    method: str
    headers: dict[str, str]
    expires_in: int
    source_uri: str


@lru_cache(maxsize=1)
def s3_client() -> Any:
    settings = get_settings()
    if not settings.object_storage_enabled:
        raise StorageNotConfiguredError(
            "Object storage is not configured (OBJECT_STORAGE_BUCKET and keys)."
        )
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=settings.object_storage_endpoint_url,
        aws_access_key_id=settings.object_storage_access_key_id,
        aws_secret_access_key=settings.object_storage_secret_access_key,
        region_name=settings.object_storage_region,
        config=Config(signature_version="s3v4", retries={"max_attempts": 3, "mode": "standard"}),
    )


def reset_client() -> None:
    """Forget the cached client (settings changed — tests, credential rotation)."""
    s3_client.cache_clear()


def upload_key(tenant_id: uuid.UUID | str, job_id: uuid.UUID | str) -> str:
    return f"uploads/{tenant_id}/{job_id}.csv"


def split_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith(S3_SCHEME):
        raise ValueError(f"not an s3:// URI: {uri!r}")
    bucket, _, key = uri[len(S3_SCHEME) :].partition("/")
    if not bucket or not key:
        raise ValueError(f"malformed s3:// URI: {uri!r}")
    return bucket, key


def presign_upload(
    tenant_id: uuid.UUID | str, job_id: uuid.UUID | str, size_bytes: int
) -> PresignedUpload:
    settings = get_settings()
    bucket = settings.object_storage_bucket
    assert bucket is not None  # object_storage_enabled guarantees it
    key = upload_key(tenant_id, job_id)
    url = s3_client().generate_presigned_url(
        "put_object",
        Params={
            "Bucket": bucket,
            "Key": key,
            "ContentType": CSV_CONTENT_TYPE,
            "ContentLength": size_bytes,
        },
        ExpiresIn=settings.presigned_url_ttl_seconds,
        HttpMethod="PUT",
    )
    return PresignedUpload(
        url=url,
        method="PUT",
        # The browser must send exactly these: they are in the signature.
        # (Content-Length is set by the browser from the body itself.)
        headers={"Content-Type": CSV_CONTENT_TYPE},
        expires_in=settings.presigned_url_ttl_seconds,
        source_uri=f"{S3_SCHEME}{bucket}/{key}",
    )


def object_size(uri: str) -> int:
    from botocore.exceptions import ClientError

    bucket, key = split_uri(uri)
    try:
        head = s3_client().head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}:
            raise UploadMissingError(uri) from exc
        raise
    return int(head["ContentLength"])


def delete_object(uri: str) -> None:
    bucket, key = split_uri(uri)
    s3_client().delete_object(Bucket=bucket, Key=key)


@contextlib.contextmanager
def local_copy(source: str, *, max_bytes: int) -> Iterator[Path]:
    """A local file path for `source`, downloading s3:// sources to a temp file.

    Streamed in chunks with the size cap enforced *while* reading, so an
    object that grew past the limit after confirmation (or was never checked)
    cannot fill the worker's disk. The temp file is removed on exit.
    """
    if not source.startswith(S3_SCHEME):
        yield Path(source)
        return

    bucket, key = split_uri(source)
    body = s3_client().get_object(Bucket=bucket, Key=key)["Body"]
    with tempfile.NamedTemporaryFile(prefix="er-upload-", suffix=".csv", delete=False) as handle:
        path = Path(handle.name)
        written = 0
        try:
            for chunk in body.iter_chunks(chunk_size=1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    raise ValueError(f"{source} exceeds the {max_bytes} byte upload limit")
                handle.write(chunk)
        finally:
            body.close()
    try:
        yield path
    finally:
        with contextlib.suppress(OSError):
            path.unlink()


def read_head(source: str, nbytes: int = 65_536) -> bytes:
    """The first `nbytes` of an upload, without fetching the rest.

    A ranged GET for s3:// sources. Used to recover the original column order
    of a file (JSONB does not keep key order) without downloading it.
    """
    if not source.startswith(S3_SCHEME):
        with Path(source).open("rb") as handle:
            return handle.read(nbytes)
    bucket, key = split_uri(source)
    body = s3_client().get_object(Bucket=bucket, Key=key, Range=f"bytes=0-{nbytes - 1}")["Body"]
    try:
        return bytes(body.read())
    finally:
        body.close()
