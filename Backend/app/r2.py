"""Cloudflare R2 (S3-compatible API) for durable run storage and downloads.

Off unless R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY and
R2_BUCKET are all set; every function is then a no-op returning False/None,
so the app runs unchanged without R2.

Objects live at "<R2_PREFIX>runs/<run_id>/<name>". Downloads are served as
presigned GET URLs that expire after R2_URL_TTL_SECONDS, so the bucket stays
private and the backend never streams large files itself.
"""
import logging
import threading
from pathlib import Path

from app.config import settings

log = logging.getLogger(__name__)

_client = None
_client_lock = threading.Lock()


def enabled() -> bool:
    return all((settings.r2_account_id, settings.r2_access_key_id,
                settings.r2_secret_access_key, settings.r2_bucket))


def _s3():
    global _client
    with _client_lock:
        if _client is None:
            import boto3
            from botocore.config import Config

            _client = boto3.client(
                "s3",
                endpoint_url=f"https://{settings.r2_account_id}.r2.cloudflarestorage.com",
                aws_access_key_id=settings.r2_access_key_id,
                aws_secret_access_key=settings.r2_secret_access_key,
                region_name="auto",
                config=Config(signature_version="s3v4", retries={"max_attempts": 3, "mode": "standard"}),
            )
        return _client


def object_key(run_id: str, name: str) -> str:
    return f"{settings.r2_prefix}runs/{run_id}/{name}"


def _is_missing(error: Exception) -> bool:
    code = str(getattr(error, "response", {}).get("Error", {}).get("Code", ""))
    return code in ("404", "NoSuchKey", "NotFound")


def upload(path: Path, run_id: str, name: str) -> bool:
    if not enabled():
        return False
    try:
        _s3().upload_file(str(path), settings.r2_bucket, object_key(run_id, name))
        return True
    except Exception as e:
        # The local copy is still intact; don't fail the run over the mirror.
        log.warning("R2 upload of %s/%s failed: %s", run_id, name, e)
        return False


def download(run_id: str, name: str, dest: Path) -> bool:
    """Fetch an object into `dest`. False if R2 is off or the object is absent."""
    if not enabled():
        return False
    tmp = dest.with_name(dest.name + ".r2tmp")
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        _s3().download_file(settings.r2_bucket, object_key(run_id, name), str(tmp))
        tmp.replace(dest)
        return True
    except Exception as e:
        if not _is_missing(e):
            log.warning("R2 download of %s/%s failed: %s", run_id, name, e)
        tmp.unlink(missing_ok=True)
        return False


def exists(run_id: str, name: str) -> bool:
    if not enabled():
        return False
    try:
        _s3().head_object(Bucket=settings.r2_bucket, Key=object_key(run_id, name))
        return True
    except Exception as e:
        if not _is_missing(e):
            log.warning("R2 lookup of %s/%s failed: %s", run_id, name, e)
        return False


def presigned_upload_url(run_id: str, name: str) -> str | None:
    """Time-limited URL the browser PUTs a file to, so large uploads skip the
    backend and the frontend proxy. The bucket needs a CORS rule allowing PUT
    from the site's origin."""
    if not enabled():
        return None
    return _s3().generate_presigned_url(
        "put_object",
        Params={"Bucket": settings.r2_bucket, "Key": object_key(run_id, name)},
        ExpiresIn=settings.r2_url_ttl_seconds,
    )


def presigned_download_url(run_id: str, name: str, filename: str) -> str | None:
    """Time-limited URL that downloads the object as `filename`."""
    if not enabled():
        return None
    safe = filename.replace('"', "")
    return _s3().generate_presigned_url(
        "get_object",
        Params={
            "Bucket": settings.r2_bucket,
            "Key": object_key(run_id, name),
            "ResponseContentDisposition": f'attachment; filename="{safe}"',
        },
        ExpiresIn=settings.r2_url_ttl_seconds,
    )
