"""Optional accounts: a name and an email address, no passwords. Every sign-in
and sign-up is confirmed with a one-time code sent by email (app/mailer.py).

Signing in is never required. Runs a signed-in user uploads or starts are
listed under their account ("My runs"); run links keep working for anyone who
has them, exactly as they do without an account. The session is a signed
cookie the frontend proxy passes through, so it never leaves the site's origin.

Off unless AUTH_SECRET is set and email can be sent. Stored beside the runs:
  accounts/users/<user id>.json     name, email, and the account's runs (mirrored to R2)
  accounts/emails/<email key>.json  which account an address belongs to (mirrored to R2)
  accounts/codes/<email key>.json   the pending code, hashed, with its limits
  accounts/sent/<yyyy-mm-dd>.json   codes sent that day, for the daily cap
Files are named by a hash of the address, so listings don't show addresses and
any address makes a safe path. Codes are stored as keyed hashes only.
"""
import base64
import hashlib
import hmac
import json
import logging
import re
import secrets
import time
import unicodedata
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from app import mailer, r2, storage
from app.config import settings

log = logging.getLogger(__name__)

COOKIE = "namtheg_session"
SESSION_SECONDS = 30 * 24 * 3600
CODE_SECONDS = 10 * 60
MAX_ATTEMPTS = 5
RESEND_SECONDS = 60
MAX_CODES_PER_HOUR = 5
MAX_RUNS = 100  # an account lists its most recent runs; older ones drop off the list, not out of storage
_EMAIL = re.compile(r"[^@\s]{1,64}@[^@\s]+\.[^@\s.]{2,}")
_USER_ID = re.compile(r"[0-9a-f]{24}")


class AuthError(Exception):
    """A refusal whose message is safe to show the user."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def enabled() -> bool:
    return bool(settings.auth_secret) and mailer.available()


def _secret() -> bytes:
    return settings.auth_secret.encode()


# -- storage -----------------------------------------------------------------------

def _path(name: str) -> Path:
    path = settings.storage_dir / "accounts" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _read(name: str, mirrored: bool = True) -> dict | None:
    storage.sync_reload()  # another container may have written it since
    path = _path(name)
    if not path.exists() and mirrored:
        r2.download_key(r2.account_key(name), path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _write(name: str, data: dict, mirrored: bool = True) -> None:
    path = _path(name)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    storage.sync_commit()
    if mirrored:
        r2.upload_key(path, r2.account_key(name))


def _delete(name: str) -> None:
    _path(name).unlink(missing_ok=True)
    storage.sync_commit()


# -- identities ----------------------------------------------------------------------

def normalize_email(email) -> str:
    value = str(email or "").strip().lower()
    if len(value) > 254 or not _EMAIL.fullmatch(value):
        raise AuthError("Enter a valid email address.")
    return value


def clean_name(name) -> str:
    """One line of at most 80 characters: control characters become spaces,
    and runs of whitespace collapse to one."""
    text = "".join(" " if unicodedata.category(ch) == "Cc" else ch for ch in str(name or ""))
    value = " ".join(text.split())[:80].rstrip()
    if not value:
        raise AuthError("Enter your name.")
    return value


def _email_key(email: str) -> str:
    # Unkeyed on purpose: changing AUTH_SECRET signs everyone out, but must not
    # orphan the accounts.
    return hashlib.sha256(email.encode()).hexdigest()[:40]


def _code_hash(email: str, code: str) -> str:
    return hmac.new(_secret(), f"{email}:{code}".encode(), hashlib.sha256).hexdigest()


def public(user: dict) -> dict:
    return {"name": user["name"], "email": user["email"]}


# -- codes ---------------------------------------------------------------------------

def _count_sent_today() -> None:
    day = time.strftime("%Y-%m-%d", time.gmtime())
    record = _read(f"sent/{day}.json", mirrored=False) or {"count": 0}
    if record["count"] >= settings.auth_max_codes_per_day:
        raise AuthError("Sign-in is busy right now. Try again later.", 429)
    record["count"] += 1
    _write(f"sent/{day}.json", record, mirrored=False)


def request_code(email: str, name: str | None = None) -> None:
    """Email a sign-in code to an existing account, or a sign-up code when a
    name comes with a new address. An unknown address without a name gets
    nothing, and the caller can't tell the difference: the answer never
    reveals whether an address has an account."""
    email = normalize_email(email)
    name = clean_name(name) if str(name or "").strip() else None
    key = _email_key(email)
    account = _read(f"emails/{key}.json")
    if account is None and name is None:
        return
    signup_name = None if account else name
    now = time.time()
    previous = _read(f"codes/{key}.json", mirrored=False)
    sent = [t for t in (previous or {}).get("sent", []) if now - t < 3600]
    if sent and now - sent[-1] < RESEND_SECONDS:
        wait = int(RESEND_SECONDS - (now - sent[-1])) + 1
        raise AuthError(f"A code was just sent. You can ask for another in {wait} seconds.", 429)
    if len(sent) >= MAX_CODES_PER_HOUR:
        raise AuthError("Too many codes for this address. Try again in an hour.", 429)
    _count_sent_today()
    code = f"{secrets.randbelow(10 ** 6):06d}"
    _write(f"codes/{key}.json", {"hash": _code_hash(email, code), "expires": now + CODE_SECONDS, "attempts": 0,
                                 "sent": sent + [now], "signup_name": signup_name}, mirrored=False)
    name_on_account = signup_name or (_read(f"users/{account['user_id']}.json") or {}).get("name", "")
    try:
        mailer.send_code(email, name_on_account, code, signup=signup_name is not None, minutes=CODE_SECONDS // 60)
    except mailer.MailError as e:
        log.error("Sign-in email to account %s failed: %s", key[:8], e)
        if previous:
            _write(f"codes/{key}.json", previous, mirrored=False)
        else:
            _delete(f"codes/{key}.json")
        raise AuthError("We couldn't send the email. Try again in a minute.", 503) from e


def verify_code(email: str, code: str) -> dict:
    """The account a correct code belongs to; a sign-up code creates it."""
    email = normalize_email(email)
    key = _email_key(email)
    pending = _read(f"codes/{key}.json", mirrored=False)
    if not pending or time.time() > pending["expires"] or pending["attempts"] >= MAX_ATTEMPTS:
        raise AuthError("That code has expired. Ask for a new one.")
    if not hmac.compare_digest(pending["hash"], _code_hash(email, re.sub(r"\D", "", str(code or "")))):
        pending["attempts"] += 1
        _write(f"codes/{key}.json", pending, mirrored=False)
        left = MAX_ATTEMPTS - pending["attempts"]
        raise AuthError(f"That code isn't right. {left} {'try' if left == 1 else 'tries'} left." if left
                        else "That code isn't right, and that was the last try. Ask for a new one.")
    _delete(f"codes/{key}.json")
    account = _read(f"emails/{key}.json")
    user = _read(f"users/{account['user_id']}.json") if account else None
    if user is None:
        user = {"id": secrets.token_hex(12), "name": pending.get("signup_name") or email.split("@")[0],
                "email": email, "created_at": time.time(), "runs": []}
        _write(f"users/{user['id']}.json", user)
        _write(f"emails/{key}.json", {"user_id": user["id"]})
    return user


# -- sessions --------------------------------------------------------------------------
# A signed token in an HttpOnly cookie: the user id and an expiry, so nothing
# about a session is stored. Signing out deletes the cookie.

def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _sign(body: str) -> str:
    return _b64(hmac.new(_secret(), body.encode(), hashlib.sha256).digest())


def session_token(user_id: str) -> str:
    body = _b64(json.dumps({"uid": user_id, "exp": int(time.time()) + SESSION_SECONDS}).encode())
    return f"{body}.{_sign(body)}"


def session_user(token: str | None) -> dict | None:
    if not token or not enabled():
        return None
    try:
        body, signature = token.split(".")
        if not hmac.compare_digest(signature, _sign(body)):
            return None
        payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
        if payload["exp"] < time.time() or not _USER_ID.fullmatch(payload["uid"]):
            return None
        return _read(f"users/{payload['uid']}.json")
    except Exception:
        return None


def current_user(request: Request) -> dict | None:
    return session_user(request.cookies.get(COOKIE))


# -- runs ------------------------------------------------------------------------------

def link_run(request: Request, run_id: str) -> None:
    """Put a run the request uploaded or started at the top of the signed-in
    user's runs. A bookmark only: run links stay open to whoever has them, so
    a shared run someone else starts again is listed for them too. Does
    nothing when nobody is signed in, and never fails the request."""
    if COOKIE not in request.cookies:
        return
    try:
        user = current_user(request)
        if user:
            runs = [r for r in user["runs"] if r["run_id"] != run_id]
            user["runs"] = (runs + [{"run_id": run_id, "added_at": time.time()}])[-MAX_RUNS:]
            _write(f"users/{user['id']}.json", user)
    except Exception:
        log.exception("Could not list run %s under an account", run_id)


def my_runs(user: dict) -> list[dict]:
    """The account's runs, newest first, with where each one got to."""
    runs = []
    for entry in reversed(user["runs"]):
        run_id = entry["run_id"]
        if not storage.run_exists(run_id):
            continue
        status = storage.read_status(run_id)
        metrics = (storage.read_json(run_id, "metrics.json") or {}) if status.get("status") == "succeeded" else {}
        runs.append({
            "run_id": run_id, "added_at": entry["added_at"], "filename": status.get("filename"),
            "status": status.get("status"), "task": status.get("task"), "target": status.get("target"),
            "model_name": metrics.get("model_name"), "score": metrics.get("score"),
            "score_metric": metrics.get("score_metric"),
        })
    return runs


# -- API ---------------------------------------------------------------------------------

router = APIRouter(prefix="/auth", tags=["accounts"])


class CodeRequest(BaseModel):
    email: str
    name: str | None = None


class VerifyRequest(BaseModel):
    email: str
    code: str


def _require_enabled() -> None:
    if not enabled():
        raise HTTPException(404, "Accounts are not enabled on this server.")


@router.get("/me")
def me(request: Request) -> dict:
    """Whether sign-in is offered here, and who is signed in."""
    user = current_user(request)
    return {"enabled": enabled(), "user": public(user) if user else None}


@router.post("/code")
def send_code(req: CodeRequest) -> dict:
    """Email a code: give `name` to create an account, leave it out to sign in.
    The answer is the same whether or not the address has an account."""
    _require_enabled()
    try:
        request_code(req.email, req.name)
    except AuthError as e:
        raise HTTPException(e.status, str(e))
    return {"sent": True}


@router.post("/verify")
def verify(req: VerifyRequest, request: Request, response: Response) -> dict:
    """Check a code; on success the session cookie is set."""
    _require_enabled()
    try:
        user = verify_code(req.email, req.code)
    except AuthError as e:
        raise HTTPException(e.status, str(e))
    forwarded = request.headers.get("x-forwarded-proto", "").split(",")[0].strip()
    secure = request.url.scheme == "https" or forwarded == "https"
    response.set_cookie(COOKIE, session_token(user["id"]), max_age=SESSION_SECONDS, httponly=True,
                        secure=secure, samesite="lax", path="/")
    return {"user": public(user)}


@router.post("/logout")
def logout(response: Response) -> dict:
    response.delete_cookie(COOKIE, path="/")
    return {"ok": True}


@router.get("/runs")
def runs(request: Request) -> dict:
    """The signed-in user's runs, newest first."""
    _require_enabled()
    user = current_user(request)
    if not user:
        raise HTTPException(401, "Sign in to see your runs.")
    return {"runs": my_runs(user)}


@router.post("/runs/{run_id}")
def save_run(run_id: str, request: Request) -> dict:
    """List a run under the account: the one that was open when the user signed in."""
    _require_enabled()
    if not current_user(request):
        raise HTTPException(401, "Sign in to save runs.")
    if not storage.run_exists(run_id):
        raise HTTPException(404, "run_id not found")
    link_run(request, run_id)
    return {"ok": True}
