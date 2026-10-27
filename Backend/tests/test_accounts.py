"""Optional accounts: codes by email, signed sessions, and each account's runs."""
import base64
import json
import re
import smtplib
import time

import pytest
from fastapi.testclient import TestClient

from app import accounts, jobs, mailer
from app.config import settings
from app.main import app as fastapi_app
from tests.conftest import INSURANCE_CSV


@pytest.fixture
def outbox(monkeypatch):
    """Accounts on, with every email captured instead of sent."""
    monkeypatch.setattr(settings, "auth_secret", "test-secret")
    monkeypatch.setattr(settings, "smtp_host", "smtp.example")
    monkeypatch.setattr(settings, "smtp_from", "Namtheg <login@example.com>")
    sent = []
    monkeypatch.setattr(mailer, "send", lambda to, subject, text: sent.append({"to": to, "subject": subject, "text": text}))
    return sent


def _code(mail: dict) -> str:
    return re.search(r"\b(\d{6})\b", mail["text"]).group(1)


def _wrong(code: str) -> str:
    return f"{(int(code) + 1) % 10 ** 6:06d}"


def _sign_up(client, outbox, email: str, name: str) -> None:
    assert client.post("/auth/code", json={"email": email, "name": name}).status_code == 200
    assert client.post("/auth/verify", json={"email": email, "code": _code(outbox[-1])}).status_code == 200


def _upload(client) -> str:
    with INSURANCE_CSV.open("rb") as f:
        r = client.post("/upload", files={"file": ("insurance.csv", f, "text/csv")})
    assert r.status_code == 200, r.text
    return r.json()["run_id"]


def _age_code(email: str, seconds: float) -> None:
    """Move a pending code `seconds` into the past, as if that time had gone by."""
    path = settings.storage_dir / "accounts" / "codes" / f"{accounts._email_key(email)}.json"
    pending = json.loads(path.read_text())
    pending["expires"] -= seconds
    pending["sent"] = [t - seconds for t in pending["sent"]]
    path.write_text(json.dumps(pending))


def test_accounts_are_off_until_configured():
    client = TestClient(fastapi_app)
    assert client.get("/auth/me").json() == {"enabled": False, "user": None}
    assert client.post("/auth/code", json={"email": "ada@example.com", "name": "Ada"}).status_code == 404
    assert client.get("/auth/runs").status_code == 404


def test_sign_up_then_sign_in_again_with_emailed_codes(outbox):
    client = TestClient(fastapi_app)
    assert client.get("/auth/me").json() == {"enabled": True, "user": None}

    r = client.post("/auth/code", json={"email": " Ada@Example.com", "name": "  Ada \n Lovelace "})
    assert r.status_code == 200 and r.json() == {"sent": True}
    (mail,) = outbox
    assert mail["to"] == "ada@example.com" and mail["subject"] == f"{_code(mail)} is your Namtheg code"
    assert mail["text"].startswith("Hi Ada Lovelace,") and "finish creating your Namtheg account" in mail["text"]

    r = client.post("/auth/verify", json={"email": "ada@example.com", "code": _wrong(_code(mail))})
    assert r.status_code == 400 and r.json()["detail"] == "That code isn't right. 4 tries left."
    assert client.get("/auth/me").json()["user"] is None

    r = client.post("/auth/verify", json={"email": "ADA@example.com", "code": f" {_code(mail)[:3]} {_code(mail)[3:]}"})
    assert r.status_code == 200 and r.json() == {"user": {"name": "Ada Lovelace", "email": "ada@example.com"}}
    cookie = r.headers["set-cookie"].lower()
    assert cookie.startswith(f"{accounts.COOKIE}=")
    assert "httponly" in cookie and "samesite=lax" in cookie and "max-age=2592000" in cookie and "secure" not in cookie
    assert client.get("/auth/me").json()["user"] == {"name": "Ada Lovelace", "email": "ada@example.com"}

    # A code works once.
    r = client.post("/auth/verify", json={"email": "ada@example.com", "code": _code(mail)})
    assert r.status_code == 400 and r.json()["detail"] == "That code has expired. Ask for a new one."

    run_id = _upload(client)
    runs = client.get("/auth/runs").json()["runs"]
    assert [(r["run_id"], r["status"], r["filename"]) for r in runs] == [(run_id, "uploaded", "insurance.csv")]

    assert client.post("/auth/logout").json() == {"ok": True}
    assert client.get("/auth/me").json()["user"] is None
    assert client.get("/auth/runs").status_code == 401

    # Signing in again takes only the address; the account and its runs are still there.
    assert client.post("/auth/code", json={"email": "ada@example.com"}).json() == {"sent": True}
    mail = outbox[-1]
    assert mail["text"].startswith("Hi Ada Lovelace,") and "sign in to Namtheg" in mail["text"]
    r = client.post("/auth/verify", json={"email": "ada@example.com", "code": _code(mail)})
    assert r.json()["user"]["name"] == "Ada Lovelace"
    assert [r["run_id"] for r in client.get("/auth/runs").json()["runs"]] == [run_id]


def test_asking_for_a_code_never_reveals_whether_an_address_has_an_account(outbox):
    client = TestClient(fastapi_app)
    r = client.post("/auth/code", json={"email": "nobody@example.com"})
    assert r.status_code == 200 and r.json() == {"sent": True}
    assert outbox == []
    r = client.post("/auth/verify", json={"email": "nobody@example.com", "code": "123456"})
    assert r.status_code == 400 and r.json()["detail"] == "That code has expired. Ask for a new one."

    # Creating an account for an address that has one signs in to it instead.
    _sign_up(client, outbox, "ada@example.com", "Ada")
    client.post("/auth/logout")
    r = client.post("/auth/code", json={"email": "ada@example.com", "name": "Mallory"})
    assert r.json() == {"sent": True}
    assert outbox[-1]["text"].startswith("Hi Ada,") and "sign in to Namtheg" in outbox[-1]["text"]
    r = client.post("/auth/verify", json={"email": "ada@example.com", "code": _code(outbox[-1])})
    assert r.json()["user"] == {"name": "Ada", "email": "ada@example.com"}


def test_codes_expire_lock_after_five_tries_and_are_rate_limited(outbox):
    client = TestClient(fastapi_app)
    email = "ada@example.com"
    client.post("/auth/code", json={"email": email, "name": "Ada"})
    r = client.post("/auth/code", json={"email": email, "name": "Ada"})
    assert r.status_code == 429 and r.json()["detail"].startswith("A code was just sent.")
    assert len(outbox) == 1

    code = _code(outbox[-1])
    details = [client.post("/auth/verify", json={"email": email, "code": _wrong(code)}).json()["detail"]
               for _ in range(accounts.MAX_ATTEMPTS)]
    assert details[0] == "That code isn't right. 4 tries left."
    assert details[3] == "That code isn't right. 1 try left."
    assert details[4] == "That code isn't right, and that was the last try. Ask for a new one."
    r = client.post("/auth/verify", json={"email": email, "code": code})
    assert r.status_code == 400 and r.json()["detail"] == "That code has expired. Ask for a new one."

    # A minute later a new code can be sent, and it works until ten minutes pass.
    _age_code(email, accounts.RESEND_SECONDS + 1)
    assert client.post("/auth/code", json={"email": email, "name": "Ada"}).status_code == 200
    _age_code(email, accounts.CODE_SECONDS + 1)
    r = client.post("/auth/verify", json={"email": email, "code": _code(outbox[-1])})
    assert r.status_code == 400 and r.json()["detail"] == "That code has expired. Ask for a new one."

    # Five codes an hour per address at most.
    for _ in range(accounts.MAX_CODES_PER_HOUR - 2):
        _age_code(email, accounts.RESEND_SECONDS + 1)
        assert client.post("/auth/code", json={"email": email, "name": "Ada"}).status_code == 200
    _age_code(email, accounts.RESEND_SECONDS + 1)
    r = client.post("/auth/code", json={"email": email, "name": "Ada"})
    assert r.status_code == 429 and r.json()["detail"] == "Too many codes for this address. Try again in an hour."
    assert len(outbox) == accounts.MAX_CODES_PER_HOUR


def test_the_daily_cap_protects_the_senders_quota(outbox, monkeypatch):
    monkeypatch.setattr(settings, "auth_max_codes_per_day", 2)
    client = TestClient(fastapi_app)
    for i in range(2):
        assert client.post("/auth/code", json={"email": f"user{i}@example.com", "name": "U"}).status_code == 200
    r = client.post("/auth/code", json={"email": "user2@example.com", "name": "U"})
    assert r.status_code == 429 and r.json()["detail"] == "Sign-in is busy right now. Try again later."
    assert len(outbox) == 2


def test_an_email_that_cannot_be_sent_is_reported_and_leaves_no_code(outbox, monkeypatch):
    def fail(to, subject, text):
        raise mailer.MailError("Could not send email: connection refused")

    monkeypatch.setattr(mailer, "send", fail)
    client = TestClient(fastapi_app)
    r = client.post("/auth/code", json={"email": "ada@example.com", "name": "Ada"})
    assert r.status_code == 503 and r.json()["detail"] == "We couldn't send the email. Try again in a minute."
    r = client.post("/auth/verify", json={"email": "ada@example.com", "code": "123456"})
    assert r.json()["detail"] == "That code has expired. Ask for a new one."

    # Nothing was sent, so the next try needn't wait.
    monkeypatch.setattr(mailer, "send", lambda to, subject, text: outbox.append({"to": to, "text": text}))
    assert client.post("/auth/code", json={"email": "ada@example.com", "name": "Ada"}).status_code == 200


def test_sessions_are_signed_expire_and_end_when_the_secret_changes(outbox, monkeypatch):
    accounts.request_code("ada@example.com", "Ada")
    user = accounts.verify_code("ada@example.com", _code(outbox[-1]))
    token = accounts.session_token(user["id"])
    assert accounts.session_user(token)["email"] == "ada@example.com"

    body, signature = token.split(".")
    payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    forged = accounts._b64(json.dumps({**payload, "uid": "f" * 24}).encode())
    assert accounts.session_user(f"{forged}.{signature}") is None
    expired = accounts._b64(json.dumps({**payload, "exp": int(time.time()) - 1}).encode())
    assert accounts.session_user(f"{expired}.{accounts._sign(expired)}") is None
    for junk in ("", "garbage", "a.b.c", f"{body}."):
        assert accounts.session_user(junk) is None

    monkeypatch.setattr(settings, "auth_secret", "another-secret")
    assert accounts.session_user(token) is None
    # The account itself survives the new secret.
    accounts.request_code("ada@example.com")
    assert accounts.verify_code("ada@example.com", _code(outbox[-1]))["id"] == user["id"]


def test_the_cookie_is_secure_behind_https(outbox):
    client = TestClient(fastapi_app)
    client.post("/auth/code", json={"email": "ada@example.com", "name": "Ada"})
    r = client.post("/auth/verify", json={"email": "ada@example.com", "code": _code(outbox[-1])},
                    headers={"x-forwarded-proto": "https"})
    assert r.status_code == 200 and "; secure" in r.headers["set-cookie"].lower()


def test_names_and_addresses_are_checked(outbox):
    client = TestClient(fastapi_app)
    for email in ("", "ada", "ada@example", "a b@example.com", "ada@@example.com", "x" * 250 + "@example.com"):
        r = client.post("/auth/code", json={"email": email, "name": "Ada"})
        assert r.status_code == 400 and r.json()["detail"] == "Enter a valid email address.", email
    r = client.post("/auth/code", json={"email": "ada@example.com", "name": "\x00\x07"})
    assert r.status_code == 400 and r.json()["detail"] == "Enter your name."
    assert accounts.clean_name("  Ada\n\tLovelace\x1b ") == "Ada Lovelace"
    assert accounts.clean_name("A" * 200) == "A" * 80
    assert outbox == []


def test_my_runs_lists_what_i_upload_or_start_newest_first(outbox, monkeypatch):
    monkeypatch.setattr(jobs, "start", lambda run_id, spec, background: {"executor": "test"})
    client = TestClient(fastapi_app)
    _upload(client)  # signed out: listed nowhere
    _sign_up(client, outbox, "ada@example.com", "Ada")
    first, second = _upload(client), _upload(client)
    assert [r["run_id"] for r in client.get("/auth/runs").json()["runs"]] == [second, first]

    assert client.post(f"/runs/{first}/start", json={"task": "auto", "target": "charges"}).status_code == 200
    runs = client.get("/auth/runs").json()["runs"]
    assert [r["run_id"] for r in runs] == [first, second]
    assert (runs[0]["status"], runs[0]["target"]) == ("queued", "charges")
    # Nothing about the account is added to the run, which anyone with its link can read.
    assert set(client.get(f"/runs/{first}/status").json()) == {"run_id", "status", "filename", "source_format",
                                                               "task", "target", "queued_at", "executor"}


def test_a_run_uploaded_before_signing_in_can_be_saved_to_the_account(outbox):
    client = TestClient(fastapi_app)
    run_id = _upload(client)
    assert client.post(f"/auth/runs/{run_id}").status_code == 401
    _sign_up(client, outbox, "ada@example.com", "Ada")
    assert client.get("/auth/runs").json()["runs"] == []
    assert client.post(f"/auth/runs/{run_id}").json() == {"ok": True}
    assert client.post(f"/auth/runs/{run_id}").json() == {"ok": True}  # saving twice lists it once
    assert [r["run_id"] for r in client.get("/auth/runs").json()["runs"]] == [run_id]
    assert client.post("/auth/runs/0123456789ab").status_code == 404
    client.post("/auth/logout")
    assert client.post(f"/auth/runs/{run_id}").status_code == 401


def test_without_smtp_codes_go_to_the_log_on_a_local_machine(monkeypatch, caplog):
    monkeypatch.setattr(settings, "auth_secret", "test-secret")
    assert not accounts.enabled()
    monkeypatch.setattr(settings, "auth_dev_log_codes", True)
    assert accounts.enabled()
    with caplog.at_level("WARNING", logger="app.mailer"):
        accounts.request_code("ada@example.com", "Ada")
    assert "AUTH_DEV_LOG_CODES" in caplog.text and "To ada@example.com" in caplog.text
    assert re.search(r"\b\d{6} is your Namtheg code", caplog.text)


class FakeSMTP:
    def __init__(self, log, kind, host, port):
        self.log = log
        log.append((kind, host, port))

    def starttls(self, context):
        self.log.append("starttls")

    def login(self, username, password):
        self.log.append(("login", username))

    def send_message(self, message):
        self.log.append(("send", message["From"], message["To"], message["Subject"]))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.log.append("quit")


def test_mail_is_sent_over_tls_on_either_port(monkeypatch):
    for key, value in {"smtp_host": "smtp.example", "smtp_from": "Namtheg <login@example.com>",
                       "smtp_username": "resend", "smtp_password": "key", "smtp_port": 465}.items():
        monkeypatch.setattr(settings, key, value)
    log = []
    monkeypatch.setattr(smtplib, "SMTP_SSL", lambda host, port, **kw: FakeSMTP(log, "tls", host, port))
    monkeypatch.setattr(smtplib, "SMTP", lambda host, port, **kw: FakeSMTP(log, "plain", host, port))

    mailer.send_code("ada@example.com", "Ada", "123456", signup=True, minutes=10)
    assert log == [("tls", "smtp.example", 465), ("login", "resend"),
                   ("send", "Namtheg <login@example.com>", "ada@example.com", "123456 is your Namtheg code"), "quit"]

    log.clear()
    monkeypatch.setattr(settings, "smtp_port", 587)
    mailer.send("ada@example.com", "Hello", "Hi")
    assert log[:3] == [("plain", "smtp.example", 587), "starttls", ("login", "resend")]

    def refused(host, port, **kw):
        raise ConnectionRefusedError("refused")

    monkeypatch.setattr(smtplib, "SMTP", refused)
    with pytest.raises(mailer.MailError, match="refused"):
        mailer.send("ada@example.com", "Hello", "Hi")
