"""Sends the sign-in codes for app/accounts.py.

Plain SMTP, so any provider works: Resend (host smtp.resend.com, username
"resend", the API key as the password), Gmail with an app password, Amazon
SES, and so on. On a local machine without SMTP, AUTH_DEV_LOG_CODES=true
writes each email to the server log instead; never set it in production.
"""
import logging
import smtplib
import ssl
from email.message import EmailMessage

from app.config import settings

log = logging.getLogger(__name__)


class MailError(RuntimeError):
    pass


def configured() -> bool:
    return bool(settings.smtp_host and settings.smtp_from)


def available() -> bool:
    return configured() or settings.auth_dev_log_codes


def send(to: str, subject: str, text: str) -> None:
    if not configured():
        if not settings.auth_dev_log_codes:
            raise MailError("Email is not configured on this server.")
        log.warning("AUTH_DEV_LOG_CODES is on, so this email was logged, not sent. To %s: %s\n%s", to, subject, text)
        return
    message = EmailMessage()
    message["From"] = settings.smtp_from
    message["To"] = to
    message["Subject"] = subject
    message.set_content(text)
    context = ssl.create_default_context()
    implicit_tls = settings.smtp_port == 465
    try:
        if implicit_tls:
            server = smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port, context=context, timeout=20)
        else:
            server = smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=20)
        with server:
            if not implicit_tls:
                server.starttls(context=context)
            if settings.smtp_username:
                server.login(settings.smtp_username, settings.smtp_password)
            server.send_message(message)
    except (smtplib.SMTPException, OSError) as e:
        raise MailError(f"Could not send email: {e}") from e


def send_code(to: str, name: str, code: str, *, signup: bool, minutes: int) -> None:
    purpose = "finish creating your Namtheg account" if signup else "sign in to Namtheg"
    text = (
        f"{f'Hi {name},' if name else 'Hi,'}\n\n"
        f"Your code to {purpose} is:\n\n    {code}\n\n"
        f"It expires in {minutes} minutes. If you didn't ask for it, you can ignore this email: "
        "nobody can sign in without the code.\n"
    )
    send(to, f"{code} is your Namtheg code", text)
