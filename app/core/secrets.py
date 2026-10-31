"""Encryption at rest for tenant secrets (their own LLM API keys).

A BYOK key is the tenant's money: anyone who reads it can spend on their
account. So it is stored only as Fernet ciphertext (AES-128-CBC + HMAC,
authenticated), decrypted only inside the worker or the verification call
that uses it, never returned by the API, and never written to a task
payload, a log line or an error message. The API shows a hint ("…3f9a")
so a person can tell which key is configured.
"""

from __future__ import annotations

from cryptography.fernet import Fernet, InvalidToken

from app.core.config import get_settings


class SecretsUnavailableError(RuntimeError):
    """No encryption key is configured, so secrets cannot be stored or read."""


def _fernet() -> Fernet:
    key = get_settings().secrets_encryption_key
    if not key:
        raise SecretsUnavailableError(
            "SECRETS_ENCRYPTION_KEY is not set; tenant API keys cannot be stored."
        )
    return Fernet(key.encode("ascii"))


def secrets_available() -> bool:
    return bool(get_settings().secrets_encryption_key)


def encrypt_secret(plaintext: str) -> str:
    return _fernet().encrypt(plaintext.encode("utf-8")).decode("ascii")


def decrypt_secret(ciphertext: str) -> str:
    try:
        return _fernet().decrypt(ciphertext.encode("ascii")).decode("utf-8")
    except InvalidToken as exc:
        # Wrong or rotated server key. Say so without echoing either value.
        raise SecretsUnavailableError(
            "A stored API key could not be decrypted (was SECRETS_ENCRYPTION_KEY changed?). "
            "Re-enter the key in Settings."
        ) from exc


def key_hint(plaintext: str) -> str:
    """Enough to recognise a key, too little to use it."""
    tail = plaintext.strip()[-4:]
    return f"…{tail}" if tail else ""


def scrub(text: str, secret: str) -> str:
    """Remove a secret from text that may be shown or logged (a provider's
    error message can quote the key it rejected)."""
    if not secret:
        return text
    redacted = text.replace(secret, "[redacted]")
    # Providers often quote a prefix and suffix of the key: drop long tails too.
    if len(secret) > 12:
        redacted = redacted.replace(secret[:8], "[redacted]").replace(secret[-6:], "[redacted]")
    return redacted
