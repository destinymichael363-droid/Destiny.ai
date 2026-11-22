from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path


DATA_DIR = Path("data")
USERS_FILE = DATA_DIR / "users_auth.json"
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_SECONDS = 300


@dataclass
class AuthResult:
    ok: bool
    message: str
    username: str | None = None


def _hash_password(password: str, salt_hex: str | None = None) -> tuple[str, str]:
    if salt_hex is None:
        salt_hex = os.urandom(16).hex()
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt_hex),
        200_000,
    )
    return salt_hex, dk.hex()


def _load_users() -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not USERS_FILE.exists():
        return {}
    try:
        return json.loads(USERS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_users(users: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    USERS_FILE.write_text(json.dumps(users, indent=2), encoding="utf-8")


def register_user(username: str, email: str, password: str) -> AuthResult:
    clean_user = username.strip().lower()
    clean_email = email.strip().lower()

    if not re.fullmatch(r"[a-z0-9_]{3,24}", clean_user):
        return AuthResult(False, "Username must be 3–24 characters (letters, numbers, underscores).")
    if "@" not in clean_email or "." not in clean_email.split("@")[-1]:
        return AuthResult(False, "Please enter a valid email address.")
    if len(password) < 8:
        return AuthResult(False, "Password must be at least 8 characters long.")

    users = _load_users()
    if clean_user in users:
        return AuthResult(False, "That username is already taken.")
    if any(u.get("email") == clean_email for u in users.values()):
        return AuthResult(False, "An account with that email already exists.")

    salt_hex, pw_hash = _hash_password(password)
    users[clean_user] = {
        "email": clean_email,
        "salt": salt_hex,
        "password_hash": pw_hash,
        "failed_attempts": 0,
        "locked_until": 0.0,
        "created_at": int(time.time()),
    }
    _save_users(users)
    return AuthResult(True, "Account created securely. You can now log in.", username=clean_user)


def authenticate_user(username_or_email: str, password: str) -> AuthResult:
    ident = username_or_email.strip().lower()
    users = _load_users()

    matched_user = None
    if ident in users:
        matched_user = ident
    else:
        for uname, record in users.items():
            if record.get("email") == ident:
                matched_user = uname
                break

    if not matched_user:
        return AuthResult(False, "Invalid credentials.")

    record = users[matched_user]
    now = time.time()
    locked_until = float(record.get("locked_until", 0.0))
    if locked_until > now:
        remaining = int(locked_until - now)
        return AuthResult(False, f"Account temporarily locked. Try again in {remaining}s.")

    salt_hex = record.get("salt", "")
    expected_hash = record.get("password_hash", "")
    _, actual_hash = _hash_password(password, salt_hex=salt_hex)

    if hmac.compare_digest(expected_hash, actual_hash):
        record["failed_attempts"] = 0
        record["locked_until"] = 0.0
        _save_users(users)
        return AuthResult(True, f"Welcome back, {matched_user}.", username=matched_user)

    failed = int(record.get("failed_attempts", 0)) + 1
    record["failed_attempts"] = failed
    if failed >= MAX_FAILED_ATTEMPTS:
        record["locked_until"] = now + LOCKOUT_SECONDS
        record["failed_attempts"] = 0
        _save_users(users)
        return AuthResult(False, f"Too many failed attempts. Locked for {LOCKOUT_SECONDS}s.")

    _save_users(users)
    return AuthResult(False, f"Invalid credentials ({MAX_FAILED_ATTEMPTS - failed} attempts left).")

