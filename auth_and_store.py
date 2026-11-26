"""
auth_and_store.py — Supabase-backed authentication for DESTINY AI.

Same crypto as before (PBKDF2-HMAC-SHA256, 200k iterations, per-user
salt, constant-time comparison, login lockout). The only change is
that users are stored in Supabase instead of a local JSON file, so
accounts survive Streamlit Cloud redeploys.

If Supabase is not configured (no SUPABASE_URL / key in env), this
module falls back to the original local-file behaviour so local
development still works without setting up Supabase.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("auth")

# ---------------------------------------------------------------- #
# Constants
# ---------------------------------------------------------------- #

DATA_DIR = Path("data")
USERS_FILE = DATA_DIR / "users_auth.json"

MAX_FAILED_ATTEMPTS = 5
LOCKOUT_SECONDS = 300

# Supabase table name — must match the SQL you ran.
_AUTH_TABLE = "auth_users"


# ---------------------------------------------------------------- #
# Public result type
# ---------------------------------------------------------------- #

@dataclass
class AuthResult:
    ok: bool
    message: str
    username: str | None = None


# ---------------------------------------------------------------- #
# Supabase helpers
# ---------------------------------------------------------------- #

def _supabase_config() -> tuple[str, str]:
    """Return (url, key) from env, or ('', '') if not configured."""
    url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
    service = os.getenv("SUPABASE_SERVICE_KEY", "").strip()
    generic = os.getenv("SUPABASE_KEY", "").strip()
    anon = os.getenv("SUPABASE_ANON_KEY", "").strip()
    key = service or generic or anon
    return url, key


def _sb_request(
    method: str,
    path: str,
    body: object | None = None,
    prefer: str | None = None,
):
    """Return (ok, parsed_json_or_None)."""
    url, key = _supabase_config()
    if not url or not key:
        return False, None

    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
    }

    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    if prefer:
        headers["Prefer"] = prefer

    req = urllib.request.Request(
        f"{url}/rest/v1/{path}",
        data=data,
        headers=headers,
        method=method,
    )

    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = resp.read().decode("utf-8")
            parsed = json.loads(raw) if raw.strip() else None
            return 200 <= resp.status < 300, parsed
    except Exception as exc:
        log.warning(
            "Supabase %s %s failed: %s",
            method,
            path.split("?")[0],
            exc,
        )
        return False, None


def _sb_get_user(username_or_email: str) -> dict | None:
    """Fetch one user row by username OR email. Returns None if not found."""
    # Try username first
    path = (
        f"{_AUTH_TABLE}?"
        f"username=eq.{urllib.parse.quote(username_or_email)}"
        "&limit=1"
    )
    ok, rows = _sb_request("GET", path)
    if ok and isinstance(rows, list) and rows:
        return rows[0]

    # Then try email
    path = (
        f"{_AUTH_TABLE}?"
        f"email=eq.{urllib.parse.quote(username_or_email)}"
        "&limit=1"
    )
    ok, rows = _sb_request("GET", path)
    if ok and isinstance(rows, list) and rows:
        return rows[0]

    return None


def _sb_user_exists_username(username: str) -> bool:
    path = (
        f"{_AUTH_TABLE}?"
        f"username=eq.{urllib.parse.quote(username)}"
        "&select=username&limit=1"
    )
    ok, rows = _sb_request("GET", path)
    return bool(ok and isinstance(rows, list) and rows)


def _sb_user_exists_email(email: str) -> bool:
    path = (
        f"{_AUTH_TABLE}?"
        f"email=eq.{urllib.parse.quote(email)}"
        "&select=email&limit=1"
    )
    ok, rows = _sb_request("GET", path)
    return bool(ok and isinstance(rows, list) and rows)


def _sb_upsert_user(record: dict) -> bool:
    ok, _ = _sb_request(
        "POST",
        f"{_AUTH_TABLE}?on_conflict=username",
        [record],
        prefer="resolution=merge-duplicates",
    )
    return ok


def _sb_update_failed_attempts(
    username: str,
    failed_attempts: int,
    locked_until: float,
) -> bool:
    path = (
        f"{_AUTH_TABLE}?"
        f"username=eq.{urllib.parse.quote(username)}"
    )
    ok, _ = _sb_request(
        "PATCH",
        path,
        {
            "failed_attempts": failed_attempts,
            "locked_until": locked_until,
        },
    )
    return ok


# ---------------------------------------------------------------- #
# Local-file fallback (only used if Supabase not configured)
# ---------------------------------------------------------------- #

def _load_users_local() -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not USERS_FILE.exists():
        return {}
    try:
        return json.loads(USERS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_users_local(users: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    USERS_FILE.write_text(json.dumps(users, indent=2), encoding="utf-8")


def _supabase_enabled() -> bool:
    url, key = _supabase_config()
    return bool(url and key)


# ---------------------------------------------------------------- #
# Crypto (unchanged from your original file — this is correct)
# ---------------------------------------------------------------- #

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


# ---------------------------------------------------------------- #
# Public API
# ---------------------------------------------------------------- #

def register_user(username: str, email: str, password: str) -> AuthResult:
    clean_user = username.strip().lower()
    clean_email = email.strip().lower()

    # ---- validation (unchanged) ----
    if not re.fullmatch(r"[a-z0-9_]{3,24}", clean_user):
        return AuthResult(
            False,
            "Username must be 3–24 characters (letters, numbers, underscores).",
        )
    if "@" not in clean_email or "." not in clean_email.split("@")[-1]:
        return AuthResult(False, "Please enter a valid email address.")
    if len(password) < 8:
        return AuthResult(False, "Password must be at least 8 characters long.")

    salt_hex, pw_hash = _hash_password(password)
    created_at = int(time.time())

    record = {
        "username": clean_user,
        "email": clean_email,
        "salt": salt_hex,
        "password_hash": pw_hash,
        "failed_attempts": 0,
        "locked_until": 0.0,
        "created_at": created_at,
    }

    # ---- Supabase path ----
    if _supabase_enabled():
        if _sb_user_exists_username(clean_user):
            return AuthResult(False, "That username is already taken.")
        if _sb_user_exists_email(clean_email):
            return AuthResult(False, "An account with that email already exists.")

        if not _sb_upsert_user(record):
            log.error("Supabase register failed for %s", clean_user)
            return AuthResult(
                False,
                "Registration could not be saved. Please try again.",
            )

        log.info("Registered user %s via Supabase", clean_user)
        return AuthResult(
            True,
            "Account created securely. You can now log in.",
            username=clean_user,
        )

    # ---- Local-file fallback ----
    users = _load_users_local()
    if clean_user in users:
        return AuthResult(False, "That username is already taken.")
    if any(u.get("email") == clean_email for u in users.values()):
        return AuthResult(False, "An account with that email already exists.")

    users[clean_user] = record
    _save_users_local(users)
    log.info("Registered user %s via local file (Supabase not configured)", clean_user)
    return AuthResult(
        True,
        "Account created securely. You can now log in.",
        username=clean_user,
    )


def authenticate_user(username_or_email: str, password: str) -> AuthResult:
    ident = username_or_email.strip().lower()

    # ---- Supabase path ----
    if _supabase_enabled():
        record = _sb_get_user(ident)
        if not record:
            return AuthResult(False, "Invalid credentials.")

        matched_user = str(record.get("username", ""))
        now = time.time()
        locked_until = float(record.get("locked_until", 0.0))
        if locked_until > now:
            remaining = int(locked_until - now)
            return AuthResult(
                False,
                f"Account temporarily locked. Try again in {remaining}s.",
            )

        salt_hex = record.get("salt", "")
        expected_hash = record.get("password_hash", "")
        _, actual_hash = _hash_password(password, salt_hex=salt_hex)

        if hmac.compare_digest(expected_hash, actual_hash):
            # Reset failed attempts on success
            if record.get("failed_attempts", 0) or record.get("locked_until", 0.0):
                _sb_update_failed_attempts(matched_user, 0, 0.0)
            log.info("Login success for %s via Supabase", matched_user)
            return AuthResult(
                True,
                f"Welcome back, {matched_user}.",
                username=matched_user,
            )

        # Failure path — increment attempts, possibly lock
        failed = int(record.get("failed_attempts", 0)) + 1
        if failed >= MAX_FAILED_ATTEMPTS:
            _sb_update_failed_attempts(
                matched_user,
                0,
                now + LOCKOUT_SECONDS,
            )
            log.warning("Lockout triggered for %s", matched_user)
            return AuthResult(
                False,
                f"Too many failed attempts. Locked for {LOCKOUT_SECONDS}s.",
            )

        _sb_update_failed_attempts(matched_user, failed, 0.0)
        return AuthResult(
            False,
            f"Invalid credentials ({MAX_FAILED_ATTEMPTS - failed} attempts left).",
        )

    # ---- Local-file fallback ----
    users = _load_users_local()

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
        return AuthResult(
            False,
            f"Account temporarily locked. Try again in {remaining}s.",
        )

    salt_hex = record.get("salt", "")
    expected_hash = record.get("password_hash", "")
    _, actual_hash = _hash_password(password, salt_hex=salt_hex)

    if hmac.compare_digest(expected_hash, actual_hash):
        record["failed_attempts"] = 0
        record["locked_until"] = 0.0
        _save_users_local(users)
        return AuthResult(
            True,
            f"Welcome back, {matched_user}.",
            username=matched_user,
        )

    failed = int(record.get("failed_attempts", 0)) + 1
    record["failed_attempts"] = failed
    if failed >= MAX_FAILED_ATTEMPTS:
        record["locked_until"] = now + LOCKOUT_SECONDS
        record["failed_attempts"] = 0
        _save_users_local(users)
        return AuthResult(
            False,
            f"Too many failed attempts. Locked for {LOCKOUT_SECONDS}s.",
        )

    _save_users_local(users)
    return AuthResult(
        False,
        f"Invalid credentials ({MAX_FAILED_ATTEMPTS - failed} attempts left).",
                                   )
