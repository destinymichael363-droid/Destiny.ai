from __future__ import annotations

import inspect
import json
import logging
import os
import signal
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd
import yfinance as yf

from broker_and_account import VirtualAccount
from ob_fvg_engine import run_ob_fvg_engine
from sr_engine import run_sr_zones_engine

log = logging.getLogger("worker")

# FIX: a failed import is now logged loudly instead of silently disabling signals.
try:
    from strategy_engine import evaluate_setups as _evaluate_setups  # type: ignore
except Exception:
    log.exception(
        "strategy_engine import failed - worker will produce NO signals"
    )
    _evaluate_setups = None

try:
    import mtf_runner as _mtf_runner  # type: ignore
except Exception:
    log.warning(
        "mtf_runner not available; continuing without MTF hooks",
        exc_info=True,
    )
    _mtf_runner = None


STATE_PATH = Path("data/worker_state.json")
VA_DIR = Path("data/virtual_accounts")

POLL_INTERVAL_SEC = int(
    os.getenv("WORKER_POLL_SEC", "30")
)

MAX_SIGNALS = int(
    os.getenv("WORKER_MAX_SIGNALS", "25")
)

MAX_AUTO_OPEN_POSITIONS = int(
    os.getenv("MAX_AUTO_OPEN_POSITIONS", "3")
)

MAX_TOTAL_RISK_PCT = float(
    os.getenv("MAX_TOTAL_RISK_PCT", "6.5")
)

MAX_RISK_PCT_PER_TRADE = float(
    os.getenv("MAX_RISK_PCT_PER_TRADE", "5.0")
)

MAX_LEVERAGE = float(
    os.getenv("MAX_LEVERAGE", "20")
)

MAX_CLOSED_LOG = int(
    os.getenv("MAX_CLOSED_LOG", "500")
)

MARGIN_RATE = 0.05

# Primary strategy timeframe.
BASE_TF_MIN = 15

# Complete strategy cascade.
STRATEGY_TIMEFRAMES = (
    "4H",
    "1H",
    "15M",
    "5M",
    "1M",
)


WORKER_WATCHLIST: list[tuple[str, str]] = [
    ("Gold (XAU)", "GC=F"),
    ("Crude Oil (WTI)", "CL=F"),
    ("EUR/USD", "EURUSD=X"),
    ("GBP/USD", "GBPUSD=X"),
    ("Bitcoin (BTC)", "BTC-USD"),
]


# FIX:
# Instruments that can represent substantially similar USD exposure.
CORRELATED_GROUPS: list[set[str]] = [
    {
        "EURUSD=X",
        "GBPUSD=X",
    }
]


OHLCV = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]


# FIX:
# The worker now knows every timeframe used by the MTF cascade.
_INTERVAL_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "1h": 60,
}


# FIX: these are currently unused; _handle_stop is never registered via
# signal.signal(), so Ctrl-C falls back to the default handler. Kept in
# place so a future main() loop can wire them up without re-adding the
# boilerplate.
_RUNNING = True
_WARNED_ANON = False


def _handle_stop(
    signum: int,
    frame: object,
) -> None:
    _ = (signum, frame)

    global _RUNNING

    _RUNNING = False


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_str() -> str:
    return _utc_now().strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )


def _utc_iso() -> str:
    return _utc_now().isoformat(
        timespec="seconds"
    )


def _get_val(
    obj: object,
    names: tuple[str, ...],
    default: object = None,
) -> object:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]

        if hasattr(obj, name):
            return getattr(obj, name)

    return default


def _to_float(
    value: object,
    default: float | None = None,
) -> float | None:
    """
    Safe numeric conversion.

    None, NaN, infinity and invalid strings become default.
    """
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default

    if out != out:
        return default

    if out in (
        float("inf"),
        float("-inf"),
    ):
        return default

    return out


def _num_or_str(
    value: object,
) -> object:
    """
    JSON-safe value.

    Real numeric values remain floats; invalid values become strings.
    """
    f = _to_float(value)

    return (
        f
        if f is not None
        else str(value)
    )


def _call_filtered(
    fn,
    **kwargs,
):
    """
    Pass only keyword arguments accepted by fn.

    This avoids the previous dangerous pattern of catching TypeError,
    which could accidentally hide a genuine bug inside the function.
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return fn(**kwargs)

    if any(
        p.kind is inspect.Parameter.VAR_KEYWORD
        for p in params.values()
    ):
        return fn(**kwargs)

    return fn(
        **{
            k: v
            for k, v in kwargs.items()
            if k in params
        }
    )


# --------------------------------------------------------------------------- #
# Position record
# --------------------------------------------------------------------------- #
class PositionRecord:
    """
    Plain position object carrying every field required by the worker.

    `reserved` is cash deducted when a position is opened and refunded
    when the position closes.
    """

    def __init__(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        size: float,
        units: float,
        margin_used: float,
        reserved: float = 0.0,
        opened_utc: str | None = None,
    ) -> None:
        self.symbol = symbol
        self.side = side
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.size = size
        self.units = units
        self.margin_used = margin_used
        self.reserved = reserved
        self.opened_utc = opened_utc

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "size": self.size,
            "units": self.units,
            "margin_used": self.margin_used,
            "reserved": self.reserved,
            "opened_utc": self.opened_utc,
        }


def _to_record(
    pos: object,
    reserved: float | None = None,
    opened_utc: str | None = None,
) -> PositionRecord:
    size = _to_float(
        _get_val(
            pos,
            (
                "size",
                "qty",
                "quantity",
                "units",
            ),
            1.0,
        ),
        1.0,
    )

    units = _to_float(
        _get_val(
            pos,
            (
                "units",
                "size",
                "qty",
                "quantity",
            ),
            size,
        ),
        size,
    )

    entry = _to_float(
        _get_val(
            pos,
            (
                "entry_price",
                "entry",
            ),
            0.0,
        ),
        0.0,
    )

    margin = _to_float(
        _get_val(
            pos,
            (
                "margin_used",
                "margin",
                "collateral",
            ),
            None,
        )
    )

    if margin is None:
        margin = round(
            abs(units * entry) * MARGIN_RATE,
            2,
        )

    if reserved is None:
        reserved = _to_float(
            _get_val(
                pos,
                ("reserved",),
                0.0,
            ),
            0.0,
        )

    return PositionRecord(
        symbol=str(
            _get_val(
                pos,
                ("symbol",),
                "",
            )
        ),
        side=str(
            _get_val(
                pos,
                ("side",),
                "LONG",
            )
        ).upper(),
        entry_price=entry,
        stop_loss=_to_float(
            _get_val(
                pos,
                (
                    "stop_loss",
                    "sl",
                ),
                0.0,
            ),
            0.0,
        ),
        take_profit=_to_float(
            _get_val(
                pos,
                (
                    "take_profit",
                    "tp",
                ),
                0.0,
            ),
            0.0,
        ),
        size=size,
        units=units,
        margin_used=margin,
        reserved=reserved,
        opened_utc=(
            opened_utc
            or _get_val(
                pos,
                ("opened_utc",),
                None,
            )
        ),
    )


def _record_from_dict(
    item: dict,
) -> PositionRecord:
    ent = _to_float(
        item.get("entry_price"),
        0.0,
    )

    # FIX: align field fallbacks with _to_record. Previously this
    # function missed "qty" / "quantity", so a persisted JSON with
    # those keys would be silently read as size=1, units=1.
    sz = _to_float(
        item.get(
            "size",
            item.get(
                "qty",
                item.get(
                    "quantity",
                    item.get("units"),
                ),
            ),
        ),
        1.0,
    )

    units = _to_float(
        item.get(
            "units",
            item.get(
                "size",
                item.get(
                    "qty",
                    item.get("quantity"),
                ),
            ),
        ),
        sz,
    )

    margin = _to_float(
        item.get("margin_used")
    )

    if margin is None:
        margin = round(
            abs(units * ent) * MARGIN_RATE,
            2,
        )

    return PositionRecord(
        symbol=str(
            item.get(
                "symbol",
                "",
            )
        ),
        side=str(
            item.get(
                "side",
                "LONG",
            )
        ).upper(),
        entry_price=ent,
        stop_loss=_to_float(
            item.get("stop_loss"),
            0.0,
        ),
        take_profit=_to_float(
            item.get("take_profit"),
            0.0,
        ),
        size=sz,
        units=units,
        margin_used=margin,
        reserved=_to_float(
            item.get("reserved"),
            0.0,
        ),
        opened_utc=item.get(
            "opened_utc"
        ),
    )


# --------------------------------------------------------------------------- #
# Supabase
# --------------------------------------------------------------------------- #
def _supabase_config() -> tuple[str, str]:
    global _WARNED_ANON

    url = os.getenv(
        "SUPABASE_URL",
        "",
    ).strip().rstrip("/")

    service = os.getenv(
        "SUPABASE_SERVICE_KEY",
        "",
    ).strip()

    generic = os.getenv(
        "SUPABASE_KEY",
        "",
    ).strip()

    anon = os.getenv(
        "SUPABASE_ANON_KEY",
        "",
    ).strip()

    key = (
        service
        or generic
        or anon
    )

    if (
        key
        and not (service or generic)
        and not _WARNED_ANON
    ):
        log.warning(
            "Using SUPABASE_ANON_KEY for server-side writes; "
            "use SUPABASE_SERVICE_KEY (server only) or writes "
            "may be rejected by row-level security."
        )

        _WARNED_ANON = True

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
        data = json.dumps(body).encode(
            "utf-8"
        )

        headers["Content-Type"] = (
            "application/json"
        )

    if prefer:
        headers["Prefer"] = prefer

    req = urllib.request.Request(
        f"{url}/rest/v1/{path}",
        data=data,
        headers=headers,
        method=method,
    )

    try:
        with urllib.request.urlopen(
            req,
            timeout=8,
        ) as resp:
            raw = resp.read().decode(
                "utf-8"
            )

            parsed = (
                json.loads(raw)
                if raw.strip()
                else None
            )

            return (
                200 <= resp.status < 300,
                parsed,
            )

    except Exception as exc:
        log.warning(
            "Supabase %s %s failed: %s",
            method,
            path.split("?")[0],
            exc,
        )

        return False, None


def _extract_payload(
    rows: object,
) -> dict | None:
    if (
        isinstance(rows, list)
        and rows
        and isinstance(rows[0], dict)
    ):
        raw = rows[0].get(
            "payload"
        )

        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except ValueError:
                return None

        if isinstance(raw, dict):
            return raw

    return None


def _supabase_load_worker_state() -> dict | None:
    ok, rows = _sb_request(
        "GET",
        "worker_state?id=eq.default&select=payload&limit=1",
    )

    return (
        _extract_payload(rows)
        if ok
        else None
    )


def _supabase_save_worker_state(
    state: dict,
) -> bool:
    ok, _ = _sb_request(
        "POST",
        "worker_state?on_conflict=id",
        [
            {
                "id": "default",
                "payload": state,
            }
        ],
        prefer="resolution=merge-duplicates",
    )

    return ok


def _supabase_load_account(
    username: str,
) -> dict | None:
    path = (
        "virtual_accounts?"
        f"username=eq."
        f"{urllib.parse.quote(username)}"
        "&select=payload&limit=1"
    )

    ok, rows = _sb_request(
        "GET",
        path,
    )

    return (
        _extract_payload(rows)
        if ok
        else None
    )


def _supabase_list_usernames() -> list[str]:
    ok, rows = _sb_request(
        "GET",
        "virtual_accounts?select=username",
    )

    if (
        not ok
        or not isinstance(rows, list)
    ):
        return []

    return [
        str(
            r.get(
                "username",
                "",
            )
        ).strip()
        for r in rows
        if (
            isinstance(r, dict)
            and str(
                r.get(
                    "username",
                    "",
                )
            ).strip()
        )
    ]


def _supabase_save_account(
    username: str,
    payload: dict,
) -> bool:
    ok, _ = _sb_request(
        "POST",
        "virtual_accounts?on_conflict=username",
        [
            {
                "username": username,
                "payload": payload,
            }
        ],
        prefer="resolution=merge-duplicates",
    )

    return ok


# --------------------------------------------------------------------------- #
# Account storage
# --------------------------------------------------------------------------- #
def _va_file_path(
    username: str,
) -> Path:
    safe_name = "".join(
        ch
        if ch.isalnum()
        or ch in ("_", "-")
        else "_"
        for ch in username
    )[:40] or "default"

    VA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    return VA_DIR / f"{safe_name}.json"


@contextmanager
def _account_lock(
    username: str,
    timeout: float = 10.0,
    stale_after: float = 60.0,
) -> Iterator[None]:
    """
    Same-host account lock around load -> modify -> save.

    This does not protect against two separate machines writing to
    Supabase simultaneously.
    """
    lock_path = _va_file_path(
        username
    ).with_suffix(".lock")

    deadline = (
        time.monotonic()
        + timeout
    )

    fd = None

    while fd is None:
        try:
            fd = os.open(
                str(lock_path),
                os.O_CREAT
                | os.O_EXCL
                | os.O_WRONLY,
            )

        except FileExistsError:
            try:
                if (
                    time.time()
                    - lock_path.stat().st_mtime
                    > stale_after
                ):
                    lock_path.unlink(
                        missing_ok=True
                    )
                    continue

            except OSError:
                pass

            if (
                time.monotonic()
                >= deadline
            ):
                raise TimeoutError(
                    f"could not lock account "
                    f"'{username}'"
                )

            time.sleep(0.05)

    try:
        yield

    finally:
        os.close(fd)

        try:
            lock_path.unlink(
                missing_ok=True
            )
        except OSError:
            pass


def _load_raw_account_payload(
    username: str,
) -> dict:
    remote = _supabase_load_account(
        username
    )

    local: dict | None = None

    acct_path = _va_file_path(
        username
    )

    if acct_path.exists():
        try:
            data = json.loads(
                acct_path.read_text(
                    encoding="utf-8"
                )
            )

            if isinstance(data, dict):
                local = data

        except Exception:
            log.warning(
                "could not read local account file %s",
                acct_path,
                exc_info=True,
            )

    # Prefer the newest copy.
    if remote and local:
        return (
            remote
            if str(
                remote.get(
                    "updated_utc",
                    "",
                )
            )
            >= str(
                local.get(
                    "updated_utc",
                    "",
                )
            )
            else local
        )

    if remote:
        return remote

    if local:
        return local

    return {
        "balance": 10000.0,
        "positions": [],
        "closed_positions": [],
        "auto_exec_virtual": True,
        "risk_pct": 2.0,
    }


def _account_to_payload(
    acct: VirtualAccount,
    closed_log: list[dict],
    prefs: dict | None = None,
) -> dict:
    p = prefs or {}

    return {
        "balance": round(
            float(
                getattr(
                    acct,
                    "balance",
                    10000.0,
                )
            ),
            2,
        ),
        "positions": [
            _to_record(pos).to_dict()
            for pos in list(
                getattr(
                    acct,
                    "positions",
                    [],
                )
                or []
            )
        ],
        "closed_positions": list(
            closed_log or []
        )[-MAX_CLOSED_LOG:],
        "auto_exec_virtual": bool(
            p.get(
                "auto_exec_virtual",
                True,
            )
        ),
        "risk_pct": float(
            p.get(
                "risk_pct",
                2.0,
            )
        ),
        "updated_utc": _utc_iso(),
    }


def _payload_to_account(
    data: dict,
) -> tuple[
    VirtualAccount,
    list[dict],
    dict,
]:
    acct = VirtualAccount()

    closed_log: list[dict] = []

    prefs = {
        "auto_exec_virtual": True,
        "risk_pct": 2.0,
    }

    if not isinstance(data, dict):
        return (
            acct,
            closed_log,
            prefs,
        )

    prefs[
        "auto_exec_virtual"
    ] = bool(
        data.get(
            "auto_exec_virtual",
            True,
        )
    )

    prefs["risk_pct"] = (
        _to_float(
            data.get(
                "risk_pct"
            ),
            2.0,
        )
    )

    if (
        "balance" in data
        and hasattr(acct, "balance")
    ):
        bal = _to_float(
            data["balance"]
        )

        if bal is not None:
            acct.balance = round(
                bal,
                2,
            )

    closed_log = list(
        data.get(
            "closed_positions"
        )
        or []
    )

    if (
        hasattr(acct, "positions")
        and isinstance(
            acct.positions,
            list,
        )
    ):
        for item in list(
            data.get(
                "positions"
            )
            or []
        ):
            if not isinstance(
                item,
                dict,
            ):
                continue

            try:
                acct.positions.append(
                    _record_from_dict(item)
                )

            except Exception:
                log.exception(
                    "could not restore saved position: %s",
                    item,
                )

    return (
        acct,
        closed_log,
        prefs,
    )


def _load_virtual_account_obj(
    username: str,
) -> tuple[
    VirtualAccount,
    list[dict],
    dict,
]:
    return _payload_to_account(
        _load_raw_account_payload(
            username
        )
    )


def _save_virtual_account_obj(
    username: str,
    acct: VirtualAccount,
    closed_log: list[dict],
    prefs: dict | None = None,
) -> None:
    payload = _account_to_payload(
        acct,
        closed_log,
        prefs=prefs,
    )

    try:
        path = _va_file_path(
            username
        )

        tmp = path.with_suffix(
            ".json.tmp"
        )

        tmp.write_text(
            json.dumps(
                payload,
                indent=2,
            ),
            encoding="utf-8",
        )

        os.replace(
            tmp,
            path,
        )

    except Exception:
        log.exception(
            "local account save failed for %s",
            username,
        )

    _supabase_save_account(
        username,
        payload,
    )


def _discover_trader_usernames() -> list[str]:
    users: set[str] = set()

    env_users = os.getenv(
        "AUTO_EXEC_USERS",
        "",
    ).strip()

    if env_users:
        for u in env_users.split(","):
            clean = u.strip()

            if clean:
                users.add(clean)

    single_user = os.getenv(
        "AUTO_EXEC_USER",
        "",
    ).strip()

    if single_user:
        users.add(single_user)

    if VA_DIR.exists():
        for p in VA_DIR.glob(
            "*.json"
        ):
            if p.stem:
                users.add(p.stem)

    users.update(
        _supabase_list_usernames()
    )

    return sorted(users)


# --------------------------------------------------------------------------- #
# Opening positions
# --------------------------------------------------------------------------- #
def _levels_valid(
    side: str,
    entry: float,
    sl: float,
    tp: float,
) -> bool:
    if side == "LONG":
        return sl < entry < tp

    if side == "SHORT":
        return tp < entry < sl

    return False


def _open_risk_dollars(
    positions: list,
) -> float:
    total = 0.0

    for pos in positions:
        rec = _to_record(pos)

        total += (
            abs(
                rec.entry_price
                - rec.stop_loss
            )
            * abs(rec.units)
        )

    return total


def _exposure_block_reason(
    positions: list,
    symbol: str,
    side: str,
    new_risk: float,
    balance: float,
) -> str | None:
    """
    Position cap + one position per symbol +
    correlation + total-risk cap.
    """
    if (
        len(positions)
        >= MAX_AUTO_OPEN_POSITIONS
    ):
        return "max_positions"

    for pos in positions:
        rec = _to_record(pos)

        if rec.symbol == symbol:
            return "symbol_already_open"

        for group in CORRELATED_GROUPS:
            if (
                symbol in group
                and rec.symbol in group
                and rec.side == side
            ):
                return "correlated_exposure"

    if (
        balance > 0
        and (
            _open_risk_dollars(
                positions
            )
            + new_risk
        )
        > balance
        * MAX_TOTAL_RISK_PCT
        / 100.0
    ):
        return "total_risk_cap"

    return None


def _call_open_fn(
    open_fn,
    symbol,
    side,
    entry_px,
    sl_px,
    tp_px,
    risk_pct,
):
    try:
        params = inspect.signature(
            open_fn
        ).parameters

    except (
        TypeError,
        ValueError,
    ):
        params = {}

    has_kwargs = any(
        p.kind
        is inspect.Parameter.VAR_KEYWORD
        for p in params.values()
    )

    if (
        "risk_pct" in params
        or has_kwargs
    ):
        return open_fn(
            symbol,
            side,
            entry_px,
            sl_px,
            tp_px,
            risk_pct=risk_pct,
        )

    if len(params) >= 6:
        return open_fn(
            symbol,
            side,
            entry_px,
            sl_px,
            tp_px,
            risk_pct,
        )

    return open_fn(
        symbol,
        side,
        entry_px,
        sl_px,
        tp_px,
    )


def _stamp_new_positions(
    acct: VirtualAccount,
    before: int,
    reserved: float,
    opened_utc: str,
) -> None:
    try:
        positions = acct.positions

        count = (
            len(positions)
            - before
        )

        if count <= 0:
            # FIX: open_position_with_risk reported success but did
            # not actually append a position. Log instead of silently
            # doing nothing — this would otherwise lose the reserved
            # margin on the next close.
            log.warning(
                "_stamp_new_positions: open_position_with_risk "
                "succeeded but no position was appended "
                "(before=%d, after=%d)",
                before,
                len(positions),
            )
            return

        per_position = round(
            reserved / count,
            2,
        )

        for i in range(
            before,
            len(positions),
        ):
            positions[i] = _to_record(
                positions[i],
                reserved=per_position,
                opened_utc=opened_utc,
            )

    except Exception:
        log.exception(
            "could not stamp new position(s)"
        )


def _open_via_virtual_account(
    acct: VirtualAccount,
    symbol: str,
    side: str,
    entry_px: float,
    sl_px: float,
    tp_px: float,
    risk_pct: float,
) -> tuple[bool, str]:
    pre_balance = round(
        float(
            getattr(
                acct,
                "balance",
                10000.0,
            )
        ),
        2,
    )

    intended_risk = round(
        pre_balance
        * (
            float(risk_pct)
            / 100.0
        ),
        2,
    )

    before = len(
        list(
            getattr(
                acct,
                "positions",
                [],
            )
            or []
        )
    )

    now_iso = _utc_iso()

    open_fn = getattr(
        acct,
        "open_position_with_risk",
        None,
    )

    if callable(open_fn):
        try:
            res = _call_open_fn(
                open_fn,
                symbol,
                side,
                entry_px,
                sl_px,
                tp_px,
                risk_pct,
            )

        except Exception:
            log.exception(
                "open_position_with_risk raised for %s",
                symbol,
            )

            return (
                False,
                "open_fn_error",
            )

        after = len(
            list(
                getattr(
                    acct,
                    "positions",
                    [],
                )
                or []
            )
        )

        if (
            isinstance(res, tuple)
            and len(res) >= 2
        ):
            ok = bool(res[0])
            msg = str(res[1])

        elif res is None:
            ok = after > before
            msg = (
                "Opened via VirtualAccount"
            )

        else:
            ok = bool(res)
            msg = (
                "Opened via VirtualAccount"
            )

        if not ok:
            return False, msg

        reserved = round(
            pre_balance
            - float(
                getattr(
                    acct,
                    "balance",
                    pre_balance,
                )
            ),
            2,
        )

        _stamp_new_positions(
            acct,
            before,
            reserved,
            now_iso,
        )

        return True, msg

    sl_dist = abs(
        entry_px
        - sl_px
    )

    if (
        sl_dist <= 0
        or intended_risk <= 0
        or pre_balance <= 0
    ):
        return (
            False,
            "signal_only_risk_guard",
        )

    units = (
        intended_risk
        / sl_dist
    )

    units = round(
        min(
            units,
            pre_balance
            * MAX_LEVERAGE
            / entry_px,
        ),
        6,
    )

    if units <= 0:
        return (
            False,
            "signal_only_risk_guard",
        )

    margin_used = round(
        abs(
            units
            * entry_px
        )
        * MARGIN_RATE,
        2,
    )

    # FIX: this fallback path used to construct PositionRecord with
    # reserved=0.0, which meant the close path would refund $0 of
    # margin and permanently lose that cash. Now it stamps
    # reserved=margin_used so the close path refunds correctly.
    rec = PositionRecord(
        symbol=symbol,
        side=side,
        entry_price=entry_px,
        stop_loss=sl_px,
        take_profit=tp_px,
        size=units,
        units=units,
        margin_used=margin_used,
        reserved=margin_used,
        opened_utc=now_iso,
    )

    positions = list(
        getattr(
            acct,
            "positions",
            [],
        )
        or []
    )

    positions.append(rec)

    acct.positions = positions

    return True, "opened_fallback"


# --------------------------------------------------------------------------- #
# Closing positions
# --------------------------------------------------------------------------- #
def _bars_since(
    source: object,
    opened_utc: str | None,
) -> list[
    tuple[
        float,
        float,
        float,
    ]
]:
    """
    Return (open, high, low) bars that started after the
    position was opened.
    """
    if isinstance(
        source,
        tuple,
    ):
        hi, lo, close = source

        return [
            (
                close,
                hi,
                lo,
            )
        ]

    df = source

    if (
        df is None
        or getattr(
            df,
            "empty",
            True,
        )
    ):
        return []

    opened_ts = (
        pd.to_datetime(
            opened_utc,
            utc=True,
            errors="coerce",
        )
        if opened_utc
        else pd.NaT
    )

    if pd.isna(opened_ts):
        last = df.iloc[-1]

        return [
            (
                float(last["Open"]),
                float(last["High"]),
                float(last["Low"]),
            )
        ]

    idx = df.index

    if getattr(
        idx,
        "tz",
        None,
    ) is None:
        idx = idx.tz_localize(
            "UTC"
        )

    sub = df[
        idx > opened_ts
    ]

    return [
        (
            float(r.Open),
            float(r.High),
            float(r.Low),
        )
        for r in sub.itertuples()
    ]


def _check_exit(
    side: str,
    sl: float,
    tp: float,
    bars: list[
        tuple[
            float,
            float,
            float,
        ]
    ],
) -> float | None:
    """
    First bar touching SL or TP.

    If both are touched in one candle, SL wins because the exact
    intrabar path is unknown.
    """
    for (
        bar_open,
        hi,
        lo,
    ) in bars:
        if side == "LONG":
            if (
                sl > 0
                and lo <= sl
            ):
                return min(
                    sl,
                    bar_open,
                )

            if (
                tp > 0
                and hi >= tp
            ):
                return tp

        elif side == "SHORT":
            if (
                sl > 0
                and hi >= sl
            ):
                return max(
                    sl,
                    bar_open,
                )

            if (
                tp > 0
                and lo <= tp
            ):
                return tp

    return None


def _evaluate_tp_sl_hits_for_users(
    price_bars: dict,
    usernames: list[str],
) -> int:
    if (
        not price_bars
        or not usernames
    ):
        return 0

    total_closed = 0

    for username in usernames:
        try:
            with _account_lock(
                username
            ):
                (
                    acct,
                    closed_log,
                    prefs,
                ) = _load_virtual_account_obj(
                    username
                )

                positions = list(
                    getattr(
                        acct,
                        "positions",
                        [],
                    )
                    or []
                )

                if not positions:
                    continue

                remaining = []
                closed_count = 0

                for pos in positions:
                    rec = _to_record(pos)

                    source = price_bars.get(
                        rec.symbol
                    )

                    if source is None:
                        remaining.append(
                            pos
                        )
                        continue

                    exit_px = _check_exit(
                        rec.side,
                        rec.stop_loss,
                        rec.take_profit,
                        _bars_since(
                            source,
                            rec.opened_utc,
                        ),
                    )

                    if exit_px is None:
                        remaining.append(
                            pos
                        )
                        continue

                    signed_units = (
                        -abs(rec.units)
                        if rec.side == "SHORT"
                        else abs(rec.units)
                    )

                    pnl_val = round(
                        float(
                            (
                                exit_px
                                - rec.entry_price
                            )
                            * signed_units
                        ),
                        2,
                    )

                    if hasattr(
                        acct,
                        "balance",
                    ):
                        # FIX: previously this refunded only
                        # `rec.reserved`. When reserved was 0
                        # (e.g. _stamp_new_positions failed, or the
                        # fallback open path was used), the margin
                        # was silently lost on close. Now we fall
                        # back to margin_used whenever reserved is
                        # missing or non-positive.
                        refund = rec.reserved

                        if (
                            not isinstance(refund, (int, float))
                            or refund <= 0
                        ):
                            refund = rec.m
