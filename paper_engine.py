
"""
DESTINY AI - VIRTUAL PAPER ENGINE
Historical simulation only. No broker connection.
"""

from datetime import datetime, timezone
import hashlib
import pandas as pd


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def make_trade_id(candidate):
    identity = "|".join([
        str(candidate["symbol"]),
        str(candidate["timeframe"]),
        str(candidate["direction"]),
        str(candidate["signal_time"]),
        str(candidate["entry"]),
    ])
    return hashlib.sha256(identity.encode()).hexdigest()[:24]


def validate_candidate(candidate):
    required = [
        "symbol",
        "timeframe",
        "direction",
        "signal_time",
        "entry",
        "stop_loss",
        "take_profit",
        "risk_amount",
    ]

    missing = [key for key in required if key not in candidate]

    if missing:
        raise ValueError(f"Missing fields: {missing}")

    item = dict(candidate)

    item["direction"] = str(item["direction"]).upper()

    if item["direction"] not in ("LONG", "SHORT"):
        raise ValueError("Direction must be LONG or SHORT.")

    for key in ["entry", "stop_loss", "take_profit", "risk_amount"]:
        item[key] = float(item[key])

    if item["risk_amount"] <= 0:
        raise ValueError("Risk amount must be positive.")

    if item["direction"] == "LONG":
        if not item["stop_loss"] < item["entry"] < item["take_profit"]:
            raise ValueError("Invalid LONG price relationships.")

    else:
        if not item["take_profit"] < item["entry"] < item["stop_loss"]:
            raise ValueError("Invalid SHORT price relationships.")

    item["trade_id"] = item.get("trade_id") or make_trade_id(item)

    return item


def simulate_candidate(candles, candidate):
    """
    Replay historical candles after the candidate signal.

    Conservative rule:
    If stop and target are both reached within one candle,
    stop is assumed to have been reached first.
    """

    item = validate_candidate(candidate)

    required_columns = {"Open", "High", "Low", "Close"}

    if not required_columns.issubset(candles.columns):
        raise ValueError("OHLC data is incomplete.")

    if candles.empty:
        raise ValueError("No candles available.")

    data = candles.copy().sort_index()
    data = data[~data.index.duplicated(keep="last")]
    data = data.dropna(subset=["Open", "High", "Low", "Close"])

    signal_time = pd.Timestamp(item["signal_time"])

    if signal_time.tzinfo is None:
        signal_time = signal_time.tz_localize("UTC")
    else:
        signal_time = signal_time.tz_convert("UTC")

    candle_index = pd.DatetimeIndex(data.index)

    if candle_index.tz is None:
        candle_index = candle_index.tz_localize("UTC")
    else:
        candle_index = candle_index.tz_convert("UTC")

    data.index = candle_index

    future = data[data.index > signal_time]

    if future.empty:
        raise ValueError("No candles after the signal time.")

    entry = item["entry"]
    stop = item["stop_loss"]
    target = item["take_profit"]
    direction = item["direction"]

    exit_price = None
    exit_time = None
    outcome = "TIME_EXIT"
    reason = "Historical replay ended"

    for timestamp, candle in future.iterrows():

        high = float(candle["High"])
        low = float(candle["Low"])
        open_price = float(candle["Open"])

        if direction == "LONG":

            stop_hit = low <= stop
            target_hit = high >= target

            if stop_hit:
                exit_price = open_price if open_price < stop else stop
                outcome = "LOSS"
                reason = "Stop reached"

            elif target_hit:
                exit_price = open_price if open_price > target else target
                outcome = "WIN"
                reason = "Target reached"

        else:

            stop_hit = high >= stop
            target_hit = low <= target

            if stop_hit:
                exit_price = open_price if open_price > stop else stop
                outcome = "LOSS"
                reason = "Stop reached"

            elif target_hit:
                exit_price = open_price if open_price < target else target
                outcome = "WIN"
                reason = "Target reached"

        if exit_price is not None:
            exit_time = timestamp.isoformat()
            break

    if exit_price is None:

        last_candle = future.iloc[-1]

        exit_price = float(last_candle["Close"])
        exit_time = future.index[-1].isoformat()
        reason = "Historical data ended"

    price_risk = abs(entry - stop)

    if direction == "LONG":
        price_result = exit_price - entry
    else:
        price_result = entry - exit_price

    result_r = price_result / price_risk

    virtual_pnl = item["risk_amount"] * result_r

    if outcome == "TIME_EXIT":

        if virtual_pnl > 0:
            outcome = "TIME_EXIT_PROFIT"

        elif virtual_pnl < 0:
            outcome = "TIME_EXIT_LOSS"

        else:
            outcome = "BREAKEVEN"

    return {
        "trade_id": item["trade_id"],
        "symbol": item["symbol"],
        "timeframe": item["timeframe"],
        "direction": direction,
        "signal_time": str(item["signal_time"]),
        "entry": entry,
        "stop_loss": stop,
        "take_profit": target,
        "exit_time": exit_time,
        "exit_price": float(exit_price),
        "risk_amount": item["risk_amount"],
        "result_r": round(result_r, 4),
        "virtual_pnl": round(virtual_pnl, 2),
        "outcome": outcome,
        "exit_reason": reason,
        "simulation_only": True,
        "recorded_at": utc_now(),
    }


def load_account(supabase, user_id):

    result = (
        supabase.table("paper_account")
        .select("balance, transactions")
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )

    if not result.data:
        raise RuntimeError("Virtual account not found.")

    account = result.data[0]

    account["balance"] = float(account.get("balance") or 0)
    account["transactions"] = account.get("transactions") or []

    return account


def record_closed_trade(supabase, user_id, trade):

    if not trade.get("simulation_only"):
        raise ValueError("Only virtual trades are accepted.")

    account = load_account(supabase, user_id)

    trade_id = trade["trade_id"]

    # Duplicate protection
    for existing in account["transactions"]:

        if (
            existing.get("Event") == "Virtual trade closed"
            and existing.get("Trade ID") == trade_id
        ):
            return {
                "saved": False,
                "duplicate": True,
                "balance": account["balance"],
            }

    pnl = float(trade["virtual_pnl"])

    new_balance = account["balance"] + pnl

    history_item = {
        "Time (UTC)": utc_now(),
        "Type": "Virtual trade closed",
        "Event": "Virtual trade closed",
        "Trade ID": trade_id,
        "Symbol": trade["symbol"],
        "Timeframe": trade["timeframe"],
        "Direction": trade["direction"],
        "Entry": trade["entry"],
        "Exit": trade["exit_price"],
        "Outcome": trade["outcome"],
        "Result (R)": trade["result_r"],
        "Amount": pnl,
        "Virtual balance": new_balance,
        "Simulation only": True,
    }

    updated_history = [history_item] + account["transactions"]

    result = (
        supabase.table("paper_account")
        .update({
            "balance": new_balance,
            "transactions": updated_history,
            "updated_at": utc_now(),
        })
        .eq("user_id", user_id)
        .execute()
    )

    if not result.data:
        raise RuntimeError("Virtual account update was not confirmed.")

    return {
        "saved": True,
        "duplicate": False,
        "balance": new_balance,
        "virtual_pnl": pnl,
    }

