from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal


@dataclass
class VirtualPosition:
    symbol: str
    side: Literal["LONG", "SHORT"]
    units: float
    margin_used: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_usd: float
    rr_ratio: float
    opened_at: str


@dataclass
class VirtualAccount:
    balance: float = 10_000.0
    starting_balance: float = 10_000.0
    positions: list[VirtualPosition] = field(default_factory=list)
    trade_history: list[dict] = field(default_factory=list)

    def reset(self, new_starting_balance: float | None = None) -> None:
        base = float(new_starting_balance) if new_starting_balance is not None else self.starting_balance
        self.starting_balance = base
        self.balance = base
        self.positions.clear()
        self.trade_history.clear()

    def open_position_with_risk(
        self,
        symbol: str,
        side: Literal["LONG", "SHORT"],
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        risk_percent: float = 2.0,
        max_leverage: float = 20.0,
    ) -> tuple[bool, str]:
        # FIX: validate max_leverage before using it. If max_leverage < 1
        # then margin_required = notional / max_leverage would be LARGER
        # than notional, which is nonsensical for a leveraged account.
        if not (isinstance(max_leverage, (int, float)) and max_leverage >= 1.0):
            return False, "max_leverage must be a number >= 1.0."

        # FIX: validate risk_percent. A value <= 0 disables trading
        # silently; a value > 100 risks more than the entire balance.
        if not (isinstance(risk_percent, (int, float)) and 0 < risk_percent <= 100):
            return False, "risk_percent must be between 0 (exclusive) and 100 (inclusive)."

        if entry_price <= 0 or stop_loss <= 0 or take_profit <= 0:
            return False, "Entry, Stop Loss, and Take Profit must all be positive."

        if side == "LONG" and not (stop_loss < entry_price < take_profit):
            return False, "For LONG: Stop Loss must be below Entry, and Take Profit above Entry."
        if side == "SHORT" and not (take_profit < entry_price < stop_loss):
            return False, "For SHORT: Take Profit must be below Entry, and Stop Loss above Entry."

        stop_distance = abs(entry_price - stop_loss)
        reward_distance = abs(take_profit - entry_price)
        if stop_distance <= 0:
            return False, "Stop Loss cannot equal Entry Price."

        risk_usd = self.balance * (risk_percent / 100.0)
        if risk_usd <= 0 or self.balance <= 0:
            return False, "Insufficient balance to risk."

        units = risk_usd / stop_distance
        notional_value = units * entry_price
        margin_required = notional_value / max_leverage

        if margin_required > self.balance:
            margin_required = self.balance
            units = (margin_required * max_leverage) / entry_price
            risk_usd = units * stop_distance

        rr_ratio = reward_distance / stop_distance
        self.balance -= margin_required

        self.positions.append(
            VirtualPosition(
                symbol=symbol,
                side=side,
                units=units,
                margin_used=margin_required,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_usd=risk_usd,
                rr_ratio=rr_ratio,
                opened_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            )
        )
        return (
            True,
            f"Opened {side} {symbol} @ {entry_price:,.5f} | SL: {stop_loss:,.5f} | TP: {take_profit:,.5f} (1:{rr_ratio:.1f} R:R)",
        )

    def close_position(
        self,
        index: int,
        exit_price: float,
        reason: str = "Manual Close",
    ) -> tuple[bool, str]:
        if index < 0 or index >= len(self.positions):
            return False, "Position not found."
        if exit_price <= 0:
            return False, "Invalid exit price."

        pos = self.positions.pop(index)
        if pos.side == "LONG":
            pnl = (exit_price - pos.entry_price) * pos.units
        else:
            pnl = (pos.entry_price - exit_price) * pos.units

        # NOTE: the max(0.0, ...) clamp below is a deliberate safety net.
        # If a stop-loss is gapped through (e.g. weekend gap or thin 1m
        # liquidity), the raw loss could exceed the posted margin. Without
        # this clamp the account balance could go negative on that single
        # trade. Clamping keeps the balance non-negative at the cost of
        # silently absorbing any loss above the margin.
        #
        # NOTE: worker.py does NOT call this method — it uses its own
        # close path in _evaluate_tp_sl_hits_for_users() which refunds
        # `reserved` (not `margin_used`). Those are kept in sync by
        # _stamp_new_positions(), so behavior is consistent as long as
        # one of the two paths is used per position. Do not mix them
        # for the same position.
        returned_cash = max(0.0, pos.margin_used + pnl)
        self.balance += returned_cash
        r_multiple = pnl / pos.risk_usd if pos.risk_usd > 0 else 0.0

        self.trade_history.append(
            {
                "closed_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "symbol": pos.symbol,
                "side": pos.side,
                "entry": round(pos.entry_price, 5),
                "exit": round(exit_price, 5),
                "sl": round(pos.stop_loss, 5),
                "tp": round(pos.take_profit, 5),
                "pnl_usd": round(pnl, 2),
                "r_result": f"{r_multiple:+.2f}R",
                "reason": reason,
            }
        )
        return True, f"Closed {pos.side} {pos.symbol} ({reason}) | P&L: {pnl:+.2f} USD ({r_multiple:+.2f}R)"

    def evaluate_tp_sl(self, symbol: str, candle_high: float, candle_low: float) -> list[str]:
        """
        Checks open positions for `symbol` against latest candle High/Low and closes any hit.

        NOTE: worker.py does NOT call this method. The worker uses its own
        bar-by-bar evaluation (_evaluate_tp_sl_hits_for_users in worker.py)
        because it needs to iterate every bar since the position opened,
        not just the most recent one. This method is kept as a public API
        for future UI use (e.g. a manual "check now" button).

        If both SL and TP fall inside a single candle, SL is checked first
        (conservative — assumes the stop was hit before the target).
        """
        messages: list[str] = []
        for idx in range(len(self.positions) - 1, -1, -1):
            pos = self.positions[idx]
            if pos.symbol != symbol:
                continue

            if pos.side == "LONG":
                if candle_low <= pos.stop_loss:
                    _, msg = self.close_position(idx, pos.stop_loss, reason="Stop Loss Hit")
                    messages.append(msg)
                elif candle_high >= pos.take_profit:
                    _, msg = self.close_position(idx, pos.take_profit, reason="Take Profit Hit")
                    messages.append(msg)
            else:
                if candle_high >= pos.stop_loss:
                    _, msg = self.close_position(idx, pos.stop_loss, reason="Stop Loss Hit")
                    messages.append(msg)
                elif candle_low <= pos.take_profit:
                    _, msg = self.close_position(idx, pos.take_profit, reason="Take Profit Hit")
                    messages.append(msg)
        return messages

    def unrealized_pnl(self, current_prices: dict[str, float]) -> float:
        total = 0.0
        for pos in self.positions:
            px = current_prices.get(pos.symbol, pos.entry_price)
            if pos.side == "LONG":
                total += (px - pos.entry_price) * pos.units
            else:
                total += (pos.entry_price - px) * pos.units
        return total


@dataclass
class BrokerSession:
    connected: bool = False
    broker_name: str = "None"
    account_label: str = ""
    api_key_masked: str = ""
    environment: Literal["Testnet / Demo", "Live"] = "Testnet / Demo"

    def connect(
        self,
        broker_name: str,
        api_key: str,
        api_secret: str,
        environment: Literal["Testnet / Demo", "Live"],
        account_label: str = "",
    ) -> tuple[bool, str]:
        clean_key = api_key.strip()
        clean_secret = api_secret.strip()
        if not clean_key or not clean_secret:
            return False, "Please enter both an API Key and API Secret."

        self.connected = True
        self.broker_name = broker_name
        self.environment = environment
        self.account_label = account_label.strip() or f"{broker_name} Account"
        self.api_key_masked = f"{clean_key[:4]}...{clean_key[-4:]}" if len(clean_key) >= 8 else "****"
        return True, f"Connected {self.account_label} ({environment})."

    def disconnect(self) -> None:
        self.connected = False
        self.broker_name = "None"
        self.account_label = ""
        self.api_key_masked = ""
