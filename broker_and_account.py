from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal


@dataclass
class VirtualPosition:
    symbol: str
    side: Literal["LONG", "SHORT"]
    size: float
    entry_price: float
    opened_at: str


@dataclass
class VirtualAccount:
    balance: float = 10_000.0
    starting_balance: float = 10_000.0
    positions: list[VirtualPosition] = field(default_factory=list)
    trade_history: list[dict] = field(default_factory=list)

    def reset(self, new_starting_balance: float | None = None) -> None:
        base = (
            float(new_starting_balance)
            if new_starting_balance is not None
            else self.starting_balance
        )
        self.starting_balance = base
        self.balance = base
        self.positions.clear()
        self.trade_history.clear()

    def open_position(
        self,
        symbol: str,
        side: Literal["LONG", "SHORT"],
        notional_usd: float,
        price: float,
    ) -> tuple[bool, str]:
        if price <= 0 or notional_usd <= 0:
            return False, "Price and order size must be greater than zero."
        if notional_usd > self.balance:
            return False, "Not enough virtual balance for that order size."

        qty = notional_usd / price
        self.balance -= notional_usd
        self.positions.append(
            VirtualPosition(
                symbol=symbol,
                side=side,
                size=qty,
                entry_price=price,
                opened_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            )
        )
        return True, f"Opened {side} {qty:.5f} {symbol} @ {price:,.5f}"

    def close_position(self, index: int, exit_price: float) -> tuple[bool, str]:
        if index < 0 or index >= len(self.positions):
            return False, "Position not found."
        if exit_price <= 0:
            return False, "Invalid exit price."

        pos = self.positions.pop(index)
        initial_notional = pos.size * pos.entry_price

        if pos.side == "LONG":
            proceeds = pos.size * exit_price
            pnl = proceeds - initial_notional
        else:
            pnl = (pos.entry_price - exit_price) * pos.size
            proceeds = max(0.0, initial_notional + pnl)

        self.balance += proceeds
        self.trade_history.append(
            {
                "closed_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "symbol": pos.symbol,
                "side": pos.side,
                "size": round(pos.size, 6),
                "entry": pos.entry_price,
                "exit": exit_price,
                "pnl_usd": round(pnl, 2),
            }
        )
        return True, f"Closed {pos.side} {pos.symbol} | P&L: {pnl:+.2f} USD"

    def unrealized_pnl(self, current_prices: dict[str, float]) -> float:
        total = 0.0
        for pos in self.positions:
            px = current_prices.get(pos.symbol, pos.entry_price)
            if pos.side == "LONG":
                total += (px - pos.entry_price) * pos.size
            else:
                total += (pos.entry_price - px) * pos.size
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
        self.api_key_masked = (
            f"{clean_key[:4]}...{clean_key[-4:]}" if len(clean_key) >= 8 else "****"
        )
        return True, f"Connected {self.account_label} ({environment})."

    def disconnect(self) -> None:
        self.connected = False
        self.broker_name = "None"
        self.account_label = ""
        self.api_key_masked = ""

