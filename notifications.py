from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal
import requests


@dataclass
class AlertItem:
    event_id: str
    category: Literal["SCALP_SIGNAL", "TP_SL_HIT", "LIQUIDITY_SWEEP", "SECURITY"]
    title: str
    detail: str
    timestamp: str


@dataclass
class NotificationCenter:
    items: list[AlertItem] = field(default_factory=list)
    seen_ids: set[str] = field(default_factory=set)

    def push(
        self,
        event_id: str,
        category: Literal["SCALP_SIGNAL", "TP_SL_HIT", "LIQUIDITY_SWEEP", "SECURITY"],
        title: str,
        detail: str,
        telegram_token: str = "",
        telegram_chat_id: str = "",
    ) -> bool:
        if event_id in self.seen_ids:
            return False

        self.seen_ids.add(event_id)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        self.items.insert(
            0,
            AlertItem(
                event_id=event_id,
                category=category,
                title=title,
                detail=detail,
                timestamp=ts,
            ),
        )
        self.items = self.items[:50]

        if telegram_token.strip() and telegram_chat_id.strip():
            try:
                requests.post(
                    f"https://api.telegram.org/bot{telegram_token.strip()}/sendMessage",
                    json={
                        "chat_id": telegram_chat_id.strip(),
                        "text": f"[{category}] {title}\n{detail}\n({ts})",
                    },
                    timeout=5,
                )
            except Exception:
                pass

        return True

