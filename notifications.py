from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal
import logging

import requests

log = logging.getLogger("notifications")


@dataclass
class AlertItem:
    event_id: str
    category: Literal["SCALP_SIGNAL", "TP_SL_HIT", "LIQUIDITY_SWEEP", "SECURITY"]
    title: str
    detail: str
    timestamp: str


# FIX: cap seen_ids so the set does not grow unbounded over weeks of
# runtime. Deque-based FIFO eviction keeps memory flat.
_MAX_SEEN_IDS = 5000


@dataclass
class NotificationCenter:
    items: list[AlertItem] = field(default_factory=list)
    seen_ids: set[str] = field(default_factory=set)

    # FIX: internal FIFO queue of recently seen IDs. When it exceeds
    # _MAX_SEEN_IDS, the oldest IDs are removed from `seen_ids`.
    _seen_order: deque[str] = field(
        default_factory=deque,
        repr=False,
    )

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

        # FIX: enforce bounded memory for seen_ids.
        self.seen_ids.add(event_id)
        self._seen_order.append(event_id)

        while len(self._seen_order) > _MAX_SEEN_IDS:
            old = self._seen_order.popleft()
            self.seen_ids.discard(old)

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
            # FIX: shorter timeout (2s) so a slow Telegram API cannot
            # freeze the Streamlit UI for the full 5s default.
            # Also log failures instead of silently swallowing them,
            # so Telegram problems are visible in the console.
            url = (
                f"https://api.telegram.org/bot"
                f"{telegram_token.strip()}/sendMessage"
            )
            payload = {
                "chat_id": telegram_chat_id.strip(),
                "text": f"[{category}] {title}\n{detail}\n({ts})",
            }
            try:
                resp = requests.post(
                    url,
                    json=payload,
                    timeout=2,
                )
                if resp.status_code >= 400:
                    log.warning(
                        "Telegram send failed: HTTP %s: %s",
                        resp.status_code,
                        resp.text[:200],
                    )
            except requests.RequestException as exc:
                log.warning(
                    "Telegram request failed: %s",
                    exc,
                )

        return True
