"""自主唤醒调度器（铁律 6：自由，但不失控）。

两种唤醒：

* ``continue_think``：**即时连环思考**——由 boot.py 在同一个事件内立即推进，
  受 ``MAX_CHAIN_TICKS`` 硬熔断（不经过本模块）；
* ``schedule_wake``：**异步定时唤醒**——“半小时后如果主人还没睡，主动问候一句”。
  本模块把预约写成明文队列，并用「每小时配额」硬性限流。

时间源可注入（``now_fn``），因此测试不需要真的 sleep。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable

from kernel.config import Config
from kernel.models import iso, now_utc, parse_iso
from kernel.paths import Workspace


class ProactiveBudgetExceeded(RuntimeError):
    """每小时自主唤醒配额已用尽。"""


@dataclass
class WakeTicket:
    id: str
    due_at: str
    created_at: str
    reason: str = ""
    target_speaker: str = "owner"
    fired: bool = False
    fired_at: str | None = None

    def due_datetime(self) -> datetime:
        return parse_iso(self.due_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "due_at": self.due_at,
            "created_at": self.created_at,
            "reason": self.reason,
            "target_speaker": self.target_speaker,
            "fired": self.fired,
            "fired_at": self.fired_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WakeTicket":
        return cls(
            id=str(d["id"]),
            due_at=str(d["due_at"]),
            created_at=str(d.get("created_at") or iso()),
            reason=str(d.get("reason") or ""),
            target_speaker=str(d.get("target_speaker") or "owner"),
            fired=bool(d.get("fired", False)),
            fired_at=d.get("fired_at"),
        )


@dataclass
class Scheduler:
    ws: Workspace
    config: Config | None = None
    now_fn: Callable[[], datetime] = now_utc
    _queue: list[WakeTicket] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.config = self.config or self.ws.config
        if not self._queue:
            self._queue = self._load()

    # ---- 持久化 -------------------------------------------------------
    def _load(self) -> list[WakeTicket]:
        path = self.ws.wake_queue_file
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return [WakeTicket.from_dict(t) for t in data.get("queue", [])]
            except (json.JSONDecodeError, OSError, KeyError):
                return []
        return []

    def _save(self) -> None:
        path = self.ws.assert_writable(self.ws.wake_queue_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"queue": [t.to_dict() for t in self._queue]}
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    # ---- 配额 ---------------------------------------------------------
    def fired_last_hour(self, now: datetime | None = None) -> int:
        now = now or self.now_fn()
        cutoff = now - timedelta(hours=1)
        count = 0
        for ticket in self._queue:
            if not ticket.fired or not ticket.fired_at:
                continue
            try:
                if parse_iso(ticket.fired_at) >= cutoff:
                    count += 1
            except ValueError:
                continue
        return count

    def quota_left(self, now: datetime | None = None) -> int:
        return max(
            0,
            int(self.config.budget.max_proactive_per_hour) - self.fired_last_hour(now),
        )

    def pending_count(self) -> int:
        return len([t for t in self._queue if not t.fired])

    def can_wake(self, now: datetime | None = None, check_queue: bool = True) -> tuple[bool, str]:
        """两道闸门：本小时已触发次数 + 待触发队列长度。

        只限“已发生”的次数容易攒出一长串待办；只限队列长度又能靠高频触发
        绕过。两把锁一起上，才算真的“自由但不失控”。
        """
        if self.quota_left(now) <= 0:
            return (
                False,
                f"本小时自主唤醒配额已用尽（{self.config.budget.max_proactive_per_hour}/小时）",
            )
        if check_queue and self.pending_count() >= int(self.config.budget.max_proactive_per_hour):
            return (
                False,
                f"待触发的自主唤醒队列已满（{self.config.budget.max_proactive_per_hour} 条），先让之前的兑现",
            )
        return True, ""

    # ---- 预约 ---------------------------------------------------------
    def schedule(
        self,
        after_sec: int | float,
        reason: str = "",
        target_speaker: str | None = None,
        now: datetime | None = None,
    ) -> WakeTicket:
        now = now or self.now_fn()
        ok, why = self.can_wake(now)
        if not ok:
            raise ProactiveBudgetExceeded(why)
        try:
            after = max(1.0, float(after_sec))
        except (TypeError, ValueError):
            after = 60.0
        ticket = WakeTicket(
            id=uuid.uuid4().hex[:8],
            due_at=iso(now + timedelta(seconds=after)),
            created_at=iso(now),
            reason=str(reason or ""),
            target_speaker=str(target_speaker or "owner"),
        )
        self._queue.append(ticket)
        self._save()
        return ticket

    # ---- 查询 ---------------------------------------------------------
    def pending(self, include_fired: bool = False) -> list[WakeTicket]:
        items = [t for t in self._queue if include_fired or not t.fired]
        return sorted(items, key=lambda t: t.due_at)

    def due(self, now: datetime | None = None) -> list[WakeTicket]:
        now = now or self.now_fn()
        return [t for t in self.pending() if t.due_datetime() <= now]

    def next_due_at(self) -> datetime | None:
        pending = self.pending()
        return pending[0].due_datetime() if pending else None

    def seconds_until_next(self, now: datetime | None = None) -> float | None:
        nxt = self.next_due_at()
        if nxt is None:
            return None
        now = now or self.now_fn()
        return max(0.0, (nxt - now).total_seconds())

    # ---- 触发 ---------------------------------------------------------
    def mark_fired(self, ticket: WakeTicket, now: datetime | None = None) -> WakeTicket:
        now = now or self.now_fn()
        ok, why = self.can_wake(now, check_queue=False)
        if not ok:
            raise ProactiveBudgetExceeded(why)
        for item in self._queue:
            if item.id == ticket.id:
                item.fired = True
                item.fired_at = iso(now)
                self._save()
                return item
        raise KeyError(f"没有这个唤醒票据: {ticket.id}")

    def cancel(self, ticket_id: str) -> bool:
        before = len(self._queue)
        self._queue = [t for t in self._queue if not (t.id == ticket_id and not t.fired)]
        if len(self._queue) != before:
            self._save()
            return True
        return False

    def prune(self, keep_fired: int = 50) -> int:
        """控制队列体积：已触发的只保留最近 N 条。"""
        fired = [t for t in self._queue if t.fired]
        alive = [t for t in self._queue if not t.fired]
        fired.sort(key=lambda t: t.fired_at or "", reverse=True)
        new = alive + fired[:keep_fired]
        removed = len(self._queue) - len(new)
        if removed:
            self._queue = new
            self._save()
        return removed
