"""时间感（消灭“感官剥夺”的第一块传感器）。

不是把 ``2026-11-27 10:39`` 这个字符串塞进 prompt，而是由代码算出现场：
现在是什么时段、距上次和这个人说话过去多久、今天是周几、忙不忙。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable

from kernel.models import iso, now_utc, parse_iso
from kernel.paths import Workspace
from memory.store import MemoryStore


def period_of(hour: int) -> str:
    if 0 <= hour < 5:
        return "深夜"
    if 5 <= hour < 8:
        return "清晨"
    if 8 <= hour < 11:
        return "上午"
    if 11 <= hour < 13:
        return "中午"
    if 13 <= hour < 17:
        return "下午"
    if 17 <= hour < 19:
        return "傍晚"
    if 19 <= hour < 23:
        return "夜晚"
    return "深夜"


def humanize_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{int(seconds)} 秒"
    minutes = seconds / 60
    if minutes < 60:
        return f"{int(minutes)} 分钟"
    hours = minutes / 60
    if hours < 24:
        return f"{int(hours)} 小时 {int(minutes % 60)} 分"
    days = hours / 24
    if days < 30:
        return f"{int(days)} 天 {int(hours % 24)} 小时"
    return f"{int(days // 30)} 个月"


@dataclass
class ClockSensor:
    ws: Workspace
    store: MemoryStore | None = None
    now_fn: Callable[[], datetime] = now_utc

    def capture(
        self, speaker_id: str | None = None, now: datetime | None = None
    ) -> dict[str, Any]:
        now = now or self.now_fn()
        local = now.astimezone()
        hour = local.hour
        last_seen: datetime | None = None
        if self.store is not None and speaker_id:
            turns = [
                t
                for t in self.store.list_turns(speaker_id)
                if t.speaker_id == speaker_id
            ]
            if turns:
                last_seen = parse_iso(max(turns, key=lambda t: t.created_at).created_at)
        gap = (now - last_seen) if last_seen else None
        weekday = "周" + "一二三四五六日"[local.weekday()]
        return {
            "now_iso": iso(now),
            "local_time": local.strftime("%Y-%m-%d %H:%M"),
            "weekday": weekday,
            "period": period_of(hour),
            "hour": hour,
            "is_late_night": hour >= 23 or hour < 5,
            "is_weekend": local.weekday() >= 5,
            "last_seen_iso": iso(last_seen) if last_seen else None,
            "since_last_seconds": gap.total_seconds() if gap else None,
            "since_last_human": humanize_duration(gap.total_seconds()) if gap else "（第一次见面）",
            "days_since": round(gap.total_seconds() / 86400, 2) if gap else None,
            "greeting_hint": self._greeting(hour, gap),
        }

    @staticmethod
    def _greeting(hour: int, gap: datetime | None) -> str:
        if gap is not None and gap > timedelta(days=3):
            return "很久没见了，可以先自然地问一句近况，不要夸张"
        if 0 <= hour < 5:
            return "这个点还醒着，多半在硬撑或赶工"
        if 5 <= hour < 9:
            return "清晨，语气可以轻一点"
        if 22 <= hour < 24:
            return "夜里，注意对方是不是该休息了"
        return "日常时段"

    def render(self, snapshot: dict[str, Any]) -> str:
        bits = [
            f"现在: {snapshot['local_time']} {snapshot['weekday']} {snapshot['period']}",
            f"距上次和当前说话人交流: {snapshot['since_last_human']}",
        ]
        if snapshot.get("is_late_night"):
            bits.append("（深夜）")
        return " ｜ ".join(bits)


def ddl_line(ddls: list[dict[str, Any]], now: datetime | None = None) -> str:
    """把 DDL 列表渲染为“还剩多久”，而不是一串日期。"""
    now = now or now_utc()
    out: list[str] = []
    for item in ddls or []:
        label = str(item.get("label", "")).strip()
        due_raw = item.get("due")
        if not label or not due_raw:
            continue
        try:
            due = parse_iso(str(due_raw))
        except ValueError:
            continue
        delta = (due - now).total_seconds() / 86400
        if delta < 0:
            out.append(f"{label} 已逾期 {abs(delta):.1f} 天")
        else:
            out.append(f"{label} 还剩 {delta:.1f} 天")
    return " · ".join(out)
