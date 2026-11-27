"""核心数据契约（纯 dataclass + 明文序列化）。

这些结构是 kernel 与 memory 之间的“宪法”：任何落盘格式都必须能无损
往返 JSON，且人类与大模型都能直接读懂（铁律 2：禁止 pickle）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

LOD2 = 2
LOD1 = 1
LOD0 = 0
LOD_COLD = -1

LOD_NAMES = {LOD2: "LOD2", LOD1: "LOD1", LOD0: "LOD0", LOD_COLD: "COLD"}

EPISTEMIC_LEVELS = ("direct", "hearsay", "inferred")
NODE_KINDS = (
    "episode",
    "belief",
    "reflex",
    "trait",
    "relation",
    "policy",
    "skill",
    "lesson",
    "evolution",
)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime | None = None) -> str:
    return (dt or now_utc()).isoformat(timespec="seconds")


def parse_iso(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def dumps(obj: Any) -> str:
    """稳定、可读的 JSON 序列化（写入磁盘的唯一格式）。"""
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# ---------------------------------------------------------------------------
# 对话轮次（焦点式 LOD 的原料）
# ---------------------------------------------------------------------------
@dataclass
class TurnRecord:
    turn_id: int
    speaker_id: str
    user_text: str
    assistant_text: str = ""
    inner_monologue: str = ""
    lod0: str = ""
    lod1: str = ""
    lod2: str = ""
    tags: list[str] = field(default_factory=list)
    visibility: list[str] = field(default_factory=list)
    epistemic: str = "direct"
    created_at: str = field(default_factory=iso)
    archived: bool = False
    refs: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.lod2:
            self.lod2 = self.render_full()
        if not self.lod1:
            self.lod1 = self.lod2[:160]
        if not self.lod0:
            self.lod0 = self.lod1[:48]
        if not self.visibility:
            self.visibility = [self.speaker_id]

    def render_full(self) -> str:
        parts = [f"{self.speaker_id}: {self.user_text}"]
        if self.assistant_text:
            parts.append(f"星璃: {self.assistant_text}")
        if self.inner_monologue:
            parts.append(f"（内心 OS: {self.inner_monologue}）")
        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "speaker_id": self.speaker_id,
            "user_text": self.user_text,
            "assistant_text": self.assistant_text,
            "inner_monologue": self.inner_monologue,
            "lod0": self.lod0,
            "lod1": self.lod1,
            "lod2": self.lod2,
            "tags": list(self.tags),
            "visibility": list(self.visibility),
            "epistemic": self.epistemic,
            "created_at": self.created_at,
            "archived": self.archived,
            "refs": list(self.refs),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TurnRecord":
        return cls(
            turn_id=int(d["turn_id"]),
            speaker_id=str(d.get("speaker_id", "owner")),
            user_text=str(d.get("user_text", "")),
            assistant_text=str(d.get("assistant_text", "")),
            inner_monologue=str(d.get("inner_monologue", "")),
            lod0=str(d.get("lod0", "")),
            lod1=str(d.get("lod1", "")),
            lod2=str(d.get("lod2", "")),
            tags=list(d.get("tags") or []),
            visibility=list(d.get("visibility") or []),
            epistemic=str(d.get("epistemic", "direct")),
            created_at=str(d.get("created_at") or iso()),
            archived=bool(d.get("archived", False)),
            refs=list(d.get("refs") or []),
        )


# ---------------------------------------------------------------------------
# 长期记忆节点（内容寻址、带生命周期）
# ---------------------------------------------------------------------------
@dataclass
class MemoryNode:
    id: str
    kind: str = "episode"
    lod0: str = ""
    lod1: str = ""
    lod2: str = ""
    source_id: str = "owner"
    about_ids: list[str] = field(default_factory=list)
    epistemic: str = "direct"
    visibility: list[str] = field(default_factory=list)
    importance: float = 0.6
    pinned: bool = False
    tags: list[str] = field(default_factory=list)
    emotion: dict[str, Any] = field(default_factory=dict)
    refs: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=iso)
    updated_at: str = field(default_factory=iso)
    last_accessed: str = field(default_factory=iso)
    access_count: int = 0
    archived: bool = False
    schema_version: int = 1
    # 派生信息（不参与内容哈希）
    score: float = 0.0
    lod: int = LOD0

    def __post_init__(self) -> None:
        if not self.lod2:
            self.lod2 = self.lod1 or self.lod0
        if not self.lod1:
            self.lod1 = self.lod2[:200]
        if not self.lod0:
            self.lod0 = self.lod1[:60]
        if self.epistemic not in EPISTEMIC_LEVELS:
            self.epistemic = "inferred"
        if not self.about_ids:
            self.about_ids = [self.source_id]
        if not self.visibility:
            self.visibility = list(dict.fromkeys([self.source_id, *self.about_ids]))
        self.importance = clamp(float(self.importance), 0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "lod0": self.lod0,
            "lod1": self.lod1,
            "lod2": self.lod2,
            "source_id": self.source_id,
            "about_ids": list(self.about_ids),
            "epistemic": self.epistemic,
            "visibility": list(self.visibility),
            "importance": round(float(self.importance), 6),
            "pinned": bool(self.pinned),
            "tags": list(self.tags),
            "emotion": dict(self.emotion),
            "refs": list(self.refs),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_accessed": self.last_accessed,
            "access_count": int(self.access_count),
            "archived": bool(self.archived),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "MemoryNode":
        return cls(
            id=str(d["id"]),
            kind=str(d.get("kind", "episode")),
            lod0=str(d.get("lod0", "")),
            lod1=str(d.get("lod1", "")),
            lod2=str(d.get("lod2", "")),
            source_id=str(d.get("source_id", "owner")),
            about_ids=list(d.get("about_ids") or []),
            epistemic=str(d.get("epistemic", "direct")),
            visibility=list(d.get("visibility") or []),
            importance=float(d.get("importance", 0.6)),
            pinned=bool(d.get("pinned", False)),
            tags=list(d.get("tags") or []),
            emotion=dict(d.get("emotion") or {}),
            refs=list(d.get("refs") or []),
            created_at=str(d.get("created_at") or iso()),
            updated_at=str(d.get("updated_at") or iso()),
            last_accessed=str(d.get("last_accessed") or iso()),
            access_count=int(d.get("access_count", 0)),
            archived=bool(d.get("archived", False)),
            schema_version=int(d.get("schema_version", 1)),
        )


# ---------------------------------------------------------------------------
# 焦点状态（可由星璃自主移动的注意力聚光灯）
# ---------------------------------------------------------------------------
@dataclass
class FocusState:
    focus_turn_id: int | None = None
    decay_before: float = 0.35
    decay_after: float = 0.20
    recency_floor: int = 2
    pinned_turn_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "focus_turn_id": self.focus_turn_id,
            "decay_before": self.decay_before,
            "decay_after": self.decay_after,
            "recency_floor": self.recency_floor,
            "pinned_turn_ids": list(self.pinned_turn_ids),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "FocusState":
        d = d or {}
        return cls(
            focus_turn_id=(
                None if d.get("focus_turn_id") is None else int(d["focus_turn_id"])
            ),
            decay_before=clamp(float(d.get("decay_before", 0.35)), 0.01, 3.0),
            decay_after=clamp(float(d.get("decay_after", 0.20)), 0.01, 3.0),
            recency_floor=max(0, int(d.get("recency_floor", 2))),
            pinned_turn_ids=[int(x) for x in (d.get("pinned_turn_ids") or [])],
        )


# ---------------------------------------------------------------------------
# 结构化认知包（System Two 的输出契约，SPEC 4.4）
# ---------------------------------------------------------------------------
@dataclass
class FocusUpdate:
    turn_id: int | None = None
    decay_before: float | None = None
    decay_after: float | None = None
    pin: list[int] = field(default_factory=list)
    unpin: list[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "FocusUpdate | None":
        if not d:
            return None
        return cls(
            turn_id=None if d.get("turn_id") is None else int(d["turn_id"]),
            decay_before=(
                None if d.get("decay_before") is None else float(d["decay_before"])
            ),
            decay_after=(
                None if d.get("decay_after") is None else float(d["decay_after"])
            ),
            pin=[int(x) for x in (d.get("pin") or [])],
            unpin=[int(x) for x in (d.get("unpin") or [])],
        )


@dataclass
class WakeSpec:
    continue_think: dict[str, Any] | None = None
    schedule_wake: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "WakeSpec":
        d = d or {}
        cont = d.get("continue_think")
        sched = d.get("schedule_wake")
        return cls(
            continue_think=dict(cont) if isinstance(cont, Mapping) else None,
            schedule_wake=dict(sched) if isinstance(sched, Mapping) else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "continue_think": self.continue_think,
            "schedule_wake": self.schedule_wake,
        }


@dataclass
class ThoughtPacket:
    """单拍认知包。字段顺序即 Schema 顺序：inner_monologue 必须第一。"""

    inner_monologue: str = ""
    focus_update: FocusUpdate | None = None
    emotion_delta: dict[str, float] = field(default_factory=dict)
    memory_ops: list[dict[str, Any]] = field(default_factory=list)
    relation_ops: list[dict[str, Any]] = field(default_factory=list)
    say: Any = None  # str | {"to":..., "text":...} | None
    wake: WakeSpec = field(default_factory=WakeSpec)
    code_action: str | None = None
    evolve_ops: list[dict[str, Any]] = field(default_factory=list)
    # 解析诊断（不落盘为“她的记忆”，仅供内核日志）
    parse_warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # 允许用普通 dict 构造（方便代码/测试），内部统一成强类型
        if isinstance(self.wake, Mapping):
            self.wake = WakeSpec.from_dict(self.wake)
        if isinstance(self.focus_update, Mapping):
            self.focus_update = FocusUpdate.from_dict(self.focus_update)
        if self.say is not None and not isinstance(self.say, (str, Mapping)):
            self.say = None

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ThoughtPacket":
        warnings: list[str] = []
        monologue = d.get("inner_monologue")
        if not isinstance(monologue, str):
            warnings.append("inner_monologue 缺失或非字符串，已置空")
            monologue = str(monologue or "")

        delta_raw = d.get("emotion_delta") or {}
        delta: dict[str, float] = {}
        if isinstance(delta_raw, Mapping):
            for k, v in delta_raw.items():
                try:
                    delta[str(k)] = float(v)
                except (TypeError, ValueError):
                    warnings.append(f"emotion_delta.{k} 不是数值，已忽略")
        else:
            warnings.append("emotion_delta 不是对象，已忽略")

        ops = d.get("memory_ops") or []
        if not isinstance(ops, list):
            warnings.append("memory_ops 不是数组，已忽略")
            ops = []
        memory_ops = [dict(op) for op in ops if isinstance(op, Mapping)]

        rel_ops = d.get("relation_ops") or []
        if not isinstance(rel_ops, list):
            warnings.append("relation_ops 不是数组，已忽略")
            rel_ops = []
        relation_ops = [dict(op) for op in rel_ops if isinstance(op, Mapping)]

        evo = d.get("evolve_ops") or []
        if not isinstance(evo, list):
            warnings.append("evolve_ops 不是数组，已忽略")
            evo = []
        evolve_ops = [dict(op) for op in evo if isinstance(op, Mapping)]

        say = d.get("say")
        if isinstance(say, Mapping):
            say = {"to": say.get("to"), "text": str(say.get("text", ""))}
        elif say is not None and not isinstance(say, str):
            warnings.append("say 类型非法，已忽略")
            say = None

        code_action = d.get("code_action")
        if code_action is not None and not isinstance(code_action, str):
            warnings.append("code_action 非字符串，已忽略")
            code_action = None

        return cls(
            inner_monologue=monologue.strip(),
            focus_update=FocusUpdate.from_dict(d.get("focus_update")),
            emotion_delta=delta,
            memory_ops=memory_ops,
            relation_ops=relation_ops,
            say=say,
            wake=WakeSpec.from_dict(d.get("wake")),
            code_action=code_action,
            evolve_ops=evolve_ops,
            parse_warnings=warnings,
        )

    def say_text(self) -> str | None:
        if self.say is None:
            return None
        if isinstance(self.say, str):
            return self.say.strip() or None
        text = str(self.say.get("text") or "").strip()
        return text or None

    def say_target(self) -> str | None:
        if isinstance(self.say, Mapping):
            target = self.say.get("to")
            return None if target in (None, "", "current") else str(target)
        return None

    def to_dict(self) -> dict[str, Any]:
        """按 SPEC 的字段顺序输出（inner_monologue 永远第一）。"""
        return {
            "inner_monologue": self.inner_monologue,
            "focus_update": (
                {
                    "turn_id": self.focus_update.turn_id,
                    "decay_before": self.focus_update.decay_before,
                    "decay_after": self.focus_update.decay_after,
                    "pin": self.focus_update.pin,
                    "unpin": self.focus_update.unpin,
                }
                if self.focus_update
                else None
            ),
            "emotion_delta": dict(self.emotion_delta),
            "memory_ops": list(self.memory_ops),
            "relation_ops": list(self.relation_ops),
            "say": self.say,
            "wake": self.wake.to_dict(),
            "code_action": self.code_action,
            "evolve_ops": list(self.evolve_ops),
        }


# ---------------------------------------------------------------------------
# 心跳输入 / 输出
# ---------------------------------------------------------------------------
@dataclass
class Event:
    """触发一拍思考的事件。"""

    kind: str = "message"  # message | wake | recall_result | internal
    speaker_id: str = "owner"
    text: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=iso)

    def describe(self) -> str:
        return f"{self.kind}:{self.speaker_id}"


@dataclass
class ReflexSignal:
    """一条模糊反射的评估结果（已过三段门控）。"""

    reflex_id: str
    gate: str  # high | mid | low
    probability: float
    confidence: float
    question: str = ""
    hint: str = ""
    actions: dict[str, Any] = field(default_factory=dict)


@dataclass
class AppliedActions:
    says: list[str] = field(default_factory=list)
    recalls: list[dict[str, Any]] = field(default_factory=list)
    memorized: list[str] = field(default_factory=list)
    teachings: list[str] = field(default_factory=list)
    evolutions: list[dict[str, Any]] = field(default_factory=list)
    relation_updates: list[str] = field(default_factory=list)
    focus_changed: bool = False
    emotion: dict[str, float] = field(default_factory=dict)
    scheduled: list[dict[str, Any]] = field(default_factory=list)
    code_output: str = ""
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "says": list(self.says),
            "recalls": list(self.recalls),
            "memorized": list(self.memorized),
            "teachings": list(self.teachings),
            "evolutions": list(self.evolutions),
            "relation_updates": list(self.relation_updates),
            "focus_changed": self.focus_changed,
            "emotion": dict(self.emotion),
            "scheduled": list(self.scheduled),
            "code_output": self.code_output,
            "errors": list(self.errors),
        }


@dataclass
class TickReport:
    tick: int
    event: Event
    frame_text: str
    frame_tokens: int
    section_tokens: dict[str, int] = field(default_factory=dict)
    packet: ThoughtPacket | None = None
    signals: list[ReflexSignal] = field(default_factory=list)
    actions: AppliedActions = field(default_factory=AppliedActions)
    duration_ms: float = 0.0
    env: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"tick#{self.tick} event={self.event.describe()} "
            f"tokens={self.frame_tokens} says={len(self.actions.says)} "
            f"recalls={len(self.actions.recalls)}"
        )


@dataclass
class HeartbeatResult:
    """一次外部事件引发的完整（可能多拍的）思考结果。"""

    event: Event
    ticks: list[TickReport] = field(default_factory=list)
    halted_reason: str = ""

    @property
    def says(self) -> list[str]:
        out: list[str] = []
        for t in self.ticks:
            out.extend(t.actions.says)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": {
                "kind": self.event.kind,
                "speaker_id": self.event.speaker_id,
                "text": self.event.text,
            },
            "ticks": [t.summary() for t in self.ticks],
            "says": self.says,
            "halted_reason": self.halted_reason,
        }


def coerce_iter(value: Any) -> Iterable[Any]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        return value
    return (value,)
