"""情绪状态机（带基线回弹 + 双重限幅）。

情绪是“活状态”而不是 prompt 里的一句话：它会被模糊反射推动、被时间拉回
基线、被单次上限约束，并且以明文 JSON 存在 ``cortex/state/emotion.json``，
可 diff、可回滚。基线来自 ``cortex/persona.py``。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from kernel.literal import load_literal
from kernel.models import clamp
from kernel.paths import Workspace


# ---------------------------------------------------------------------------
# 情绪状态机（带基线回弹 + 双重限幅）
# ---------------------------------------------------------------------------
@dataclass
class EmotionState:
    values: dict[str, float] = field(default_factory=dict)
    baseline: dict[str, float] = field(default_factory=dict)
    decay_rate: float = 0.08
    max_delta_per_tick: float = 0.25

    def __post_init__(self) -> None:
        if not self.baseline:
            self.baseline = dict(DEFAULT_EMOTION_BASELINE)
        for k, v in self.baseline.items():
            self.values.setdefault(k, v)
        for k, v in self.values.items():
            self.baseline.setdefault(k, 0.5)
            self.values[k] = clamp(float(v), 0.0, 1.0)

    def apply(self, deltas: Mapping[str, float], max_delta: float | None = None) -> dict[str, float]:
        """限幅后应用情绪增量；返回实际生效的变化。"""
        cap = max_delta if max_delta is not None else self.max_delta_per_tick
        applied: dict[str, float] = {}
        for key, delta in deltas.items():
            try:
                d = float(delta)
            except (TypeError, ValueError):
                continue
            # 双重限幅：单次反射的 max_delta 与每拍全局上限取更严者
            limited = clamp(d, -abs(cap), abs(cap))
            limited = clamp(limited, -self.max_delta_per_tick, self.max_delta_per_tick)
            before = self.values.get(key, self.baseline.get(key, 0.5))
            after = clamp(before + limited, 0.0, 1.0)
            self.values[key] = after
            applied[key] = round(after - before, 4)
        return applied

    def tick_decay(self) -> None:
        """向基线回弹（指数衰减），防止自激震荡与人格漂移。"""
        for key, base in self.baseline.items():
            cur = self.values.get(key, base)
            self.values[key] = clamp(cur + (base - cur) * self.decay_rate, 0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": {k: round(v, 4) for k, v in self.values.items()},
            "baseline": {k: round(v, 4) for k, v in self.baseline.items()},
            "decay_rate": self.decay_rate,
            "max_delta_per_tick": self.max_delta_per_tick,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "EmotionState":
        d = d or {}
        return cls(
            values=dict(d.get("values") or {}),
            baseline=dict(d.get("baseline") or {}),
            decay_rate=float(d.get("decay_rate", 0.08)),
            max_delta_per_tick=float(d.get("max_delta_per_tick", 0.25)),
        )

    def residual(self, threshold: float = 0.06) -> str:
        """“情绪余温”：偏离基线最明显的那几维。"""
        diffs = sorted(
            ((k, v - self.baseline.get(k, 0.5)) for k, v in self.values.items()),
            key=lambda kv: -abs(kv[1]),
        )
        hot = [f"{k}{'+' if d > 0 else ''}{d:.2f}" for k, d in diffs if abs(d) >= threshold]
        return " ".join(hot[:4]) if hot else "平静（贴近基线）"


DEFAULT_EMOTION_BASELINE = {
    "warmth": 0.6,
    "caring": 0.55,
    "playfulness": 0.45,
    "energy": 0.55,
    "focus": 0.6,
    "fatigue": 0.2,
}


class EmotionStore:
    def __init__(self, ws: Workspace):
        self.ws = ws
        self.path = ws.emotion_file

    def _persona_defaults(self) -> tuple[dict[str, float], float, float]:
        baseline = load_literal(self.ws.persona_file, "EMOTION_BASELINE", default=None)
        decay = load_literal(self.ws.persona_file, "EMOTION_DECAY_PER_TICK", default=0.08)
        cap = load_literal(self.ws.persona_file, "EMOTION_MAX_DELTA_PER_TICK", default=0.25)
        if not isinstance(baseline, dict) or not baseline:
            baseline = dict(DEFAULT_EMOTION_BASELINE)
        return (
            {k: float(v) for k, v in baseline.items()},
            float(decay) if isinstance(decay, (int, float)) else 0.08,
            float(cap) if isinstance(cap, (int, float)) else 0.25,
        )

    def load(self) -> EmotionState:
        baseline, decay, cap = self._persona_defaults()
        if self.path.is_file():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                state = EmotionState.from_dict(data)
                state.baseline = baseline
                state.decay_rate = decay
                state.max_delta_per_tick = cap
                for k in baseline:
                    state.values.setdefault(k, baseline[k])
                return state
            except (json.JSONDecodeError, OSError, ValueError):
                pass
        return EmotionState(baseline=baseline, decay_rate=decay, max_delta_per_tick=cap)

    def save(self, state: EmotionState) -> None:
        path = self.ws.assert_writable(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(state.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
