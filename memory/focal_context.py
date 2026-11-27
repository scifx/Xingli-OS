"""焦点式双向 LOD 衰减器（SPEC 4.2 的核心实现）。

近期对话不再按扁平数组硬塞，而是围绕一个**可由星璃自主移动的焦点**
``focus_turn_id`` 计算清晰度：

    d = i - f
    R(i) = exp(-decay_before·|d|)   (d < 0，焦点之前/更早)
    R(i) = 1.0                      (d = 0)
    R(i) = exp(-decay_after·|d|)    (d > 0，焦点之后/更新)

再叠加「最近 recency_floor 轮强制 LOD2」「pinned 钉选」两条保底规则，
最后按阈值 θ2/θ1/θ0 映射到 LOD2/LOD1/LOD0/冷区。

预算永远优先：当渲染超过 Token 预算时，从“分数最低的那一层”开始降级
（LOD2→LOD1→LOD0→冷区），因此帧长严格受控，与对话总轮数无关。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from kernel.config import Config
from kernel.literal import load_literal
from kernel.models import (
    LOD0,
    LOD1,
    LOD2,
    LOD_COLD,
    LOD_NAMES,
    FocusState,
    FocusUpdate,
    TurnRecord,
)
from kernel.paths import Workspace
from kernel.models import clamp
from kernel.textutil import estimate_tokens, truncate
from memory.store import MemoryStore


def clarity(index: int, focus: int, decay_before: float, decay_after: float) -> float:
    """清晰度 R(i) ∈ [0,1]，严格按 SPEC 4.2 的公式。"""
    d = index - focus
    if d == 0:
        return 1.0
    if d < 0:
        return math.exp(-max(0.0, decay_before) * abs(d))
    return math.exp(-max(0.0, decay_after) * abs(d))


@dataclass
class FocalEntry:
    turn_id: int
    level: int
    score: float
    speaker_id: str
    pinned: bool = False
    floor: bool = False
    text: str = ""
    lod0: str = ""
    lod1: str = ""
    lod2: str = ""

    @property
    def level_name(self) -> str:
        return LOD_NAMES[self.level]


@dataclass
class FocalRender:
    text: str
    entries: list[FocalEntry] = field(default_factory=list)
    focus_turn_id: int | None = None
    visible_turns: int = 0
    cold_turns: list[int] = field(default_factory=list)
    tokens: int = 0
    state: FocusState = field(default_factory=FocusState)

    def level_counts(self) -> dict[str, int]:
        counts = {"LOD2": 0, "LOD1": 0, "LOD0": 0, "COLD": len(self.cold_turns)}
        for e in self.entries:
            counts[e.level_name] = counts.get(e.level_name, 0) + 1
        return counts


class FocalContext:
    """对话上下文的视距控制器（读写焦点状态 + 渲染）。"""

    def __init__(self, ws: Workspace, store: MemoryStore, config: Config | None = None):
        self.ws = ws
        self.store = store
        self.config = config or ws.config

    # ------------------------------------------------------------------
    # 焦点状态
    # ------------------------------------------------------------------
    def defaults(self) -> FocusState:
        data = load_literal(self.ws.focus_policy_file, "FOCUS_DEFAULTS", default={}) or {}
        if not isinstance(data, dict):
            data = {}
        return FocusState.from_dict(data)

    def load_state(self) -> FocusState:
        path = self.ws.focus_state_file
        if path.is_file():
            try:
                import json

                return FocusState.from_dict(
                    json.loads(path.read_text(encoding="utf-8"))
                )
            except (ValueError, OSError):
                pass
        return self.defaults()

    def save_state(self, state: FocusState) -> None:
        import json

        path = self.ws.assert_writable(self.ws.focus_state_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(state.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def apply_update(
        self,
        update: FocusUpdate,
        visible_ids: Sequence[int],
        state: FocusState | None = None,
    ) -> tuple[FocusState, list[str]]:
        """应用星璃自己的 ``set_focus(...)``；越权/越界的部分被安全裁剪。"""
        state = state or self.load_state()
        warnings: list[str] = []
        visible = sorted(visible_ids)
        if update.turn_id is not None:
            if not visible:
                warnings.append("当前没有可见轮次，焦点未改变")
            elif update.turn_id in visible:
                state.focus_turn_id = int(update.turn_id)
            else:
                nearest = min(visible, key=lambda v: abs(v - int(update.turn_id)))
                warnings.append(
                    f"焦点轮次 {update.turn_id} 不可见或不存在，已锚定到最近的 #{nearest}"
                )
                state.focus_turn_id = nearest
        if update.decay_before is not None:
            state.decay_before = clamp(float(update.decay_before), 0.01, 3.0)
        if update.decay_after is not None:
            state.decay_after = clamp(float(update.decay_after), 0.01, 3.0)
        for tid in update.pin:
            if tid in visible and tid not in state.pinned_turn_ids:
                state.pinned_turn_ids.append(int(tid))
            elif tid not in visible:
                warnings.append(f"钉选失败：轮次 {tid} 不可见")
        for tid in update.unpin:
            if tid in state.pinned_turn_ids:
                state.pinned_turn_ids.remove(int(tid))
        state.pinned_turn_ids = sorted(set(state.pinned_turn_ids))
        self.save_state(state)
        return state, warnings

    # ------------------------------------------------------------------
    # 计算
    # ------------------------------------------------------------------
    def effective_focus(self, state: FocusState, visible_ids: Sequence[int]) -> int | None:
        """把焦点裁剪到“当前说话人可见”的轮次上。

        隐私墙的一个细节：焦点可能指向别人的私密轮次，此时必须回退到
        最近一个可见轮次，绝不能让不可见轮次通过焦点泄漏存在性。
        """
        visible = sorted(visible_ids)
        if not visible:
            return None
        if state.focus_turn_id is None:
            return visible[-1]
        if state.focus_turn_id in visible:
            return state.focus_turn_id
        earlier = [v for v in visible if v <= state.focus_turn_id]
        return earlier[-1] if earlier else visible[0]

    def distribution(
        self,
        turns: Sequence[TurnRecord],
        state: FocusState | None = None,
    ) -> tuple[list[FocalEntry], int | None]:
        state = state or self.load_state()
        ids = [t.turn_id for t in turns]
        focus = self.effective_focus(state, ids)
        if focus is None:
            return [], None
        latest = max(ids)
        cfg = self.config.render
        entries: list[FocalEntry] = []
        for turn in turns:
            r = clarity(turn.turn_id, focus, state.decay_before, state.decay_after)
            pinned = turn.turn_id in state.pinned_turn_ids
            recent = state.recency_floor > 0 and turn.turn_id > latest - state.recency_floor
            if pinned or recent or r >= cfg.theta2:
                level = LOD2
            elif r >= cfg.theta1:
                level = LOD1
            elif r >= cfg.theta0:
                level = LOD0
            else:
                level = LOD_COLD
            entries.append(
                FocalEntry(
                    turn_id=turn.turn_id,
                    level=level,
                    score=r,
                    speaker_id=turn.speaker_id,
                    pinned=pinned,
                    floor=recent,
                    lod0=turn.lod0,
                    lod1=turn.lod1,
                    lod2=turn.lod2,
                )
            )
        return entries, focus

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------
    def render(
        self,
        speaker_id: str,
        budget_tokens: int | None = None,
        state: FocusState | None = None,
    ) -> FocalRender:
        state = state or self.load_state()
        turns = self.store.list_turns(speaker_id)
        if not turns:
            return FocalRender(text="（这是第一次对话，还没有历史轮次）", state=state)
        entries, focus = self.distribution(turns, state)
        budget = budget_tokens or self.config.budget.section_focal
        cfg = self.config.render

        # 预算紧张时的降级顺序：
        #   ① 牺牲清晰度最低的非保底轮次（LOD2→LOD1→LOD0→冷区）
        #   ② 仍然超支 → 缩短全文截断长度
        #   ③ 实在装不下 → 才动“最近 K 轮保底”（钉选永远不动）
        shrink = cfg.turn_lod2_chars
        for _ in range(len(entries) * 4 + 12):
            text, cold = self._compose(entries, focus, state, shrink)
            tokens = estimate_tokens(text)
            if tokens <= budget:
                break
            if self._demote_one(entries):
                continue
            if shrink > 30:
                shrink = max(30, shrink // 2)
                continue
            if self._demote_one(entries, allow_floor=True):
                continue
            break

        text, cold = self._compose(entries, focus, state, shrink)
        tokens = estimate_tokens(text)
        kept = [e for e in entries if e.level != LOD_COLD]
        return FocalRender(
            text=text,
            entries=sorted(kept, key=lambda e: e.turn_id),
            focus_turn_id=focus,
            visible_turns=len(turns),
            cold_turns=sorted(e.turn_id for e in entries if e.level == LOD_COLD),
            tokens=tokens,
            state=state,
        )

    @staticmethod
    def _demote_one(entries: list[FocalEntry], allow_floor: bool = False) -> bool:
        """把“最该被牺牲”的一条降一级；返回是否发生了降级。

        钉选（pinned）是星璃显式给出的承诺，永不因预算被降级；“最近 K 轮保底”
        是即时连贯性的底线，只有别的都牺牲光了（且已缩短截断长度）才动。
        """
        for level, target in ((LOD2, LOD1), (LOD1, LOD0), (LOD0, LOD_COLD)):
            pool = [
                e
                for e in entries
                if e.level == level and not e.pinned and (allow_floor or not e.floor)
            ]
            if not pool:
                continue
            pool.sort(key=lambda e: e.score)
            pool[0].level = target
            return True
        return False

    def _compose(
        self,
        entries: list[FocalEntry],
        focus: int | None,
        state: FocusState,
        lod2_chars: int,
    ) -> tuple[str, list[int]]:
        visible = len(entries)
        lines = [
            f"焦点=#{focus} ｜ 前衰减={state.decay_before:.2f} 后衰减={state.decay_after:.2f}"
            f" ｜ 可见轮次={visible}"
        ]
        if state.pinned_turn_ids:
            lines[0] += f" ｜ 钉选={state.pinned_turn_ids}"
        cold: list[int] = []
        for e in sorted(entries, key=lambda x: x.turn_id):
            if e.level == LOD_COLD:
                cold.append(e.turn_id)
                continue
            if e.level == LOD2:
                body = truncate(e.lod2, lod2_chars)
                pin = " ★" if e.pinned else ""
                lines.append(f"#{e.turn_id} 〔LOD2 R={e.score:.2f}{pin}〕\n{body}")
            elif e.level == LOD1:
                lines.append(f"#{e.turn_id} 〔LOD1 R={e.score:.2f}〕{e.lod1}")
            else:
                lines.append(f"#{e.turn_id} 〔LOD0 R={e.score:.2f}〕{e.lod0}")
        if cold:
            ids = ", ".join(f"#{i}" for i in cold[:8])
            more = "" if len(cold) <= 8 else f" 等 {len(cold)} 轮"
            lines.append(
                f"[冷区] {len(cold)} 轮已折叠（{ids}{more}）—— 可用 recall_turn(id) 拉回全文"
            )
        return "\n".join(lines), cold

    # ------------------------------------------------------------------
    # 只读诊断（/focus 指令用）
    # ------------------------------------------------------------------
    def snapshot(self, speaker_id: str) -> dict:
        state = self.load_state()
        turns = self.store.list_turns(speaker_id)
        entries, focus = self.distribution(turns, state)
        return {
            "focus_turn_id": focus,
            "stored_focus_turn_id": state.focus_turn_id,
            "decay_before": state.decay_before,
            "decay_after": state.decay_after,
            "recency_floor": state.recency_floor,
            "pinned": state.pinned_turn_ids,
            "visible_turns": len(turns),
            "levels": [
                {
                    "turn_id": e.turn_id,
                    "R": round(e.score, 4),
                    "lod": e.level_name,
                }
                for e in sorted(entries, key=lambda x: x.turn_id)
            ],
        }


def decay_table(
    turn_ids: Iterable[int],
    focus: int,
    decay_before: float,
    decay_after: float,
) -> dict[int, float]:
    """纯粹的数学习题入口（测试与 /focus 诊断共用）。"""
    return {
        i: clarity(int(i), focus, decay_before, decay_after) for i in turn_ids
    }
