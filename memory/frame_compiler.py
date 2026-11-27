"""认知帧编译器（SPEC 4.1）。

每次唤醒 Sys2，**不**发送历史 chat messages 数组，而是把此刻的世界状态
压缩成一帧恒定极短、高密度的快照：

    [SITUATIONAL_ENV]     时间/间隔/说话人/关系/情绪余温/环境
    [SYS1_INTUITION]      潜意识反射命中的直觉
    [FOCAL_DIALOGUE]      焦点式双向衰减的近况
    [LONG_TERM_LANDSCAPE] 长期记忆地貌 + 准则/技能目录
    [CURRENT_EVENT]       此刻唤醒她的事件（发言 / 召回结果 / 自主闹钟）

总长度锁在 ``budget.frame_tokens`` 以内，与对话总轮数无关。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from kernel.config import Config
from kernel.models import Event, ReflexSignal
from kernel.paths import Workspace
from kernel.sys2_llm import SYSTEM_CONTRACT
from kernel.textutil import estimate_tokens, truncate
from memory.focal_context import FocalContext, FocalRender
from memory.lod_renderer import LandscapeRender, LodRenderer
from memory.store import MemoryStore
from sensors.clock import ClockSensor
from sensors.social import SocialSensor
from sensors.world import WorldSensor


@dataclass
class Frame:
    system: str
    user: str
    section_tokens: dict[str, int] = field(default_factory=dict)
    total_tokens: int = 0
    focal: FocalRender | None = None
    landscape: LandscapeRender | None = None
    env: dict[str, Any] = field(default_factory=dict)
    signals: list[ReflexSignal] = field(default_factory=list)
    event_text: str = ""
    speaker_id: str = "owner"

    def messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]

    def summary(self) -> str:
        parts = " ".join(f"{k}={v}" for k, v in self.section_tokens.items())
        return f"frame {self.total_tokens} tokens ({parts})"


class FrameCompiler:
    def __init__(
        self,
        ws: Workspace,
        store: MemoryStore,
        config: Config | None = None,
        focal: FocalContext | None = None,
        landscape: LodRenderer | None = None,
        clock: ClockSensor | None = None,
        social: SocialSensor | None = None,
        world: WorldSensor | None = None,
    ):
        self.ws = ws
        self.store = store
        self.config = config or ws.config
        self.focal = focal or FocalContext(ws, store, self.config)
        self.landscape = landscape or LodRenderer(ws, store, self.config)
        self.clock = clock or ClockSensor(ws, store)
        self.social = social or SocialSensor(ws, self._relations())
        self.world = world or WorldSensor(ws)

    def _relations(self):
        from memory.relations import RelationsStore

        return RelationsStore(self.ws)

    # ------------------------------------------------------------------
    def compile(
        self,
        event: Event,
        signals: Sequence[ReflexSignal] = (),
        speaker_id: str | None = None,
        carried: Iterable[Mapping[str, str]] = (),
        now=None,
    ) -> Frame:
        speaker = speaker_id or event.speaker_id
        cfg = self.config
        system = SYSTEM_CONTRACT
        system_tokens = estimate_tokens(system)
        user_budget = max(600, cfg.budget.frame_tokens - system_tokens)

        env_snapshot = {
            "clock": self.clock.capture(speaker, now=now),
            "social": self.social.capture(speaker),
            "world": self.world.capture(),
        }
        env_text = self._render_env(env_snapshot, user_budget)
        sys1_text = self._render_sys1(signals)
        event_text = self._render_event(event, carried)

        base = {
            "env": cfg.budget.section_env,
            "sys1": cfg.budget.section_sys1,
            "event": cfg.budget.section_event,
        }
        reserve = sum(base.values())
        focal_budget = max(250, min(cfg.budget.section_focal, user_budget - reserve - 250))
        land_budget = max(250, min(cfg.budget.section_landscape, user_budget - reserve - focal_budget))
        if focal_budget + land_budget + reserve > user_budget:
            scale = user_budget / (focal_budget + land_budget + reserve)
            focal_budget = int(focal_budget * scale)
            land_budget = int(land_budget * scale)

        query = event.text or ""
        boost = [tag for s in signals for tag in (s.actions.get("boost_tags") or [])]

        focal_render = self.focal.render(speaker, focal_budget)
        land_render = self.landscape.render(speaker, query, land_budget, boost_tags=boost)

        for _ in range(6):
            user_text = "\n".join(
                [
                    "[SITUATIONAL_ENV]",
                    env_text,
                    "[SYS1_INTUITION]",
                    sys1_text,
                    "[FOCAL_DIALOGUE]",
                    focal_render.text,
                    "[LONG_TERM_LANDSCAPE]",
                    land_render.text,
                    "[CURRENT_EVENT]",
                    event_text,
                ]
            )
            total = estimate_tokens(user_text) + system_tokens
            if total <= cfg.budget.frame_tokens:
                break
            overflow = total - cfg.budget.frame_tokens
            # 先压远景，再压近况，最后压直觉；环境与当前事件绝不牺牲
            if land_budget > 250 and overflow > 0:
                cut = min(overflow, land_budget - 250)
                land_budget -= cut
                overflow -= cut
                land_render = self.landscape.render(
                    speaker, query, land_budget, boost_tags=boost
                )
                if overflow <= 0:
                    continue
            if focal_budget > 250 and overflow > 0:
                focal_budget = max(250, focal_budget - overflow)
                focal_render = self.focal.render(speaker, focal_budget)
                if overflow - (focal_budget - 250) <= 0:
                    continue
            break

        section_tokens = {
            "env": estimate_tokens(env_text),
            "sys1": estimate_tokens(sys1_text),
            "focal": estimate_tokens(focal_render.text),
            "landscape": estimate_tokens(land_render.text),
            "event": estimate_tokens(event_text),
        }
        return Frame(
            system=system,
            user=user_text,
            section_tokens=section_tokens,
            total_tokens=estimate_tokens(user_text) + system_tokens,
            focal=focal_render,
            landscape=land_render,
            env=env_snapshot,
            signals=list(signals),
            event_text=event_text,
            speaker_id=speaker,
        )

    # ------------------------------------------------------------------
    def _render_env(self, snapshot: Mapping[str, Any], budget: int) -> str:
        lines = [
            self.clock.render(snapshot["clock"]),
            self.social.render(snapshot["social"]),
            self.world.render(snapshot["world"]),
        ]
        text = "\n".join(lines)
        return truncate(text, max(200, budget * 2))

    @staticmethod
    def _render_sys1(signals: Sequence[ReflexSignal]) -> str:
        if not signals:
            return "（潜意识安静：本轮没有模糊反射命中）"
        lines: list[str] = []
        for sig in signals:
            if sig.gate == "high":
                lines.append(f"● 高置信[{sig.reflex_id}] p={sig.probability:.2f}：{sig.hint}")
            elif sig.gate == "mid":
                lines.append(
                    f"○ 中置信[{sig.reflex_id}] p={sig.probability:.2f}（不确定，仅供参考）：{sig.hint}"
                )
        if not lines:
            lines.append("（潜意识安静：本轮没有达到门控阈值的反射）")
        return "\n".join(lines)

    @staticmethod
    def _render_event(
        event: Event, carried: Iterable[Mapping[str, str]]
    ) -> str:
        lines: list[str] = []
        for item in carried:
            label = item.get("label", "事件")
            body = item.get("text", "")
            lines.append(f"[{label}]\n{body}")
        if event.kind == "message":
            lines.append(f"[{event.speaker_id} 发言]\n{event.text}")
        elif event.kind == "wake":
            reason = event.extra.get("reason") or "（没有写原因）"
            lines.append(
                f"[自主唤醒 · 这是你自己预约的时刻，不是有人发消息]\n预约理由：{reason}"
            )
            if event.text:
                lines.append(event.text)
        elif event.kind == "recall_result":
            lines.append(f"[召回结果]\n{event.text}")
        else:
            lines.append(f"[{event.kind}]\n{event.text}")
        return "\n".join(lines)
