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
from kernel.literal import load_literal
from kernel.models import Event, ReflexSignal
from kernel.paths import Workspace
from kernel.sys2_llm import SYSTEM_CONTRACTS, SYSTEM_CONTRACT
from kernel.textutil import estimate_tokens, one_line, truncate
from memory.focal_context import FocalContext, FocalRender
from memory.lod_renderer import LandscapeRender, LodRenderer
from memory.store import MemoryStore
from sensors.clock import ClockSensor
from sensors.social import SocialSensor
from sensors.world import WorldSensor


#: 性格维度的中文名（cortex/persona.py 里是英文键，便于程序处理）
TRAIT_NAMES = {
    "curiosity": "好奇",
    "directness": "直率",
    "warmth": "温和",
    "playfulness": "俏皮",
    "self_respect": "自尊",
    "empathy": "共情",
    "patience": "耐心",
    "focus": "专注",
}


OUTPUT_REMINDER = (
    "[OUTPUT_FORMAT] 只输出一个 JSON 对象（不要解释、不要围栏）："
    '{"inner_monologue":"…","say":"…或空串","memory_ops":[],"focus_update":null,'
    '"emotion_delta":{},"wake":{}}'
)


def render_identity(persona: Mapping[str, Any], max_chars: int = 480) -> str:
    """把 cortex/persona.py 压成一段“她是谁”（恒定短，可被进化改写）。"""
    name = str(persona.get("NAME") or "星璃")
    concept = one_line(str(persona.get("SELF_CONCEPT") or ""), 130)
    values = [
        one_line(str(v).rstrip("。；; "), 36) for v in (persona.get("VALUES") or [])
    ][:4]
    traits = persona.get("TRAITS") or {}
    trait_text = " · ".join(
        f"{TRAIT_NAMES.get(str(k), str(k))} {float(v):.2f}"
        for k, v in traits.items()
        if isinstance(v, (int, float))
    )
    lines = [f"【你是谁】{name}（Xingli）"]
    if concept:
        lines.append(f"你怎么理解自己：{concept}")
    if values:
        lines.append("你的底线：" + "；".join(values))
    if trait_text:
        lines.append("性格倾向：" + trait_text)
    lines.append(
        "说话用第一人称，有判断、有温度；不客套、不背说明书、不假装记得；"
        "被问到自己是谁时，就讲你怎么记事（焦点视距 / 长期地貌 / 自己的 Git 账本）。"
    )
    return truncate("\n".join(lines), max_chars)


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
    identity: str = ""

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

    def persona(self) -> dict[str, Any]:
        """读取 cortex/persona.py 的字面量（按 mtime 缓存，进化后自动失效）。"""
        path = self.ws.persona_file
        try:
            mtime = path.stat().st_mtime_ns
        except OSError:
            mtime = 0
        cached = getattr(self, "_persona_cache", None)
        if cached and cached[0] == mtime:
            return cached[1]
        data: dict[str, Any] = {}
        for key in ("NAME", "SELF_CONCEPT", "VALUES", "TRAITS"):
            value = load_literal(path, key, None)
            if value is not None:
                data[key] = value
        self._persona_cache = (mtime, data)
        return data

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
        identity = render_identity(
            self.persona(), max_chars=max(160, cfg.budget.section_identity * 2)
        )
        contract = SYSTEM_CONTRACTS.get(cfg.sys2.contract, SYSTEM_CONTRACT)
        system = f"{identity}\n\n{contract}"
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
            blocks = [
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
            if cfg.sys2.contract_in_user:
                blocks.extend([OUTPUT_REMINDER])
            user_text = "\n".join(blocks)
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
            identity=identity,
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
