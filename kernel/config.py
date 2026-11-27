"""配置加载器（纯标准库，tomllib）。

设计原则：配置驱动、Mock 友好。不配置任何 API Key 时，系统也能以
``sys2.provider = "mock"`` + ``sys1.provider = "local"`` 离线跑通全流程
（用于测试与理论模型验证）；配置真实端点后零代码改动切换。

配置文件优先级：
1. 显式传入的 path；
2. 环境变量 ``XINGLI_CONFIG``；
3. 工作区根目录下的 ``xingli.toml`` / ``config.toml``；
4. 缺省值（本文件中的 dataclass 默认值）。
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _as_str(value: Any, default: str) -> str:
    return default if value is None else str(value)


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


@dataclass(frozen=True)
class Sys2Config:
    """System Two：生成式大模型（前额叶）配置。"""

    provider: str = "mock"  # mock | openai_compatible
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "XINGLI_SYS2_KEY"
    api_key: str = ""  # 一般留空，走环境变量；仅供本地临时调试
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 1400
    timeout_sec: float = 90.0
    json_mode: bool = True
    #: 结构化输出方式：auto（json_schema → json_object → 关）/ json_schema / json_object / off
    structured_output: str = "auto"
    #: 输出像坏掉的 JSON 时追加一次修复请求（纯散文不浪费这一趟）
    repair: bool = True
    #: 网络/连接失败的重试次数（不含上面那种降级重试）
    retries: int = 1
    #: 原样并入请求体的额外字段，例如 { chat_template_kwargs = { enable_thinking = false } }
    extra_body: Mapping[str, Any] = field(default_factory=dict)
    #: 行动契约版本：full（默认）| compact（小模型 / 紧张上下文）
    contract: str = "full"
    #: 在认知帧末尾再贴一次“只要 JSON”的短提醒（有些模型的 chat template 会吞掉 system）
    contract_in_user: bool = False
    #: 预填 assistant 开头，逼模型“接着 JSON 往下写”。off | say
    #:   say: 预填 {"inner_monologue": "", "say": "  —— 让它先说话，思考模型也没法长篇叙事
    prefill: str = "off"
    #: 检测到“模型在叙述分析过程”时，自动用预填再试一次（只多一趟短请求）
    auto_prefill: bool = True
    #: 弱模型补洞：模型没写 memory_ops 时，内核再发一次极短的“记忆抽取”请求
    memory_pass: bool = True

    @property
    def resolved_api_key(self) -> str:
        return self.api_key or os.environ.get(self.api_key_env, "")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Sys2Config":
        d = d or {}
        return cls(
            provider=_as_str(d.get("provider"), cls.provider),
            base_url=_as_str(d.get("base_url"), cls.base_url).rstrip("/"),
            api_key_env=_as_str(d.get("api_key_env"), cls.api_key_env),
            api_key=_as_str(d.get("api_key"), cls.api_key),
            model=_as_str(d.get("model"), cls.model),
            temperature=_as_float(d.get("temperature"), cls.temperature),
            max_tokens=_as_int(d.get("max_tokens"), cls.max_tokens),
            timeout_sec=_as_float(d.get("timeout_sec"), cls.timeout_sec),
            json_mode=_as_bool(d.get("json_mode"), cls.json_mode),
            structured_output=_as_str(
                d.get("structured_output"),
                "auto" if _as_bool(d.get("json_mode"), cls.json_mode) else "off",
            ),
            repair=_as_bool(d.get("repair"), cls.repair),
            retries=_as_int(d.get("retries"), cls.retries),
            extra_body=_as_mapping(d.get("extra_body")),
            contract=_as_str(d.get("contract"), cls.contract),
            contract_in_user=_as_bool(d.get("contract_in_user"), cls.contract_in_user),
            prefill=_as_str(d.get("prefill"), cls.prefill).lower(),
            auto_prefill=_as_bool(d.get("auto_prefill"), cls.auto_prefill),
            memory_pass=_as_bool(d.get("memory_pass"), cls.memory_pass),
        )


@dataclass(frozen=True)
class Sys1Config:
    """System One：非生成式模糊决策模型（丘脑/杏仁核）配置。

    provider:
      - ``local``            : 离线词法近似适配器（无 Key 时的开发/测试降级）
      - ``jev``              : TypeSafe Jev / OpenRouter 的 System One 端点
      - ``openai_decisions`` : OpenAI Decisions API
      - ``llm_fallback``     : 用轻量生成式模型模拟概率输出
    """

    provider: str = "local"
    base_url: str = ""
    api_key_env: str = "XINGLI_SYS1_KEY"
    api_key: str = ""
    model: str = ""
    timeout_sec: float = 8.0
    max_tokens: int = 300
    #: 原样并入请求体的额外字段（与 sys2.extra_body 同义）
    extra_body: Mapping[str, Any] = field(default_factory=dict)

    @property
    def resolved_api_key(self) -> str:
        return self.api_key or os.environ.get(self.api_key_env, "")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Sys1Config":
        d = d or {}
        return cls(
            provider=_as_str(d.get("provider"), cls.provider),
            base_url=_as_str(d.get("base_url"), cls.base_url).rstrip("/"),
            api_key_env=_as_str(d.get("api_key_env"), cls.api_key_env),
            api_key=_as_str(d.get("api_key"), cls.api_key),
            model=_as_str(d.get("model"), cls.model),
            timeout_sec=_as_float(d.get("timeout_sec"), cls.timeout_sec),
            max_tokens=_as_int(d.get("max_tokens"), cls.max_tokens),
            extra_body=_as_mapping(d.get("extra_body")),
        )


@dataclass(frozen=True)
class BudgetConfig:
    """Token 预算与自主唤醒硬熔断（铁律 6）。"""

    frame_tokens: int = 6000
    section_env: int = 700
    section_identity: int = 240
    section_sys1: int = 700
    section_focal: int = 2300
    section_landscape: int = 2000
    section_event: int = 400
    # 单次外部事件触发的最大连环思考步数（硬熔断）
    max_chain_ticks: int = 4
    # 一轮外部事件内，连环思考允许花掉的总墙钟时间（秒）。本地 4B 一拍就 15~40s，
    # 没有这个刹车就会出现“她没有输入窗口、你只能干等”的长时间自转。
    max_chain_seconds: int = 60
    # 自主唤醒触发的连环思考上限（默认 1：没人等着听，就别自转）
    max_wake_chain_ticks: int = 1
    # 自主唤醒的最短预约间隔（秒）：模型常写 after_sec=0..5，会把对话变成自激循环
    min_wake_sec: int = 30
    # 每小时允许的自主唤醒（schedule_wake 真正触发）配额
    max_proactive_per_hour: int = 6

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "BudgetConfig":
        d = d or {}
        return cls(
            frame_tokens=_as_int(d.get("frame_tokens"), cls.frame_tokens),
            section_env=_as_int(d.get("section_env"), cls.section_env),
            section_identity=_as_int(d.get("section_identity"), cls.section_identity),
            section_sys1=_as_int(d.get("section_sys1"), cls.section_sys1),
            section_focal=_as_int(d.get("section_focal"), cls.section_focal),
            section_landscape=_as_int(d.get("section_landscape"), cls.section_landscape),
            section_event=_as_int(d.get("section_event"), cls.section_event),
            max_chain_ticks=_as_int(d.get("max_chain_ticks"), cls.max_chain_ticks),
            max_chain_seconds=_as_int(d.get("max_chain_seconds"), cls.max_chain_seconds),
            max_wake_chain_ticks=_as_int(
                d.get("max_wake_chain_ticks"), cls.max_wake_chain_ticks
            ),
            min_wake_sec=_as_int(d.get("min_wake_sec"), cls.min_wake_sec),
            max_proactive_per_hour=_as_int(
                d.get("max_proactive_per_hour"), cls.max_proactive_per_hour
            ),
        )


@dataclass(frozen=True)
class RenderConfig:
    """焦点 LOD 与长期记忆地貌的渲染参数。"""

    # 焦点式对话 LOD 的阈值（SPEC 4.2）
    theta2: float = 0.65
    theta1: float = 0.25
    theta0: float = 0.08
    # 长期记忆 LOD 视野容量
    landscape_n2: int = 4
    landscape_n1: int = 10
    landscape_n0: int = 40
    # 单条记忆/轮次的字符上限（防止单条巨物撑爆预算，超出部分保留哈希句柄）
    turn_lod2_chars: int = 700
    node_lod2_chars: int = 600
    # 心理距离打分权重（SPEC 4.5 / mhtml 讨论中的 Score(m) 公式）
    weight_semantic: float = 0.40
    weight_recency: float = 0.20
    weight_importance: float = 0.25
    weight_social: float = 0.10
    weight_boost: float = 0.05
    # 语义匹配的字符二元组阈值
    semantic_min_score: float = 0.0

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RenderConfig":
        d = d or {}
        return cls(
            theta2=_as_float(d.get("theta2"), cls.theta2),
            theta1=_as_float(d.get("theta1"), cls.theta1),
            theta0=_as_float(d.get("theta0"), cls.theta0),
            landscape_n2=_as_int(d.get("landscape_n2"), cls.landscape_n2),
            landscape_n1=_as_int(d.get("landscape_n1"), cls.landscape_n1),
            landscape_n0=_as_int(d.get("landscape_n0"), cls.landscape_n0),
            turn_lod2_chars=_as_int(d.get("turn_lod2_chars"), cls.turn_lod2_chars),
            node_lod2_chars=_as_int(d.get("node_lod2_chars"), cls.node_lod2_chars),
            weight_semantic=_as_float(d.get("weight_semantic"), cls.weight_semantic),
            weight_recency=_as_float(d.get("weight_recency"), cls.weight_recency),
            weight_importance=_as_float(d.get("weight_importance"), cls.weight_importance),
            weight_social=_as_float(d.get("weight_social"), cls.weight_social),
            weight_boost=_as_float(d.get("weight_boost"), cls.weight_boost),
            semantic_min_score=_as_float(
                d.get("semantic_min_score"), cls.semantic_min_score
            ),
        )


@dataclass(frozen=True)
class SleepConfig:
    """离线睡眠整合参数。"""

    half_life_days: float = 7.0
    min_importance: float = 0.05
    distill_window: int = 4  # 每 N 轮（或会话中断）聚合成一个 episode
    session_gap_minutes: int = 45
    sample_size: int = 3
    # 反失忆抽查的召回覆盖率阈值（唯一可证伪的“不失忆”定义）
    coverage_threshold: float = 0.6
    lod1_chars: int = 220
    lod0_chars: int = 60

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SleepConfig":
        d = d or {}
        return cls(
            half_life_days=_as_float(d.get("half_life_days"), cls.half_life_days),
            min_importance=_as_float(d.get("min_importance"), cls.min_importance),
            distill_window=_as_int(d.get("distill_window"), cls.distill_window),
            session_gap_minutes=_as_int(
                d.get("session_gap_minutes"), cls.session_gap_minutes
            ),
            sample_size=_as_int(d.get("sample_size"), cls.sample_size),
            coverage_threshold=_as_float(
                d.get("coverage_threshold"), cls.coverage_threshold
            ),
            lod1_chars=_as_int(d.get("lod1_chars"), cls.lod1_chars),
            lod0_chars=_as_int(d.get("lod0_chars"), cls.lod0_chars),
        )


@dataclass(frozen=True)
class Config:
    sys1: Sys1Config = field(default_factory=Sys1Config)
    sys2: Sys2Config = field(default_factory=Sys2Config)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    sleep: SleepConfig = field(default_factory=SleepConfig)
    # 相对路径的文件名（相对工作区根目录）
    cortex_dir: str = "cortex"
    runtime_dir: str = ".xingli"

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Config":
        d = d or {}
        return cls(
            sys1=Sys1Config.from_dict(d.get("sys1", {})),
            sys2=Sys2Config.from_dict(d.get("sys2", {})),
            budget=BudgetConfig.from_dict(d.get("budget", {})),
            render=RenderConfig.from_dict(d.get("render", {})),
            sleep=SleepConfig.from_dict(d.get("sleep", {})),
            cortex_dir=_as_str(d.get("cortex_dir"), cls.cortex_dir),
            runtime_dir=_as_str(d.get("runtime_dir"), cls.runtime_dir),
        )

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None, root: str | os.PathLike[str] | None = None) -> "Config":
        """从 TOML 加载配置；找不到文件时返回默认配置。"""
        candidates: list[Path] = []
        if path:
            candidates.append(Path(path))
        elif os.environ.get("XINGLI_CONFIG"):
            candidates.append(Path(os.environ["XINGLI_CONFIG"]))
        elif root:
            base = Path(root)
            candidates.extend([base / "xingli.toml", base / "config.toml"])
        for c in candidates:
            if c.is_file():
                with c.open("rb") as fh:
                    return cls.from_dict(tomllib.load(fh))
        return cls()

    def replace(self, **kwargs: Any) -> "Config":
        return replace(self, **kwargs)

    def with_sys2(self, **kwargs: Any) -> "Config":
        return replace(self, sys2=replace(self.sys2, **kwargs))

    def with_sys1(self, **kwargs: Any) -> "Config":
        return replace(self, sys1=replace(self.sys1, **kwargs))

    def with_budget(self, **kwargs: Any) -> "Config":
        return replace(self, budget=replace(self.budget, **kwargs))

    def with_render(self, **kwargs: Any) -> "Config":
        return replace(self, render=replace(self.render, **kwargs))
