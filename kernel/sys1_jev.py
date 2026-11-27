"""System One：模糊反射引擎（铁律 5）。

对外只暴露四个原语：``noul`` / ``choose`` / ``score`` / ``fan_out``。
底层供应商可切换（Jev、OpenAI Decisions、LLM 降级、离线词法近似），
上层只认「一次 HTTP 并行扇出 + 三段式置信度门控 + 冷却/限幅」。

三段门控（绝不硬切 p > 0.5）：

* ``p >= high_threshold``          → 高置信：反射动作直接生效；
* ``low_threshold < p < high``     → 中置信：只注入“直觉提示”交给 Sys2 慢思考；
* ``p <= low_threshold`` 或置信太低 → 静默忽略。

每条反射强制带 ``cooldown_sec`` / ``decay`` / ``max_delta``，杜绝自激震荡。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Mapping, Protocol

from kernel.config import Config, Sys1Config
from kernel.jsonx import loads_lenient, message_text
from kernel.literal import load_literal
from kernel.models import ReflexSignal, clamp, iso, now_utc, parse_iso
from kernel.paths import Workspace

# ---------------------------------------------------------------------------
# 数据契约
# ---------------------------------------------------------------------------
@dataclass
class QuestionSpec:
    id: str
    kind: str  # noul | choose | score
    question: str
    options: list[str] | None = None
    levels: list[str] | None = None
    cues: dict[str, list[str]] | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "question": self.question,
            "type": self.kind,
        }
        if self.options:
            payload["options"] = list(self.options)
        if self.levels:
            payload["levels"] = list(self.levels)
        return payload


@dataclass
class DecisionResult:
    spec_id: str
    kind: str
    value: Any
    distribution: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def probability(self) -> float:
        """noul 的“是”概率；其他类型用置信度兜底。"""
        if self.kind == "noul":
            try:
                return clamp(float(self.value), 0.0, 1.0)
            except (TypeError, ValueError):
                return 0.0
        return clamp(self.confidence, 0.0, 1.0)


class Sys1Error(RuntimeError):
    pass


class Sys1Client(Protocol):
    name: str

    def fan_out(
        self, state: str, specs: list[QuestionSpec]
    ) -> dict[str, DecisionResult]: ...


# ---------------------------------------------------------------------------
# 离线词法近似适配器（无 Key 时的可复现降级）
# ---------------------------------------------------------------------------
class LocalLexiconClient:
    """用词法线索模拟模糊感知器。

    诚实声明：这**不是**模糊决策模型，它只是让整条门控/冷却/限幅管线在
    离线与测试环境里可复现。真实语义判断请配置 ``provider = "jev"``。
    """

    name = "local_lexicon"

    def fan_out(self, state: str, specs: list[QuestionSpec]) -> dict[str, DecisionResult]:
        return {spec.id: self._evaluate(state, spec) for spec in specs}

    @staticmethod
    def _hits(text: str, cues: Iterable[str] | None) -> int:
        if not cues:
            return 0
        return sum(text.count(str(c)) for c in cues if c)

    def _evaluate(self, state: str, spec: QuestionSpec) -> DecisionResult:
        cues = spec.cues or {}
        if spec.kind == "noul":
            pos = self._hits(state, cues.get("positive"))
            neg = self._hits(state, cues.get("negative"))
            if pos == 0 and neg == 0:
                return DecisionResult(
                    spec_id=spec.id,
                    kind=spec.kind,
                    value=0.5,
                    confidence=0.1,
                    raw={"reason": "no_cue"},
                )
            p = 0.5 + 0.3 * (pos - neg) / (1 + pos + neg) + 0.15 * min(1, pos) - 0.15 * min(1, neg)
            conf = min(0.9, 0.6 + 0.1 * min(3, pos + neg))
            return DecisionResult(
                spec_id=spec.id,
                kind=spec.kind,
                value=clamp(p, 0.02, 0.98),
                confidence=conf,
                distribution={"yes": clamp(p, 0.02, 0.98), "no": 1 - clamp(p, 0.02, 0.98)},
                raw={"pos": pos, "neg": neg},
            )
        if spec.kind == "choose":
            options = spec.options or []
            option_cues = {o: list(cues.get(o, [])) for o in options}
            scores = {o: self._hits(state, c) for o, c in option_cues.items()}
            total = sum(scores.values())
            if total == 0:
                dist = {o: 1.0 / max(1, len(options)) for o in options}
                return DecisionResult(
                    spec_id=spec.id,
                    kind=spec.kind,
                    value=options[0] if options else None,
                    distribution=dist,
                    confidence=0.1,
                    raw={"reason": "no_cue"},
                )
            dist = {o: scores[o] / total for o in options}
            selected = max(scores.items(), key=lambda kv: kv[1])[0]
            return DecisionResult(
                spec_id=spec.id,
                kind=spec.kind,
                value=selected,
                distribution=dist,
                confidence=min(0.9, 0.55 + 0.15 * total),
                raw={"scores": scores},
            )
        # score
        levels = spec.levels or ["1", "2", "3", "4", "5"]
        pos = self._hits(state, cues.get("positive"))
        neg = self._hits(state, cues.get("negative"))
        if pos == 0 and neg == 0:
            value = 3.0
            conf = 0.1
        else:
            value = 1 + 4 * (pos / (pos + neg + 1)) + 0.5 * min(1, pos)
            value = clamp(value, 1.0, float(len(levels)))
            conf = min(0.9, 0.55 + 0.1 * (pos + neg))
        return DecisionResult(
            spec_id=spec.id,
            kind=spec.kind,
            value=value,
            distribution={lvl: (1.0 if abs(float(lvl) - value) < 0.6 else 0.0) for lvl in levels},
            confidence=conf,
            raw={"pos": pos, "neg": neg},
        )


# ---------------------------------------------------------------------------
# HTTP 适配器（Jev / OpenAI Decisions / LLM 降级）
# ---------------------------------------------------------------------------
def _post_json(url: str, payload: dict[str, Any], api_key: str, timeout: float) -> dict[str, Any]:
    headers = {"Content-Type": "application/json", "User-Agent": "xingli-os/0.2"}
    if api_key:  # 本地 llama.cpp 通常不校验 Key
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "ignore")[:300]
        raise Sys1Error(f"HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise Sys1Error(f"网络失败: {exc}") from exc
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise Sys1Error(f"返回不是 JSON: {body[:200]}") from exc
    if not isinstance(parsed, dict):
        raise Sys1Error("返回不是 JSON 对象")
    return parsed


def _parse_results(payload: dict[str, Any], specs: list[QuestionSpec]) -> dict[str, DecisionResult]:
    """尽量宽容地解析不同供应商的返回结构。

    期望（Jev 风格）::

        {"results": {"<id>": {"yes": 0.86, "confidence": 0.72}}}

    也兼容 ``{"answers": [{"id": ..., ...}]}`` 与直接以 id 为键的返回。
    """
    by_id = {s.id: s for s in specs}
    raw_items: dict[str, Any] = {}

    results = payload.get("results")
    if isinstance(results, dict):
        raw_items = results
    elif isinstance(results, list):
        for item in results:
            if isinstance(item, dict) and "id" in item:
                raw_items[str(item["id"])] = item
    answers = payload.get("answers") or payload.get("decisions") or payload.get("data")
    if not raw_items and isinstance(answers, list):
        for item in answers:
            if isinstance(item, dict) and "id" in item:
                raw_items[str(item["id"])] = item
    if not raw_items:
        raw_items = {
            k: v
            for k, v in payload.items()
            if k in by_id and isinstance(v, dict)
        }

    parsed: dict[str, DecisionResult] = {}
    for spec_id, item in raw_items.items():
        spec = by_id.get(spec_id)
        if spec is None or not isinstance(item, dict):
            continue
        confidence = float(item.get("confidence", item.get("conf", 0.6)) or 0.6)
        if spec.kind == "noul":
            yes = item.get("yes", item.get("probability", item.get("p")))
            if yes is None and isinstance(item.get("distribution"), dict):
                yes = (item["distribution"] or {}).get("yes")
            try:
                yes_f = clamp(float(yes), 0.0, 1.0)
            except (TypeError, ValueError):
                continue
            parsed[spec_id] = DecisionResult(
                spec_id=spec_id,
                kind="noul",
                value=yes_f,
                distribution={"yes": yes_f, "no": 1 - yes_f},
                confidence=clamp(confidence, 0.0, 1.0),
                raw=item,
            )
        elif spec.kind == "choose":
            dist_raw = item.get("distribution") or item.get("dist") or {}
            dist = {}
            if isinstance(dist_raw, dict):
                for k, v in dist_raw.items():
                    try:
                        dist[str(k)] = float(v)
                    except (TypeError, ValueError):
                        continue
            selected = item.get("selected") or item.get("choice") or item.get("value")
            if selected is None and dist:
                selected = max(dist.items(), key=lambda kv: kv[1])[0]
            parsed[spec_id] = DecisionResult(
                spec_id=spec_id,
                kind="choose",
                value=str(selected) if selected is not None else None,
                distribution=dist,
                confidence=clamp(confidence, 0.0, 1.0),
                raw=item,
            )
        else:
            value = item.get("value", item.get("score", item.get("grade")))
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                continue
            dist_raw = item.get("distribution") or {}
            dist = {}
            if isinstance(dist_raw, dict):
                for k, v in dist_raw.items():
                    try:
                        dist[str(k)] = float(v)
                    except (TypeError, ValueError):
                        continue
            parsed[spec_id] = DecisionResult(
                spec_id=spec_id,
                kind="score",
                value=value_f,
                distribution=dist,
                confidence=clamp(confidence, 0.0, 1.0),
                raw=item,
            )
    return parsed


class HttpDecisionClient:
    """Jev /v1/systemone 或 OpenAI /v1/decisions 适配器（单请求并行扇出）。"""

    def __init__(self, config: Sys1Config):
        self.config = config
        self.name = config.provider
        if not config.base_url:
            raise Sys1Error("sys1.base_url 未配置")
        # 本地 Jev / llama.cpp 可能不校验 Key：没有 Key 也能连

    def _url(self) -> str:
        base = self.config.base_url.rstrip("/")
        if self.config.provider == "openai_decisions":
            return base + "/decisions"
        return base + "/systemone"

    def fan_out(self, state: str, specs: list[QuestionSpec]) -> dict[str, DecisionResult]:
        payload: dict[str, Any] = {
            "state": state,
            "questions": [s.to_payload() for s in specs],
        }
        if self.config.model:
            payload["model"] = self.config.model
        data = _post_json(
            self._url(), payload, self.config.resolved_api_key, self.config.timeout_sec
        )
        return _parse_results(data, specs)


class LLMFallbackClient:
    """用普通聊天模型模拟概率输出（任何 OpenAI 兼容端点，含本地 llama.cpp）。

    与 ``jev`` 的区别：不需要 ``/v1/systemone`` 这种专用结构，只发一次
    chat/completions 并让它返回 ``{"results": {...}}``。适合“我的 Jev 其实
    只是个 OpenAI 兼容端点”的场景。
    """

    name = "llm_fallback"

    def __init__(self, config: Sys1Config):
        self.config = config
        if not config.base_url:
            raise Sys1Error("sys1.base_url 未配置")

    def _payload(self, prompt: str, json_mode: bool) -> dict[str, Any]:
        cfg = self.config
        payload: dict[str, Any] = {
            "model": cfg.model or "local-model",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是模糊感知器。阅读 STATE，对每个问题给出校准后的概率/分布/打分。"
                        "只输出 JSON 对象，形如 {\"results\": {\"<id>\": {...}}}，不要解释。"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": cfg.max_tokens,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        payload.update(dict(cfg.extra_body or {}))
        return payload

    def fan_out(self, state: str, specs: list[QuestionSpec]) -> dict[str, DecisionResult]:
        schema = {
            s.id: {"yes": "0~1", "confidence": "0~1"} if s.kind == "noul" else (
                {"selected": s.options} if s.kind == "choose" else {"value": s.levels}
            )
            for s in specs
        }
        prompt = (
            f"问题与期望字段: {json.dumps([s.to_payload() for s in specs], ensure_ascii=False)}\n"
            f"字段模板: {json.dumps(schema, ensure_ascii=False)}\n"
            f"STATE:\n{state[:4000]}"
        )
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        json_mode = True
        try:
            data = _post_json(
                url,
                self._payload(prompt, json_mode),
                self.config.resolved_api_key,
                self.config.timeout_sec,
            )
        except Sys1Error as exc:
            if "response_format" not in str(exc).lower() and "json_object" not in str(exc).lower():
                raise
            json_mode = False
            data = _post_json(
                url,
                self._payload(prompt, json_mode),
                self.config.resolved_api_key,
                self.config.timeout_sec,
            )
        content = message_text(data)
        inner = loads_lenient(content) if content else None
        if inner is None:
            raise Sys1Error(f"LLM 降级返回不可解析: {content[:160]!r}")
        if isinstance(inner, list):
            inner = {"answers": inner}
        if isinstance(inner, dict) and "results" not in inner and "answers" not in inner:
            # 有些模型会直接返回 {id: {...}}
            inner = {"results": inner}
        return _parse_results(inner, specs)


class ChainClient:
    """主适配器失败时自动降级（例如 Jev 超时 → 离线词法），保证一拍不卡死。

    连续失败 ``max_failures`` 次后打开断路器：接下来 ``cooldown_sec`` 秒内
    直接走本地词法，不再为每一拍白等一次超时；冷却结束后自动试探恢复。
    """

    def __init__(
        self,
        primary: Sys1Client,
        fallback: Sys1Client,
        *,
        max_failures: int = 2,
        cooldown_sec: float = 60.0,
        now_fn=time.monotonic,
    ):
        self.primary = primary
        self.fallback = fallback
        self.name = f"{getattr(primary, 'name', 'primary')}->{getattr(fallback, 'name', 'fallback')}"
        self.max_failures = max_failures
        self.cooldown_sec = cooldown_sec
        self.now_fn = now_fn
        self.failures = 0
        self.open_until = 0.0
        self.last_error: str | None = None

    @property
    def circuit_open(self) -> bool:
        return self.failures >= self.max_failures and self.now_fn() < self.open_until

    def fan_out(self, state: str, specs: list[QuestionSpec]) -> dict[str, DecisionResult]:
        if self.circuit_open:
            self.last_error = (
                f"断路器打开中（再等 {self.open_until - self.now_fn():.0f}s）：{self.last_error or ''}"
            )
            return self.fallback.fan_out(state, specs)
        try:
            result = self.primary.fan_out(state, specs)
        except Exception as exc:  # noqa: BLE001 - 降级是有意为之
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.failures += 1
            if self.failures >= self.max_failures:
                self.open_until = self.now_fn() + self.cooldown_sec
            return self.fallback.fan_out(state, specs)
        self.failures = 0
        self.open_until = 0.0
        return result


#: 常见写法都认：用户写 llm-fallback / openai / chat 都当成 llm_fallback
_PROVIDER_ALIASES = {
    "llm": "llm_fallback",
    "fallback": "llm_fallback",
    "llm-fallback": "llm_fallback",
    "llmfallback": "llm_fallback",
    "openai": "llm_fallback",
    "openai_compatible": "llm_fallback",
    "openai-chat": "llm_fallback",
    "chat": "llm_fallback",
    "chat_completions": "llm_fallback",
    "lexicon": "local",
    "off": "local",
    "none": "local",
}


def _normalize_provider(provider: str) -> str:
    value = (provider or "").strip().lower()
    return _PROVIDER_ALIASES.get(value, value)


def build_client(config: Config) -> Sys1Client:
    """按配置造丘脑；接不上时**把原因留在 client 上**，不要静默消失。

    以前这里 `except Sys1Error: return local`，于是“我明明配了模型却没生效”
    在界面上完全看不出来。现在退回本地词法时会带上 ``build_error``，
    CLI 的 banner / `/sys1` 会把它原样打出来。
    """
    local = LocalLexiconClient()
    provider = _normalize_provider(config.sys1.provider)
    attempts: list[tuple[str, Any]] = []
    if provider in {"jev", "openai_decisions"}:
        attempts.append(("jev", HttpDecisionClient))
    elif provider == "llm_fallback":
        attempts.append(("llm_fallback", LLMFallbackClient))
    elif provider not in {"local", ""}:
        local.build_error = (
            f"不认识的 [sys1] provider = {provider!r}（可选 local / llm_fallback / jev）"
        )
        return local
    for _name, factory in attempts:
        try:
            return ChainClient(factory(config.sys1), local)
        except Sys1Error as exc:
            local.build_error = f"{provider} 初始化失败：{exc}"
            return local
    if config.sys1.model and not config.sys1.base_url:
        local.build_error = (
            f"配了 model={config.sys1.model!r} 但 [sys1] base_url 是空的，"
            "所以丘脑退回本地词法；补上 base_url，或在聊天里 /sys1 on <模型>"
        )
    return local


# ---------------------------------------------------------------------------
# 基础感知器（不写在 cortex，属于内核的“先天感官”）
# ---------------------------------------------------------------------------
BASE_PERCEPTIONS: list[QuestionSpec] = [
    QuestionSpec(
        id="base_intent",
        kind="choose",
        question="说话人这句话的主要意图是什么？",
        options=["闲聊", "求助", "倾诉", "吐槽", "指令", "教学"],
        cues={
            "闲聊": ["在吗", "你好", "哈哈", "随便聊聊", "天气"],
            "求助": ["帮我", "怎么办", "能不能", "如何", "求"],
            "倾诉": ["难受", "累", "烦", "压力", "委屈", "焦虑"],
            "吐槽": ["服了", "无语", "离谱", "又", "烂"],
            "指令": ["去", "执行", "运行", "改", "重启", "删"],
            "教学": ["记住", "以后", "应该", "不要", "教我", "规则", "别再"],
        },
    ),
    QuestionSpec(
        id="base_importance",
        kind="score",
        question="这句话值得写入长期记忆的重要度（1~5）？",
        levels=["1", "2", "3", "4", "5"],
        cues={
            "positive": ["重要", "记住", "永远", "第一次", "承诺", "约定", "别忘", "关键"],
            "negative": ["随便", "算了", "没事"],
        },
    ),
    QuestionSpec(
        id="base_teaching",
        kind="noul",
        question="说话人是否在教导、纠偏或给我立一条新规则？",
        cues={
            "positive": ["记住", "以后", "不要", "别再", "应该", "教我", "规则", "不准"],
            "negative": ["开玩笑", "随便说"],
        },
    ),
]

# ---------------------------------------------------------------------------
# 门控 + 冷却
# ---------------------------------------------------------------------------
@dataclass
class GatedScan:
    signals: list[ReflexSignal] = field(default_factory=list)
    perception: dict[str, DecisionResult] = field(default_factory=dict)
    emotion: dict[str, float] = field(default_factory=dict)
    hints: list[str] = field(default_factory=list)
    boost_tags: list[str] = field(default_factory=list)
    fired: list[str] = field(default_factory=list)
    mids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    results: dict[str, DecisionResult] = field(default_factory=dict)
    client: str = ""

    def intent(self) -> str | None:
        res = self.perception.get("base_intent")
        return str(res.value) if res and res.value else None

    def importance(self, default: float = 0.6) -> float:
        res = self.perception.get("base_importance")
        if res is None:
            return default
        try:
            return clamp(float(res.value) / 5.0, 0.05, 1.0)
        except (TypeError, ValueError):
            return default


class ReflexEngine:
    """把 cortex/reflexes.py 里的模糊反射编译成一次并行扫描。"""

    def __init__(
        self,
        ws: Workspace,
        config: Config | None = None,
        client: Sys1Client | None = None,
        now_fn=now_utc,
    ):
        self.ws = ws
        self.config = config or ws.config
        self.client = client or build_client(self.config)
        self.now_fn = now_fn

    # ---- 定义与状态 ---------------------------------------------------
    def reflex_defs(self) -> list[dict[str, Any]]:
        raw = load_literal(self.ws.reflexes_file, "REFLEXES", default=[]) or []
        return [r for r in raw if isinstance(r, dict) and r.get("enabled", True)]

    def _load_state(self) -> dict[str, Any]:
        import json

        path = self.ws.reflex_state_file
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_state(self, state: Mapping[str, Any]) -> None:
        import json

        path = self.ws.assert_writable(self.ws.reflex_state_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def cooling(self, reflex: Mapping[str, Any], state: Mapping[str, Any], now: datetime) -> bool:
        entry = state.get(str(reflex.get("id"))) or {}
        last = entry.get("last_fired")
        if not last:
            return False
        try:
            elapsed = (now - parse_iso(str(last))).total_seconds()
        except ValueError:
            return False
        return elapsed < float(reflex.get("cooldown_sec", 0) or 0)

    # ---- 扫描 ---------------------------------------------------------
    def specs(self, state: Mapping[str, Any], now: datetime) -> tuple[list[QuestionSpec], list[dict[str, Any]]]:
        specs = list(BASE_PERCEPTIONS)
        active: list[dict[str, Any]] = []
        for reflex in self.reflex_defs():
            if self.cooling(reflex, state, now):
                continue
            sensor = reflex.get("sensor") or {}
            kind = str(sensor.get("type", "noul"))
            if kind not in {"noul", "choose", "score"}:
                continue
            specs.append(
                QuestionSpec(
                    id=str(reflex.get("id", f"reflex_{len(specs)}")),
                    kind=kind,
                    question=str(sensor.get("question", "")),
                    options=sensor.get("options"),
                    levels=sensor.get("levels"),
                    cues=sensor.get("cues"),
                )
            )
            active.append(reflex)
        return specs, active

    def scan(
        self,
        state_text: str,
        speaker_id: str = "owner",
        now: datetime | None = None,
        extra_specs: list[QuestionSpec] | None = None,
    ) -> GatedScan:
        now = now or self.now_fn()
        stored = self._load_state()
        specs, active = self.specs(stored, now)
        if extra_specs:
            specs.extend(extra_specs)
        result = GatedScan(client=getattr(self.client, "name", "unknown"))
        if not specs:
            return result
        try:
            # 铁律：所有反射合并为一次请求，严禁循环里串行调用
            raw = self.client.fan_out(state_text, specs)
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"Sys1 扫描失败: {type(exc).__name__}: {exc}")
            return result
        result.results = raw
        for spec in BASE_PERCEPTIONS:
            if spec.id in raw:
                result.perception[spec.id] = raw[spec.id]

        for reflex in active:
            rid = str(reflex.get("id"))
            res = raw.get(rid)
            if res is None:
                result.errors.append(f"反射 {rid} 没有返回结果")
                continue
            p = res.probability
            conf = clamp(res.confidence, 0.0, 1.0)
            high = float(reflex.get("high_threshold", 0.8))
            low = float(reflex.get("low_threshold", 0.25))
            if conf < 0.15:
                gate = "low"
            elif p >= high:
                gate = "high"
            elif p > low:
                gate = "mid"
            else:
                gate = "low"
            actions = dict(reflex.get("on_high") or {}) if gate == "high" else {}
            signal = ReflexSignal(
                reflex_id=rid,
                gate=gate,
                probability=round(p, 4),
                confidence=round(conf, 4),
                question=str((reflex.get("sensor") or {}).get("question", "")),
                hint=(
                    str(actions.get("inject_hint", ""))
                    if gate == "high"
                    else (str(reflex.get("hint_on_mid", "")) if gate == "mid" else "")
                ),
                actions=actions,
            )
            result.signals.append(signal)
            entry = stored.setdefault(rid, {})
            if gate == "high":
                result.fired.append(rid)
                entry["last_fired"] = iso(now)
                entry["fires"] = int(entry.get("fires", 0)) + 1
                max_delta = abs(float(reflex.get("max_delta", 0.2)))
                for key, delta in (actions.get("emotion_delta") or {}).items():
                    try:
                        d = clamp(float(delta), -max_delta, max_delta)
                    except (TypeError, ValueError):
                        continue
                    result.emotion[key] = result.emotion.get(key, 0.0) + d
                for tag in actions.get("boost_tags") or []:
                    if tag not in result.boost_tags:
                        result.boost_tags.append(str(tag))
                if signal.hint:
                    result.hints.append(f"[{rid}] {signal.hint}")
            elif gate == "mid":
                result.mids.append(rid)
                entry["mids"] = int(entry.get("mids", 0)) + 1
                if signal.hint:
                    result.hints.append(f"[{rid}·不确定] {signal.hint}")
            else:
                entry["ignores"] = int(entry.get("ignores", 0)) + 1
            entry["last_seen"] = iso(now)

        # 冷却状态只保留最近 200 条反射，防止状态文件无限增长
        if len(stored) > 200:
            keep = sorted(
                stored.items(), key=lambda kv: str(kv[1].get("last_seen", "")), reverse=True
            )[:200]
            stored = dict(keep)
        self._save_state(stored)
        return result

    # ---- 四个原语的便捷封装（也用于测试） -----------------------------
    def noul(self, state: str, question: str, cues: dict[str, list[str]] | None = None) -> DecisionResult:
        spec = QuestionSpec(id="q", kind="noul", question=question, cues=cues)
        return self.client.fan_out(state, [spec]).get("q", DecisionResult("q", "noul", 0.0))

    def choose(
        self, state: str, question: str, options: list[str], cues: dict[str, list[str]] | None = None
    ) -> DecisionResult:
        spec = QuestionSpec(id="q", kind="choose", question=question, options=options, cues=cues)
        return self.client.fan_out(state, [spec]).get("q", DecisionResult("q", "choose", None))

    def score(
        self, state: str, question: str, levels: list[str], cues: dict[str, list[str]] | None = None
    ) -> DecisionResult:
        spec = QuestionSpec(id="q", kind="score", question=question, levels=levels, cues=cues)
        return self.client.fan_out(state, [spec]).get("q", DecisionResult("q", "score", 0.0))
