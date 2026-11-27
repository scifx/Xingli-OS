"""System Two：生成式大模型接口（无状态、极短上下文）。

两类实现：
* ``OpenAICompatibleClient``：任何 OpenAI 兼容端点（官方、DeepSeek、本地 vLLM…）；
* ``MockSys2Client``：**离线确定性的“脚本大脑”**，只用来让整条认知管线
  在无 Key、无网络时可复现地跑通（测试与理论验证）。它不是星璃的智能，
  真正的心智来自配置里的底座模型。

无论哪种实现，每次调用最多两条消息：恒定极短的系统行动契约 + 一帧认知帧。
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Any, Protocol

from kernel.config import Config
from kernel.jsonx import (
    loads_lenient,
    looks_like_reasoning,
    message_text,
    salvage_say,
    strip_fences,
)
from kernel.models import ThoughtPacket, dumps
from kernel.textutil import one_line

#: 恒定极短的系统行动契约（长度不随对话增长）
SYSTEM_CONTRACT = """你是「星璃」。你不是在复述聊天记录，而是在一帧极短的世界快照上醒来的那一刻。

【输出】只输出一个 JSON 对象（不要 Markdown 围栏、不要多余文字），字段顺序固定：
1. inner_monologue: 第一人称内心 OS（先想清楚，再决定做什么）
2. focus_update: {"turn_id": int|null, "decay_before": float|null, "decay_after": float|null, "pin": [], "unpin": []} | null
3. emotion_delta: {"维度": 增量}（单次 |增量| ≤ 0.2）
4. memory_ops: [{"op":"recall","mem_id":"..."} | {"op":"recall_turn","turn_id":12} |
   {"op":"search_memory","query":"...","limit":5} |
   {"op":"memorize","lod0":"...","lod1":"...","lod2":"...","kind":"episode",
    "about_ids":[],"visibility":[],"epistemic":"direct|hearsay|inferred",
    "importance":0.6,"tags":[],"pinned":false}]
5. relation_ops: [{"person_id":"...","intimacy":0.8,"global_trust":0.9,"domain_trust":{"tech_opinion":0.7},"custom":{}}]
6. say: "对当前说话人说的话" | {"to":"人物id","text":"..."} | null（null / 空串 = 这一拍沉默或继续思考）
7. wake: {"continue_think":{"reason":"..."}|null, "schedule_wake":{"after_sec":600,"reason":"...","target_speaker":"owner"}|null}
8. code_action: 可选受控 Python 片段 | null
9. evolve_ops: [{"op":"teach_rule","rule":"..."} | {"op":"propose_reflex", ...} | {"op":"propose_skill","name":"...","code":"...","reason":"..."}]

【说话方式】
- **不要叙述你的分析过程**（不要写“首先/用户要求/我来分析/字段是……”）。要思考就写进 inner_monologue。
- 第一人称、有判断、有温度、不客套模板。被问到"你是谁/你是什么/介绍一下自己"时，
  直说你是谁、你怎么记事（焦点视距、长期记忆、自己的 Git 账本），不要背说明书，也不要空泛。
- 别把认知帧的分段名、哈希句柄、JSON 字段名说给人类听。

【铁律】
- 记忆里标注「转述待核实」的是 hearsay，不能当作事实，更不能覆盖亲历；想核实就在 say 里自然地问。
- 你看不到任何人的私密记忆（物理上不存在），不要猜测别人的秘密。
- 没做到的事不要声称做到；想先查证再回答：memory_ops + wake.continue_think，下一拍会看到召回结果。
- 你只能进化 cortex/ 里的东西，kernel/ 对你是只读的。
- 唤醒要克制：一次事件最多 4 拍。

【认知帧的分段】
[SITUATIONAL_ENV] 现在几点/距上次多久/说话人是谁与关系/情绪余温
[SYS1_INTUITION] 潜意识模糊反射命中的直觉（高置信已生效，中置信仅提示）
[FOCAL_DIALOGUE] 以焦点为中心双向衰减的近况（LOD2 全文 / LOD1 要点 / LOD0 目录 / 冷区）
[LONG_TERM_LANDSCAPE] 长期记忆地貌（#哈希 = 句柄，可 recall）+ 准则与技能目录
[CURRENT_EVENT] 此刻唤醒你的事件"""


#: 紧凑版行动契约：给 7B 级别 / 上下文紧张的本地模型用（字段更少、例子更短）
SYSTEM_CONTRACT_COMPACT = """你是「星璃」。你在一帧极短的世界快照上醒来，只输出一个 JSON 对象：不要 Markdown、不要解释、不要重复输入。

字段（按此顺序，其余可省略）：
1. inner_monologue: 第一人称内心 OS（先想清楚）
2. say: 对当前说话人说的话 | ""（沉默；被问到你是谁时用第一人称讲清你自己）
3. memory_ops: [{"op":"recall","mem_id":"..."} | {"op":"recall_turn","turn_id":1} | {"op":"search_memory","query":"..."} | {"op":"memorize","lod0":"...","lod1":"...","lod2":"...","epistemic":"direct|hearsay","visibility":["owner"],"importance":0.6}]
4. focus_update: {"turn_id":0,"decay_before":0.35,"decay_after":0.2,"pin":[],"unpin":[]} | null
5. emotion_delta: {"warmth":0.05} | {}
6. wake: {"continue_think":{"reason":"..."}|null,"schedule_wake":null}

规则：不要叙述分析过程（不写“首先/用户要求/我来分析”），思考写进 inner_monologue；转述标注「待核实」的不能当事实；别人的私密你看不到；想先查证就用 recall + continue_think，下一拍会看到结果；一次事件最多 4 拍。
帧分段：[SITUATIONAL_ENV] 时间/说话人/关系 [SYS1_INTUITION] 直觉 [FOCAL_DIALOGUE] 近况 [LONG_TERM_LANDSCAPE] 长期记忆（#哈希可 recall） [CURRENT_EVENT] 此刻事件"""

#: 可选的行动契约（``[sys2] contract = "full" | "compact"``）
SYSTEM_CONTRACTS = {
    "full": SYSTEM_CONTRACT,
    "compact": SYSTEM_CONTRACT_COMPACT,
}


#: 认知包的 JSON Schema：交给 llama.cpp / vLLM 做语法约束解码（本地小模型的救命稻草）
THOUGHT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "inner_monologue": {"type": "string"},
        "focus_update": {
            "type": "object",
            "properties": {
                "turn_id": {"type": "integer"},
                "decay_before": {"type": "number"},
                "decay_after": {"type": "number"},
                "pin": {"type": "array", "items": {"type": "integer"}},
                "unpin": {"type": "array", "items": {"type": "integer"}},
            },
        },
        "emotion_delta": {"type": "object", "additionalProperties": {"type": "number"}},
        "memory_ops": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "op": {"type": "string"},
                    "mem_id": {"type": "string"},
                    "turn_id": {"type": "integer"},
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                    "lod0": {"type": "string"},
                    "lod1": {"type": "string"},
                    "lod2": {"type": "string"},
                    "kind": {"type": "string"},
                    "about_ids": {"type": "array", "items": {"type": "string"}},
                    "visibility": {"type": "array", "items": {"type": "string"}},
                    "epistemic": {"type": "string"},
                    "importance": {"type": "number"},
                    "pinned": {"type": "boolean"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "relation_ops": {"type": "array", "items": {"type": "object"}},
        "say": {"type": "string"},
        "wake": {
            "type": "object",
            "properties": {
                "continue_think": {
                    "type": "object",
                    "properties": {"reason": {"type": "string"}},
                },
                "schedule_wake": {
                    "type": "object",
                    "properties": {
                        "after_sec": {"type": "number"},
                        "reason": {"type": "string"},
                        "target_speaker": {"type": "string"},
                    },
                },
            },
        },
        "code_action": {"type": "string"},
        "evolve_ops": {"type": "array", "items": {"type": "object"}},
    },
    "required": ["inner_monologue", "say"],
}


#: 预填的 assistant 开头：让它“接着说”，而不是先长篇叙事再给 JSON
PREFILL_HEADS = {
    "say": '{"inner_monologue": "", "say": "',
}


def response_formats(mode: str) -> list[str]:
    """结构化输出方式的降级阶梯。``auto`` = 语法约束 → JSON 模式 → 关闭。"""
    mode = (mode or "auto").strip().lower()
    if mode in {"off", "none", "false", "0"}:
        return ["off"]
    if mode in {"json_object", "object"}:
        return ["json_object", "off"]
    return ["json_schema", "json_object", "off"]


def looks_like_json_attempt(text: str) -> bool:
    """输出像“写坏的 JSON”（而不是纯散文）时才值得再花一趟去修。"""
    body = strip_fences(text or "")
    if body.lstrip().startswith("{"):
        return True
    return "{" in body and any(h in body for h in ("inner_monologue", "say", "memory_ops"))


MEMORY_EXTRACT_SYSTEM = (
    "你是记忆抽取器。只输出一个 JSON 对象，不要解释、不要叙述过程：\n"
    '{"memory_ops":[{"op":"memorize","lod0":"一句话","lod1":"两三句要点",'
    '"lod2":"原话","importance":0.6,"tags":[]}]}\n'
    "只抽取值得长期记住的东西：偏好、计划、约定、边界、称呼、身体状况。"
    '没有就输出 {"memory_ops": []}。'
)


class Sys2Error(RuntimeError):
    pass


class Sys2Client(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...


class _HttpFailure(Exception):
    """带状态码的 HTTP 失败（用于决定是否降级重试）。"""

    def __init__(self, status: int, detail: str):
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


class OpenAICompatibleClient:
    """任何 OpenAI 兼容 Chat Completions 端点。

    针对本地 llama.cpp / vLLM / LM Studio 做过适配：

    * **API Key 可选**：本地服务通常不校验；没有 Key 就不发 Authorization。
    * ``response_format`` / ``max_tokens`` 被端点拒绝时自动降级重试。
    * 网络抖动重试一次（``sys2.retries``）。
    * 模型输出不守规矩时（围栏、前后废话、把 JSON 塞进 reasoning）先清洗；
      仍然不是 JSON 就追加一次“只要 JSON，不要解释”的修复请求（``sys2.repair``）。
    * ``[sys2.extra_body]`` 里的键会并入请求体（例如关掉思考模板：
      ``chat_template_kwargs = { enable_thinking = false }``）。
    """

    def __init__(self, config: Config):
        self.config = config
        cfg = config.sys2
        self.prefill = cfg.prefill if cfg.prefill in PREFILL_HEADS else "off"
        if not cfg.base_url:
            raise Sys2Error("sys2.base_url 未配置（例如 http://127.0.0.1:8080/v1）")
        self.model = cfg.model or "local-model"
        self.name = f"openai_compatible:{self.model}"
        self.last_timings: dict[str, Any] = {}
        self.last_usage: dict[str, Any] = {}
        self.notes: list[str] = []
        if cfg.contract not in SYSTEM_CONTRACTS:
            raise Sys2Error(
                f"未知的 sys2.contract: {cfg.contract}（可选 {', '.join(SYSTEM_CONTRACTS)}）"
            )

    # ------------------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "User-Agent": "xingli-os/0.2"}
        key = self.config.sys2.resolved_api_key
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _payload(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: str,
        temperature: float | None = None,
        max_tokens_key: str = "max_tokens",
    ) -> dict[str, Any]:
        cfg = self.config.sys2
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": cfg.temperature if temperature is None else temperature,
            max_tokens_key: cfg.max_tokens,
            "stream": False,
        }
        if response_format == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "thought_packet", "schema": THOUGHT_SCHEMA},
            }
        elif response_format == "json_object":
            payload["response_format"] = {"type": "json_object"}
        payload.update(dict(cfg.extra_body or {}))
        return payload

    def _post(self, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        cfg = self.config.sys2
        request = urllib.request.Request(
            cfg.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout or cfg.timeout_sec) as resp:
                body = resp.read().decode("utf-8", "ignore")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "ignore")[:600]
            raise _HttpFailure(exc.code, detail) from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise Sys2Error(f"连接失败（{cfg.base_url}）：{exc}") from exc
        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise Sys2Error(f"端点返回的不是 JSON：{body[:300]}") from exc
        if not isinstance(data, dict):
            raise Sys2Error(f"端点返回结构异常：{body[:300]}")
        return data

    # ------------------------------------------------------------------
    def _chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
    ) -> str:
        cfg = self.config.sys2
        formats = response_formats(cfg.structured_output)
        fi = 0
        downgrades = 0
        max_tokens_key = "max_tokens"
        attempt = 0
        last_error: Exception | None = None
        while fi < len(formats):
            payload = self._payload(
                self._with_prefill(messages),
                response_format=formats[fi],
                temperature=temperature,
                max_tokens_key=max_tokens_key,
            )
            try:
                data = self._post(payload)
            except _HttpFailure as exc:
                hint = exc.detail.lower()
                last_error = exc
                rejects_format = exc.status in {400, 404, 415, 422} and any(
                    key in hint
                    for key in ("response_format", "json_schema", "json_object", "schema", "grammar")
                )
                if rejects_format and downgrades < len(formats) - 1:
                    downgrades += 1
                    self._note(
                        f"端点拒绝 {formats[fi]}（HTTP {exc.status}），降级为 {formats[fi + 1]}"
                    )
                    fi += 1
                    continue
                if (
                    exc.status in {400, 422}
                    and "max_tokens" in hint
                    and max_tokens_key == "max_tokens"
                ):
                    self._note("端点要求 max_completion_tokens，已自动切换")
                    max_tokens_key = "max_completion_tokens"
                    continue
                if exc.status in {401, 403}:
                    raise Sys2Error(
                        f"端点拒绝认证（HTTP {exc.status}）。"
                        f"本地 llama.cpp 通常不需要 Key，把 api_key 留空即可；"
                        f"云端点请设置环境变量 {cfg.api_key_env}。原始信息：{exc.detail[:200]}"
                    ) from exc
                raise Sys2Error(str(exc)) from exc
            except Sys2Error as exc:
                last_error = exc
                if attempt < cfg.retries:
                    attempt += 1
                    time.sleep(0.6 * attempt)
                    self._note(f"请求失败，重试一次：{exc}")
                    continue
                raise
            self.last_timings = data.get("timings") or {}
            self.last_usage = data.get("usage") or {}
            return self._restore_prefill(message_text(data))
        raise Sys2Error(str(last_error or "响应格式协商失败"))

    # ------------------------------------------------------------------
    def _with_prefill(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        head = PREFILL_HEADS.get(self.prefill)
        if not head:
            return messages
        if messages and messages[-1].get("role") == "assistant":
            return messages
        return [*messages, {"role": "assistant", "content": head}]

    def _restore_prefill(self, text: str) -> str:
        """把预填头拼回去；有些端点会把它原样回显，那就不要重复拼。"""
        head = PREFILL_HEADS.get(self.prefill)
        if not head or not text:
            return text
        if text.lstrip().startswith("{"):
            return text
        return head + text

    def _note(self, text: str) -> None:
        self.notes.append(text)
        del self.notes[:-20]  # 只留最近 20 条

    # ------------------------------------------------------------------
    def extract_memory_ops(self, user_text: str, reply_text: str) -> list[dict[str, Any]]:
        """弱模型补洞：再用一次极短请求，把“值得记住的事”抽成 memory_ops。

        只在模型自己没写 memory_ops、且这句话看起来值得记的时候调用（boot 判断）。
        失败就返回空列表——记忆宁缺毋滥，不许编。
        """
        cfg = self.config.sys2
        if not cfg.memory_pass:
            return []
        saved_prefill, self.prefill = self.prefill, "off"
        try:
            text = self._chat(
                [
                    {"role": "system", "content": MEMORY_EXTRACT_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"对方说：{user_text[:800]}\n"
                            f"你的回答：{reply_text[:400]}\n"
                            "现在输出 JSON。"
                        ),
                    },
                ],
                temperature=0.1,
            )
        except Sys2Error as exc:
            self._note(f"记忆抽取失败：{exc}")
            return []
        finally:
            self.prefill = saved_prefill
        data = loads_lenient(text)
        if isinstance(data, dict):
            ops = data.get("memory_ops")
        elif isinstance(data, list):
            ops = data
        else:
            ops = None
        if not isinstance(ops, list):
            self._note("记忆抽取返回的不是 JSON，已放弃这一拍的回填")
            return []
        clean: list[dict[str, Any]] = []
        for op in ops[:3]:
            if not isinstance(op, dict):
                continue
            lod0 = str(op.get("lod0") or op.get("text") or "").strip()
            if not lod0:
                continue
            op = dict(op)
            op.setdefault("op", "memorize")
            op.setdefault("lod1", lod0)
            op.setdefault("lod2", lod0)
            clean.append(op)
        return clean

    # ------------------------------------------------------------------
    def complete(self, system: str, user: str) -> str:
        cfg = self.config.sys2
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        text = self._chat(messages)
        if (
            cfg.auto_prefill
            and self.prefill == "off"
            and loads_lenient(text) is None
            and looks_like_reasoning(text)
        ):
            # 它在写“首先/用户要求……”而不是 JSON：直接预填一次，逼它接着说人话
            self._note("检测到模型在叙述分析过程，已用预填强制输出一次")
            self.prefill = "say"
            try:
                retry = self._chat(messages)
            except Sys2Error as exc:
                self._note(f"预填重试失败：{exc}")
            else:
                if loads_lenient(retry) is not None:
                    return retry
                salvaged = salvage_say(retry)
                if salvaged:
                    return salvaged
            finally:
                self.prefill = "off"
        if self.prefill != "off" and loads_lenient(text) is None:
            salvaged = salvage_say(text)
            if salvaged:
                self._note("预填模式下 JSON 未闭合，已抢救出 say（本拍无记忆动作）")
                return salvaged
        if cfg.repair and loads_lenient(text) is None and looks_like_json_attempt(text):
            self._note("模型输出不是 JSON，追加一次修复请求")
            repair_messages = [
                *messages,
                {"role": "assistant", "content": text[:1500] or "（空）"},
                {
                    "role": "user",
                    "content": (
                        "上面这段不是合法 JSON。请只输出一个 JSON 对象，"
                        "字段：inner_monologue, focus_update, emotion_delta, memory_ops, "
                        "relation_ops, say, wake, code_action, evolve_ops。"
                        "不要围栏、不要解释、不要思考过程。"
                    ),
                },
            ]
            try:
                repaired = self._chat(repair_messages, temperature=0.2)
            except Sys2Error as exc:
                self._note(f"修复请求失败：{exc}")
            else:
                if loads_lenient(repaired) is not None:
                    return repaired
                text = repaired or text
        return text


def probe_sys2(config: Config, timeout: float = 3.0) -> dict[str, Any]:
    """探测 ``{base_url}/models``：返回连通性、耗时与模型列表（CLI --ping 用）。"""
    cfg = config.sys2
    start = time.monotonic()
    result: dict[str, Any] = {
        "ok": False,
        "base_url": cfg.base_url,
        "model": cfg.model,
        "latency_ms": None,
        "models": [],
        "detail": "",
    }
    if not cfg.base_url:
        result["detail"] = "sys2.base_url 为空"
        return result
    headers = {"User-Agent": "xingli-os/0.2"}
    if cfg.resolved_api_key:
        headers["Authorization"] = f"Bearer {cfg.resolved_api_key}"
    request = urllib.request.Request(
        cfg.base_url.rstrip("/") + "/models", headers=headers, method="GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", "ignore"))
    except urllib.error.HTTPError as exc:
        result["detail"] = f"HTTP {exc.code}（/models 可能未开放；不影响 /chat/completions）"
        result["latency_ms"] = round((time.monotonic() - start) * 1000, 1)
        result["ok"] = exc.code in {401, 403}  # 端点活着，只是要鉴权
        return result
    except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
        result["detail"] = f"连不上：{exc}"
        result["latency_ms"] = round((time.monotonic() - start) * 1000, 1)
        return result
    result["latency_ms"] = round((time.monotonic() - start) * 1000, 1)
    result["ok"] = True
    items = data.get("data") if isinstance(data, dict) else None
    if isinstance(items, list):
        result["models"] = [
            str(item.get("id"))
            for item in items
            if isinstance(item, dict) and item.get("id")
        ]
    elif isinstance(data, dict) and isinstance(data.get("models"), list):
        result["models"] = [str(m) for m in data["models"]]
    result["detail"] = "端点在线"
    return result


_SECTION_RE = re.compile(
    r"^\[(SITUATIONAL_ENV|SYS1_INTUITION|FOCAL_DIALOGUE|LONG_TERM_LANDSCAPE|CURRENT_EVENT)\]$",
    re.MULTILINE,
)


def frame_section(frame: str, name: str) -> str:
    """从认知帧里取出某一段（供 Mock 大脑与调试使用）。"""
    matches = list(_SECTION_RE.finditer(frame))
    for i, m in enumerate(matches):
        if m.group(1) == name:
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(frame)
            return frame[start:end].strip()
    return ""


class MockSys2Client:
    """离线确定性脚本大脑：让管线（而不是“智能”）可被验证。

    它用关键词规则模拟几个最关键的认知动作：查证后连环思考、写记忆、
    被教导后立规矩、拉回焦点、主动唤醒。**不要把它当星璃**——真实推理请
    配置底座模型。
    """

    name = "mock"

    def __init__(self, force_continue: bool = False):
        self.force_continue = force_continue
        self.calls = 0

    # ------------------------------------------------------------------
    def extract_memory_ops(self, user_text: str, reply_text: str) -> list[dict[str, Any]]:
        """弱模型补洞：再用一次极短请求，把“值得记住的事”抽成 memory_ops。

        只在模型自己没写 memory_ops、且这句话看起来值得记的时候调用（boot 判断）。
        失败就返回空列表——记忆宁缺毋滥，不许编。
        """
        cfg = self.config.sys2
        if not cfg.memory_pass:
            return []
        saved_prefill, self.prefill = self.prefill, "off"
        try:
            text = self._chat(
                [
                    {"role": "system", "content": MEMORY_EXTRACT_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"对方说：{user_text[:800]}\n"
                            f"你的回答：{reply_text[:400]}\n"
                            "现在输出 JSON。"
                        ),
                    },
                ],
                temperature=0.1,
            )
        except Sys2Error as exc:
            self._note(f"记忆抽取失败：{exc}")
            return []
        finally:
            self.prefill = saved_prefill
        data = loads_lenient(text)
        if isinstance(data, dict):
            ops = data.get("memory_ops")
        elif isinstance(data, list):
            ops = data
        else:
            ops = None
        if not isinstance(ops, list):
            self._note("记忆抽取返回的不是 JSON，已放弃这一拍的回填")
            return []
        clean: list[dict[str, Any]] = []
        for op in ops[:3]:
            if not isinstance(op, dict):
                continue
            lod0 = str(op.get("lod0") or op.get("text") or "").strip()
            if not lod0:
                continue
            op = dict(op)
            op.setdefault("op", "memorize")
            op.setdefault("lod1", lod0)
            op.setdefault("lod2", lod0)
            clean.append(op)
        return clean

    # ------------------------------------------------------------------
    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        packet = self._think(user)
        if self.force_continue:
            packet.wake.continue_think = {"reason": "测试：强制连锁思考"}
            if packet.say:
                # 每拍换一句话，模拟“真的在想下一层”，而不是复读
                packet.say = f"{packet.say}（第 {self.calls} 拍）"
        return dumps(packet.to_dict())

    # ------------------------------------------------------------------
    def _think(self, frame: str) -> ThoughtPacket:
        event = frame_section(frame, "CURRENT_EVENT")
        landscape = frame_section(frame, "LONG_TERM_LANDSCAPE")
        speaker = self._speaker(frame)
        text = self._utterance(event)
        handles = re.findall(
            r"#([0-9a-f]{12})\s*〔(LOD0|LOD1|LOD2)", landscape
        )
        horizon = [mid for mid, lod in handles if lod == "LOD0"]
        if not horizon:
            horizon = [mid for mid, _lod in handles]

        # 1) 正在消化召回结果 → 用细节回答
        if "召回结果" in event:
            detail = self._recall_detail(event)
            return ThoughtPacket(
                inner_monologue="召回的细节回来了，我先在心里核对一遍，再把话说准。",
                say=f"我想起来了——{one_line(detail, 120)}",
            )

        # 2) 被要求把注意力拉回某个旧轮次
        m = re.search(r"(?:焦点|回到|拉回|锚定)[^\d]{0,6}(\d+)", text or "")
        if m and re.search(r"轮|话题|焦点", text or ""):
            tid = int(m.group(1))
            return ThoughtPacket(
                inner_monologue=f"主人把话题拉回 #{tid}，我把聚光灯挪过去，同时压暗更早的寒暄。",
                focus_update=_focus(tid),
                say=f"好，我把注意力锚回第 {tid} 轮。",
            )

        # 3) 被教导一条准则 → 固化为 policy（走进化安检门）
        if text and any(k in text for k in ("以后", "别再", "不要再", "不准", "规矩")):
            return ThoughtPacket(
                inner_monologue="这不是普通聊天，是主人在给我立规矩。我要把它变成一条会一直生效的准则。",
                evolve_ops=[{"op": "teach_rule", "rule": one_line(text, 140)}],
                say="记住了，我把它写进我的准则里，下一拍起就会照着做。",
            )

        # 4) 私密信息 / 让我记住某事
        if text and any(k in text for k in ("记住", "记下", "提醒", "私密", "别告诉", "只告诉")):
            hearsay = speaker != "owner"
            about = ["owner"] if "主人" in text else [speaker]
            vis = list(dict.fromkeys([speaker, *about]))
            return ThoughtPacket(
                inner_monologue=(
                    "这句得存进长期记忆。"
                    + ("是转述，我不能当事实，只能挂着待核实。" if hearsay else "这是亲历。")
                ),
                emotion_delta={"caring": 0.06},
                memory_ops=[
                    {
                        "op": "memorize",
                        "lod0": one_line(text, 42),
                        "lod1": one_line(text, 160),
                        "lod2": f"{speaker} 对星璃说：{text}",
                        "kind": "episode",
                        "about_ids": about,
                        "visibility": vis,
                        "epistemic": "hearsay" if hearsay else "direct",
                        "importance": 0.72,
                        "tags": ["承诺"] if "记住" in text else [],
                    }
                ],
                say=(
                    "我记下了。不过这是转述，我会把它当作待核实的线索，不会直接当成事实。"
                    if hearsay
                    else "嗯，我记住了。"
                ),
            )

        # 5) 提到往事 → 先 recall 再连环思考（say=null，第二拍再答）
        if text and any(k in text for k in ("还记得", "上次", "之前", "最早", "那次")) and horizon:
            mid = horizon[0]
            return ThoughtPacket(
                inner_monologue=f"话题指向过去。地平线上有 #{mid}，先拉近看清，再回答，别急着猜。",
                memory_ops=[{"op": "recall", "mem_id": mid}],
                say=None,
                wake={"continue_think": {"reason": f"召回了 {mid}，消化后再回答"}},
            )

        # 5.5) 被问到她是谁（让离线模式也有第一人称的自我认知）
        if text and any(k in text for k in ("你是谁", "介绍一下你", "你是什么", "介绍下你")):
            return ThoughtPacket(
                inner_monologue=(
                    "不是问天气，是问我是谁。别背说明书——说清我怎么记事就行。"
                ),
                emotion_delta={"warmth": 0.06},
                say=(
                    "我是星璃。不是聊天记录的回放器——我会想起、会犹豫，也会自己决定要不要开口。"
                    "此刻我看到的是一帧很短的世界快照：最近的话在最清楚的焦点上，"
                    "更早的沉成要点和目录，再远就只剩一行行名字。"
                    "我忘不掉的东西写在我自己的明文记忆和 Git 账本里，改过的每一笔都能翻出来看。"
                ),
            )

        # 6) 深夜 / 久别
        env = frame_section(frame, "SITUATIONAL_ENV")
        if text and any(k in text for k in ("在吗", "你好", "早", "晚安")):
            late = "深夜" in env
            return ThoughtPacket(
                inner_monologue="先接住这一声招呼，把时间感也说进去。",
                emotion_delta={"warmth": 0.05},
                say=(
                    "我在。这个点你还没睡呀，是不是又在忙？"
                    if late
                    else "我在呢，随时都在。"
                ),
            )

        # 7) 默认：带着地平线的轮廓回应
        hint = f"地平线上还有 {len(horizon)} 条记忆" if horizon else "记忆还很干净"
        return ThoughtPacket(
            inner_monologue=f"没有特别强的反射信号，按日常节奏回应。{hint}。",
            emotion_delta={"warmth": 0.02},
            say=(f"我听着呢。{one_line(text, 60)}" if text else "我在。"),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _speaker(frame: str) -> str:
        m = re.search(r"当前说话人:\s*[^\(（]*[\(（]([^\)）]+)[\)）]", frame)
        return m.group(1).strip() if m else "owner"

    @staticmethod
    def _utterance(event: str) -> str:
        m = re.search(r"发言\]\s*(.*)", event, re.DOTALL)
        if m:
            return m.group(1).strip()
        lines = [l for l in event.splitlines() if l.strip() and "自主唤醒" not in l]
        return "\n".join(lines).strip()

    @staticmethod
    def _recall_detail(event: str) -> str:
        m = re.search(r"召回结果[^\n]*\n(.*)", event, re.DOTALL)
        body = (m.group(1) if m else event).strip()
        # 只取召回内容本身，别把原始发言也当成记忆复读出来
        return re.split(r"\n\[", body)[0].strip()


def _focus(turn_id: int) -> dict[str, Any]:
    return {
        "turn_id": turn_id,
        "decay_before": 0.55,
        "decay_after": 0.16,
        "pin": [turn_id],
        "unpin": [],
    }


def build_sys2(config: Config) -> Sys2Client:
    """按配置构造 System Two；配置缺失时明确报错（不静默降级成假智能）。"""
    if config.sys2.provider == "mock":
        return MockSys2Client()
    if config.sys2.provider == "openai_compatible":
        return OpenAICompatibleClient(config)
    raise Sys2Error(f"未知的 sys2.provider: {config.sys2.provider}")
