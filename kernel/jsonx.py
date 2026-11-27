"""模型输出的宽容解析工具。

真实端点（尤其是本地 llama.cpp 上的小模型）不会永远守规矩：它们会加
Markdown 围栏、在 JSON 前后补一句人话、留下尾随逗号，或者把 JSON 塞进
``reasoning_content`` 而让 ``content`` 为空。

这里的函数只做“**取回她写的那一个 JSON 对象**”，不做任何语义修补；
真正解析失败时仍然返回原文，让内核如实记录错误（绝不半执行）。
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

_FENCE = re.compile(r"```[ \t]*(?:python|py|json|JSON)?[ \t]*\r?\n?(.*?)```", re.DOTALL)
_OPEN_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*", re.MULTILINE)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")

#: 认知包的特征键：用于从“废话 + JSON”里认出正确的那一段
PACKET_HINTS: tuple[str, ...] = (
    "inner_monologue",
    "memory_ops",
    "focus_update",
    "emotion_delta",
    "relation_ops",
    "say",
)


def strip_fences(text: str) -> str:
    """去掉 Markdown 围栏；没有围栏时原样返回（去首尾空白）。"""
    if not text:
        return ""
    m = _FENCE.search(text)
    if m:
        return m.group(1).strip()
    if "```" in text:
        return _OPEN_FENCE.sub("", text).replace("```", "").strip()
    return text.strip()


def find_json_object(text: str) -> str | None:
    """从任意文本里截出第一个括号配平的 ``{...}``（感知字符串与转义）。"""
    search_from = 0
    while True:
        start = text.find("{", search_from)
        if start < 0:
            return None
        depth = 0
        in_str = False
        quote = ""
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == quote:
                    in_str = False
                continue
            if ch in {'"', "'"}:
                in_str = True
                quote = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        search_from = start + 1


def loads_lenient(text: str) -> Any | None:
    """尽量解析 JSON；允许尾随逗号。失败返回 ``None``（不抛异常）。"""
    if not text:
        return None
    for candidate in (text, _TRAILING_COMMA.sub(r"\1", text)):
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def clean_model_output(text: str, hints: Iterable[str] = PACKET_HINTS) -> str:
    """围栏 → 整段 JSON？→ 内部的对象？→ 原样返回。"""
    body = strip_fences(text or "")
    if loads_lenient(body) is not None:
        return body
    candidate = find_json_object(body)
    if candidate is not None and loads_lenient(candidate) is not None:
        if not hints or any(h in candidate for h in hints):
            return candidate
    return body


def message_text(payload: dict[str, Any], hints: Iterable[str] = PACKET_HINTS) -> str:
    """从 OpenAI 兼容响应里取出可用的正文（兼容 reasoning_content / text 通道）。"""
    choices = payload.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    choice = choices[0]
    message = choice.get("message") or {}
    candidates: list[str] = []
    for key in ("content", "reasoning_content", "reasoning", "refusal"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value)
    if isinstance(choice.get("text"), str):
        candidates.append(choice["text"])
    cleaned = [clean_model_output(c, hints) for c in candidates]
    for text in cleaned:
        if loads_lenient(text) is not None:
            return text
    return cleaned[0] if cleaned else ""
