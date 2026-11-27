"""纯标准库文本工具：Token 估算、CJK 友好的相似度、抽取式摘要。

为什么不用向量库：M1 的目标是让“视距渲染”的数学与预算先跑通。语义匹配
在 MVP 里用字符二元组（对中文天然友好）近似，接口保持可替换——将来换成
嵌入模型只需要换掉 :func:`similarity` 的实现，不影响其他模块。
"""

from __future__ import annotations

import math
import re
from collections import Counter

_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff]")
_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;\n])|(?<=\.)\s+")
_WORD = re.compile(r"[A-Za-z0-9_]+")
_NUMBER = re.compile(r"\d+(?:[.,:：/-]\d+)*")
_PUNCT = re.compile(r"[\s\u3000，。！？；：、“”‘’（）《》【】\[\]{}<>~`!@#$%^&*()_+\-=|\\/.,:;'\"]")


def estimate_tokens(text: str) -> int:
    """粗略但稳定的 Token 估算。

    CJK 字符按 1 token/字，其余按 3.6 字符/token。对预算控制来说，
    “稳定 + 偏保守”比“绝对精确”更重要：宁可多留余量。
    """
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    other = max(0, len(text) - cjk)
    return cjk + math.ceil(other / 3.6)


def truncate(text: str, max_chars: int, marker: str = "…（截断）") -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars <= len(marker):
        return text[:max_chars]
    return text[: max_chars - len(marker)] + marker


def normalize(text: str) -> str:
    return _PUNCT.sub("", (text or "").lower())


def bigrams(text: str) -> Counter:
    """字符二元组（跨语言通用）。对中文比空格分词靠谱得多。"""
    s = normalize(text)
    if len(s) < 2:
        return Counter([s]) if s else Counter()
    return Counter(s[i : i + 2] for i in range(len(s) - 1))


def similarity(a: str, b: str) -> float:
    """0~1 的相似度：二元组余弦（不用外部依赖）。"""
    if not a or not b:
        return 0.0
    ba, bb = bigrams(a), bigrams(b)
    if not ba or not bb:
        return 0.0
    common = set(ba) & set(bb)
    num = sum(ba[k] * bb[k] for k in common)
    den = math.sqrt(sum(v * v for v in ba.values()) * sum(v * v for v in bb.values()))
    return num / den if den else 0.0


def sentences(text: str) -> list[str]:
    raw = [s.strip() for s in _SENT_SPLIT.split(text or "") if s and s.strip()]
    return raw or ([text.strip()] if text and text.strip() else [])


def key_tokens(text: str) -> set[str]:
    """抽取“事实性”标记：数字、拉丁词、以及中文双字词片段。

    用于反失忆抽查：这些标记代表一条记忆里“必须能被回忆起来”的骨架。
    """
    t = text or ""
    out: set[str] = set(_NUMBER.findall(t))
    out.update(w.lower() for w in _WORD.findall(t) if len(w) >= 3)
    for seg in re.findall(r"[\u4e00-\u9fff]{2,}", t):
        for i in range(len(seg) - 1):
            out.add(seg[i : i + 2])
    return out


def coverage(view_text: str, source_text: str, limit: int = 60) -> float:
    """视图（LOD0+LOD1）对原文关键事实的覆盖率，0~1。

    这就是 README 里“反失忆抽查通过率”的微观定义：在当前视距下，
    一条记忆的事实骨架还能被召回多少。
    """
    keys = key_tokens(source_text)
    if not keys:
        return 1.0
    if len(keys) > limit:
        # 取“最不常见”的片段作为代表：长数字与拉丁词优先，其次中文片段
        keys = set(sorted(keys, key=lambda k: (-len(k), k))[:limit])
    view = normalize(view_text)
    hit = sum(1 for k in keys if normalize(k) in view)
    return hit / len(keys)


def extractive_summary(text: str, max_chars: int = 220) -> str:
    """抽取式摘要：选关键事实密度最高的句子，按原文顺序拼接。

    没有任何 LLM 依赖，因此睡眠周期的反失忆抽查可以离线、可复现地跑。
    """
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    sents = sentences(text)
    if not sents:
        return truncate(text, max_chars)
    scored: list[tuple[float, int, str]] = []
    total = Counter()
    for s in sents:
        total.update(key_tokens(s))
    for idx, s in enumerate(sents):
        toks = key_tokens(s)
        # 事实密度 = 关键标记数 / 长度；略偏爱靠前的句子
        density = len(toks) / (len(s) + 12)
        recency_bonus = 0.15 * (1.0 - idx / max(1, len(sents)))
        scored.append((density + recency_bonus, idx, s))
    scored.sort(key=lambda x: (-x[0], x[1]))
    chosen: list[tuple[int, str]] = []
    used = 0
    for _, idx, s in scored:
        if used + len(s) > max_chars and chosen:
            continue
        chosen.append((idx, s))
        used += len(s)
    chosen.sort(key=lambda x: x[0])
    out = "".join(s for _, s in chosen)
    return truncate(out, max_chars) if len(out) > max_chars else out


def one_line(text: str, max_chars: int = 60) -> str:
    """LOD0 一句话锚点：压掉换行与多余空白。"""
    flat = re.sub(r"\s+", " ", (text or "").strip())
    return truncate(flat, max_chars, marker="…")


def squash(text: str, max_chars: int) -> str:
    return one_line(text, max_chars)
