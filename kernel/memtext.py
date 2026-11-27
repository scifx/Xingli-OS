"""记忆文本的“同一件事”判定：去重、升级、以及脏数据识别。

真实 4B 端点给我们上了一课：它会写出

* ``lod0="一句话"`` / ``lod1="两三句要点"`` —— 把 schema 里的示例值当内容抄回来；
* ``lod1="0.85"`` / ``lod2="0.7"`` —— 把 LOD 当成“等级分数”；
* ``lod0="主人亲密度0.85，信任值1.00"`` —— 把自己的社交状态回显成“记忆”；
* 同一件事的七八种说法：``主人偏好：喜欢菠萝和芒果``、
  ``owner喜欢食用菠萝和芒果，偏好清甜的果香口感``、``主人喜欢菠萝和芒果``……

这些都不是记忆。放在这里而不是散在 `boot.py` 里，是因为“写入时该拒”和
“存量该清洗”必须用**同一套判据**，否则修完新数据、旧数据还在地上滚。
"""

from __future__ import annotations

import re

#: 模型把 schema 里的示例值当成内容抄回来的典型痕迹
PLACEHOLDER_LOD = {
    "一句话", "两三句要点", "原话", "...", "…", "一句话事实", "要点", "摘要",
    "无", "null", "none", "n/a", "text", "string", "内容", "具体内容",
}

#: 这不是记忆，是她把自己的社交状态回显了一遍
STATE_ECHO = re.compile(
    r"^(?:主人|owner|对方)?(?:的)?(?:亲密度|信任值|信任|关系|情绪)[^\n]{0,20}\d"
)

#: 同一事实的不同说法：这些词不携带事实，比对时抹掉
PREDICATE_WORDS = re.compile(
    "主人|owner|对方|我|的|了|是|叫|名字|称呼|喜欢|偏爱|偏好|钟爱|爱吃|爱|"
    "和|与|、|跟|同|口味|水果|吃点|吃|食用|表示|提到|明确|确认|记入|记事|自身"
)

_TIME_PREFIX = re.compile(r"^\s*\d{4}[-/]\d{1,2}[-/]\d{1,2}[ T]?[\d:：]{0,8}\s*")
_PUNCT = re.compile(r"[\s，。、：；！？,.:;!?（）()\[\]【】…—\-~\"'“”‘’]")


def looks_like_placeholder(text: str) -> bool:
    """占位符 / 纯数字 / 纯类型名 —— 一律不是事实。"""
    bare = (text or "").strip().strip("。.，,").lower()
    if not bare:
        return True
    if bare in PLACEHOLDER_LOD:
        return True
    if re.fullmatch(r"[\d\s.,%:：\-+]+", bare):
        return True
    return bool(re.fullmatch(r"(?:epistemic|confidence|importance)\W*\w+", bare))


def looks_like_state_echo(text: str) -> bool:
    return bool(STATE_ECHO.match((text or "").strip()))


def mem_key(text: str) -> str:
    """去重用的骨架：抹掉时间前缀、谓词虚词、标点、同义换词。

    ``2026-11-27 20:21 主人偏好：喜欢菠萝和芒果`` 与 ``主人喜欢菠萝和芒果``
    在事实层面是同一句，骨架都应该是 ``菠萝芒果``。
    """
    text = _TIME_PREFIX.sub("", text or "")
    text = PREDICATE_WORDS.sub("", text)
    text = _PUNCT.sub("", text)
    return text.replace("凤梨", "菠萝").lower()


def chars(text: str) -> set[str]:
    return set(mem_key(text))


_POSITIVE = ("喜欢", "偏爱", "偏好", "钟爱", "爱吃", "爱吃")
_NEGATIVE = ("不喜欢", "讨厌", "受不了", "不爱", "不吃", "戒了")


def polarity(text: str) -> int:
    """+1 喜欢 / -1 不喜欢 / 0 中性。相反极性永远不算“同一件事”。"""
    text = text or ""
    if any(word in text for word in _NEGATIVE):
        return -1
    if any(word in text for word in _POSITIVE):
        return 1
    return 0


def same_fact(a: str, b: str) -> bool:
    """两条记忆是不是在说同一件事（不做语义理解，只做骨架重合）。"""
    ka, kb = mem_key(a), mem_key(b)
    if not ka or not kb:
        return False
    if ka == kb:
        return True
    pa, pb = polarity(a), polarity(b)
    if pa * pb < 0:
        return False  # 喜欢 vs 不喜欢，绝不能合并
    ca, cb = set(ka), set(kb)
    jaccard = len(ca & cb) / max(1, len(ca | cb))
    if jaccard >= 0.8:
        return True
    # 一方是另一方的子集：同一件事，只是其中一边多写了细节
    small, big = (ca, cb) if len(ca) <= len(cb) else (cb, ca)
    if len(small) >= 2 and small <= big:
        return True
    return jaccard >= 0.6 and min(len(ca), len(cb)) >= 3


def is_richer(new_text: str, old_text: str, *, slack: int = 4) -> bool:
    """新写法是否比旧写法更完整（同一事实，但多写了细节）。"""
    new_key, old_key = mem_key(new_text), mem_key(old_text)
    if not old_key or not new_key:
        return False
    if len(new_key) < len(old_key) + slack:
        return False
    return set(old_key) <= set(new_key)
