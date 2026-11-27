"""显式事实抽取：从主人这句话里，直接读出“可纠正的事实”。

为什么放在内核而不是 Sys2：4B 级模型经常把“我叫什么”读反（把她的名字当成主人的名字），
而**称呼、偏好、生日**这类事实一旦记错，会污染后面所有召回。所以这里用确定性规则抓：

    我(的)名字是 X / 我叫 X / 以后叫我 X      → 对方的称呼
    主人叫 X                                 → 对方的称呼（次优先）
    你的名字是 X                             → 她的名字（只记录，不擅自改人格）
    我喜欢 / 我爱 / 我讨厌 / 我对 X 过敏       → 偏好与禁忌
    我的生日是 X                             → 生日

规则只做**抽取**，不写库：写不写、怎么写、要不要作废旧记忆，由 ``Heartbeat`` 决定
（隐私与信源规则仍在那里统一执行）。抽取失败就是没抽到，绝不猜。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

#: 这些词出现在“名字”位置时，说明她在提问或指代，不是事实
_NAME_STOPWORDS = {
    "什么", "啥", "谁", "哪里", "哪儿", "哪个", "你", "我", "他", "她", "自己",
    "星璃", "xingli", "名字", "谁呀", "什么呢",
}

#: 句尾语气词，抽名字 / 偏好时去掉
_TAIL = " \t。，,.!?！？；;：:、\"'“”‘’）)】]"

_NAME_PATTERNS = [
    # 我(的)名字(是|叫) X ／ 我叫 X
    re.compile(r"我(?:的)?名字(?:叫做|叫|是)?\s*[:：]?\s*([^\s，。,.!?！？、；;:：]{1,16})"),
    re.compile(r"我(?:就)?叫\s*([^\s，。,.!?！？、；;:：]{1,16})"),
    re.compile(r"(?:以后|之后|从今以后|请|你可以|你就)?叫我\s*([^\s，。,.!?！？、；;:：]{1,16})"),
    # 你的主人叫 X ／ 主人的名字是 X
    re.compile(r"主人(?:的)?名字(?:叫做|叫|是)?\s*[:：]?\s*([^\s，。,.!?！？、；;:：]{1,16})"),
    re.compile(r"(?:你的)?主人(?:就)?叫\s*([^\s，。,.!?！？、；;:：]{1,16})"),
    # English
    re.compile(r"my name is\s+([A-Za-z][A-Za-z0-9_\-]{0,15})", re.IGNORECASE),
    re.compile(r"call me\s+([A-Za-z][A-Za-z0-9_\-]{0,15})", re.IGNORECASE),
]

_SELF_NAME_PATTERNS = [
    re.compile(r"你(?:的)?名字(?:叫做|叫|是|应该是|才叫)?\s*[:：]?\s*([^\s，。,.!?！？、；;:：]{1,16})"),
    re.compile(r"你(?:才)?叫\s*([^\s，。,.!?！？、；;:：]{1,16})"),
]

_LIKE_PATTERNS = [
    re.compile(r"我(?:很|最|特别|超|比较|就)?(?:喜欢|爱吃|爱|偏爱|钟爱|偏好)\s*([^。，,.!?！？；;\n]{1,40})"),
    re.compile(r"我(?:很|最|特别|超)?(?:讨厌|不喜欢|受不了|不爱|不吃)\s*([^。，,.!?！？；;\n]{1,40})"),
    re.compile(r"我对\s*([^。，,.!?！？；;\s]{1,20})\s*过敏"),
]
_BIRTHDAY_PATTERNS = [
    re.compile(r"我(?:的)?生日(?:是|在|就是)\s*([^。，,.!?！？；;\n]{1,24})"),
]

#: 主人明确在纠错时说这些词（用来决定“旧的要不要作废”）
CORRECTION_CUES = ("说错", "不对", "纠正", "不是叫", "记错", "搞错", "更正", "改口", "应该说")


@dataclass
class Claim:
    kind: str  # name | self_name | like | dislike | allergy | birthday
    value: str
    raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "value": self.value, "raw": self.raw}


_QUESTION_TAILS = ("？", "?", "吗", "呢", "么")


def _clean(value: str) -> str:
    value = value.strip().strip(_TAIL)
    for filler in ("才是", "才叫", "才是叫", "是", "叫"):
        if value.startswith(filler) and len(value) > len(filler) + 1:
            value = value[len(filler) :].strip()
    for tail in ("的", "了", "呀", "啊", "吧", "呢", "哦", "嘛"):
        if value.endswith(tail) and len(value) > 2:
            value = value[: -len(tail)].strip()
    return value


def _plausible_name(value: str, *, strict: bool = True) -> bool:
    """strict=True 用于别人的名字；她的名字允许叫“星璃”。"""
    if not 1 <= len(value) <= 16:
        return False
    if value.lower() in {"什么", "啥", "谁", "哪里", "哪儿", "哪个", "名字", "谁呀", "什么呢"}:
        return False
    if strict and (value.lower() in _NAME_STOPWORDS or value in _NAME_STOPWORDS):
        return False
    if value.endswith("来着") or any(value.startswith(w) for w in ("什么", "啥", "谁", "哪")):
        return False
    # 纯标点 / 纯语气词不算
    return bool(re.search(r"[A-Za-z\u4e00-\u9fa5]", value))


_SEGMENT_SPLIT = re.compile(r"[。！!\n]+")
_QUESTION_HINT = re.compile(r"[？?]|(?:吗|呢|什么|啥|谁|哪)[，,。\s]*$")


def _declarative_segments(text: str) -> list[str]:
    """只保留陈述句：问句里的“我叫什么”不是事实。"""
    out: list[str] = []
    for segment in _SEGMENT_SPLIT.split(text):
        segment = segment.strip()
        if not segment or _QUESTION_HINT.search(segment):
            continue
        out.append(segment)
    return out


def extract_claims(text: str) -> list[Claim]:
    """从一句话里抽出显式事实。按“名字 > 别名”去重，保持首次出现顺序。"""
    text = (text or "").strip()
    if not text:
        return []
    claims: list[Claim] = []
    seen: set[tuple[str, str]] = set()

    def add(kind: str, value: str) -> None:
        value = _clean(value)
        if kind in {"like", "dislike"} and value[:1] in {"吃", "喝"} and len(value) > 2:
            value = value[1:]
        if not value:
            return
        key = (kind, value.lower())
        if key in seen:
            return
        seen.add(key)
        claims.append(Claim(kind=kind, value=value, raw=text))

    def add_name(kind: str, value: str) -> None:
        strict = kind != "self_name"
        if _plausible_name(_clean(value), strict=strict):
            add(kind, value)

    segments = _declarative_segments(text)
    for segment in segments:
        for pattern in _SELF_NAME_PATTERNS:
            for match in pattern.finditer(segment):
                add_name("self_name", match.group(1))
        for pattern in _NAME_PATTERNS:
            for match in pattern.finditer(segment):
                add_name("name", match.group(1))

    for index, pattern in enumerate(_LIKE_PATTERNS):
        kind = "like" if index == 0 else ("dislike" if index == 1 else "allergy")
        for match in pattern.finditer(text):
            value = _clean(match.group(1))
            if len(value) <= 2 and value.lstrip("吃").strip() in {"你", "我", "他", "她", "它"}:
                continue
            if value in {"你", "我", "他", "她", "它", "你们", "我们"}:
                continue
            add(kind, value)

    for pattern in _BIRTHDAY_PATTERNS:
        for match in pattern.finditer(text):
            add("birthday", match.group(1))

    # “你的主人叫星璃”这类把她的名字安到主人头上的句子：名字claim 若与 self_name 相同，丢掉
    self_names = {c.value.lower() for c in claims if c.kind == "self_name"}
    return [
        c
        for c in claims
        if not (c.kind == "name" and c.value.lower() in self_names)
    ]


def is_correction(text: str) -> bool:
    text = text or ""
    return any(cue in text for cue in CORRECTION_CUES)
