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
import urllib.error
import urllib.request
from typing import Any, Protocol

from kernel.config import Config
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
6. say: "对当前说话人说的话" | {"to":"人物id","text":"..."} | null（null = 这一拍沉默/继续思考）
7. wake: {"continue_think":{"reason":"..."}|null, "schedule_wake":{"after_sec":600,"reason":"...","target_speaker":"owner"}|null}
8. code_action: 可选受控 Python 片段 | null
9. evolve_ops: [{"op":"teach_rule","rule":"..."} | {"op":"propose_reflex", ...} | {"op":"propose_skill","name":"...","code":"...","reason":"..."}]

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


class Sys2Error(RuntimeError):
    pass


class Sys2Client(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...


class OpenAICompatibleClient:
    """OpenAI 兼容 Chat Completions（默认 JSON 模式）。"""

    def __init__(self, config: Config):
        self.config = config
        self.name = "openai_compatible"
        if not config.sys2.base_url:
            raise Sys2Error("sys2.base_url 未配置")
        if not config.sys2.resolved_api_key:
            raise Sys2Error(
                f"缺少 System Two API Key（环境变量 {config.sys2.api_key_env}）"
            )

    def complete(self, system: str, user: str) -> str:
        cfg = self.config.sys2
        payload: dict[str, Any] = {
            "model": cfg.model or "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
        }
        if cfg.json_mode:
            payload["response_format"] = {"type": "json_object"}
        request = urllib.request.Request(
            cfg.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {cfg.resolved_api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=cfg.timeout_sec) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "ignore")[:400]
            raise Sys2Error(f"HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise Sys2Error(f"调用失败: {exc}") from exc
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise Sys2Error(f"返回结构异常: {data}") from exc


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
    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        packet = self._think(user)
        if self.force_continue:
            packet.wake.continue_think = {"reason": "测试：强制连锁思考"}
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
    """按配置构造 System Two；缺少 Key 时明确报错（不静默降级成假智能）。"""
    if config.sys2.provider == "mock":
        return MockSys2Client()
    if config.sys2.provider == "openai_compatible":
        return OpenAICompatibleClient(config)
    raise Sys2Error(f"未知的 sys2.provider: {config.sys2.provider}")
