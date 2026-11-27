"""M1 验收：代码即思考、结构化认知包、连环思考熔断、自主唤醒。"""

from __future__ import annotations

import json
import time

import pytest

from kernel.boot import Heartbeat
from kernel.config import Config, Sys2Config
from kernel.runtime_exec import audit_code, parse_thought_output
from kernel.scheduler import ProactiveBudgetExceeded, Scheduler
from kernel.sys2_llm import MockSys2Client, build_sys2


class JsonBrain:
    """返回固定 JSON 认知包（验证声明式通道）。"""

    name = "json_brain"

    def __init__(self, payload):
        self.payload = payload

    def complete(self, system, user):
        payload = dict(self.payload)
        payload.setdefault("inner_monologue", "测试用内心 OS")
        return json.dumps(payload, ensure_ascii=False)


class CodeBrain:
    """返回纯 Python 代码（验证代码即思考通道）。"""

    name = "code_brain"

    def __init__(self, code):
        self.code = code

    def complete(self, system, user):
        return self.code


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------
def test_json_packet_roundtrip_and_field_order():
    payload = {
        "inner_monologue": "先想清楚再说话",
        "emotion_delta": {"caring": 0.1},
        "say": "嗯",
        "wake": {"continue_think": {"reason": "还想再想一步"}},
    }
    packet, mode = parse_thought_output(json.dumps(payload))
    assert mode == "json"
    assert packet.inner_monologue == "先想清楚再说话"
    assert packet.say_text() == "嗯"
    assert packet.wake.continue_think["reason"] == "还想再想一步"
    out = packet.to_dict()
    assert list(out)[0] == "inner_monologue", "inner_monologue 必须是第一个字段"


def test_code_mode_extracts_monologue_and_primitives():
    code = """
# 主人今晚的语气有点沉，先把情绪放软一点。
feel(caring=0.12)
say("别急着一个人扛。")
memorize(lod0="主人今晚情绪偏低", lod1="他说没事但语气很沉", lod2="完整记录")
continue_think("先确认他是不是真的要聊")
"""
    packet, mode = parse_thought_output(code, "owner")
    assert mode == "code"
    assert "语气有点沉" in packet.inner_monologue
    assert packet.say_text() == "别急着一个人扛。"
    assert packet.emotion_delta["caring"] == pytest.approx(0.12)
    assert packet.memory_ops[0]["op"] == "memorize"
    assert packet.wake.continue_think is not None


@pytest.mark.parametrize(
    "code",
    [
        "import os\nsay('hi')",
        "say(open('/etc/passwd').read())",
        "eval('1+1')",
        "__import__('os').system('ls')",
        "(1).__class__.__bases__",
    ],
)
def test_code_audit_rejects_dangerous_calls(code):
    assert audit_code(code), f"应当被拒绝: {code}"
    packet, mode = parse_thought_output(code)
    assert packet.say is None
    assert packet.parse_warnings


def test_infinite_loop_is_interrupted():
    code = "for i in range(10**9):\n    x = i\n"
    start = time.monotonic()
    packet, mode = parse_thought_output(code)
    elapsed = time.monotonic() - start
    assert elapsed < 10
    assert any("预算" in w for w in packet.parse_warnings)


def test_syntax_error_is_reported_not_executed():
    packet, mode = parse_thought_output("say('hi'")
    assert packet.say is None
    assert any("语法错误" in w for w in packet.parse_warnings)


# ---------------------------------------------------------------------------
# 代码即思考的端到端执行
# ---------------------------------------------------------------------------
def test_code_brain_can_write_memory_and_reply(ws):
    code = """
# 先把这份承诺记下来，再说一句。
memorize(lod0="主人要求以后先查证再回答", lod1="原话：以后先查证", lod2="完整对话", importance=0.8, tags=["policy"])
say("好，我先查再说。")
"""
    hb = Heartbeat.create(ws, ws.config, sys2=CodeBrain(code))
    result = hb.handle_message("owner", "以后先查证再回答我")
    assert result.says == ["好，我先查再说。"]
    nodes = hb.store.list_nodes("owner")
    assert any("先查证" in n.lod0 for n in nodes)


# ---------------------------------------------------------------------------
# 连环思考 + 硬熔断
# ---------------------------------------------------------------------------
def test_chain_thinking_is_hard_capped(ws):
    cfg = ws.config.with_budget(max_chain_ticks=3)
    hb = Heartbeat.create(ws, cfg, sys2=MockSys2Client(force_continue=True))
    result = hb.handle_message("owner", "你在吗")
    assert len(result.ticks) == 3
    assert "MAX_CHAIN_TICKS" in result.halted_reason


def test_recall_then_answer_in_two_ticks(hb):
    node = hb.store.put(
        lod0="最早讨论过的哈希方案",
        lod1="用 sha256 内容寻址 + 游戏 LOD 视距来管理记忆",
        lod2="最早我们讨论的是 sha256 内容寻址 + 游戏 LOD 视距来管理长期记忆，避免上下文爆炸。",
        source_id="owner",
        importance=0.9,
    )
    result = hb.handle_message("owner", "你还记得我们上次聊的哈希方案吗？")
    assert len(result.ticks) == 2, [t.summary() for t in result.ticks]
    first, second = result.ticks
    assert first.actions.recalls and first.actions.recalls[0]["id"] == node.id
    assert first.actions.says == [], "第一拍应该先查证，不急着回答"
    assert "召回结果" in second.frame_text
    assert "sha256" in (second.actions.says[0] if second.actions.says else "")


def test_recall_of_missing_memory_is_reported(hb):
    payload = {
        "inner_monologue": "试着召回一个不存在的记忆",
        "memory_ops": [{"op": "recall", "mem_id": "deadbeefcafe"}],
        "say": "嗯",
    }
    hb2 = Heartbeat.create(hb.ws, hb.ws.config, sys2=JsonBrain(payload), now_fn=hb.now_fn)
    result = hb2.handle_message("owner", "查一下那个记忆")
    assert any("不存在" in e for e in result.ticks[0].actions.errors)


# ---------------------------------------------------------------------------
# 随心所动：即时 / 定时唤醒
# ---------------------------------------------------------------------------
def test_schedule_wake_fires(ws, clock):
    cfg = ws.config.with_budget(max_proactive_per_hour=3)
    payload = {
        "inner_monologue": "他半小时后可能还没睡，到时候问一句",
        "say": "那我等下再来问你。",
        "wake": {"schedule_wake": {"after_sec": 1800, "reason": "确认主人是否休息", "target_speaker": "owner"}},
    }
    hb = Heartbeat.create(ws, cfg, sys2=JsonBrain(payload), now_fn=clock)
    hb.handle_message("owner", "我先去忙了")
    tickets = hb.scheduler.pending()
    assert len(tickets) == 1

    # 还没到点：不触发
    assert hb.poll_wakes() == []
    clock.advance(1801)
    results = hb.poll_wakes()
    assert len(results) == 1
    assert results[0].event.kind == "wake"
    assert results[0].says, "自主唤醒应该真的开口"


def test_proactive_hourly_budget(ws, clock):
    cfg = ws.config.with_budget(max_proactive_per_hour=2)
    payload = {
        "inner_monologue": "预约",
        "say": "好",
        "wake": {"schedule_wake": {"after_sec": 10, "reason": "测试"}},
    }
    hb = Heartbeat.create(ws, cfg, sys2=JsonBrain(payload), now_fn=clock)
    hb.handle_message("owner", "一")
    hb.handle_message("owner", "二")
    third = hb.handle_message("owner", "三")
    assert len(hb.scheduler.pending()) == 2, "队列长度受配额约束"
    assert any("队列已满" in e for e in third.ticks[0].actions.errors)

    clock.advance(11)
    fired = hb.poll_wakes()
    assert len(fired) == 2
    # 本小时配额用尽后不再主动开口
    clock.advance(11)
    assert hb.poll_wakes() == []


def test_scheduler_quota_recovers_after_an_hour(ws, clock):
    cfg = ws.config.with_budget(max_proactive_per_hour=2)
    scheduler = Scheduler(ws, cfg, now_fn=clock)
    for _ in range(2):
        ticket = scheduler.schedule(1, "x")
        scheduler.mark_fired(ticket)
    with pytest.raises(ProactiveBudgetExceeded):
        scheduler.schedule(1, "c")
    clock.advance(3601)
    assert scheduler.quota_left() == 2
    scheduler.schedule(1, "d")


def test_any_json_packet_applies_all_fields(ws, clock):
    payload = {
        "inner_monologue": "同时做很多事",
        "focus_update": {"turn_id": 1, "decay_before": 0.5, "decay_after": 0.1, "pin": [1]},
        "emotion_delta": {"warmth": 0.1},
        "relation_ops": [{"person_id": "owner", "intimacy": 0.9, "custom": {"note": "今天很坦诚"}}],
        "memory_ops": [{"op": "memorize", "lod0": "一次坦诚的对话", "lod1": "他讲了压力", "importance": 0.7}],
        "say": "谢谢你愿意讲这些。",
    }
    hb = Heartbeat.create(ws, ws.config, sys2=JsonBrain(payload), now_fn=clock)
    hb.handle_message("owner", "先聊一句")
    result = hb.handle_message("owner", "其实我最近挺累的")
    assert result.says == ["谢谢你愿意讲这些。"]
    assert hb.relations.get("owner")["custom"]["note"] == "今天很坦诚"
    assert result.ticks[0].actions.focus_changed
    assert result.ticks[0].actions.memorized
    assert hb.emotion.load().values["warmth"] > 0.6


def test_build_sys2_requires_key_without_mock(ws):
    from kernel.sys2_llm import Sys2Error

    with pytest.raises(Sys2Error):
        build_sys2(Config(sys2=Sys2Config(provider="openai_compatible", api_key_env="DEFINITELY_NOT_SET")))
