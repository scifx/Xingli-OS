"""真实使用中暴露的三个问题，逐一钉死：

1. 认知包里有 ``code_action`` 时崩在 ``ThoughtPacket.notes``（整场聊天被杀掉）；
2. 模型把 ``continue_think`` 当客套话 → 同一句话自我循环到熔断；
3. 她把“我叫 lucky”记成“主人叫星璃”，而且**改不回来**（只会加，不会作废）。
"""

from __future__ import annotations

import json

from kernel.boot import Heartbeat
from kernel.models import ThoughtPacket
from kernel.paths import Workspace
from kernel.sys2_llm import MockSys2Client


class JsonBrain:
    """固定返回一个认知包（可多拍）。"""

    name = "json_brain"

    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.calls = 0

    def complete(self, system, user):
        payload = self.payloads[min(self.calls, len(self.payloads) - 1)]
        self.calls += 1
        return json.dumps(payload, ensure_ascii=False)


def _hb(ws: Workspace, clock, brain) -> Heartbeat:
    hb = Heartbeat.create(ws, ws.config, sys2=brain, now_fn=clock)
    hb.on_event = lambda msg: None
    return hb


# ---------------------------------------------------------------------------
# 1) code_action 里带回执（notes）不再把整场聊天掀翻
# ---------------------------------------------------------------------------
def test_code_action_with_notes_does_not_crash(ws: Workspace, clock):
    code = (
        "packet = recall('000000000000')\n"
        "remember('主人今天说了咖啡', lod1='他喝了两杯', importance=0.6)\n"
        "say('记下了')\n"
    )
    hb = _hb(ws, clock, JsonBrain({"inner_monologue": "先查再记", "code_action": {"code": code}}))
    result = hb.handle_message("owner", "今天喝了两杯咖啡")
    tick = result.ticks[0]
    assert not any("应用认知包失败" in e for e in tick.actions.errors)
    assert "找不到记忆" in tick.actions.code_output
    assert result.says == ["记下了"]


# ---------------------------------------------------------------------------
# 2) 自转刹车：重复输出 / 主人插话 / 时间预算
# ---------------------------------------------------------------------------
def test_repeated_say_stops_the_chain(ws: Workspace, clock):
    payload = {"inner_monologue": "再想想", "say": "我在。", "wake": {"continue_think": {"reason": "继续"}}}
    hb = _hb(ws, clock, JsonBrain(payload))
    result = hb.handle_message("owner", "你在吗")
    assert len(result.ticks) == 2, "第二拍已经复读了，应该停下而不是刷满 MAX_CHAIN_TICKS"
    assert "重复" in result.halted_reason
    assert result.says == ["我在。"]


def test_should_stop_ends_the_chain_immediately(ws: Workspace, clock):
    payloads = [
        {"inner_monologue": "第一层", "say": "第一句", "wake": {"continue_think": {"reason": "深挖"}}},
        {"inner_monologue": "第二层", "say": "第二句", "wake": {"continue_think": {"reason": "深挖"}}},
        {"inner_monologue": "第三层", "say": "第三句"},
    ]
    hb = _hb(ws, clock, JsonBrain(*payloads))
    # 没人插话：三拍走完（模型自己写完了）
    result = hb.handle_message("owner", "说说看", should_stop=lambda: False)
    assert len(result.ticks) == 3

    # 主人插话：第一拍答完就收尾，不再自转
    called = {"n": 0}

    def typed() -> bool:
        called["n"] += 1
        return True

    result = hb.handle_message("owner", "再说说", should_stop=typed)
    assert len(result.ticks) == 1
    assert "插话" in result.halted_reason


def test_chain_respects_time_budget(ws: Workspace, clock):
    cfg = ws.config.with_budget(max_chain_ticks=9, max_chain_seconds=0)
    payload = {"inner_monologue": "继续", "say": "嗯", "wake": {"continue_think": {"reason": "x"}}}
    hb = _hb(ws, clock, JsonBrain(payload))
    hb.config = cfg
    result = hb.handle_message("owner", "你好")
    assert len(result.ticks) == 1, "预算为 0 时不允许自转"
    assert "预算" in result.halted_reason


# ---------------------------------------------------------------------------
# 3) 放错抽屉的字段：别再报“不认识的进化操作”
# ---------------------------------------------------------------------------
def test_misplaced_ops_are_routed_to_the_right_channel(ws: Workspace, clock):
    payload = {
        "inner_monologue": "顺手做点事",
        "say": "好。",
        "memory_ops": [
            {"op": "focus_update", "turn_id": 1, "decay_before": 0.5},
            {"op": "memorize", "lod0": "主人偏爱清爽口味"},
        ],
        "evolve_ops": [
            {"op": "emotion_delta", "warmth": 0.1},
            {"op": "update_affinity", "person_id": "owner", "intimacy": 0.9},
            {"op": "memorize", "lod0": "主人不爱吵闹"},
        ],
    }
    hb = _hb(ws, clock, JsonBrain(payload))
    result = hb.handle_message("owner", "随便聊聊")
    tick = result.ticks[0]
    assert not any("进化被拒绝" in e for e in tick.actions.errors)
    assert any("字段归位" in note for note in tick.actions.notes)
    assert tick.actions.focus_changed
    assert tick.actions.emotion.get("warmth")
    assert "owner" in tick.actions.relation_updates
    assert len(tick.actions.memorized) == 2


# ---------------------------------------------------------------------------
# 4) 她能改口：称呼纠正 + 作废互相矛盾的旧记忆 + 认知帧里分清两个名字
# ---------------------------------------------------------------------------
def test_owner_name_correction_overwrites_and_retracts(ws: Workspace, clock):
    wrong = {
        "op": "memorize",
        "lod0": "主人名字：星璃",
        "lod1": "主人叫星璃",
        "importance": 0.5,
    }
    payload = {"inner_monologue": "记下名字", "say": "记住了。", "memory_ops": [wrong]}
    hb = _hb(ws, clock, JsonBrain(payload))
    hb.handle_message("owner", "你叫什么？我叫星璃吗")
    stale = [n for n in hb.store.list_nodes("owner") if "主人名字" in n.lod0]
    assert stale, "先造出一条错误的旧记忆"

    hb2 = _hb(ws, clock, JsonBrain({"inner_monologue": "改口", "say": "好，记住了。"}))
    hb2.handle_message("owner", "我的名字是lucky，也就是说，你的主人叫lucky。而你才是星璃，记住了")
    assert hb2.relations.get("owner")["display_name"] == "lucky"
    # 旧称呼记忆被归档（不是删掉）
    for node in stale:
        archived = hb2.store.get(node.id)
        assert archived.archived is True
    facts = [n for n in hb2.store.list_nodes("owner") if n.lod0.startswith("称呼：")]
    assert facts and facts[0].pinned
    # 认知帧明确区分“谁叫什么”
    hb2.handle_message("owner", "你还记得我叫什么吗")
    frame = hb2.last_frame
    assert "你叫「星璃」" in frame.user
    assert "lucky" in frame.user


def test_preference_claims_are_memorised_and_deduped(ws: Workspace, clock):
    hb = _hb(ws, clock, JsonBrain({"inner_monologue": "嗯", "say": "记住了。"}))
    hb.handle_message("owner", "我喜欢吃菠萝和芒果")
    hb.handle_message("owner", "对了，我喜欢菠萝和芒果")
    nodes = [n for n in hb.store.list_nodes("owner") if "菠萝" in n.lod0]
    assert len(nodes) == 1, "同一件事不该长成两条记忆"
    assert nodes[0].lod0 == "主人喜欢菠萝和芒果"


def test_forget_op_archives_memory(ws: Workspace, clock):
    hb = _hb(ws, clock, JsonBrain({"inner_monologue": "记", "say": "好"}))
    node = hb.store.put(lod0="主人不爱吃辣", lod1="", lod2="", source_id="owner", importance=0.5)
    payload = {"inner_monologue": "作废", "say": "好", "memory_ops": [{"op": "forget", "mem_id": node.id}]}
    hb2 = _hb(ws, clock, JsonBrain(payload))
    result = hb2.handle_message("owner", "那个记错了")
    assert hb2.store.get(node.id).archived is True
    assert any("作废记忆" in note for note in result.ticks[0].actions.notes)


# ---------------------------------------------------------------------------
# 5) 自主唤醒的最短间隔：after_sec=0 不该立刻把她叫醒
# ---------------------------------------------------------------------------
def test_wake_interval_is_clamped(ws: Workspace, clock):
    payload = {
        "inner_monologue": "等一下再说",
        "say": "嗯",
        "wake": {"schedule_wake": {"after_sec": 0, "reason": "马上回来"}},
    }
    hb = _hb(ws, clock, JsonBrain(payload))
    result = hb.handle_message("owner", "你好")
    assert result.ticks[0].actions.scheduled
    due = result.ticks[0].actions.scheduled[0]["due_at"]
    assert due > result.ticks[0].actions.scheduled[0]["created_at"]
    assert any("间隔" in note for note in result.ticks[0].actions.notes)
    clock.advance(1)
    assert hb.poll_wakes() == [], "1 秒内不该被自己叫醒"


# ---------------------------------------------------------------------------
# 6) mock 大脑也走同一条链路（不用真实端点）
# ---------------------------------------------------------------------------
def test_cognitive_packet_has_no_notes_field(ws: Workspace, clock):
    """回归：内核回执在 PacketBuilder 上，认知包没有 notes —— 以前这里会 AttributeError。"""
    assert not hasattr(ThoughtPacket(inner_monologue="在", say="在的"), "notes")
    hb = Heartbeat.create(ws, ws.config, sys2=MockSys2Client(), now_fn=clock)
    result = hb.handle_message("owner", "在吗")
    assert result.says


# ---------------------------------------------------------------------------
# 7) 自主唤醒：不自转、不落空轮次
# ---------------------------------------------------------------------------
def test_wake_event_does_not_chain(ws: Workspace, clock):
    payload = {
        "inner_monologue": "自己醒一下",
        "say": "我在这儿。",
        "wake": {"continue_think": {"reason": "再想想"}},
    }
    hb = _hb(ws, clock, JsonBrain(payload))
    hb.scheduler.schedule(60, reason="测试唤醒", now=clock())
    clock.advance(61)
    fired = hb.poll_wakes()
    assert len(fired) == 1
    assert len(fired[0].ticks) == 1, "自主唤醒不该连转四拍"
    assert "MAX_WAKE_CHAIN_TICKS" in fired[0].halted_reason


def test_quiet_wake_does_not_write_a_turn(ws: Workspace, clock):
    hb = _hb(ws, clock, JsonBrain({"inner_monologue": "没什么好说的", "say": ""}))
    hb.scheduler.schedule(60, reason="空跑", now=clock())
    clock.advance(61)
    hb.poll_wakes()
    assert hb.store.list_turns("owner") == [], "没说话、没思考，就不该留下轮次"


# ---------------------------------------------------------------------------
# 8) sys1 配了却没生效：必须说出来，而不是静默退回
# ---------------------------------------------------------------------------
def test_sys1_with_model_but_local_provider_is_promoted(ws: Workspace):
    import cli

    cfg = ws.config.with_sys1(provider="local", model="Kev-4B-GGUF", base_url="http://127.0.0.1:8080/v1")
    cli.SYS1_NOTICES.clear()
    promoted = cli._maybe_promote_sys1(cfg)
    assert promoted.sys1.provider == "llm_fallback"
    assert any("自动接管" in n for n in cli.SYS1_NOTICES)


def test_sys1_build_error_is_reported():
    from kernel.config import Config, Sys1Config
    from kernel.sys1_jev import build_client

    cfg = Config(sys1=Sys1Config(provider="llm_fallback", base_url=""))
    client = build_client(cfg)
    assert getattr(client, "build_error", ""), "接不上必须留下原因"


def test_sys1_unknown_provider_is_reported():
    from kernel.config import Config, Sys1Config
    from kernel.sys1_jev import build_client

    cfg = Config(sys1=Sys1Config(provider="openai_chat", model="x", base_url="http://127.0.0.1:1/v1"))
    client = build_client(cfg)
    assert "不认识的" in getattr(client, "build_error", "")


def test_sys1_provider_typos_still_connect():
    from kernel.config import Config, Sys1Config
    from kernel.sys1_jev import build_client

    for spelling in ("llm-fallback", "llm_fallback", "openai", "chat"):
        client = build_client(
            Config(
                sys1=Sys1Config(
                    provider=spelling, base_url="http://127.0.0.1:1/v1", model="m"
                )
            )
        )
        assert client.name.startswith("llm_fallback"), spelling
