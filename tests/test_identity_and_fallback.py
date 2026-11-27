"""她是谁（身份块）、结构化输出阶梯、以及“不肯输出 JSON”时的降级。"""

from __future__ import annotations

import json
from pathlib import Path

from fake_openai import FakeOpenAI, chat_reply, packet

from kernel.boot import Heartbeat
from kernel.config import Config
from kernel.paths import Workspace
from kernel.sys2_llm import OpenAICompatibleClient


def make_config(base_url: str, **sys2) -> Config:
    data = {
        "provider": "openai_compatible",
        "base_url": base_url,
        "model": "fake-local",
        "api_key_env": "XINGLI_TEST_KEY_NOT_SET",
        "timeout_sec": 10,
    }
    data.update(sys2)
    return Config.from_dict({"sys2": data, "sys1": {"provider": "local"}})


# ---------------------------------------------------------------------------
# 身份块
# ---------------------------------------------------------------------------
def test_identity_block_comes_from_persona(tmp_path: Path, clock):
    with FakeOpenAI() as server:
        cfg = make_config(server.base_url)
        ws = Workspace.open(tmp_path / "xingli", cfg)
        hb = Heartbeat.create(ws, cfg, now_fn=clock)
        hb.handle_message("owner", "你是谁")
    system = hb.last_frame.system
    assert system.startswith("【你是谁】星璃")
    assert "聊天记录的回放器" in system          # 来自 cortex/persona.py 的 SELF_CONCEPT
    assert "不假装记得" in system                     # 来自 VALUES
    assert "好奇 0.75" in system                      # 来自 TRAITS
    assert "第一人称" in system
    assert hb.last_frame.identity  # /frame 里能单独看到


def test_identity_is_mutable_cortex_state(tmp_path: Path, clock):
    """人格写在 cortex 里（可被进化改写），改完下一帧就生效。"""
    with FakeOpenAI() as server:
        cfg = make_config(server.base_url)
        ws = Workspace.open(tmp_path / "xingli", cfg)
        hb = Heartbeat.create(ws, cfg, now_fn=clock)
        hb.handle_message("owner", "第一句")

        persona = ws.persona_file.read_text(encoding="utf-8")
        ws.persona_file.write_text(
            persona.replace(
                "我不是一个聊天记录的回放器。",
                "我是一个会自己挑话题的同伴。",
            ),
            encoding="utf-8",
        )
        hb.handle_message("owner", "第二句")
    assert "会自己挑话题的同伴" in hb.last_frame.system
    assert "会自己挑话题的同伴" in hb.last_frame.system


# ---------------------------------------------------------------------------
# 结构化输出阶梯
# ---------------------------------------------------------------------------
def test_structured_output_off_sends_no_response_format():
    with FakeOpenAI() as server:
        client = OpenAICompatibleClient(make_config(server.base_url, structured_output="off"))
        json.loads(client.complete("S", "U"))
        assert "response_format" not in server.requests[0]


def test_structured_output_json_object_only():
    with FakeOpenAI() as server:
        client = OpenAICompatibleClient(
            make_config(server.base_url, structured_output="json_object")
        )
        json.loads(client.complete("S", "U"))
        assert server.requests[0]["response_format"] == {"type": "json_object"}


# ---------------------------------------------------------------------------
# 不肯输出 JSON：降级成“纯发言”，但绝不装死、也绝不假装执行
# ---------------------------------------------------------------------------
def test_prose_output_still_speaks(tmp_path: Path, clock):
    with FakeOpenAI(lambda p, h: chat_reply("我是星璃。会想起，也会犹豫。")) as server:
        cfg = make_config(server.base_url)
        ws = Workspace.open(tmp_path / "xingli", cfg)
        hb = Heartbeat.create(ws, cfg, now_fn=clock)
        result = hb.handle_message("owner", "你是谁")
    assert result.says == ["我是星璃。会想起，也会犹豫。"]
    assert result.ticks[0].mode == "prose"
    assert any("纯发言" in e for e in result.ticks[0].actions.errors)
    # 降级的那一拍不许偷偷写记忆
    assert result.ticks[0].actions.memorized == []
    assert not list(ws.episodes.glob("*.json"))


def test_structure_first_still_wins_when_model_obeys(tmp_path: Path, clock):
    ops = [
        {
            "op": "memorize",
            "lod0": "下周三看牙医",
            "lod1": "要点",
            "lod2": "全文",
            "epistemic": "direct",
            "visibility": ["owner"],
            "importance": 0.8,
        }
    ]
    with FakeOpenAI(
        lambda p, h: chat_reply(json.dumps(packet(say="记下了。", memory_ops=ops), ensure_ascii=False))
    ) as server:
        cfg = make_config(server.base_url)
        ws = Workspace.open(tmp_path / "xingli", cfg)
        hb = Heartbeat.create(ws, cfg, now_fn=clock)
        result = hb.handle_message("owner", "记住：下周三看牙医")
    assert result.ticks[0].mode == "json"
    assert result.says == ["记下了。"]
    assert list(ws.episodes.glob("*.json"))


def test_contract_reminder_can_ride_in_the_frame(tmp_path: Path, clock):
    """有些 chat template 会吞掉 system 消息：把格式提醒贴到帧末尾兜底。"""
    with FakeOpenAI() as server:
        cfg = make_config(server.base_url, contract_in_user=True)
        ws = Workspace.open(tmp_path / "on", cfg)
        hb = Heartbeat.create(ws, cfg, now_fn=clock)
        hb.handle_message("owner", "你好")
        assert "[OUTPUT_FORMAT]" in hb.last_frame.user

    with FakeOpenAI() as server:
        cfg = make_config(server.base_url)
        ws = Workspace.open(tmp_path / "off", cfg)
        hb = Heartbeat.create(ws, cfg, now_fn=clock)
        hb.handle_message("owner", "你好")
        assert "[OUTPUT_FORMAT]" not in hb.last_frame.user


# ---------------------------------------------------------------------------
# 思考型模型：预填 JSON 开头，逼它直接说人话
# ---------------------------------------------------------------------------
REASONING_TEXT = "首先，用户要求只输出合法JSON，字段有 inner_monologue, say。首先分析当前场景……"


def test_reasoning_output_triggers_one_auto_prefill_retry():
    state = {"n": 0}

    def handler(payload, headers):
        state["n"] += 1
        last = payload["messages"][-1]
        if last["role"] == "assistant":
            # 模型在 prefill 之后只负责把话说下去
            return chat_reply('我是星璃。我会记住你的牙医预约。"}')
        return chat_reply(REASONING_TEXT)

    with FakeOpenAI(handler) as server:
        cfg = make_config(server.base_url, auto_prefill=True)
        client = OpenAICompatibleClient(cfg)
        text = client.complete("S", "U")
        assert json.loads(text)["say"] == "我是星璃。我会记住你的牙医预约。"
        assert state["n"] == 2
        assert server.requests[1]["messages"][-1]["role"] == "assistant"
        assert server.requests[1]["messages"][-1]["content"].startswith('{"inner_monologue"')
        assert any("预填" in note for note in client.notes)


def test_auto_prefill_can_be_disabled():
    with FakeOpenAI(lambda p, h: chat_reply(REASONING_TEXT)) as server:
        client = OpenAICompatibleClient(make_config(server.base_url, auto_prefill=False))
        assert "首先" in client.complete("S", "U")
        assert server.calls == 1


def test_prefill_truncated_output_salvages_the_sentence():
    with FakeOpenAI(lambda p, h: chat_reply("我是星璃，我记得你下周三要看牙医")) as server:
        client = OpenAICompatibleClient(make_config(server.base_url, prefill="say"))
        assert client.complete("S", "U") == "我是星璃，我记得你下周三要看牙医"
        assert server.requests[0]["messages"][-1]["role"] == "assistant"
        assert any("抢救" in note for note in client.notes)


def test_prefill_packet_reaches_the_chat(tmp_path: Path, clock):
    def handler(payload, headers):
        if payload["messages"][-1]["role"] == "assistant":
            return chat_reply('是的，我记得。"}')
        return chat_reply(REASONING_TEXT)

    with FakeOpenAI(handler) as server:
        cfg = make_config(server.base_url)
        ws = Workspace.open(tmp_path / "xingli", cfg)
        hb = Heartbeat.create(ws, cfg, now_fn=clock)
        result = hb.handle_message("owner", "你还记得我喜欢吃什么吗")
    assert result.says == ["是的，我记得。"]
    assert result.ticks[0].mode == "json"


# ---------------------------------------------------------------------------
# 弱模型补洞：模型不写 memory_ops 时，内核补抽一次
# ---------------------------------------------------------------------------
EXTRACT_SYSTEM_HINT = "记忆抽取器"


def _extract_reply() -> str:
    return json.dumps(
        {
            "memory_ops": [
                {
                    "op": "memorize",
                    "lod0": "主人喜欢吃菠萝和芒果",
                    "lod1": "主人明确说过喜欢菠萝和芒果。",
                    "lod2": "主人：你记住，我喜欢吃菠萝和芒果。",
                    "importance": 0.75,
                    "tags": ["偏好"],
                }
            ]
        },
        ensure_ascii=False,
    )


def test_memory_pass_fills_in_for_weak_models(tmp_path: Path, clock):
    def handler(payload, headers):
        messages = payload["messages"]
        if EXTRACT_SYSTEM_HINT in messages[0]["content"]:
            return chat_reply(_extract_reply())
        if messages[-1]["role"] == "assistant":
            return chat_reply('好，菠萝和芒果我记住了。"}')
        return chat_reply(REASONING_TEXT)

    with FakeOpenAI(handler) as server:
        cfg = make_config(server.base_url)
        ws = Workspace.open(tmp_path / "xingli", cfg)
        hb = Heartbeat.create(ws, cfg, now_fn=clock)
        result = hb.handle_message("owner", "你记住，我喜欢吃菠萝和芒果")
        assert result.says == ["好，菠萝和芒果我记住了。"]
        assert result.ticks[0].actions.memorized
        assert any("记忆回填" in e for e in result.ticks[0].actions.errors)
        episodes = sorted(ws.episodes.glob("*.json"))
        assert episodes
        node = json.loads(episodes[0].read_text(encoding="utf-8"))
        assert node["lod0"] == "主人喜欢吃菠萝和芒果"
        assert node["visibility"] == ["owner"]
        assert node["epistemic"] == "direct"
        # 抽取只在“值得记 + 模型没写 ops”时发生：这一轮共 3 次请求
        assert server.calls == 3


def test_memory_pass_skipped_when_model_already_wrote_ops(tmp_path: Path, clock):
    ops = [{"op": "memorize", "lod0": "喜欢菠萝", "lod1": "要点", "lod2": "全文"}]
    with FakeOpenAI(
        lambda p, h: chat_reply(json.dumps(packet(say="记下了。", memory_ops=ops), ensure_ascii=False))
    ) as server:
        cfg = make_config(server.base_url)
        ws = Workspace.open(tmp_path / "xingli", cfg)
        hb = Heartbeat.create(ws, cfg, now_fn=clock)
        hb.handle_message("owner", "记住：我喜欢菠萝")
    assert server.calls == 1
    assert list(ws.episodes.glob("*.json"))


def test_memory_pass_can_be_disabled(tmp_path: Path, clock):
    with FakeOpenAI(lambda p, h: chat_reply("先说重点：我在。")) as server:
        cfg = make_config(server.base_url, memory_pass=False)
        ws = Workspace.open(tmp_path / "xingli", cfg)
        hb = Heartbeat.create(ws, cfg, now_fn=clock)
        result = hb.handle_message("owner", "记住，我喜欢菠萝")
    assert not result.ticks[0].actions.memorized
    assert not list(ws.episodes.glob("*.json"))
    assert server.calls == 1
