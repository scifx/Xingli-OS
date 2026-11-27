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
