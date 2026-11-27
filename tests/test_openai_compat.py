"""真实端点路径（llama.cpp / vLLM / OpenAI 兼容）的适配测试。

全部打在一个本地假服务器上，所以离线、确定性、不依赖任何 Key。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fake_openai import FakeOpenAI, chat_reply, packet

from kernel.boot import Heartbeat
from kernel.config import Config
from kernel.paths import Workspace
from kernel.sys2_llm import OpenAICompatibleClient, Sys2Error, probe_sys2

NOKEY = "XINGLI_TEST_KEY_NOT_SET"


def make_config(base_url: str, **sys2) -> Config:
    data = {
        "provider": "openai_compatible",
        "base_url": base_url,
        "model": "fake-local",
        "api_key_env": NOKEY,
        "timeout_sec": 10,
    }
    data.update(sys2)
    return Config.from_dict({"sys2": data, "sys1": {"provider": "local"}})


@pytest.fixture(autouse=True)
def _no_key(monkeypatch):
    monkeypatch.delenv(NOKEY, raising=False)


# ---------------------------------------------------------------------------
# 基本往返
# ---------------------------------------------------------------------------
def test_roundtrip_without_api_key():
    with FakeOpenAI() as server:
        client = OpenAICompatibleClient(make_config(server.base_url))
        text = client.complete("SYSTEM", "USER")
        data = json.loads(text)
        assert data["say"] == "收到，我在这里。"
        assert data["inner_monologue"]
        # 本地服务不校验 Key：不应凭空发一个 Authorization
        assert "authorization" not in server.headers[0]
        sent = server.requests[0]
        assert sent["model"] == "fake-local"
        assert sent["stream"] is False
        # auto 模式优先语法约束（llama.cpp 会据此做 GBNF 解码）
        assert sent["response_format"]["type"] == "json_schema"
        assert "inner_monologue" in sent["response_format"]["json_schema"]["schema"]["properties"]
        assert [m["role"] for m in sent["messages"]] == ["system", "user"]
        assert client.last_timings["predicted_per_second"] == 42.0


def test_api_key_used_when_present(monkeypatch):
    monkeypatch.setenv(NOKEY, "sk-local-test")
    with FakeOpenAI() as server:
        client = OpenAICompatibleClient(make_config(server.base_url))
        client.complete("S", "U")
        assert server.headers[0]["authorization"] == "Bearer sk-local-test"


def test_extra_body_is_merged():
    with FakeOpenAI() as server:
        client = OpenAICompatibleClient(
            make_config(
                server.base_url, extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )
        )
        client.complete("S", "U")
        assert server.requests[0]["chat_template_kwargs"] == {"enable_thinking": False}


# ---------------------------------------------------------------------------
# 端点能力差异：自动降级
# ---------------------------------------------------------------------------
def test_schema_downgrades_to_json_object_then_off():
    """三轮降级：json_schema（被拒）→ json_object（被拒）→ 关掉约束。"""

    def handler(payload, headers):
        if "response_format" in payload:
            return 400, {"error": {"message": "response_format is not supported"}}
        return chat_reply(json.dumps(packet()))

    with FakeOpenAI(handler) as server:
        client = OpenAICompatibleClient(make_config(server.base_url))
        data = json.loads(client.complete("S", "U"))
        assert data["say"] == "收到，我在这里。"
        assert server.calls == 3
        assert server.requests[0]["response_format"]["type"] == "json_schema"
        assert server.requests[1]["response_format"] == {"type": "json_object"}
        assert "response_format" not in server.requests[2]
        assert any("json_schema" in note for note in client.notes)


def test_schema_only_downgrades_one_step():
    def handler(payload, headers):
        if payload.get("response_format", {}).get("type") == "json_schema":
            return 400, {"error": {"message": "json_schema is not supported by this build"}}
        return chat_reply(json.dumps(packet()))

    with FakeOpenAI(handler) as server:
        client = OpenAICompatibleClient(make_config(server.base_url))
        json.loads(client.complete("S", "U"))
        assert server.calls == 2
        assert server.requests[1]["response_format"] == {"type": "json_object"}


def test_max_tokens_downgrade():
    def handler(payload, headers):
        if "max_tokens" in payload:
            return 400, {"error": {"message": "max_tokens is deprecated; use max_completion_tokens"}}
        return chat_reply(json.dumps(packet()))

    with FakeOpenAI(handler) as server:
        client = OpenAICompatibleClient(make_config(server.base_url))
        client.complete("S", "U")
        assert "max_completion_tokens" in server.requests[1]
        assert "max_tokens" not in server.requests[1]


def test_auth_failure_message_mentions_local_servers():
    with FakeOpenAI(lambda p, h: (401, {"error": {"message": "invalid api key"}})) as server:
        client = OpenAICompatibleClient(make_config(server.base_url))
        with pytest.raises(Sys2Error) as excinfo:
            client.complete("S", "U")
    assert "本地 llama.cpp" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 模型输出不守规矩
# ---------------------------------------------------------------------------
def test_fenced_json_with_chatter_is_cleaned():
    content = (
        "好的，我按格式输出：\n```json\n"
        + json.dumps(packet(say="围栏也挡不住我。"), ensure_ascii=False)
        + "\n```\n以上。"
    )
    with FakeOpenAI(lambda p, h: chat_reply(content)) as server:
        client = OpenAICompatibleClient(make_config(server.base_url))
        assert json.loads(client.complete("S", "U"))["say"] == "围栏也挡不住我。"


def test_json_inside_reasoning_content_is_recovered():
    reasoning = "让我想想……\n" + json.dumps(packet(say="思考通道里的回答。"), ensure_ascii=False)
    with FakeOpenAI(lambda p, h: chat_reply(None, reasoning=reasoning)) as server:
        client = OpenAICompatibleClient(make_config(server.base_url))
        assert json.loads(client.complete("S", "U"))["say"] == "思考通道里的回答。"


def test_broken_json_triggers_one_repair_request():
    broken = '{"inner_monologue": "想了", "say": '

    def handler(payload, headers):
        last = payload["messages"][-1]["content"]
        if "不是合法 JSON" in last:
            return chat_reply(json.dumps(packet(say="修好了。"), ensure_ascii=False))
        return chat_reply(broken)

    with FakeOpenAI(handler) as server:
        client = OpenAICompatibleClient(make_config(server.base_url))
        assert json.loads(client.complete("S", "U"))["say"] == "修好了。"
        assert server.calls == 2
        assert "不是合法 JSON" in server.requests[1]["messages"][-1]["content"]


def test_pure_prose_does_not_waste_a_repair_request():
    """纯散文不是“写坏的 JSON”：不重发（省一整趟生成），交给内核降级成发言。"""
    with FakeOpenAI(lambda p, h: chat_reply("你好呀。我是星璃，我在这里。")) as server:
        client = OpenAICompatibleClient(make_config(server.base_url))
        assert "星璃" in client.complete("S", "U")
        assert server.calls == 1
        assert client.notes == []


def test_repair_can_be_disabled():
    with FakeOpenAI(lambda p, h: chat_reply("就不是 JSON")) as server:
        client = OpenAICompatibleClient(make_config(server.base_url, repair=False))
        assert "就不是 JSON" in client.complete("S", "U")
        assert server.calls == 1


def test_transient_error_is_retried_once():
    client = OpenAICompatibleClient(make_config("http://127.0.0.1:9/v1"))
    seen = {"n": 0}

    def fake_post(payload, timeout=None):
        seen["n"] += 1
        if seen["n"] == 1:
            raise Sys2Error("连接失败")
        return {"choices": [{"message": {"content": json.dumps(packet())}}]}

    client._post = fake_post  # type: ignore[method-assign]
    assert json.loads(client.complete("S", "U"))["say"] == "收到，我在这里。"
    assert seen["n"] == 2


# ---------------------------------------------------------------------------
# 探测与端到端心跳（记忆文件真的长出来）
# ---------------------------------------------------------------------------
def test_probe_reports_model_list():
    with FakeOpenAI() as server:
        probe = probe_sys2(make_config(server.base_url))
    assert probe["ok"] is True
    assert probe["models"] == ["fake-local"]
    assert probe["latency_ms"] is not None


def test_heartbeat_writes_plaintext_memory_that_reaches_next_frame(tmp_path: Path, clock):
    title = "下周三要去看牙医"
    ops = [
        {
            "op": "memorize",
            "lod0": title,
            "lod1": "主人说下周三上午要去医院看牙，别在那天安排别的事。",
            "lod2": "主人：记住，下周三要去看牙医，别安排别的事。",
            "kind": "episode",
            "about_ids": ["owner"],
            "visibility": ["owner"],
            "epistemic": "direct",
            "importance": 0.82,
            "tags": ["承诺"],
        }
    ]

    def handler(payload, headers):
        return chat_reply(
            json.dumps(
                packet(say="记住了，下周三不给你安排别的事。", memory_ops=ops),
                ensure_ascii=False,
            )
        )

    with FakeOpenAI(handler) as server:
        cfg = make_config(server.base_url)
        ws = Workspace.open(tmp_path / "xingli", cfg)
        hb = Heartbeat.create(ws, cfg, now_fn=clock)

        first = hb.handle_message("owner", "记住，下周三要去看牙医，别安排别的事")
        assert first.says == ["记住了，下周三不给你安排别的事。"]

        episodes = sorted(ws.episodes.glob("*.json"))
        assert len(episodes) == 1
        node = json.loads(episodes[0].read_text(encoding="utf-8"))
        assert episodes[0].stem == node["id"]
        assert node["lod0"] == title
        assert node["visibility"] == ["owner"]
        assert node["epistemic"] == "direct"
        assert node["importance"] == pytest.approx(0.82)
        assert node["created_at"]
        assert "主人" in node["lod2"]

        turns = sorted(ws.turns.glob("*.json"))
        assert len(turns) == 1
        turn = json.loads(turns[0].read_text(encoding="utf-8"))
        assert turn["user_text"].startswith("记住")
        assert turn["assistant_text"] == "记住了，下周三不给你安排别的事。"

        # 下一次请求的认知帧里应该出现这条记忆的句柄（记忆 = 改变下一拍的计算）
        hb.handle_message("owner", "我们下周三有安排吗？")
        frame = server.requests[-1]["messages"][-1]["content"]
        assert f"#{node['id']}" in frame
        assert title in frame
        # 恒定锐度：每次仍然只有 system + user 两条消息
        assert len(server.requests[-1]["messages"]) == 2
        assert hb.last_frame.total_tokens <= cfg.budget.frame_tokens


# ---------------------------------------------------------------------------
# 行动契约版本
# ---------------------------------------------------------------------------
def test_compact_contract_shrinks_the_system_prompt(tmp_path: Path, clock):
    from kernel.sys2_llm import SYSTEM_CONTRACT_COMPACT

    with FakeOpenAI() as server:
        cfg = make_config(server.base_url, contract="compact")
        ws = Workspace.open(tmp_path / "xingli", cfg)
        hb = Heartbeat.create(ws, cfg, now_fn=clock)
        hb.handle_message("owner", "你好")
        assert SYSTEM_CONTRACT_COMPACT in hb.last_frame.system
        assert hb.last_frame.identity.startswith("【你是谁】")
        compact_tokens = hb.last_frame.total_tokens

    with FakeOpenAI() as server:
        cfg = make_config(server.base_url, contract="full")
        ws = Workspace.open(tmp_path / "full", cfg)
        hb = Heartbeat.create(ws, cfg, now_fn=clock)
        hb.handle_message("owner", "你好")
        assert SYSTEM_CONTRACT_COMPACT not in hb.last_frame.system
        assert hb.last_frame.total_tokens > compact_tokens


def test_unknown_contract_is_rejected_early():
    with pytest.raises(Sys2Error):
        OpenAICompatibleClient(make_config("http://127.0.0.1:9/v1", contract="tiny"))
