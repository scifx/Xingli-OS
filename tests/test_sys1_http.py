"""System One 的 HTTP 适配器：本地端点、宽容解析、断路器降级。"""

from __future__ import annotations

import json

from fake_openai import FakeOpenAI, chat_reply

from kernel.config import Config
from kernel.sys1_jev import (
    ChainClient,
    LLMFallbackClient,
    LocalLexiconClient,
    QuestionSpec,
)

SPECS = [
    QuestionSpec(id="tired", kind="noul", question="她现在是不是很累？"),
    QuestionSpec(
        id="intent", kind="choose", question="主要意图？", options=["闲聊", "求助", "倾诉"]
    ),
]

RESULTS = {
    "results": {
        "tired": {"yes": 0.91, "confidence": 0.83},
        "intent": {"distribution": {"闲聊": 0.1, "求助": 0.2, "倾诉": 0.7}, "confidence": 0.7},
    }
}


def make_config(base_url: str) -> Config:
    return Config.from_dict(
        {
            "sys1": {
                "provider": "llm_fallback",
                "base_url": base_url,
                "model": "fake-local",
                "timeout_sec": 10,
            }
        }
    )


def test_llm_fallback_parses_results_without_api_key():
    with FakeOpenAI(lambda p, h: chat_reply(json.dumps(RESULTS, ensure_ascii=False))) as server:
        client = LLMFallbackClient(make_config(server.base_url).sys1)
        decisions = client.fan_out("她今天很累，说话有气无力", SPECS)
        assert decisions["tired"].probability == 0.91
        assert decisions["tired"].confidence == 0.83
        assert decisions["intent"].value == "倾诉"
        assert "authorization" not in server.headers[0]
        assert server.requests[0]["stream"] is False
        assert server.requests[0]["max_tokens"] > 0


def test_llm_fallback_retries_without_response_format():
    def handler(payload, headers):
        if "response_format" in payload:
            return 400, {"error": {"message": "response_format is not supported"}}
        return chat_reply(json.dumps(RESULTS, ensure_ascii=False))

    with FakeOpenAI(handler) as server:
        client = LLMFallbackClient(make_config(server.base_url).sys1)
        decisions = client.fan_out("她今天很累", SPECS)
        assert decisions["tired"].probability == 0.91
        assert server.calls == 2
        assert "response_format" not in server.requests[1]


def test_llm_fallback_accepts_fenced_json():
    content = "```json\n" + json.dumps(RESULTS, ensure_ascii=False) + "\n```"
    with FakeOpenAI(lambda p, h: chat_reply(content)) as server:
        client = LLMFallbackClient(make_config(server.base_url).sys1)
        assert client.fan_out("她今天很累", SPECS)["intent"].value == "倾诉"


def test_circuit_breaker_stops_calling_a_dead_endpoint():
    class Boom:
        name = "boom"

        def __init__(self):
            self.calls = 0

        def fan_out(self, state, specs):
            self.calls += 1
            raise RuntimeError("connection refused")

    primary = Boom()
    chain = ChainClient(primary, LocalLexiconClient(), max_failures=2, cooldown_sec=60)
    chain.fan_out("她今天很累", SPECS)
    chain.fan_out("她今天很累", SPECS)
    assert primary.calls == 2
    assert chain.circuit_open

    decisions = chain.fan_out("她今天很累", SPECS)
    assert primary.calls == 2  # 断路器打开后不再白等
    assert "tired" in decisions
    assert "断路器" in (chain.last_error or "")


def test_circuit_breaker_recovers_after_cooldown():
    class Flaky:
        name = "flaky"

        def __init__(self):
            self.calls = 0

        def fan_out(self, state, specs):
            self.calls += 1
            raise RuntimeError("down")

    now = {"t": 0.0}
    primary = Flaky()
    chain = ChainClient(
        primary, LocalLexiconClient(), max_failures=1, cooldown_sec=10, now_fn=lambda: now["t"]
    )
    chain.fan_out("她今天很累", SPECS)
    assert chain.circuit_open
    now["t"] = 11.0
    assert not chain.circuit_open
    chain.fan_out("她今天很累", SPECS)
    assert primary.calls == 2
