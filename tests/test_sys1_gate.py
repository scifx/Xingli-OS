"""M2 验收：System One 模糊反射 —— 单次扇出、三段门控、冷却、限幅。"""

from __future__ import annotations

from kernel.literal import write_list_module
from kernel.models import now_utc
from kernel.sys1_jev import LocalLexiconClient, ReflexEngine
from memory.emotion import EmotionState, EmotionStore


class CountingClient:
    """记录每次 fan_out 的规格，用来证明“一轮只发一次请求”。"""

    name = "counting"

    def __init__(self):
        self.inner = LocalLexiconClient()
        self.calls: list[list[str]] = []

    def fan_out(self, state, specs):
        self.calls.append([s.id for s in specs])
        return self.inner.fan_out(state, specs)


def make_engine(ws):
    client = CountingClient()
    engine = ReflexEngine(ws, ws.config, client=client)
    return engine, client


def test_single_fan_out_per_tick(ws):
    engine, client = make_engine(ws)
    scan = engine.scan("我又熬夜了，压力好大", "owner", now_utc())
    assert len(client.calls) == 1, "所有反射必须合并成一次并行扇出"
    ids = client.calls[0]
    # 基础感知 + 出厂反射都在同一批
    assert {"base_intent", "base_importance", "base_teaching"} <= set(ids)
    assert {"masking_fatigue", "teaching_signal", "urgency_deadline"} <= set(ids)
    assert scan.fired, "熬夜+压力应当命中高置信反射"


def test_high_confidence_fires_and_cools_down(ws):
    engine, client = make_engine(ws)
    scan = engine.scan("我又熬夜了，压力好大", "owner", now_utc())
    assert "masking_fatigue" in scan.fired
    assert scan.emotion.get("caring", 0) > 0
    assert any("共情陪伴" in h for h in scan.hints)

    # 冷却期内：既不进请求，也不会再次触发
    scan2 = engine.scan("我又熬夜了，压力好大", "owner", now_utc())
    assert "masking_fatigue" not in client.calls[-1]
    assert "masking_fatigue" not in scan2.fired


def test_mid_confidence_only_hints(ws):
    engine, _ = make_engine(ws)
    scan = engine.scan("今天先把这个弄完", "owner", now_utc())
    gates = {s.reflex_id: s.gate for s in scan.signals}
    assert gates.get("urgency_deadline") == "mid"
    assert "urgency_deadline" not in scan.fired
    assert not scan.emotion, "中置信不允许直接改情绪"
    assert any("不确定" in h for h in scan.hints)


def test_low_confidence_is_silent(ws):
    engine, _ = make_engine(ws)
    scan = engine.scan("今天天气不错", "owner", now_utc())
    gates = {s.reflex_id: s.gate for s in scan.signals}
    assert gates.get("masking_fatigue") == "low"
    assert "masking_fatigue" not in scan.fired
    assert not scan.emotion


def test_max_delta_caps_emotion(ws):
    spec = {
        "id": "explosive_joy",
        "enabled": True,
        "sensor": {"type": "noul", "question": "对方在狂喜？", "cues": {"positive": ["爆炸"]}},
        "high_threshold": 0.5,
        "low_threshold": 0.2,
        "cooldown_sec": 60,
        "max_delta": 0.1,
        "on_high": {"emotion_delta": {"caring": 0.9, "warmth": -0.9}},
    }
    write_list_module(ws.reflexes_file, "REFLEXES", [spec])
    engine, _ = make_engine(ws)
    scan = engine.scan("我爆炸开心", "owner", now_utc())
    assert "explosive_joy" in scan.fired
    assert scan.emotion["caring"] <= 0.1 + 1e-9
    assert scan.emotion["warmth"] >= -0.1 - 1e-9


def test_emotion_decays_toward_baseline(ws):
    state = EmotionState(baseline={"caring": 0.5}, values={"caring": 0.5})
    state.apply({"caring": 0.2})
    peak = state.values["caring"]
    assert peak == 0.7
    for _ in range(5):
        state.tick_decay()
    assert 0.5 < state.values["caring"] < peak
    for _ in range(200):
        state.tick_decay()
    assert abs(state.values["caring"] - 0.5) < 0.01


def test_emotion_store_respects_persona_baseline(ws):
    store = EmotionStore(ws)
    state = store.load()
    assert set(state.baseline) == {"warmth", "caring", "playfulness", "energy", "focus", "fatigue"}
    applied = state.apply({"caring": 0.5})  # 超过单拍上限
    assert applied["caring"] <= state.max_delta_per_tick + 1e-9
    store.save(state)
    assert ws.emotion_file.is_file()


def test_noul_choose_score_primitives(ws):
    engine, _ = make_engine(ws)
    noul = engine.noul("我又熬夜了", "在硬撑？", cues={"positive": ["熬夜"]})
    assert 0 <= noul.value <= 1 and noul.confidence > 0.5
    choose = engine.choose(
        "帮我看看这个 bug",
        "意图？",
        ["求助", "闲聊"],
        cues={"求助": ["帮我", "bug"], "闲聊": ["天气"]},
    )
    assert choose.value == "求助"
    score = engine.score("这很重要，记住", "重要度？", ["1", "2", "3", "4", "5"], cues={"positive": ["重要", "记住"]})
    assert score.value > 3
