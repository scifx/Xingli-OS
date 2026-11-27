"""小模型的“不按契约命名”：接住它，但把降级记录下来。"""

from __future__ import annotations

from kernel.models import ThoughtPacket


def test_canonical_packet_has_no_warnings():
    packet = ThoughtPacket.from_dict(
        {
            "inner_monologue": "先听清楚。",
            "emotion_delta": {"warmth": 0.1},
            "memory_ops": [{"op": "recall", "mem_id": "abc123"}],
            "say": "在的。",
            "wake": {},
        }
    )
    assert packet.inner_monologue == "先听清楚。"
    assert packet.say == "在的。"
    assert packet.parse_warnings == []


def test_alias_fields_are_accepted_with_a_warning():
    packet = ThoughtPacket.from_dict(
        {
            "monologue": "想一想",
            "reply": "我在听",
            "emotion": {"caring": 0.1},
            "memories": [{"op": "recall", "mem_id": "abcdef"}],
        }
    )
    assert packet.inner_monologue == "想一想"
    assert packet.say == "我在听"
    assert packet.emotion_delta == {"caring": 0.1}
    assert packet.memory_ops[0]["mem_id"] == "abcdef"
    assert any("monologue→inner_monologue" in w for w in packet.parse_warnings)


def test_nested_shell_is_unwrapped():
    packet = ThoughtPacket.from_dict({"thought_packet": {"inner_monologue": "解包", "say": "好"}})
    assert packet.inner_monologue == "解包"
    assert packet.say == "好"
    assert any("已解包" in w for w in packet.parse_warnings)


def test_single_memory_op_object_is_wrapped_into_list():
    packet = ThoughtPacket.from_dict(
        {"inner_monologue": "记一下", "memory_ops": {"op": "memorize", "lod0": "某事"}}
    )
    assert isinstance(packet.memory_ops, list)
    assert packet.memory_ops[0]["lod0"] == "某事"
    assert any("单元素数组" in w for w in packet.parse_warnings)


def test_unknown_shape_stays_empty_but_records_warning():
    packet = ThoughtPacket.from_dict({"foo": 1, "bar": [1, 2]})
    assert packet.inner_monologue == ""
    assert packet.say is None
    assert packet.parse_warnings  # 不猜测、不半执行，只记录
