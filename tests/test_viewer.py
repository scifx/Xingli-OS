"""CLI 的记忆可视化：快照、求差、`/cat` 路径解析。"""

from __future__ import annotations

import json

import viewer
from memory.store import MemoryStore


def test_snapshot_and_diff(tmp_path):
    root = tmp_path / "cortex"
    (root / "episodes").mkdir(parents=True)
    (root / "episodes" / "a.json").write_text("{}", encoding="utf-8")
    before = viewer.snapshot(root)
    assert set(before) == {"episodes/a.json"}

    (root / "episodes" / "b.json").write_text('{"lod0": "新记忆"}', encoding="utf-8")
    (root / "episodes" / "a.json").write_text('{"x": 1}', encoding="utf-8")
    changed = viewer.diff(before, viewer.snapshot(root))
    assert changed.added == ["episodes/b.json"]
    assert changed.modified == ["episodes/a.json"]
    assert changed.changed and changed.count == 2


def test_diff_ignores_pycache(tmp_path):
    root = tmp_path / "cortex"
    (root / "__pycache__").mkdir(parents=True)
    (root / "__pycache__" / "x.pyc").write_bytes(b"junk")
    assert viewer.snapshot(root) == {}


def test_format_diff_shows_episode_preview(ws):
    store = MemoryStore(ws)
    store.put(
        lod0="下周三要去看牙医",
        lod1="要点",
        lod2="全文",
        source_id="owner",
        epistemic="direct",
        visibility=["owner"],
        importance=0.8,
    )
    changed = viewer.diff({}, viewer.snapshot(ws.cortex))
    text = viewer.format_diff(changed, ws.cortex, store)
    assert "记忆变化" in text
    assert "episodes/" in text
    assert "下周三要去看牙医" in text
    assert "direct" in text


def test_format_diff_empty_when_nothing_changed(ws):
    assert viewer.format_diff(viewer.diff({}, {}), ws.cortex, None) == ""


def test_tree_and_list_memories(ws):
    store = MemoryStore(ws)
    store.put(lod0="第一条", lod1="要点", lod2="全文", source_id="owner")
    tree = viewer.tree(ws.cortex)
    assert "episodes/" in tree and "个文件" in tree
    listing = viewer.list_memories(store, "owner", ws.episodes)
    assert "第一条" in listing and "对话轮次" in listing


def test_resolve_memory_path(ws, hb):
    store = MemoryStore(ws)
    node = store.put(lod0="可被 cat 的记忆", lod1="要点", lod2="全文", source_id="owner")
    assert viewer.resolve_memory_path(node.id, ws, store) == ws.episodes / f"{node.id}.json"
    assert viewer.resolve_memory_path(f"episodes/{node.id}.json", ws, store) is not None

    hb.handle_message("owner", "你好")
    turns = sorted(ws.turns.glob("*.json"))
    assert turns
    resolved = viewer.resolve_memory_path(turns[0].stem, ws, store)
    assert resolved == turns[0]
    data = json.loads(resolved.read_text(encoding="utf-8"))
    assert data["speaker_id"] == "owner"
    assert viewer.resolve_memory_path("不存在的东西", ws, store) is None


def test_human_size():
    assert viewer.human_size(512) == "512B"
    assert viewer.human_size(2048) == "2.0KB"
