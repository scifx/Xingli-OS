"""记忆“该拒的拒、该并的并”：用你真实跑出来的脏数据当夹具。

线上那 22 条记忆里：``lod0="一句话"``、``lod1="0.85"``、
``lod0="主人亲密度0.85，信任值1.00"``、同一句名字的 7 个变体……
这里把这条路径逐个钉死。
"""

from __future__ import annotations

import json

from kernel.boot import Heartbeat
from kernel.memtext import is_richer, mem_key, same_fact
from kernel.paths import Workspace
from memory.hygiene import tidy
from memory.store import MemoryStore


class Brain:
    name = "brain"

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


def _memorize(*ops):
    return {"inner_monologue": "记", "say": "好", "memory_ops": list(ops)}


# ---------------------------------------------------------------------------
# 文本判据
# ---------------------------------------------------------------------------
def test_mem_key_collapses_real_world_variants():
    variants = [
        "2026-11-27 20:21 主人偏好：喜欢菠萝和芒果",
        "主人喜欢菠萝和芒果",
        "主人偏好菠萝与芒果",
        "owner喜欢吃菠萝和芒果",
    ]
    keys = {mem_key(v) for v in variants}
    assert keys == {"菠萝芒果"}
    assert same_fact(variants[0], variants[3])
    assert not same_fact("主人喜欢菠萝", "主人不喜欢菠萝")


def test_is_richer_only_for_more_detail():
    assert is_richer("owner喜欢食用菠萝和芒果，偏好清甜的果香口感", "主人喜欢菠萝和芒果")
    assert not is_richer("主人喜欢菠萝", "主人喜欢菠萝和芒果")


# ---------------------------------------------------------------------------
# 写入端：垃圾进不来
# ---------------------------------------------------------------------------
def test_placeholder_and_numeric_memory_is_rejected(ws: Workspace, clock):
    hb = _hb(
        ws,
        clock,
        Brain(
            _memorize(
                {"op": "memorize", "lod0": "一句话", "lod1": "两三句要点", "lod2": "原话"},
                {"op": "memorize", "lod0": "0.85"},
                {"op": "memorize", "lod0": "主人亲密度0.85，信任值1.00"},
            )
        ),
    )
    result = hb.handle_message("owner", "随便聊聊")
    assert not result.ticks[0].actions.memorized
    notes = " ".join(result.ticks[0].actions.notes)
    assert "占位符" in notes and "状态回显" in notes
    assert not list(ws.episodes.glob("*.json"))


def test_numeric_lod1_lod2_are_repaired_with_lod0(ws: Workspace, clock):
    hb = _hb(
        ws,
        clock,
        Brain(
            _memorize(
                {"op": "memorize", "lod0": "主人喜欢菠萝", "lod1": "0.85", "lod2": "0.7"}
            )
        ),
    )
    result = hb.handle_message("owner", "我喜欢菠萝")
    node_id = result.ticks[0].actions.memorized[0]
    node = hb.store.get(node_id)
    assert node.lod1 == "主人喜欢菠萝"
    assert node.lod2 == "主人喜欢菠萝"
    assert "记忆清洗" in " ".join(result.ticks[0].actions.notes)


def test_same_fact_written_twice_does_not_create_a_second_file(ws: Workspace, clock):
    hb = _hb(
        ws,
        clock,
        Brain(
            _memorize({"op": "memorize", "lod0": "2026-11-27 20:21 主人偏好：喜欢菠萝和芒果"}),
            _memorize({"op": "memorize", "lod0": "主人偏好菠萝与芒果"}),
        ),
    )
    hb.handle_message("owner", "我喜欢菠萝和芒果")
    result = hb.handle_message("owner", "对了，我喜欢菠萝和芒果")
    assert len(list(ws.episodes.glob("*.json"))) == 1
    assert any("去重" in n for n in result.ticks[0].actions.notes)


def test_richer_version_upgrades_and_archives_the_older(ws: Workspace, clock):
    hb = _hb(
        ws,
        clock,
        Brain(
            _memorize({"op": "memorize", "lod0": "主人喜欢菠萝和芒果"}),
            _memorize(
                {
                    "op": "memorize",
                    "lod0": "主人喜欢菠萝和芒果，还偏好清甜的果香口感",
                }
            ),
        ),
    )
    first = hb.handle_message("owner", "我喜欢菠萝和芒果")
    old_id = first.ticks[0].actions.memorized[0]
    second = hb.handle_message("owner", "我喜欢菠萝和芒果，要清甜的那种")
    assert hb.store.get(old_id).archived is True
    assert any("记忆升级" in n for n in second.ticks[0].actions.notes)
    live = [n for n in hb.store.list_nodes("owner") if "菠萝" in n.lod0]
    assert len(live) == 1 and "清甜" in live[0].lod0


# ---------------------------------------------------------------------------
# 存量端：/tidy
# ---------------------------------------------------------------------------
POLLUTED = [
    ("485d69b0fd85", "2026-11-27 20:21 主人偏好：喜欢菠萝和芒果", "0.85", 0.6),
    ("87b910025fd7", "一句话", "两三句要点", 0.62),
    ("bf939ed76198", "主人名字：星璃", "主人偏好：菠萝、芒果", 0.86),
    ("a1154935a10e", "主人名字为星璃", "主人偏好菠萝与芒果", 0.88),
    ("62ab811929dd", "主人亲密度0.85，信任值1.00", "主人喜欢菠萝、芒果", 0.86),
    ("ddbfa43f922f", "主人喜欢菠萝和芒果", "主人名字是星璃", 0.97),
]


def _seed(ws: Workspace, rows) -> MemoryStore:
    ws.episodes.mkdir(parents=True, exist_ok=True)
    for node_id, lod0, lod1, importance in rows:
        payload = {
            "id": node_id, "kind": "episode", "lod0": lod0, "lod1": lod1, "lod2": lod1,
            "source_id": "owner", "about_ids": ["owner"], "epistemic": "direct",
            "visibility": ["owner"], "importance": importance, "pinned": False,
            "tags": [], "emotion": {}, "refs": [],
            "created_at": "2026-11-27T12:21:31+00:00",
            "updated_at": "2026-11-27T12:21:31+00:00",
            "last_accessed": "2026-11-27T12:21:31+00:00",
            "access_count": 3, "archived": False, "schema_version": 1,
        }
        (ws.episodes / f"{node_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
    store = MemoryStore(ws)
    store.reindex()
    return store


def test_tidy_dry_run_changes_nothing(ws: Workspace, clock):
    store = _seed(ws, POLLUTED)
    report = tidy(store, "owner", archive=False)
    assert report.junk and report.duplicates
    assert len(store.list_nodes("owner")) == len(POLLUTED)


def test_tidy_archives_junk_and_merges_duplicates(ws: Workspace, clock):
    store = _seed(ws, POLLUTED)
    report = tidy(store, "owner")
    live = store.list_nodes("owner")
    assert len(live) == 2, [n.lod0 for n in live]
    texts = " ".join(n.lod0 for n in live)
    assert "菠萝" in texts
    assert "亲密度" not in texts and "一句话" not in texts
    assert all(n.lod1 for n in live)
    assert "0.85" not in " ".join(n.lod1 for n in live)
    # 归档而不是删除：文件还在，能 /rollback
    archived = [n for n in store.list_nodes("owner", include_archived=True) if n.archived]
    assert len(archived) == len(POLLUTED) - len(live)
    assert report.changed == len(POLLUTED) - len(live) + len(report.repaired)


# ---------------------------------------------------------------------------
# 重要度不该被“召回次数”顶到天花板
# ---------------------------------------------------------------------------
def test_recall_reinforcement_is_bounded(ws: Workspace, clock):
    store = MemoryStore(ws)
    node = store.put(lod0="主人今天喝了咖啡", lod1="", lod2="", importance=0.5)
    for _ in range(20):
        store.touch(node.id)
    refreshed = store.get(node.id)
    assert refreshed.importance <= 0.62 + 1e-6, refreshed.importance
    assert refreshed.access_count == 20
