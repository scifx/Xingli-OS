"""记忆库与睡眠整合：内容寻址、明文可重建、衰减、反失忆抽查。"""

from __future__ import annotations

from datetime import timedelta

from kernel.textutil import coverage, estimate_tokens
from memory.sleep_cycle import SleepCycle
from memory.store import MemoryStore


def test_content_addressed_and_deduplicated(ws):
    store = MemoryStore(ws)
    kwargs = dict(
        lod0="同一件事",
        lod1="要点",
        lod2="完整内容",
        source_id="owner",
        created_at="2026-11-26T00:00:00+00:00",
    )
    a = store.put(**kwargs)
    b = store.put(**kwargs)
    assert a.id == b.id
    assert len(list(ws.episodes.glob("*.json"))) == 1
    assert b.access_count >= 1


def test_memory_is_plaintext_and_reindexable(ws):
    store = MemoryStore(ws)
    node = store.put(lod0="明文记忆", lod1="要点", lod2="全文", source_id="owner")
    path = ws.episodes / f"{node.id}.json"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "明文记忆" in text and node.id in text
    assert not list(ws.episodes.glob("*.pkl"))

    # 索引丢了也不怕：明文文件是正典
    for suffix in ("", "-wal", "-shm"):
        p = ws.db_path.with_name(ws.db_path.name + suffix)
        p.unlink(missing_ok=True)
    fresh = MemoryStore(ws)
    assert fresh.list_nodes(None) == []
    assert fresh.get(node.id) is not None
    counts = fresh.reindex()
    assert counts["nodes"] == 1
    assert [n.id for n in fresh.list_nodes(None)] == [node.id]
    # 索引重建后隐私过滤仍然生效
    assert fresh.list_nodes("colleague_b") == []


def test_decay_is_ebbinghaus_and_spares_pinned(ws):
    store = MemoryStore(ws)
    normal = store.put(lod0="普通记忆", lod1="x", lod2="y", source_id="owner", importance=0.8)
    pinned = store.put(
        lod0="钉选承诺", lod1="x", lod2="y", source_id="owner", importance=0.8, pinned=True
    )
    future = __import__("kernel.models", fromlist=["now_utc"]).now_utc() + timedelta(days=30)
    changes = store.decay(7.0, 0.05, now=future)
    ids = {c[0] for c in changes}
    assert normal.id in ids
    assert pinned.id not in ids
    assert store.get(normal.id).importance <= 0.06
    assert store.get(pinned.id).importance == 0.8
    # 永不物理删除
    assert (ws.episodes / f"{normal.id}.json").is_file()


def test_sleep_distills_turns_into_episode(hb):
    for i in range(3):
        hb.handle_message("owner", f"第{i + 1}轮：我们在设计记忆的 LOD 视距，重点是 {i + 1}")
    assert hb.store.list_turns("owner")

    report = hb.sleep()
    assert report.distilled, "睡眠周期应当把原始轮次蒸馏成情景节点"
    assert report.archived_turn_ids
    assert hb.store.list_turns("owner") == [], "原始轮次已归档"
    nodes = [n for n in hb.store.list_nodes("owner") if "sleep" in n.tags]
    assert nodes
    node = nodes[0]
    assert node.lod0 and node.lod1 and node.lod2
    # 在认知帧里，长期地貌能看到它，而近期对话不再重复原文
    hb.handle_message("owner", "继续")
    frame = hb.last_frame.user
    assert node.id in frame or node.lod0[:10] in frame


def test_anti_amnesia_repairs_distorted_index(ws):
    store = MemoryStore(ws)
    node = store.put(
        lod0="主人要出差",
        lod1="主人下周三出差",
        lod2=(
            "主人说他下周三要去杭州参加 LOD 记忆会议，酒店订在西湖边，"
            "预算 3200 元，还要带上笔记本和转接头。"
        ),
        source_id="owner",
        importance=0.7,
    )
    before = coverage(f"{node.lod0}\n{node.lod1}", node.lod2)
    assert before < ws.config.sleep.coverage_threshold

    report = SleepCycle(ws, ws.config, store=store, seed=42).run(commit=False)
    assert len(report.samples) == 1
    sample = report.samples[0]
    assert sample.node_id == node.id
    assert not sample.passed
    assert sample.repaired and sample.new_id
    assert report.pass_rate_before == 0.0
    assert report.pass_rate_after == 1.0

    new_node = store.get(sample.new_id)
    assert new_node is not None
    assert coverage(f"{new_node.lod0}\n{new_node.lod1}", node.lod2) >= ws.config.sleep.coverage_threshold
    # 旧节点不是删除，而是降权沉入冷库
    old = store.get(node.id)
    assert old.archived and old.importance < 0.7
    assert node.id in new_node.refs


def test_sleep_commits_and_writes_report(ws):
    store = MemoryStore(ws)
    store.put(lod0="一条正常记忆", lod1="内容足够长以参与抽查", lod2="一条正常记忆，内容足够长以参与抽查。", source_id="owner")
    report = SleepCycle(ws, ws.config, store=store).run()
    assert report.ledger_commit
    reports = list(ws.sleep_reports.glob("*.json"))
    assert reports, "睡眠报告应当落盘"
    assert report.pass_rate_before == 1.0 == report.pass_rate_after


def test_sleep_on_empty_brain_is_safe(ws):
    report = SleepCycle(ws, ws.config, store=MemoryStore(ws)).run()
    assert report.pass_rate_before == 1.0
    assert report.pass_rate_after == 1.0
    assert not report.errors
    assert report.ledger_commit  # 至少留下一个健康快照


def test_frame_tokens_stay_below_budget_with_many_memories(hb):
    for i in range(30):
        hb.store.put(
            lod0=f"第{i}条长期记忆：关于并发、LOD 与哈希的设计讨论 {i}",
            lod1=f"第{i}条记忆的要点，涉及项目X的并发与记忆视距",
            lod2=f"第{i}条记忆的完整内容。" * 12,
            source_id="owner",
            importance=0.5 + (i % 5) / 10,
        )
    hb.handle_message("owner", "我们继续聊并发和 LOD 的设计")
    frame = hb.last_frame
    assert frame.total_tokens <= hb.config.budget.frame_tokens
    assert frame.landscape.level_counts()["LOD0"] > 0
    assert estimate_tokens(frame.user) < hb.config.budget.frame_tokens
