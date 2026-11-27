"""M1 验收：焦点式双向 LOD 的数学正确性与 Token 预算。"""

from __future__ import annotations

import math

import pytest

from kernel.models import LOD2, FocusState, FocusUpdate, TurnRecord
from kernel.textutil import estimate_tokens
from memory.focal_context import FocalContext, clarity
from memory.store import MemoryStore


def add_turns(store: MemoryStore, count: int, speaker: str = "owner") -> list[TurnRecord]:
    turns = []
    for i in range(1, count + 1):
        turn = TurnRecord(
            turn_id=store.next_turn_id(),
            speaker_id=speaker,
            user_text=f"第{i}轮：我们在讨论 LOD 记忆的第 {i} 个细节",
            assistant_text=f"收到第{i}轮的回应",
        )
        turn.lod1 = f"第{i}轮要点"
        turn.lod0 = f"第{i}轮"
        turns.append(store.add_turn(turn))
    return turns


# ---------------------------------------------------------------------------
# 1. 公式
# ---------------------------------------------------------------------------
def test_clarity_matches_formula_exactly():
    f, db, da = 10, 0.35, 0.2
    assert clarity(f, f, db, da) == 1.0
    for d in (1, 2, 5, 30):
        assert clarity(f - d, f, db, da) == pytest.approx(math.exp(-db * d))
        assert clarity(f + d, f, db, da) == pytest.approx(math.exp(-da * d))
    # 同样距离，焦点之前衰减更快（刚发生的事更清楚）
    assert clarity(f - 5, f, db, da) < clarity(f + 5, f, db, da)


# ---------------------------------------------------------------------------
# 2. LOD 分层与阈值
# ---------------------------------------------------------------------------
def test_lod_levels_follow_thresholds(ws):
    store = MemoryStore(ws)
    add_turns(store, 20)
    focal = FocalContext(ws, store, ws.config)
    state = FocusState(focus_turn_id=15, decay_before=0.35, decay_after=0.2, recency_floor=2)
    focal.save_state(state)
    snap = focal.snapshot("owner")
    by_id = {row["turn_id"]: row for row in snap["levels"]}
    cfg = ws.config.render

    for tid, row in by_id.items():
        if tid > 18:  # recency floor
            assert row["lod"] == "LOD2", (tid, row)
            continue
        r = clarity(tid, 15, 0.35, 0.2)
        assert row["R"] == pytest.approx(round(r, 4), abs=1e-4)
        if r >= cfg.theta2:
            assert row["lod"] == "LOD2"
        elif r >= cfg.theta1:
            assert row["lod"] == "LOD1"
        elif r >= cfg.theta0:
            assert row["lod"] == "LOD0"
        else:
            assert row["lod"] == "COLD"

    # 焦点之前更远 → 更早进入冷区；焦点之后的第 16~18 轮仍然清楚
    assert by_id[1]["lod"] == "COLD"
    assert by_id[16]["lod"] == "LOD2"


# ---------------------------------------------------------------------------
# 3. 自主调焦
# ---------------------------------------------------------------------------
def test_set_focus_moves_spotlight(ws):
    store = MemoryStore(ws)
    add_turns(store, 20)
    focal = FocalContext(ws, store, ws.config)
    focal.save_state(FocusState(focus_turn_id=20, recency_floor=2))

    before = {row["turn_id"]: row for row in focal.snapshot("owner")["levels"]}
    assert before[3]["lod"] == "COLD"

    visible = [t.turn_id for t in store.list_turns("owner")]
    state, warnings = focal.apply_update(
        FocusUpdate(turn_id=3, decay_before=0.8, decay_after=0.12, pin=[3]), visible
    )
    assert not warnings
    assert state.focus_turn_id == 3
    after = {row["turn_id"]: row for row in focal.snapshot("owner")["levels"]}
    assert after[3]["lod"] == "LOD2"
    assert after[3]["R"] == 1.0
    # 焦点之后的近况被保留（decay_after 小）
    assert after[10]["lod"] in {"LOD1", "LOD2"}
    # 更早的寒暄被压暗（同一距离下比焦点之后衰减快得多）
    assert after[1]["lod"] in {"LOD0", "COLD"}
    assert after[1]["R"] < after[10]["R"]


def test_focus_update_rejects_invisible_turn(ws):
    store = MemoryStore(ws)
    add_turns(store, 3)
    focal = FocalContext(ws, store, ws.config)
    state, warnings = focal.apply_update(FocusUpdate(turn_id=99), [1, 2, 3])
    assert warnings and state.focus_turn_id == 3


# ---------------------------------------------------------------------------
# 4. 钉选
# ---------------------------------------------------------------------------
def test_pinned_turn_stays_lod2(ws):
    store = MemoryStore(ws)
    add_turns(store, 15)
    focal = FocalContext(ws, store, ws.config)
    focal.save_state(
        FocusState(focus_turn_id=15, decay_before=0.35, decay_after=0.2, pinned_turn_ids=[2])
    )
    snap = {row["turn_id"]: row for row in focal.snapshot("owner")["levels"]}
    assert snap[2]["lod"] == "LOD2"


# ---------------------------------------------------------------------------
# 5. 预算：降级而不是溢出
# ---------------------------------------------------------------------------
def test_render_respects_tight_budget(ws):
    store = MemoryStore(ws)
    add_turns(store, 12)
    focal = FocalContext(ws, store, ws.config)
    render = focal.render("owner", budget_tokens=120)
    assert render.tokens <= 120
    # 仍然保留最近两轮的保底
    levels = {e.turn_id: e.level for e in render.entries}
    assert levels.get(11) == LOD2 and levels.get(12) == LOD2


def test_cold_region_has_handles(ws):
    store = MemoryStore(ws)
    add_turns(store, 20)
    focal = FocalContext(ws, store, ws.config)
    focal.save_state(FocusState(focus_turn_id=20, decay_before=0.6))
    render = focal.render("owner", budget_tokens=1500)
    assert "冷区" in render.text
    assert "recall_turn" in render.text
    # 冷区里的轮次仍可从存储器里拉回全文
    cold = render.cold_turns[0]
    assert store.get_turn(cold) is not None


# ---------------------------------------------------------------------------
# 6. 30 轮：帧长受控，且不随轮数增长
# ---------------------------------------------------------------------------
def test_thirty_turn_frame_budget(hb):
    tokens = []
    for i in range(30):
        result = hb.handle_message("owner", f"第{i + 1}轮：继续聊 LOD 记忆的工程细节，编号 {i + 1}")
        assert result.ticks, "每轮至少一拍"
        tokens.append(hb.last_frame.total_tokens)
        assert hb.last_frame.total_tokens <= hb.config.budget.frame_tokens

    assert max(tokens) <= hb.config.budget.frame_tokens
    early = sum(tokens[:10]) / 10
    late = sum(tokens[-10:]) / 10
    # 帧长不随对话增长（30 轮后与 10 轮时同一量级）
    assert late <= early * 1.25 + 40, (early, late)

    # 继续聊 40 轮：帧长仍然不增长（这是“恒定锐度”的核心承诺）
    for i in range(30, 70):
        hb.handle_message("owner", f"第{i + 1}轮：继续聊 LOD 记忆的工程细节，编号 {i + 1}")
        assert hb.last_frame.total_tokens <= hb.config.budget.frame_tokens
    later = hb.last_frame.total_tokens
    assert later <= early * 1.4 + 60, (early, later)

    # 而且确实远小于“把 70 轮原始对话塞进去”
    raw = "\n".join(
        t.lod2 for t in hb.store.list_turns("owner", include_archived=True)
    )
    assert estimate_tokens(raw) > estimate_tokens(hb.last_frame.user) * 2


# ---------------------------------------------------------------------------
# 7. 帧里确实包含五段
# ---------------------------------------------------------------------------
def test_frame_sections(hb):
    hb.handle_message("owner", "你好")
    frame = hb.last_frame
    for marker in (
        "[SITUATIONAL_ENV]",
        "[SYS1_INTUITION]",
        "[FOCAL_DIALOGUE]",
        "[LONG_TERM_LANDSCAPE]",
        "[CURRENT_EVENT]",
    ):
        assert marker in frame.user
    assert len(frame.messages()) == 2
    assert frame.section_tokens["event"] > 0
