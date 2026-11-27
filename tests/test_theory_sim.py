"""理论沙盒的验收：手写算得对不对、模型是否真有界、Jev 是否真的在决定分叉。

这些测试不碰产品代码（不 import cli/kernel/memory），只验 theory/ 里的模型。
"""

from __future__ import annotations

import pytest

from theory import sim as S

GRAPH = S.load_graph("theory/seed_memory.json")


def drained(**over):
    st = dict(S.STATE_DRAINED)
    st.update(over)
    return st


# ---------------------------------------------------------------------------
# 1. 图本身：手写的文件必须是可用的
# ---------------------------------------------------------------------------
def test_graph_is_well_formed():
    ids = {n.id for n in GRAPH.nodes}
    for n in GRAPH.nodes:
        for e in n.edges:
            assert e.to in ids, f"{n.id} -> {e.to} 指向不存在的节点"
        # w 的含义是「把这个源节点这一拍的多少激活交出去」，所以出边和 ≤ 1
        assert sum(abs(e.w) for e in n.edges) <= 1.0 + 1e-9, n.id
    branches = [n.branch.id for n in GRAPH.nodes if n.branch]
    assert branches == ["b_fatigue", "b_ddl_pressure"]


def test_boundedness_premise_gamma_less_than_every_lambda():
    for row in GRAPH.audit():
        assert row["bounded"], row
        assert row["a_star"] <= 1.0


def test_unnormalised_linear_model_would_need_impossible_lambda():
    """不做除性归一化的话，λ 必须大于入边和；这张图有节点做不到。"""
    rows = [r for r in GRAPH.audit() if not r["linear_ok"]]
    assert len(rows) >= 5
    assert max(r["fanin"] for r in rows) > 1.0


# ---------------------------------------------------------------------------
# 2. 手写核对：每个数都从纸上抄进代码，不用循环、不用模拟器
# ---------------------------------------------------------------------------
def test_hand_transcribed_first_two_ticks_match_simulator():
    g = S.Sim(GRAPH, S.StubOracle())

    # ── t1：只有 cue_msg 被注入 1.0，别的都是 0 ─────────────────────
    t1 = g.tick(drained(), {"cue_msg": 1.0})
    lam = {n.id: n.lam for n in GRAPH.nodes}
    G = GRAPH.gamma
    a_cue = 1.0
    hand1 = {
        "cue_msg": (1 - lam["cue_msg"]) * a_cue,
        "n_fatigue": G * (0.45 * a_cue) / (1 + 0.45 * a_cue),
        "n_deadline": G * (0.30 * a_cue) / (1 + 0.30 * a_cue),
        "n_eyes": G * (0.15 * a_cue) / (1 + 0.15 * a_cue),
        "n_praise": G * (0.10 * a_cue) / (1 + 0.10 * a_cue),
    }
    for k, v in hand1.items():
        assert t1.a[k] == pytest.approx(v, abs=1e-12), k
    assert sum(1 for v in t1.a.values() if v > 0) == 5

    # ── t2：岔口 b_fatigue 被点亮（a=0.186 ≥ θ=0.10），p 由 stub 给 ──
    p = S.StubOracle().ask(
        [(next(n for n in GRAPH.nodes if n.id == "n_fatigue"),
          next(n for n in GRAPH.nodes if n.branch and n.branch.id == "b_fatigue").branch,
          drained())], 0)["b_fatigue"]
    assert p["care_sleep"] == pytest.approx(0.6875, abs=1e-3)   # 手算：1/(1+e^-0.7886)

    f = t1.a["n_fatigue"]
    d = t1.a["n_deadline"]
    e_ = t1.a["n_eyes"]
    pr = t1.a["n_praise"]

    s_sleep = 0.50 * p["care_sleep"] * f + 1.00 * e_ + 0.60 * pr
    s_stay = 0.50 * p["care_stay"] * f
    s_press = 0.65 * d
    s_fat = 0.45 * 1.0 + 0.35 * d

    hand2 = {
        "n_sleep_ritual": G * s_sleep / (1 + s_sleep),
        "n_stay_up": G * s_stay / (1 + s_stay),
        "n_pressure": G * s_press / (1 + s_press),
        "n_fatigue": (1 - lam["n_fatigue"]) * f + G * s_fat / (1 + s_fat),
        "n_deadline": (1 - lam["n_deadline"]) * d + G * 0.30 / 1.30,
        "n_eyes": (1 - lam["n_eyes"]) * e_ + G * 0.15 / 1.15,
    }
    t2 = g.tick(drained(), {"cue_msg": 1.0})
    for k, v in hand2.items():
        assert t2.a[k] == pytest.approx(v, abs=1e-12), k
    assert t2.branch_log and t2.branch_log[0]["branch"] == "b_fatigue"
    assert t2.oracle_calls == 1


def test_inhibition_pushes_the_other_policy_to_zero():
    g = S.Sim(GRAPH, S.StubOracle())
    for _ in range(12):
        st = g.tick(drained(), {"cue_msg": 1.0})
    assert st.a["n_sleep_ritual"] > 0.10
    assert st.a["n_stay_up"] == 0.0          # 被 -0.20 的抑制项压到 0（硬抑制）


# ---------------------------------------------------------------------------
# 3. 发散：有界、收敛、够不着的地方就不亮
# ---------------------------------------------------------------------------
def test_spread_is_bounded_and_converges():
    g = S.Sim(GRAPH, S.StubOracle(), k=len(GRAPH.nodes))
    mass = []
    for _ in range(30):
        mass.append(sum(g.tick(drained(), {"cue_msg": 1.0}).a.values()))
    bound = sum(GRAPH.gamma / n.lam for n in GRAPH.nodes)
    assert all(m <= bound + 1e-9 for m in mass)
    assert all(max(g.a.values()) <= GRAPH.gamma / 0.70 + 1e-9 for _ in [0])
    assert abs(mass[-1] - mass[-2]) < 1e-3
    # 冷区永远不亮：没有通路就是没有记忆，不能靠"猜"
    for cold in ("n_food", "n_movie", "n_exam", "n_work", "n_rain"):
        assert g.a[cold] == 0.0


def test_workmemory_cap_keeps_at_most_k():
    g = S.Sim(GRAPH, S.StubOracle(), k=4)
    st = g.tick(drained(), {"cue_msg": 1.0})
    assert sum(1 for v in st.a.values() if v > 0) == 4      # 5 个非零，只能留 4 个
    assert st.truncated                                     # 截断必须被记录下来


# ---------------------------------------------------------------------------
# 4. 岔口 = 决定来源；没有岔口，两个状态输出一模一样
# ---------------------------------------------------------------------------
def _margin(sim: S.Sim, state) -> tuple[float, list[str]]:
    for _ in range(6):
        st = sim.tick(state, {"cue_msg": 1.0})
    sl, su = st.a["n_sleep_ritual"], st.a["n_stay_up"]
    return (sl - su) / (sl + su) if sl + su else 0.0, [r["id"] for r in st.outputs["reflex"]]


def test_branch_is_what_makes_the_two_states_differ():
    with_branch = {}
    for name, state in (("drained", S.STATE_DRAINED), ("ok", S.STATE_OK)):
        with_branch[name] = _margin(S.Sim(GRAPH, S.StubOracle()), state)
    assert with_branch["drained"][0] > 0.8
    assert abs(with_branch["ok"][0]) < 0.15
    assert with_branch["drained"][1] != with_branch["ok"][1]

    without = {}
    for name, state in (("drained", S.STATE_DRAINED), ("ok", S.STATE_OK)):
        without[name] = _margin(S.Sim(GRAPH, S.StubOracle(), use_branch=False), state)
    assert without["drained"] == without["ok"]   # 不用 Jev → 输出与状态无关


def test_branch_question_only_asked_when_node_is_lit():
    g = S.Sim(GRAPH, S.StubOracle())
    st = g.tick(drained(), {"cue_msg": 1.0})     # n_fatigue=0.186 < θ? 不，0.186 ≥ 0.10
    assert st.oracle_calls == 0                  # 但 t1 的岔口判断用的是**上一拍**的激活
    st2 = g.tick(drained(), {"cue_msg": 1.0})
    assert st2.oracle_calls == 1                 # 一拍两个岔口也只发一次


# ---------------------------------------------------------------------------
# 5. 烘焙 → 探针 → 解冻
# ---------------------------------------------------------------------------
def test_bake_freezes_then_probe_catches_drift():
    g = S.Sim(GRAPH, S.StubOracle(), probe_every=1)
    for _ in range(6):
        g.tick(drained(), {"cue_msg": 1.0})
    assert any(b.branch_id == "b_fatigue" for b in g.baked.values())
    frozen = {b.branch_id: b.option for b in g.baked.values()}
    assert frozen["b_fatigue"] == "care_sleep"
    assert g.asks.get("b_fatigue", 0) <= 4       # 冻上之后就靠探针，不再每拍问

    for _ in range(6):
        g.tick(drained(ddl_days=0.0), {"cue_msg": 1.0})   # DDL 变成今晚
    assert g.unbaked_history, "状态漂移后探针没有抓到，冻结会一直错下去"
    rec = g.unbaked_history[0]
    assert rec["frozen"] == "care_sleep"
    assert rec["probe"]["care_stay"] > rec["probe"]["care_sleep"]
    assert rec["hits"] >= 3


def test_undecided_branch_is_never_baked():
    g = S.Sim(GRAPH, S.StubOracle(), probe_every=0)
    for _ in range(15):
        g.tick(drained(), {"cue_msg": 1.0})
    assert "b_ddl_pressure" not in {b.branch_id for b in g.baked.values()}
    assert g.asks["b_ddl_pressure"] >= 10        # 一直不确定 → 一直问


def test_granularity_error_is_monotone():
    rows = S.exp_granularity(GRAPH)["rows"]
    flips = [r["flip"] for r in rows]
    assert flips == sorted(flips, reverse=True)
    assert flips[0] > 0.3                        # 一个固定习惯 → 四成情况选错方向
    assert rows[-1]["cells"] > rows[0]["cells"]


# ---------------------------------------------------------------------------
# 6. 输出参数：是数字，不是话
# ---------------------------------------------------------------------------
def test_outputs_are_parameters():
    g = S.Sim(GRAPH, S.StubOracle())
    for _ in range(6):
        st = g.tick(drained(), {"cue_msg": 1.0})
    o = st.outputs
    assert o["event"] == ["cue_msg"]
    assert len(o["lod2"]) == 3 and not (set(o["lod2"]) & {"cue_msg"})
    assert all(r["id"].startswith("r_") for r in o["reflex"])
    assert all(abs(v) <= 0.20 for v in o["emotion"].values())
    assert "softness" in o["emotion"]            # 只有反射放电才产生情绪偏移
    assert isinstance(o["lod0"], list) and o["lod0"]


# ---------------------------------------------------------------------------
# 7. 绝不静默降级
# ---------------------------------------------------------------------------
def test_jev_failure_is_loud():
    g = S.Sim(GRAPH, S.HttpOracle("http://127.0.0.1:9/v1/systemone", timeout=0.2))
    with pytest.raises(SystemExit) as exc:
        g.tick(drained(), {"cue_msg": 1.0})
        g.tick(drained(), {"cue_msg": 1.0})
    assert "jev" in str(exc.value)


def test_jev_request_is_batched():
    o = S.HttpOracle("http://unused")
    node = next(n for n in GRAPH.nodes if n.id == "n_fatigue")
    br = node.branch
    other = next(n for n in GRAPH.nodes if n.id == "n_pressure")
    body = o.payload([(node, br, drained()), (other, other.branch, drained())])
    assert len(body["inputs"]) == 2
    assert body["inputs"][0]["type"] == "choose"
    assert set(body["inputs"][0]["options"]) == {"care_sleep", "care_stay"}
