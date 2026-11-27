#!/usr/bin/env python3
"""最小记忆节点图模拟器 —— 只做理论验证，不接产品、不碰 CLI/DB/LLM。

四件必须看得见的事：

1. **发散激活**：一条线索注入手写的记忆图（``theory/seed_memory.json``），
   逐拍扩散；每一拍打印「哪个节点、经由哪条边、拿到多少激活」。
2. **岔口问 Jev**：只有被点亮到阈值以上的岔口节点才生成问题，一拍的全部
   岔口**打包成一次请求**；返回的分布直接决定激活往哪个子节点流。
3. **烘焙（髓鞘化）**：同一岔口 + 同一状态签名连续 N 拍同向 → 冻结成
   ``(岔口, 签名) → 分布``，此后不再问 Jev；探针发现失配 → 解冻。
4. **输出参数**：收敛的激活向量 → LOD2/LOD1/LOD0、放电的反射、情绪偏移。

传播律（归一化，见 MODEL.md §2）::

    S_v = Σ_u  w_uv · g_uv · a_u            # 汇聚输入（g=1 普通边；g=p 岔口边）
    a'_v = clamp( (1-λ_v)·a_v + Γ·S_v/(1+|S_v|) )   # 除性归一化 + 漏电

这个形式让 a' 有上界 Γ/λ_v，与图的形状/入度无关；代价是它是非线性
（归一化），所以"稳定"是逐节点证的，不是谱半径证的。

纯 stdlib。时间单位 = tick（1 tick = 一条用户消息）。离线用 ``StubOracle``
（手写 logistic，标注 ⟨stub⟩）；``--jev-url`` 接真 Jev，失败即抛错，
**绝不静默降级**。

用法::

    python theory/sim.py --exp all
    python theory/sim.py --exp spread --verbose
    python theory/sim.py --exp bake --probe-every 2
    python theory/sim.py --exp granularity
    python theory/sim.py --print-request --jev-url http://127.0.0.1:8000/v1/systemone
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

FLOOR = 0.01          # 低于此值视为没点亮（避免浮点尾巴污染 top-k）
DEFAULT_GAMMA = 0.60  # 广播增益 Γ：一拍里一个节点把多少激活散出去


# ---------------------------------------------------------------------------
# 0. 量纲工具（全部无量纲，时间以 tick 为单位）
# ---------------------------------------------------------------------------
def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def lam_from_half_life(t_half: float) -> float:
    """λ = 1 − 2^(−1/T½)。T½ 以 tick 计。"""
    return 1.0 - 2.0 ** (-1.0 / t_half)


def half_life(lam: float) -> float:
    return math.log(0.5) / math.log(1.0 - lam)


def softmax(scores: Mapping[str, float]) -> dict[str, float]:
    m = max(scores.values())
    e = {k: math.exp(v - m) for k, v in scores.items()}
    z = sum(e.values())
    return {k: v / z for k, v in e.items()}


def is_late(hour: float) -> float:
    """深夜传感器（23:00–05:00）。这是传感器，不是 Jev 的输出。"""
    return 1.0 if (hour >= 23.0 or hour < 5.0) else 0.0


def ddl_near(ddl_days: float) -> float:
    return clamp(1.0 - ddl_days / 7.0)


def band(value: float, edges: list[float], names: list[str]) -> str:
    for edge, name in zip(edges, names):
        if value < edge:
            return name
    return names[-1]


# 烘焙签名用的离散化。**这是模型唯一的自由参数（粒度）**，见 VERDICT.md §4
BANDS: dict[str, Callable[[Mapping[str, float]], str]] = {
    "late": lambda s: "late" if is_late(s["hour"]) else "awake",
    "drain_band": lambda s: band(s["drain"], [0.4, 0.6], ["drain_low", "drain_mid", "drain_high"]),
    "ddl_band": lambda s: band(ddl_near(s["ddl_days"]), [0.4, 0.75],
                               ["ddl_far", "ddl_mid", "ddl_near"]),
}


# ---------------------------------------------------------------------------
# 1. 数据结构
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Edge:
    to: str
    w: float                 # 无量纲 [-1,1]，负 = 抑制；同一个源节点的 |w| 之和应 ≤ 1
    kind: str = "about"
    opt: str | None = None   # 属于哪个岔口选项；None = 普通边，恒通


@dataclass(frozen=True)
class Branch:
    id: str
    ask: str
    theta: float             # 与 a 同量纲、同量级
    options: dict[str, str]
    bake_sig: tuple[str, ...]
    stub: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class Node:
    id: str
    type: str
    lod0: str
    s: float                 # 存储强度 ∈[0,1]（不是传导率，不能当边权用）
    lam: float               # 每拍漏电率 (0,1]
    edges: tuple[Edge, ...]
    branch: Branch | None = None
    reflex: dict[str, Any] | None = None
    lod1: str = ""


@dataclass
class Graph:
    nodes: list[Node]
    meta: dict[str, Any] = field(default_factory=dict)
    gamma: float = DEFAULT_GAMMA

    def __post_init__(self) -> None:
        self.index = {n.id: n for n in self.nodes}
        self._inc: dict[str, list[tuple[Node, Edge]]] | None = None

    def incoming(self) -> dict[str, list[tuple[Node, Edge]]]:
        if self._inc is None:
            inc: dict[str, list[tuple[Node, Edge]]] = {n.id: [] for n in self.nodes}
            for n in self.nodes:
                for e in n.edges:
                    if e.to not in inc:
                        raise ValueError(f"边指向不存在的节点：{n.id} -> {e.to}")
                    inc[e.to].append((n, e))
            self._inc = inc
        return self._inc

    def out_sum(self, n: Node) -> float:
        return sum(abs(e.w) for e in n.edges)

    def audit(self) -> list[dict[str, Any]]:
        """两个可机械检查的前提：
        (a) Γ < λ_v            → 该节点有界（a* ≤ Γ/λ_v）
        (b) Σ_in|w| ≤ λ_v 的线性松弛 → 若不做归一化，需要多少次（只用来说明
            "线性模型必须归一化"，见 MODEL.md §2.3）
        (c) 每个源的 Σ_out|w| ≤ 1 → w 才配叫"传导比例"
        """
        inc = self.incoming()
        rows = []
        for n in self.nodes:
            fanin = sum(abs(e.w) for _, e in inc[n.id])
            rows.append(
                {
                    "id": n.id,
                    "lam": n.lam,
                    "fanin": round(fanin, 3),
                    "out": round(self.out_sum(n), 3),
                    "bounded": self.gamma < n.lam + 1e-12,
                    "a_star": round(self.gamma / n.lam, 3),
                    "linear_ok": n.lam > fanin + 1e-12,
                    "w_ok": self.out_sum(n) <= 1.0 + 1e-9,
                }
            )
        return rows


def load_graph(path: str | Path) -> Graph:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    nodes: list[Node] = []
    for item in raw["nodes"]:
        br = item.get("branch")
        branch = None
        if br:
            branch = Branch(
                id=br["id"],
                ask=br["ask"],
                theta=float(br["theta"]),
                options=dict(br["options"]),
                bake_sig=tuple(br.get("bake_sig", ())),
                stub=dict(br.get("stub", {})),
            )
        nodes.append(
            Node(
                id=item["id"],
                type=item["type"],
                lod0=item["lod0"],
                lod1=item.get("lod1", ""),
                s=float(item["s"]),
                lam=float(item["lam"]),
                edges=tuple(
                    Edge(to=e["to"], w=float(e["w"]), kind=e.get("kind", "about"),
                         opt=e.get("opt"))
                    for e in item.get("edges", [])
                ),
                branch=branch,
                reflex=item.get("reflex"),
            )
        )
    return Graph(nodes=nodes, meta=raw.get("meta", {}),
                 gamma=float(raw.get("meta", {}).get("gamma", DEFAULT_GAMMA)))


# ---------------------------------------------------------------------------
# 2. 岔口 oracle
# ---------------------------------------------------------------------------
class Oracle:
    """一次 ask = 一拍里**全部**岔口，打包成一个请求。"""

    tag = "?"

    def __init__(self) -> None:
        self._calls = 0

    def ask(self, batch, t: int) -> dict[str, dict[str, float]]:
        raise NotImplementedError

    def calls(self) -> int:
        return self._calls


class StubOracle(Oracle):
    """手写 logistic 替身：语义是「Jev 大概会怎么答」，不是「Jev 答了什么」。

    特征只来自场景给定的传感器数值（hour/drain/ddl_days）。真实 Jev 应直接读
    原话 + 上下文；所以这里**有语义损失**——离线跑通 ≠ 真实 Jev 也这样答。
    """

    tag = "stub"

    def __init__(self) -> None:
        super().__init__()

    def ask(self, batch, t):
        self._calls += 1
        out: dict[str, dict[str, float]] = {}
        for _node, br, state in batch:
            feats = {
                "drain": state["drain"],
                "not_drain": 1.0 - state["drain"],
                "late": is_late(state["hour"]),
                "ddl_near": ddl_near(state["ddl_days"]),
            }
            scores = {
                opt: float(spec.get("bias", 0.0))
                + sum(float(w) * feats.get(f, 0.0) for f, w in spec.get("w", {}).items())
                for opt, spec in br.stub.items()
            }
            out[br.id] = softmax(scores)
        return out


class HttpOracle(Oracle):
    """把一拍的所有岔口打包成一次 HTTP 请求。**协议未在真实 Jev 上验证过**。

    ``--print-request`` 只打印请求体、不发送。把 body 贴给 Jev、告诉我真实形状，
    我再改 —— 上一轮就是猜了个 /systemone 然后当成已验证，不重复。
    """

    tag = "jev"

    def __init__(self, url: str, api_key: str = "", timeout: float = 3.0) -> None:
        super().__init__()
        self.url, self.api_key, self.timeout = url, api_key, timeout
        self.dry = False

    def payload(self, batch) -> dict[str, Any]:
        return {
            "inputs": [
                {"id": br.id, "type": "choose", "question": br.ask,
                 "options": list(br.options)}
                for _node, br, _state in batch
            ]
        }

    def ask(self, batch, t):
        body = json.dumps(self.payload(batch), ensure_ascii=False, indent=2)
        if self.dry:
            print("── 将要发送的请求体（未发送）────────────")
            print(body)
            raise SystemExit(0)
        req = urllib.request.Request(
            self.url, data=body.encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        self._calls += 1
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise SystemExit(f"⟨jev⟩ 请求失败：{exc}（没有静默降级到 stub）") from exc
        return _parse_choose(data, batch)


def _parse_choose(payload, batch) -> dict[str, dict[str, float]]:
    if isinstance(payload, list):
        items: Iterable[Any] = payload
    elif isinstance(payload, dict):
        items = payload.get("outputs") or payload.get("results") or payload.get("inputs") or []
    else:
        raise SystemExit(f"⟨jev⟩ 无法理解的返回类型：{type(payload).__name__}")
    by_id = {br.id: br for _n, br, _s in batch}
    out: dict[str, dict[str, float]] = {}
    for item in items:
        bid = item.get("id") or item.get("spec_id")
        probs = item.get("probabilities") or item.get("distribution") or item.get("probs")
        if bid in by_id and isinstance(probs, dict):
            out[bid] = {k: float(v) for k, v in probs.items()}
    missing = set(by_id) - set(out)
    if missing:
        raise SystemExit(f"⟨jev⟩ 返回缺岔口：{sorted(missing)}（不做默认填充）")
    return out


# ---------------------------------------------------------------------------
# 3. 烘焙账本
# ---------------------------------------------------------------------------
@dataclass
class Baked:
    branch_id: str
    sig: tuple[str, ...]
    option: str
    p: float
    first_tick: int
    last_tick: int
    hits: int = 1
    suspects: int = 0

    @property
    def key(self) -> tuple[str, tuple[str, ...]]:
        return (self.branch_id, self.sig)

    def label(self) -> str:
        return f"{self.branch_id}@{'/'.join(self.sig) or '-'}"


# ---------------------------------------------------------------------------
# 4. 模拟器
# ---------------------------------------------------------------------------
@dataclass
class Step:
    t: int
    a: dict[str, float]
    spread: list[str]
    branch_log: list[dict[str, Any]]
    truncated: list[str]
    outputs: dict[str, Any]
    oracle_calls: int
    baked_used: list[str]


class Sim:
    def __init__(self, graph: Graph, oracle: Oracle, k: int = 10, bake_after: int = 3,
                 bake_p: float = 0.68, probe_every: int = 0, suspend: int = 2,
                 verbose: bool = False, use_branch: bool = True) -> None:
        self.g = graph
        self.oracle = oracle
        self.k = k
        self.bake_after = bake_after
        self.bake_p = bake_p
        self.probe_every = probe_every
        self.suspend = suspend
        self.verbose = verbose
        self.use_branch = use_branch
        self.a: dict[str, float] = {n.id: 0.0 for n in graph.nodes}
        self.baked: dict[tuple[str, tuple[str, ...]], Baked] = {}
        self.streak: dict[tuple[str, tuple[str, ...]], list[str]] = {}
        self.prob_ema: dict[tuple[str, tuple[str, ...]], float] = {}
        self.uses: dict[tuple[str, tuple[str, ...]], int] = {}
        self.unbaked_history: list[dict[str, Any]] = []
        self.asks: dict[str, int] = {}     # 每个岔口「真问了几次」
        self.probes: dict[str, int] = {}   # 每个岔口「冻结后探针问了几次」
        self.log: list[Step] = []

    # -- 4.1 一拍 ---------------------------------------------------------
    def tick(self, state: Mapping[str, float], inject: Mapping[str, float] | None = None,
             label: str = "") -> Step:
        t = len(self.log) + 1
        a0 = dict(self.a)
        for nid, v in (inject or {}).items():
            a0[nid] = clamp(a0[nid] + v)

        branch_log: list[dict[str, Any]] = []
        baked_used: list[str] = []
        factors: dict[tuple[str, int], float] = {}

        # (1) 岔口：只有被点亮的才问；一拍全部打包
        batch: list[tuple[Node, Branch, Mapping[str, float]]] = []
        plan: list[tuple[Node, int, tuple[str, ...], Baked | None]] = []
        for n in self.g.nodes:
            if n.branch is None or not self.use_branch or a0[n.id] < n.branch.theta:
                continue
            sig = tuple(BANDS[f](state) for f in n.branch.bake_sig) if n.branch.bake_sig else ()
            key = (n.branch.id, sig)
            hit = self.baked.get(key)
            if hit is None:
                batch.append((n, n.branch, state))
                plan.append((n, len(batch) - 1, sig, None))
                continue
            hit.last_tick = t
            hit.hits += 1
            uses = self.uses[key] = self.uses.get(key, 0) + 1
            baked_used.append(f"{hit.label()} → {hit.option} (p={hit.p:.2f}, 冻于 t{hit.first_tick}, 命中 {hit.hits})")
            if self.probe_every > 0 and uses % self.probe_every == 0:
                batch.append((n, n.branch, state))
                plan.append((n, len(batch) - 1, sig, hit))
            else:
                plan.append((n, -1, sig, hit))

        answers = self.oracle.ask(batch, t) if batch else {}
        calls = 1 if batch else 0

        for node, bi, sig, hit in plan:
            if bi < 0:
                continue
            br = node.branch
            assert br is not None
            probs = answers[br.id]
            if hit is None:
                self.asks[br.id] = self.asks.get(br.id, 0) + 1
                self._learn(br, sig, probs, t, branch_log)
            else:
                self.probes[br.id] = self.probes.get(br.id, 0) + 1
                self._probe((br.id, sig), probs, t, branch_log)

        for node, _bi, _sig, hit in plan:
            br = node.branch
            assert br is not None
            probs = {hit.option: hit.p} if hit is not None else answers[br.id]
            for idx, e in enumerate(node.edges):
                if e.opt is not None:
                    factors[(node.id, idx)] = probs.get(e.opt, 0.0)

        # (2) 汇聚输入 S_v，再除性归一化
        s_in = {n.id: 0.0 for n in self.g.nodes}
        spread: list[str] = []
        for n in self.g.nodes:
            src = a0[n.id]
            for idx, e in enumerate(n.edges):
                gate = 1.0 if (e.opt is None or not self.use_branch) else factors.get((n.id, idx), 0.0)
                c = e.w * gate * src
                if abs(c) < 1e-12:
                    continue
                s_in[e.to] += c
                tag = "" if gate == 1.0 else f"×p={gate:.2f}"
                spread.append(f"{n.id}({src:.3f}) {'抑制' if c < 0 else '→'} {e.to}"
                              f"  w={e.w:+.2f}{tag} = {c:+.4f}")

        # (3) 漏电 + 归一化 + 工作记忆截断
        new = {}
        g_ = self.g.gamma
        for n in self.g.nodes:
            s = s_in[n.id]
            new[n.id] = clamp((1.0 - n.lam) * a0[n.id] + g_ * s / (1.0 + abs(s)))
        ordered = sorted(new.items(), key=lambda kv: (-kv[1], kv[0]))
        keep = {k for k, v in ordered[: self.k] if v >= FLOOR}
        truncated = [k for k, v in ordered if k not in keep and v > 0]
        self.a = {k: (v if k in keep else 0.0) for k, v in new.items()}

        step = Step(t, dict(self.a), spread, branch_log, truncated,
                    outputs_of(self.g, self.a), calls, baked_used)
        self.log.append(step)
        if self.verbose:
            self._print(step, state, label)
        return step

    # -- 4.2 烘焙 / 探针 ------------------------------------------------
    def _learn(self, br: Branch, sig: tuple[str, ...], probs: Mapping[str, float],
               t: int, log: list[dict[str, Any]]) -> None:
        key = (br.id, sig)
        opt = max(probs, key=lambda k: probs[k])
        p = probs[opt]
        log.append({"tick": t, "branch": br.id, "sig": list(sig),
                    "probs": {k: round(v, 3) for k, v in probs.items()},
                    "choice": opt, "source": self.oracle.tag, "baked": False})
        if key in self.baked:
            return
        streak = self.streak.setdefault(key, [])
        if streak and streak[-1] != opt:
            streak.clear()
        streak.append(opt)
        ema = self.prob_ema.get(key, p)
        self.prob_ema[key] = 0.5 * ema + 0.5 * p
        if len(streak) >= self.bake_after and self.prob_ema[key] >= self.bake_p:
            entry = Baked(branch_id=br.id, sig=sig, option=opt, p=self.prob_ema[key],
                          first_tick=t, last_tick=t)
            self.baked[key] = entry
            streak.clear()
            log.append({"tick": t, "branch": br.id, "sig": list(sig), "choice": opt,
                        "source": "烘焙", "baked": True,
                        "note": f"{self.bake_after} 拍同向、p̄={entry.p:.2f} ≥ {self.bake_p} → 冻结"})

    def _probe(self, key, probs, t, log) -> None:
        entry = self.baked.get(key)
        if entry is None:
            return
        opt = max(probs, key=lambda k: probs[k])
        p = probs[opt]
        if opt != entry.option or p < 0.5:
            entry.suspects += 1
            log.append({"tick": t, "branch": entry.branch_id, "sig": list(entry.sig),
                        "probs": {k: round(v, 3) for k, v in probs.items()}, "choice": opt,
                        "source": f"探针({self.oracle.tag})", "baked": True,
                        "note": f"冻结={entry.option}(p={entry.p:.2f}) vs 现在={opt}(p={p:.2f})"
                                f" → 失配 {entry.suspects}/{self.suspend}"})
            if entry.suspects >= self.suspend:
                del self.baked[key]
                self.unbaked_history.append({
                    "branch": entry.branch_id, "sig": list(entry.sig),
                    "frozen": entry.option, "p_at_bake": round(entry.p, 3),
                    "tick": t, "probe": {k: round(v, 3) for k, v in probs.items()},
                    "lifetime": entry.last_tick - entry.first_tick + 1, "hits": entry.hits,
                })
                log.append({"tick": t, "branch": entry.branch_id, "sig": list(entry.sig),
                            "choice": opt, "source": "解冻", "baked": False,
                            "note": "连续失配 → 撤回冻结点；出处保留在 unbaked_history"})
                self.streak.pop(key, None)
                self.prob_ema.pop(key, None)
        elif entry.suspects:
            entry.suspects = 0

    # -- 4.3 打印 -------------------------------------------------------
    def _print(self, st: Step, state, label: str) -> None:
        head = f"── t{st.t}"
        if label:
            head += f"  {label}"
        head += f"   (hour={state['hour']:.1f} drain={state['drain']:.2f} ddl={state['ddl_days']:.1f})"
        print(head)
        for bl in st.branch_log:
            probs = "/".join(f"{k} {v:.2f}" for k, v in bl.get("probs", {}).items())
            print(f"   ⟨岔口⟩ {bl['branch']:<14} sig={','.join(bl['sig']) or '-'}  "
                  f"[{bl['source']}] {probs}" + (f"  ⤷ {bl['note']}" if bl.get("note") else ""))
        for line in st.spread:
            print("     " + line)
        lit = [f"{k}={v:.3f}" for k, v in sorted(st.a.items(), key=lambda kv: -kv[1]) if v > 0]
        print("   激活  " + (" ".join(lit) if lit else "（全灭）"))
        for b in st.baked_used:
            print(f"   ⟨烘焙⟩ {b}    ← 0 请求")
        if st.truncated:
            print(f"   ⟨截断⟩ {' '.join(st.truncated)}")
        o = st.outputs
        print(f"   输出  event={o['event']}  lod2={o['lod2']}  lod1={o['lod1']}")
        print(f"         reflex={[r['id'] for r in o['reflex']]}  emotion={o['emotion']}"
              f"  请求={st.oracle_calls}")


# ---------------------------------------------------------------------------
# 5. 输出参数
# ---------------------------------------------------------------------------
def outputs_of(g: Graph, a: Mapping[str, float], cap: float = 0.20) -> dict[str, Any]:
    """激活向量 → 下一拍要用的**数字**。用排序而不是绝对阈值：绝对阈值要标定。"""
    live = sorted(((n.id, a[n.id]) for n in g.nodes
                   if a[n.id] >= FLOOR and n.type != "sensor"),
                  key=lambda kv: (-kv[1], kv[0]))
    event = [n.id for n in g.nodes if n.type == "sensor" and a[n.id] > 0]
    lod2 = [k for k, _ in live[:3]]
    lod1 = [k for k, _ in live[3:7]]
    lod0 = [n.id for n in g.nodes
            if n.type != "sensor" and n.id not in lod2 and n.id not in lod1]
    fired, emotion = [], {}
    for n in g.nodes:
        if not n.reflex:
            continue
        thr = float(n.reflex.get("threshold", 0.05))
        if a[n.id] >= thr:
            # 满量程增益：a 是 [0,1] 的比例量，但实测稳态只有 0.1–0.2，
            # 直接乘 max_delta 永远用不满。full = 「这条反射满强度的激活量」。
            full = float(n.reflex.get("full", 0.25))
            gain = clamp(a[n.id] / full) if full > 0 else 1.0
            fired.append({"id": n.id, "a": round(a[n.id], 3), "gain": round(gain, 3),
                          "hint": n.reflex.get("hint", "")})
            for k, d in n.reflex.get("delta", {}).items():
                emotion[k] = emotion.get(k, 0.0) + float(d) * gain
    emotion = {k: round(max(-cap, min(cap, v)), 3) for k, v in sorted(emotion.items())}
    return {"event": event, "lod2": lod2, "lod1": lod1, "lod0": lod0,
            "reflex": fired, "emotion": emotion}


# ---------------------------------------------------------------------------
# 6. 实验
# ---------------------------------------------------------------------------
STATE_DRAINED = {"hour": 2.2, "drain": 0.80, "ddl_days": 2.0}
STATE_OK = {"hour": 2.2, "drain": 0.25, "ddl_days": 2.0}
STATE_DDL_TONIGHT = {"hour": 2.2, "drain": 0.80, "ddl_days": 0.0}


def run_ticks(sim: Sim, state, ticks: int, label: str = "", inject: str = "cue_msg",
              sustain: bool = True):
    """sustain=True：这条消息在整个片段里都是「当前事件」（每拍重新注入）。

    这才对应真实用法：a_t 是**本轮**的激活场，一句还在进行中的话不会突然凉掉。
    """
    steps = []
    for i in range(ticks):
        inj = {inject: 1.0} if (i == 0 or sustain) else None
        steps.append(sim.tick(state, inj, label=f"{label}#{i + 1}"))
    return steps


def exp_spread(g: Graph, verbose: bool = True) -> dict[str, Any]:
    print("=" * 78)
    print("实验 1 · 发散激活：一句话能点亮多少节点，几拍收敛，会不会淹掉全图")
    print("=" * 78)
    sim = Sim(g, StubOracle(), verbose=verbose)
    steps = run_ticks(sim, STATE_DRAINED, 6, label="发散")
    mass = [round(sum(s.a.values()), 4) for s in steps]
    lit = [sum(1 for v in s.a.values() if v > 0) for s in steps]
    print()
    print(f"  激活总量/拍 : {mass}")
    print(f"  点亮节点数  : {lit}  （全图 {len(g.nodes)}）")
    audit = g.audit()
    bad = [r["id"] for r in audit if not r["bounded"]]
    print(f"  有界性 Γ<λ  : {'全部通过' if not bad else '违规 ' + str(bad)}"
          f"   （Γ={g.gamma:.2f}，λ_min={min(n.lam for n in g.nodes):.2f}，"
          f"a* 上界 {max(r['a_star'] for r in audit):.3f}）")
    lin = [r["id"] for r in audit if not r["linear_ok"]]
    print(f"  若不做归一化，必须 λ>Σ|w|：{len(lin)}/{len(audit)} 个节点做不到"
          f"（例：{lin[0]} 需要 λ>{next(r['fanin'] for r in audit if r['id'] == lin[0]):.2f}）"
          if lin else "  线性模型也能成立")
    wbar = statistics.fmean([abs(e.w) for n in g.nodes for e in n.edges])
    print(f"  平均边权 w̄={wbar:.3f}；按 a_d ≈ Γ·w̄^d 估可见深度 = "
          f"{math.log(FLOOR / g.gamma) / math.log(wbar):.1f} 跳（θ={FLOOR}）")
    ever = {k for st in steps for k, v in st.a.items() if v > 0}
    dark = [n.id for n in g.nodes if n.id not in ever]
    print(f"  六拍内从未点亮（够不着）: {dark}")
    print(f"  收敛：|Δ总量|(t5→t6) = {abs(mass[-1] - mass[-2]):.5f}"
          f"   稳态总量 {mass[-1]}（上界 Σ Γ/λ = {sum(r['a_star'] for r in audit):.2f}）")
    return {"mass": mass, "lit": lit, "dark": dark, "audit": audit}


def exp_decisive(g: Graph, use_branch: bool = True, tag: str = "") -> dict[str, Any]:
    print("=" * 78)
    print(("实验 2 · 自我抉择：同一句话 + 两个内部状态" if use_branch
           else "实验 2' · 对照：岔口常通（等价于不用 Jev）") + tag)
    print("=" * 78)
    out: dict[str, Any] = {}
    table = []
    for name, state in (("疲惫 drain=.80", STATE_DRAINED), ("还行 drain=.25", STATE_OK)):
        sim = Sim(g, StubOracle(), verbose=True, use_branch=use_branch)
        steps = run_ticks(sim, state, 6, label=name)
        f = steps[-1].outputs
        a = steps[-1].a
        probs = {}
        for st in steps:
            for bl in st.branch_log:
                if bl.get("probs"):
                    probs[bl["branch"]] = bl["probs"]
        sl, su = a.get("n_sleep_ritual", 0.0), a.get("n_stay_up", 0.0)
        margin = (sl - su) / (sl + su) if (sl + su) > 0 else 0.0
        out[name] = {"lod2": f["lod2"], "reflex": [r["id"] for r in f["reflex"]],
                     "emotion": f["emotion"], "margin": round(margin, 3), "probs": probs}
        table.append((name, probs, sl, su, margin, f))
        print()
    print("  ── 输出参数对照 ────────────────────────────────────────────")
    print("  状态            b_fatigue p(劝睡)   a(劝睡)   a(陪熬)   偏向   放电的反射")
    for name, probs, sl, su, margin, f in table:
        p_sleep = probs.get("b_fatigue", {}).get("care_sleep", float("nan"))
        print(f"  {name:<16} {p_sleep:>10.2f}   {sl:>8.3f}  {su:>8.3f}  {margin:>+6.2f}   "
              f"{[r['id'] for r in f['reflex']]}")
    a, b = out["疲惫 drain=.80"], out["还行 drain=.25"]
    same = a["lod2"] == b["lod2"] and a["emotion"] == b["emotion"]
    out["same"] = same
    print(f"\n  两组输出是否相同：{'相同 → 决定不来自图，只来自岔口' if same else '不同 → 岔口把状态翻译成了不同通路'}")
    print(f"  偏向 margin = (a_劝睡 − a_陪熬)/(a_劝睡 + a_陪熬)：{a['margin']:+.2f} vs {b['margin']:+.2f}"
          f"   ← 置信度不用问 LLM，从场的形状里直接读")
    return out


def exp_bake(g: Graph, probe_every: int = 2) -> dict[str, Any]:
    print("=" * 78)
    print(f"实验 3 · 烘焙：连续同向 → 冻结 → 状态变了 → 探针失配 → 解冻（探针每 {probe_every} 次用一次）")
    print("=" * 78)
    sim = Sim(g, StubOracle(), verbose=True, probe_every=probe_every)
    s1 = run_ticks(sim, STATE_DRAINED, 6, label="同向")
    print()
    print("  ── 状态突变：DDL 从 2 天变成今晚（hour/drain 不变 → 烘焙签名相同）")
    s2 = run_ticks(sim, STATE_DDL_TONIGHT, 5, label="突变")
    calls = [s.oracle_calls for s in s1] + [s.oracle_calls for s in s2]
    print()
    print(f"  每拍请求数：{calls}   合计 {sum(calls)}（打包：一拍最多 1 次）")
    print("  岔口                真问次数   探针次数")
    for bid in sorted(sim.asks, key=lambda k: -sim.asks[k]):
        print(f"  {bid:<18} {sim.asks[bid]:>8}   {sim.probes.get(bid, 0):>8}")
    print(f"  仍冻结的：{[b.label() + '→' + b.option for b in sim.baked.values()]}")
    print(f"  解冻记录：{json.dumps(sim.unbaked_history, ensure_ascii=False)}")
    return {"calls": calls, "asks": sim.asks, "probes": sim.probes,
            "baked": [(b.label(), b.option, round(b.p, 3), b.first_tick)
                      for b in sim.baked.values()],
            "unbaked": sim.unbaked_history}


def exp_granularity(g: Graph) -> dict[str, Any]:
    """烘焙 = 对 Jev 函数做「按签名分片取常值」的近似。

    两个指标：Δp（冻结值与原值的差）和**翻转率**（argmax 变了没有）。
    真正要命的是后者：Δp 再大，只要 argmax 不变，行为就一样。
    """
    print("=" * 78)
    print("实验 4 · 粒度：签名越细 → 冻结点越多、误差与翻转率越低（可量的权衡）")
    print("=" * 78)
    grid = [{"hour": h, "drain": round(0.05 * i, 2), "ddl_days": float(d)}
            for h in (1.0, 15.0, 23.0) for i in range(2, 20) for d in range(0, 7)]
    variants = {
        "无签名": (),
        "late": ("late",),
        "drain": ("drain_band",),
        "late+drain": ("late", "drain_band"),
        "late+drain+ddl": ("late", "drain_band", "ddl_band"),
    }
    stub = StubOracle()
    node = next(n for n in g.nodes if n.branch and n.branch.id == "b_fatigue")
    br = node.branch
    assert br is not None
    truth = {i: stub.ask([(node, br, st)], 0)[br.id] for i, st in enumerate(grid)}
    rows = []
    for name, sig in variants.items():
        cells: dict[tuple, list[int]] = {}
        for i, st in enumerate(grid):
            cells.setdefault(tuple(BANDS[f](st) for f in sig), []).append(i)
        dp, flips, worst = [], 0, 0.0
        for key, idxs in cells.items():
            frozen = {o: statistics.fmean(truth[i][o] for i in idxs) for o in br.options}
            for i in idxs:
                d = abs(truth[i]["care_sleep"] - frozen["care_sleep"])
                dp.append(d)
                worst = max(worst, d)
                if max(truth[i], key=truth[i].get) != max(frozen, key=frozen.get):
                    flips += 1
        rows.append({"sig": name, "cells": len(cells),
                     "dp_mean": round(statistics.fmean(dp), 3),
                     "dp_max": round(worst, 3),
                     "flip": round(flips / len(grid), 3)})
        print(f"  {name:<18} 单元 {len(cells):>3}   Δp 平均 {statistics.fmean(dp):.3f}  "
              f"最坏 {worst:.3f}   决策翻转率 {flips / len(grid):.1%}")
    print(f"\n  网格 {len(grid)} 个状态点（hour × drain × ddl）。")
    print("  『无签名』一行就是「靠一个固定习惯处理所有情况」的错误率上限。")
    print("  样本成本 = 单元数 × 烘焙所需拍数；翻转率才是验收指标。")
    return {"rows": rows, "grid": len(grid)}


def exp_stability(g: Graph) -> dict[str, Any]:
    """压力测试：把边权整体放大（等价于提高入度/汇聚输入），看激活场何时被钉在 1。

    定理只给上界 a* ≤ Γ/λ；能不能逼近上界，取决于 S = Σ w·a 有多大。
    """
    print("=" * 78)
    print("实验 5 · 有界性压力测试：边权放大 → S 变大 → 激活被截断在 1（显著性消失）")
    print("=" * 78)
    rows = []
    print("  w×     最大 a   top3 占比   a₃/a₁₀   最小/最大   饱和节点")
    for scale in (1.0, 2.0, 3.0, 5.0, 8.0):
        d = json.loads(json.dumps(_as_dict(g)))
        for n in d["nodes"]:
            for e in n["edges"]:
                e["w"] = round(e["w"] * scale, 4)
        gg = _from_dict(d)
        sim = Sim(gg, StubOracle(), k=len(gg.nodes))
        st = None
        for _ in range(40):
            st = sim.tick(STATE_DRAINED, {"cue_msg": 1.0})
        assert st is not None
        vals = sorted((v for v in st.a.values() if v > 0), reverse=True)
        tot = sum(vals)
        share = sum(vals[:3]) / tot
        ratio = vals[2] / vals[9] if len(vals) > 9 else float("nan")
        sat = sum(1 for v in vals if v >= 0.999)
        rows.append({"w_scale": scale, "max_a": round(vals[0], 3),
                     "top3_share": round(share, 3), "a3_a10": round(ratio, 2), "saturated": sat})
        print(f"  {scale:<4.1f}  {vals[0]:>7.3f}  {share:>8.3f}  {ratio:>6.2f}  "
              f"{vals[-1] / vals[0]:>8.3f}  {sat:>8}")
    print("\n  失败模式不是「发散爆炸」，而是「对比度塌缩」：入边越强，除性归一化把大家压得越平，")
    print("  top-3 的排序失去信息 —— 这时 LOD2 选谁、反射先后，都变成噪声。")
    print(f"  本图 Γ={g.gamma:.2f}、λ_min={min(n.lam for n in g.nodes):.2f}，上界 a*≤Γ/λ="
          f"{g.gamma / min(n.lam for n in g.nodes):.3f}（从未触及，因为 S 一直被压在 1 以下）")
    return {"rows": rows}


def exp_cost(g: Graph) -> dict[str, Any]:
    print("=" * 78)
    print("实验 6 · 代价：一拍几次请求、几个 token")
    print("=" * 78)
    sim = Sim(g, StubOracle())
    calls = [sim.tick(STATE_DRAINED, {"cue_msg": 1.0}).oracle_calls for _ in range(8)]
    branches = [n.branch for n in g.nodes if n.branch]
    q = sum(len(b.ask) + sum(len(o) for o in b.options.values()) + 24 for b in branches)
    print(f"  岔口数 {len(branches)}（{', '.join(b.id for b in branches)}）")
    print(f"  每拍请求数：{calls}（上限 1，与节点数无关）")
    print(f"  一拍的问题文本 ≈ {q} 字（≈{q} tokens，CJK 1:1）")
    for rtt in (0.12, 0.30):
        print(f"  RTT={rtt * 1000:.0f}ms → 1 拍额外 {rtt:.2f}s")
    return {"calls": calls, "q_tokens": q}


# ---------------------------------------------------------------------------
# 7. JSON 往返（粒度和稳定性实验要改参数）
# ---------------------------------------------------------------------------
def _as_dict(g: Graph) -> dict[str, Any]:
    return {
        "meta": {**g.meta, "gamma": g.gamma},
        "nodes": [
            {
                "id": n.id, "type": n.type, "lod0": n.lod0, "lod1": n.lod1,
                "s": n.s, "lam": n.lam,
                "edges": [{"to": e.to, "w": e.w, "kind": e.kind, "opt": e.opt} for e in n.edges],
                **({"branch": {"id": n.branch.id, "ask": n.branch.ask,
                               "theta": n.branch.theta, "bake_sig": list(n.branch.bake_sig),
                               "options": n.branch.options, "stub": n.branch.stub}}
                   if n.branch else {}),
                **({"reflex": n.reflex} if n.reflex else {}),
            }
            for n in g.nodes
        ],
    }


def _from_dict(d: dict[str, Any]) -> Graph:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                     encoding="utf-8") as fh:
        json.dump(d, fh, ensure_ascii=False)
        name = fh.name
    try:
        return load_graph(name)
    finally:
        Path(name).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 8. main
# ---------------------------------------------------------------------------
EXPERIMENTS = {
    "spread": exp_spread,
    "decisive": exp_decisive,
    "nobranch": lambda g: exp_decisive(g, use_branch=False, tag="（对照）"),
    "bake": exp_bake,
    "granularity": exp_granularity,
    "stability": exp_stability,
    "cost": exp_cost,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="最小记忆节点图模拟器（理论验证）")
    ap.add_argument("--graph", default=str(Path(__file__).with_name("seed_memory.json")))
    ap.add_argument("--exp", default="all", choices=[*EXPERIMENTS, "all"])
    ap.add_argument("--jev-url", default="")
    ap.add_argument("--jev-key", default="")
    ap.add_argument("--print-request", action="store_true", help="只打印请求体，不发送")
    ap.add_argument("--probe-every", type=int, default=2)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    g = load_graph(args.graph)
    audit = g.audit()
    hard = [r for r in audit if not r["bounded"]]
    if hard:
        print("⟨拒绝运行⟩ Γ ≥ λ：这些节点无界，激活会饱和到 1，显著性消失")
        for r in hard:
            print(f"   {r['id']:<16} Γ={g.gamma:.2f}  λ={r['lam']:.2f}")
        return 2
    over = [r for r in audit if not r["w_ok"]]
    if over:
        print(f"⟨警告⟩ 这些节点的出边 |w| 之和 > 1，w 不再是「传导比例」：{[r['id'] for r in over]}")

    oracle: Oracle | None = None
    if args.jev_url:
        h = HttpOracle(args.jev_url, args.jev_key)
        h.dry = args.print_request
        oracle = h
    elif args.print_request:
        oracle = HttpOracle("http://unused")
        oracle.dry = True

    results: dict[str, Any] = {}
    names = list(EXPERIMENTS) if args.exp == "all" else [args.exp]
    for name in names:
        if oracle is not None and isinstance(oracle, HttpOracle):
            results[name] = _with_oracle(name, g, oracle, args)
        else:
            results[name] = EXPERIMENTS[name](g)
        print()
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    return 0


def _with_oracle(name: str, g: Graph, oracle: Oracle, args) -> Any:
    if name in ("spread", "decisive", "nobranch"):
        sim = Sim(g, oracle, verbose=True,
                  use_branch=(name != "nobranch"))
        run_ticks(sim, STATE_DRAINED, 6, label=name)
        return {"oracle": oracle.tag}
    if name == "bake":
        sim = Sim(g, oracle, verbose=True, probe_every=args.probe_every)
        run_ticks(sim, STATE_DRAINED, 6, label="同向")
        run_ticks(sim, STATE_DDL_TONIGHT, 5, label="突变")
        return {"oracle": oracle.tag,
                "baked": [b.label() for b in sim.baked.values()],
                "unbaked": sim.unbaked_history}
    return EXPERIMENTS[name](g)


if __name__ == "__main__":
    sys.exit(main())
