"""长期记忆地貌渲染器（游戏 LOD）+ 物理级 Visibility ACL（铁律 3）。

核心不是“检索 Top-K”，而是“渲染一整片有纵深的地貌”：

* 近景 LOD2：与此刻最相关 / 被钉选的少数节点，展开全文；
* 中景 LOD1：结构性摘要，成本低一个量级；
* 远景 LOD0：一句话目录 + 哈希句柄，常驻视野地平线——
  星璃**始终看得见自己有什么**，需要细节再 ``recall(id)`` 把镜头拉近。

物理防火墙：``store.list_nodes(speaker_id)`` 在 SQL 层就过滤掉越权节点，
本模块拿到的候选集里根本不存在别人的私密记忆——不靠 prompt 祈祷。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Iterable, Sequence

from kernel.config import Config
from kernel.literal import load_literal
from kernel.models import LOD0, LOD1, LOD2, LOD_COLD, LOD_NAMES, MemoryNode, now_utc, parse_iso
from kernel.paths import Workspace
from kernel.textutil import estimate_tokens, one_line, similarity, truncate
from memory.store import MemoryStore

TrustLookup = Callable[[str], float]


def _default_trust(person: str) -> float:
    return {"owner": 1.0}.get(person, 0.5)


@dataclass
class LandscapeEntry:
    node: MemoryNode
    score: float
    level: int
    line: str = ""

    @property
    def level_name(self) -> str:
        return LOD_NAMES[self.level]


@dataclass
class LandscapeRender:
    text: str
    entries: list[LandscapeEntry] = field(default_factory=list)
    tokens: int = 0
    candidates: int = 0
    hidden_by_acl: int = 0
    policy_lines: list[str] = field(default_factory=list)
    skill_lines: list[str] = field(default_factory=list)

    def level_counts(self) -> dict[str, int]:
        counts = {"LOD2": 0, "LOD1": 0, "LOD0": 0}
        for e in self.entries:
            counts[e.level_name] = counts.get(e.level_name, 0) + 1
        return counts

    def horizon_ids(self) -> list[str]:
        return [e.node.id for e in self.entries if e.level == LOD0]


class LodRenderer:
    def __init__(
        self,
        ws: Workspace,
        store: MemoryStore,
        config: Config | None = None,
        trust_lookup: TrustLookup | None = None,
    ):
        self.ws = ws
        self.store = store
        self.config = config or ws.config
        self.trust_lookup = trust_lookup or _default_trust

    # ------------------------------------------------------------------
    # 心理距离打分（mhtml 讨论中的 Score(m) 公式）
    # ------------------------------------------------------------------
    def score(
        self,
        node: MemoryNode,
        speaker_id: str,
        query: str = "",
        boost_tags: Iterable[str] = (),
        now: datetime | None = None,
    ) -> float:
        cfg = self.config.render
        now = now or now_utc()
        semantic = similarity(query, f"{node.lod0} {node.lod1} {' '.join(node.tags)}")
        age_days = max(
            0.0, (now - parse_iso(node.created_at)).total_seconds() / 86400.0
        )
        half_life = max(1e-6, self.config.sleep.half_life_days)
        recency = 0.5 ** (age_days / half_life)
        importance = node.importance
        if node.source_id == speaker_id or speaker_id in node.about_ids:
            social = 1.0
        elif "owner" in node.about_ids:
            social = 0.6
        else:
            social = 0.25
        boosts = set(boost_tags)
        boost = 1.0 if boosts & set(node.tags) else 0.0
        total = (
            cfg.weight_semantic * semantic
            + cfg.weight_recency * recency
            + cfg.weight_importance * importance
            + cfg.weight_social * social
            + cfg.weight_boost * boost
        )
        # 钉选与“被反复想起”的节点轻微上调，避免刚写下的重要承诺被淹没
        if node.pinned:
            total += 0.08
        return min(1.0, total)

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------
    def render(
        self,
        speaker_id: str,
        query: str = "",
        budget_tokens: int | None = None,
        boost_tags: Iterable[str] = (),
        now: datetime | None = None,
        include_directory: bool = True,
    ) -> LandscapeRender:
        cfg = self.config.render
        budget = budget_tokens or self.config.budget.section_landscape
        visible = self.store.list_nodes(speaker_id)
        total_nodes = len(self.store.list_nodes(None))
        hidden = max(0, total_nodes - len(visible))

        ranked: list[LandscapeEntry] = []
        for node in visible:
            s = self.score(node, speaker_id, query, boost_tags, now)
            ranked.append(LandscapeEntry(node=node, score=s, level=LOD_COLD))
        # 稳定排序：分数 → 重要度 → 新近
        ranked.sort(key=lambda e: (-e.score, -e.node.importance, e.node.created_at))

        pinned = [e for e in ranked if e.node.pinned]
        rest = [e for e in ranked if not e.node.pinned]
        n2 = max(0, cfg.landscape_n2 - len(pinned))
        n1 = cfg.landscape_n1
        n0 = cfg.landscape_n0
        for e in pinned:
            e.level = LOD2
        for e in rest[:n2]:
            e.level = LOD2
        for e in rest[n2 : n2 + n1]:
            e.level = LOD1
        for e in rest[n2 + n1 : n2 + n1 + n0]:
            e.level = LOD0

        policy_lines, skill_lines = (
            self.directory(speaker_id) if include_directory else ([], [])
        )

        for _ in range(len(ranked) * 2 + 8):
            text = self._compose(ranked, speaker_id, policy_lines, skill_lines)
            tokens = estimate_tokens(text)
            if tokens <= budget:
                break
            if not self._demote_one(ranked):
                text = self._compose(ranked, speaker_id, policy_lines, skill_lines)
                tokens = estimate_tokens(text)
                break

        text = self._compose(ranked, speaker_id, policy_lines, skill_lines)
        tokens = estimate_tokens(text)
        kept = [e for e in ranked if e.level != LOD_COLD]
        kept.sort(key=lambda e: (e.level != LOD2, -e.score))
        return LandscapeRender(
            text=text,
            entries=kept,
            tokens=tokens,
            candidates=len(visible),
            hidden_by_acl=hidden,
            policy_lines=policy_lines,
            skill_lines=skill_lines,
        )

    @staticmethod
    def _demote_one(entries: list[LandscapeEntry]) -> bool:
        for level, target in ((LOD2, LOD1), (LOD1, LOD0), (LOD0, LOD_COLD)):
            pool = [e for e in entries if e.level == level and not e.node.pinned]
            if not pool:
                continue
            pool.sort(key=lambda e: e.score)
            pool[0].level = target
            return True
        return False

    def _annotation(self, node: MemoryNode, speaker_id: str) -> str:
        parts = [f"重要{node.importance:.2f}"]
        source = node.source_id
        if node.epistemic == "direct":
            parts.append(f"来源:{source}｜亲历")
        elif node.epistemic == "hearsay":
            trust = self.trust_lookup(source)
            parts.append(f"来源:{source}｜转述待核实｜信源信任{trust:.2f}")
        else:
            parts.append(f"来源:{source}｜推测")
        if node.visibility and "*" not in node.visibility:
            parts.append(f"可见:{'/'.join(node.visibility)}")
        if node.pinned:
            parts.append("钉选")
        return "〔" + "·".join(parts) + "〕"

    def _compose(
        self,
        entries: Sequence[LandscapeEntry],
        speaker_id: str,
        policy_lines: Sequence[str],
        skill_lines: Sequence[str],
    ) -> str:
        cfg = self.config.render
        counts = {"LOD2": 0, "LOD1": 0, "LOD0": 0}
        for e in entries:
            if e.level != LOD_COLD:
                counts[e.level_name] += 1
        shown = counts["LOD2"] + counts["LOD1"] + counts["LOD0"]
        lines = [
            f"近景 LOD2={counts['LOD2']} ｜ 中景 LOD1={counts['LOD1']} ｜"
            f" 远景 LOD0={counts['LOD0']} ｜ 视野内 {shown} 条"
        ]
        for e in entries:
            if e.level == LOD_COLD:
                continue
            node = e.node
            ann = self._annotation(node, speaker_id)
            if e.level == LOD2:
                body = truncate(node.lod2 or node.lod1, cfg.node_lod2_chars)
                lines.append(f"#{node.id} 〔LOD2·相关{e.score:.2f}〕{ann}\n{body}")
            elif e.level == LOD1:
                lines.append(f"#{node.id} 〔LOD1·相关{e.score:.2f}〕{ann} {node.lod1}")
            else:
                lines.append(f"#{node.id} 〔LOD0·相关{e.score:.2f}〕{ann} {node.lod0}")
        if policy_lines:
            lines.append("已内化的准则：" + " ｜ ".join(policy_lines))
        if skill_lines:
            lines.append("已学会的技能：" + " ｜ ".join(skill_lines))
        if counts["LOD0"] == 0 and counts["LOD1"] == 0 and counts["LOD2"] == 0:
            lines.append("（长期记忆还是空的）")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 技能 / 准则目录（也属于“地平线”的一部分）
    # ------------------------------------------------------------------
    def directory(self, speaker_id: str = "owner") -> tuple[list[str], list[str]]:
        policies_raw = load_literal(self.ws.policies_file, "POLICIES", default=[]) or []
        policy_lines: list[str] = []
        visible: list[str] = []
        for item in policies_raw:
            if isinstance(item, dict):
                text = str(item.get("rule") or item.get("text") or "").strip()
                if not text:
                    continue
                vis = [str(v) for v in (item.get("visibility") or [])]
                if vis and "*" not in vis and speaker_id not in vis:
                    continue  # 准则是某个人的私事 → 对另一个人物理不存在
            elif isinstance(item, str):
                text = item
            else:
                continue
            visible.append(one_line(text, 70))
        policy_lines = visible[-6:]

        skill_lines: list[str] = []
        for path in sorted(self.ws.skills.glob("*.py")):
            if path.name.startswith("_"):
                continue
            sig = self._skill_signature(path)
            if sig:
                skill_lines.append(sig)
        return policy_lines, skill_lines[:10]

    @staticmethod
    def _skill_signature(path) -> str:
        import ast

        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, OSError):
            return ""
        doc = ast.get_docstring(tree) or ""
        fns = [
            n.name
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith("_")
        ]
        if not fns and not doc:
            return ""
        desc = one_line(doc, 40) if doc else ""
        return f"{path.stem}({'/'.join(fns[:2])}) {desc}".strip()
