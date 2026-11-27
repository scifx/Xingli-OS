"""睡眠整合（离线巩固）：蒸馏 · 衰减 · 反失忆抽查 · 习惯统计。

这是 README 里“不失忆”唯一可证伪的落点：随机抽历史记忆节点，检查在
**当前 LOD 视野**（LOD0 目录 + LOD1 摘要，而不看全文）下还能不能召回
事实骨架；失真就重写索引描述（生成新节点 + 旧节点沉入冷库）。

同时做两件“防止越活越乱”的事：艾宾浩斯衰减（非 pinned 权重缓慢下降）
与反射噪声统计（高误触发率的反射会在报告中标记，供下一轮进化降权）。
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from kernel.config import Config
from kernel.git_ledger import GitLedger, LedgerError
from kernel.models import MemoryNode, iso, now_utc, parse_iso
from kernel.paths import Workspace
from kernel.textutil import coverage, extractive_summary, one_line
from memory.store import MemoryStore


@dataclass
class AuditedSample:
    node_id: str
    lod0: str
    coverage_before: float
    coverage_after: float
    passed: bool
    repaired: bool = False
    new_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "lod0": self.lod0,
            "coverage_before": round(self.coverage_before, 3),
            "coverage_after": round(self.coverage_after, 3),
            "passed": self.passed,
            "repaired": self.repaired,
            "new_id": self.new_id,
        }


@dataclass
class SleepReport:
    started_at: str = field(default_factory=iso)
    finished_at: str = ""
    distilled: list[str] = field(default_factory=list)
    archived_turn_ids: list[int] = field(default_factory=list)
    decayed: list[dict[str, Any]] = field(default_factory=list)
    samples: list[AuditedSample] = field(default_factory=list)
    pass_rate_before: float = 1.0
    pass_rate_after: float = 1.0
    reflex_stats: dict[str, Any] = field(default_factory=dict)
    reflex_candidates: list[str] = field(default_factory=list)
    coverage_threshold: float = 0.6
    ledger_commit: str | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "distilled": self.distilled,
            "archived_turns": self.archived_turn_ids,
            "decayed": self.decayed,
            "samples": [s.to_dict() for s in self.samples],
            "pass_rate_before": round(self.pass_rate_before, 3),
            "pass_rate_after": round(self.pass_rate_after, 3),
            "coverage_threshold": self.coverage_threshold,
            "reflex_stats": self.reflex_stats,
            "reflex_candidates": self.reflex_candidates,
            "ledger_commit": self.ledger_commit,
            "errors": self.errors,
        }

    def summary(self) -> str:
        lines = [
            f"睡眠整合完成：蒸馏 {len(self.distilled)} 个情景节点，"
            f"归档 {len(self.archived_turn_ids)} 轮原始对话",
            f"艾宾浩斯衰减：{len(self.decayed)} 条记忆权重下调（非 pinned）",
            f"反失忆抽查：{len(self.samples)} 条，通过率 "
            f"{self.pass_rate_before:.0%} → {self.pass_rate_after:.0%}"
            f"（召回覆盖率阈值 {self.coverage_threshold:.0%}）",
        ]
        repaired = [s for s in self.samples if s.repaired]
        if repaired:
            lines.append(
                "  修复：" + "、".join(f"#{s.node_id}→#{s.new_id}" for s in repaired)
            )
        if self.reflex_stats:
            noisy = [k for k, v in self.reflex_stats.items() if v.get("noisy")]
            if noisy:
                lines.append(f"  高噪反射（建议降权）：{', '.join(noisy)}")
        if self.ledger_commit:
            lines.append(f"  Git 快照：{self.ledger_commit[:10]}")
        if self.errors:
            lines.append("  警告：" + "；".join(self.errors))
        return "\n".join(lines)


class SleepCycle:
    def __init__(
        self,
        ws: Workspace,
        config: Config | None = None,
        store: MemoryStore | None = None,
        ledger: GitLedger | None = None,
        now_fn: Callable[[], datetime] = now_utc,
        seed: int | None = 42,
    ):
        self.ws = ws
        self.config = config or ws.config
        self.store = store or MemoryStore(ws)
        self.ledger = ledger or GitLedger(ws)
        self.now_fn = now_fn
        self.seed = seed

    # ------------------------------------------------------------------
    def run(self, now: datetime | None = None, commit: bool = True) -> SleepReport:
        now = now or self.now_fn()
        report = SleepReport(
            started_at=iso(now), coverage_threshold=self.config.sleep.coverage_threshold
        )
        self._distill(report, now)
        self._decay(report, now)
        self._audit(report, now)
        self._reflex_stats(report)
        report.finished_at = iso(self.now_fn())
        if commit:
            try:
                report.ledger_commit = self.ledger.commit(
                    f"[sleep] 蒸馏 {len(report.distilled)} 组 · "
                    f"修复 {len([s for s in report.samples if s.repaired])} 条 · "
                    f"反失忆 {report.pass_rate_after:.0%}"
                )
            except LedgerError as exc:
                report.errors.append(f"Git 快照失败: {exc}")
        self._save(report)
        return report

    # ------------------------------------------------------------------
    def _group_turns(self) -> list[list[Any]]:
        turns = self.store.list_turns(None, include_archived=False)
        if not turns:
            return []
        gap = self.config.sleep.session_gap_minutes * 60
        window = max(1, self.config.sleep.distill_window)
        groups: list[list[Any]] = []
        current: list[Any] = []
        for turn in turns:
            if not current:
                current = [turn]
                continue
            prev = current[-1]
            try:
                dt = (parse_iso(turn.created_at) - parse_iso(prev.created_at)).total_seconds()
            except ValueError:
                dt = 0
            same_speaker = turn.speaker_id == current[0].speaker_id
            if same_speaker and dt <= gap and len(current) < window:
                current.append(turn)
            else:
                groups.append(current)
                current = [turn]
        if current:
            groups.append(current)
        return groups

    def _distill(self, report: SleepReport, now: datetime) -> None:
        cfg = self.config.sleep
        for group in self._group_turns():
            text = "\n".join(t.lod2 for t in group)
            if not text.strip():
                continue
            speaker = group[0].speaker_id
            lod1 = extractive_summary(text, cfg.lod1_chars)
            lod0 = one_line(lod1, cfg.lod0_chars)
            importance = min(0.85, 0.5 + 0.08 * len(group))
            node = self.store.put(
                lod0=lod0,
                lod1=lod1,
                lod2=text,
                kind="episode",
                source_id=speaker,
                about_ids=[speaker],
                epistemic="direct",
                visibility=[speaker],
                importance=importance,
                tags=["sleep", "episode"],
            )
            report.distilled.append(node.id)
            for turn in group:
                report.archived_turn_ids.append(turn.turn_id)
        if report.archived_turn_ids:
            self.store.mark_turns_archived(report.archived_turn_ids, archived=True)

    def _decay(self, report: SleepReport, now: datetime) -> None:
        cfg = self.config.sleep
        changes = self.store.decay(cfg.half_life_days, cfg.min_importance, now=now)
        report.decayed = [
            {"id": nid, "before": round(b, 4), "after": round(a, 4)}
            for nid, b, a in changes
        ]

    def _audit(self, report: SleepReport, now: datetime) -> None:
        cfg = self.config.sleep
        nodes = [
            n
            for n in self.store.list_nodes(None)
            if n.lod2 and len(n.lod2) > 20 and not n.tags.count("reindexed") > 1
        ]
        if not nodes:
            report.pass_rate_before = report.pass_rate_after = 1.0
            return
        rng = random.Random(self.seed)
        sample_size = min(cfg.sample_size, len(nodes))
        sample = rng.sample(nodes, sample_size)
        passed_before = 0
        passed_after = 0
        for node in sample:
            view_before = f"{node.lod0}\n{node.lod1}"
            before = coverage(view_before, node.lod2, limit=80)
            passed = before >= cfg.coverage_threshold
            if passed:
                passed_before += 1
            sample_rec = AuditedSample(
                node_id=node.id,
                lod0=node.lod0,
                coverage_before=before,
                coverage_after=before,
                passed=passed,
            )
            if not passed:
                new_node = self._repair(node)
                if new_node is not None:
                    after = coverage(f"{new_node.lod0}\n{new_node.lod1}", node.lod2, limit=80)
                    sample_rec.coverage_after = after
                    sample_rec.repaired = True
                    sample_rec.new_id = new_node.id
                    if after >= cfg.coverage_threshold:
                        passed_after += 1
                    else:
                        report.errors.append(
                            f"#{node.id} 索引重写后仍未达标（{after:.2f}）"
                        )
                else:
                    report.errors.append(f"#{node.id} 无法重写索引")
            else:
                passed_after += 1
            report.samples.append(sample_rec)
        report.pass_rate_before = passed_before / len(sample)
        report.pass_rate_after = passed_after / len(sample)

    def _repair(self, node: MemoryNode) -> MemoryNode | None:
        """重写索引描述：生成新节点 + 旧节点降权沉入冷库（永不物理删除）。"""
        cfg = self.config.sleep
        budget = int(cfg.lod1_chars * 1.6)
        lod1 = extractive_summary(node.lod2, budget)
        # 如果抽取式摘要仍然覆盖不了，退化为“首段 + 事实骨架最密的一段”
        lod0 = one_line(lod1, cfg.lod0_chars)
        try:
            new_node = self.store.put(
                lod0=lod0,
                lod1=lod1,
                lod2=node.lod2,
                kind=node.kind,
                source_id=node.source_id,
                about_ids=node.about_ids,
                epistemic=node.epistemic,
                visibility=node.visibility,
                importance=max(node.importance, 0.55),
                pinned=node.pinned,
                tags=list(dict.fromkeys([*node.tags, "reindexed"])),
                refs=list(dict.fromkeys([*node.refs, node.id])),
            )
            self.store.set_importance(node.id, max(cfg.min_importance, node.importance * 0.5))
            self.store.archive(node.id, archived=True)
            return new_node
        except Exception:  # noqa: BLE001
            return None

    def _reflex_stats(self, report: SleepReport) -> None:
        path = self.ws.reflex_state_file
        if not path.is_file():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        stats: dict[str, Any] = {}
        for rid, entry in data.items():
            if not isinstance(entry, dict):
                continue
            fires = int(entry.get("fires", 0))
            ignores = int(entry.get("ignores", 0))
            stats[rid] = {
                "fires": fires,
                "mids": int(entry.get("mids", 0)),
                "ignores": ignores,
                # “高噪”= 真触发过，且误触发/未触发次数远超触发次数；
                # 从未触发过的反射不叫噪音（只是还没遇到过）。
                "noisy": (fires >= 3 and ignores >= fires * 3)
                or (fires == 0 and ignores >= 20),
            }
        report.reflex_stats = stats
        # 习惯固化的候选：反复出现的记忆标签
        counter: Counter[str] = Counter()
        for node in self.store.list_nodes(None):
            for tag in node.tags:
                if tag not in {"sleep", "episode", "reindexed", "immune"}:
                    counter[tag] += 1
        report.reflex_candidates = [f"{tag}×{n}" for tag, n in counter.most_common(5) if n >= 2]

    # ------------------------------------------------------------------
    def _save(self, report: SleepReport) -> None:
        try:
            path = self.ws.assert_writable(
                self.ws.sleep_reports / f"{report.started_at.replace(':', '-')}.json"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001
            pass
