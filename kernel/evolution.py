"""受控自进化：提案 → 安检 → 落盘 → Git 提交（失败则回滚 + 免疫记忆）。

这里把 SPEC 4.6 的护栏串成一条真正的管线：

    提案 ──静态审计──▶ 子进程试跑 ──▶ 回归探针 ──▶ 写 cortex/ ──▶ git commit
                          │              │               │
                          └──── 任一失败 ─┴──▶ 恢复旧文件 + 写“免疫记忆”

外部选择压力（铁律/README §六）在代码里的落点就是这条管线加“你的真实反馈”：
没有通过物理存活测试的变异，连进入她记忆的资格都没有。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from kernel.config import Config
from kernel.git_ledger import GitLedger, LedgerError
from kernel.literal import append_list_item, load_literal
from kernel.models import iso, now_utc
from kernel.paths import Workspace
from kernel.sandbox import Sandbox
from kernel.textutil import one_line


@dataclass
class EvolutionReport:
    ok: bool
    op: str
    target: str
    message: str
    commit: str | None = None
    violations: list[str] = field(default_factory=list)
    trial: str = ""
    immune_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "op": self.op,
            "target": self.target,
            "message": self.message,
            "commit": self.commit,
            "violations": list(self.violations),
            "trial": self.trial,
            "immune_id": self.immune_id,
        }


class Evolver:
    def __init__(
        self,
        ws: Workspace,
        store,
        config: Config | None = None,
        ledger: GitLedger | None = None,
        sandbox: Sandbox | None = None,
    ):
        self.ws = ws
        self.store = store
        self.config = config or ws.config
        self.ledger = ledger or GitLedger(ws)
        self.sandbox = sandbox or Sandbox(ws)

    # ------------------------------------------------------------------
    def _immune(
        self,
        op: str,
        reason: str,
        detail: str,
        source_id: str = "kernel",
        subject: str = "",
    ) -> str:
        tag = f"「{subject}」" if subject else ""
        node = self.store.put(
            lod0=f"进化失败{tag}（{op}）：{one_line(reason, 36)}",
            lod1=f"提案{tag}被安检门拦截：{detail[:260]}",
            lod2=(
                f"[免疫记忆] op={op} target={subject or '-'}\n提案理由: {reason}\n"
                f"失败细节: {detail}\n"
                "下次遇到同类想法时先检查这条教训，不要重复提案。"
            ),
            kind="lesson",
            source_id=source_id,
            about_ids=[source_id],
            epistemic="direct",
            importance=0.8,
            tags=["immune", op],
            visibility=["owner"],
        )
        return node.id

    def _commit(self, message: str) -> str | None:
        try:
            return self.ledger.commit(message)
        except LedgerError as exc:
            raise LedgerError(f"账本提交失败: {exc}") from exc

    def _restore(self, path: Path, previous: str | None, created: bool = False) -> None:
        if created:
            path.unlink(missing_ok=True)
            return
        if previous is not None:
            path.write_text(previous, encoding="utf-8")

    def _provenance(
        self, source_id: str, reason: str, inner: str
    ) -> dict[str, str]:
        return {
            "source_id": source_id,
            "reason": one_line(reason, 120),
            "inner_monologue": one_line(inner, 120),
            "at": iso(now_utc()),
        }

    # ------------------------------------------------------------------
    def teach_rule(
        self,
        rule: str,
        *,
        source_id: str = "owner",
        reason: str = "",
        inner: str = "",
    ) -> EvolutionReport:
        violations = self.sandbox.validate_policy(rule)
        if violations:
            detail = "; ".join(violations)
            return EvolutionReport(
                ok=False,
                op="teach_rule",
                target=str(self.ws.policies_file),
                message="准则未通过校验",
                violations=violations,
                immune_id=self._immune("teach_rule", rule, detail, source_id),
            )
        path = self.ws.policies_file
        previous = path.read_text(encoding="utf-8") if path.is_file() else None
        entry = {
            "rule": rule.strip(),
            "source_id": source_id,
            "created_at": iso(),
            "visibility": [source_id],
            "provenance": self._provenance(source_id, reason, inner),
        }
        try:
            append_list_item(path, "POLICIES", entry, doc="星璃从被教导/强化中沉淀的行为准则。")
            reg = self.sandbox.regression()
            if not reg.ok:
                self._restore(path, previous)
                return EvolutionReport(
                    ok=False,
                    op="teach_rule",
                    target=str(path),
                    message=f"回归未通过：{reg.detail}",
                    violations=[reg.detail],
                    trial=reg.stage,
                    immune_id=self._immune("teach_rule", rule, reg.detail, source_id),
                )
            commit = self._commit(f"[evolve] teach_rule: {one_line(rule, 50)}")
        except Exception as exc:  # noqa: BLE001
            self._restore(path, previous)
            return EvolutionReport(
                ok=False,
                op="teach_rule",
                target=str(path),
                message=f"进化失败：{type(exc).__name__}: {exc}",
                violations=[str(exc)],
                immune_id=self._immune("teach_rule", rule, str(exc), source_id),
            )
        node = self.store.put(
            lod0=f"准则：{one_line(rule, 50)}",
            lod1=rule.strip(),
            lod2=f"被 {source_id} 教导：{rule}\n进化理由：{reason or '（未说明）'}",
            kind="policy",
            source_id=source_id,
            about_ids=[source_id],
            epistemic="direct",
            importance=0.9,
            pinned=True,
            tags=["policy", "evolve"],
            visibility=entry["visibility"],
        )
        return EvolutionReport(
            ok=True,
            op="teach_rule",
            target=str(path),
            message=f"已固化为准则 #{node.id}",
            commit=commit,
            trial="regression passed",
        )

    # ------------------------------------------------------------------
    def propose_reflex(
        self,
        spec: Mapping[str, Any],
        *,
        source_id: str = "owner",
        reason: str = "",
        inner: str = "",
    ) -> EvolutionReport:
        existing = load_literal(self.ws.reflexes_file, "REFLEXES", default=[]) or []
        existing_ids = [str(r.get("id")) for r in existing if isinstance(r, Mapping)]
        violations = self.sandbox.validate_reflex(spec, existing_ids)
        if violations:
            detail = "; ".join(violations)
            return EvolutionReport(
                ok=False,
                op="propose_reflex",
                target=str(self.ws.reflexes_file),
                message="反射未通过校验",
                violations=violations,
                immune_id=self._immune(
                    "propose_reflex",
                    reason or str(spec.get("id")),
                    detail,
                    source_id,
                    subject=str(spec.get("id") or ""),
                ),
            )
        entry = dict(spec)
        entry["enabled"] = bool(entry.get("enabled", True))
        entry["created_by"] = source_id
        entry["created_at"] = iso()
        entry["provenance"] = self._provenance(source_id, reason, inner)
        path = self.ws.reflexes_file
        previous = path.read_text(encoding="utf-8") if path.is_file() else None
        try:
            append_list_item(path, "REFLEXES", entry, doc="Jev 模糊条件反射清单。")
            reg = self.sandbox.regression()
            if not reg.ok:
                self._restore(path, previous)
                return EvolutionReport(
                    ok=False,
                    op="propose_reflex",
                    target=str(path),
                    message=f"回归未通过：{reg.detail}",
                    violations=[reg.detail],
                    trial=reg.stage,
                    immune_id=self._immune("propose_reflex", reason, reg.detail, source_id),
                )
            commit = self._commit(f"[evolve] reflex {entry['id']}: {one_line(reason, 50)}")
        except Exception as exc:  # noqa: BLE001
            self._restore(path, previous)
            return EvolutionReport(
                ok=False,
                op="propose_reflex",
                target=str(path),
                message=f"进化失败：{type(exc).__name__}: {exc}",
                violations=[str(exc)],
                immune_id=self._immune("propose_reflex", reason, str(exc), source_id),
            )
        node = self.store.put(
            lod0=f"新反射：{entry['id']}",
            lod1=str((entry.get("sensor") or {}).get("question", "")),
            lod2=json.dumps(entry, ensure_ascii=False, indent=2),
            kind="reflex",
            source_id=source_id,
            about_ids=[source_id],
            epistemic="direct",
            importance=0.85,
            tags=["reflex", "evolve"],
            visibility=["owner"],
        )
        return EvolutionReport(
            ok=True,
            op="propose_reflex",
            target=str(path),
            message=f"反射 {entry['id']} 已生效（#{node.id}）",
            commit=commit,
            trial="regression passed",
        )

    # ------------------------------------------------------------------
    def propose_skill(
        self,
        name: str,
        code: str,
        *,
        source_id: str = "owner",
        reason: str = "",
        inner: str = "",
    ) -> EvolutionReport:
        if not isinstance(name, str) or not name.isidentifier() or name.startswith("_"):
            detail = "技能名必须是合法 Python 标识符且不以 _ 开头"
            return EvolutionReport(
                ok=False,
                op="propose_skill",
                target=str(self.ws.skills),
                message=detail,
                violations=[detail],
                immune_id=self._immune(
                    "propose_skill", reason or str(name), detail, source_id, subject=str(name)
                ),
            )
        if len(code or "") > 20000:
            detail = "技能代码超过 20000 字符"
            return EvolutionReport(
                ok=False,
                op="propose_skill",
                target=str(self.ws.skills),
                message=detail,
                violations=[detail],
                immune_id=self._immune(
                    "propose_skill", reason, detail, source_id, subject=str(name)
                ),
            )
        trial = self.sandbox.trial_skill(code, name)
        if not trial.ok:
            return EvolutionReport(
                ok=False,
                op="propose_skill",
                target=str(self.ws.skills / f"{name}.py"),
                message=f"{trial.stage} 未通过：{trial.detail}",
                violations=trial.violations or [trial.detail],
                trial=trial.stage,
                immune_id=self._immune(
                    "propose_skill", reason or name, trial.detail, source_id, subject=str(name)
                ),
            )
        path = self.ws.skills / f"{name}.py"
        previous = path.read_text(encoding="utf-8") if path.is_file() else None
        header = (
            f'"""{one_line(reason or name, 60)}\n\n'
            f"由 {source_id} 提出、经安检门通过后固化（{iso()}）。\n"
            f'"""\n\n'
        )
        try:
            self.ws.assert_writable(path)
            path.write_text(header + code.strip() + "\n", encoding="utf-8")
            reg = self.sandbox.regression()
            if not reg.ok:
                self._restore(path, previous, created=previous is None)
                return EvolutionReport(
                    ok=False,
                    op="propose_skill",
                    target=str(path),
                    message=f"回归未通过：{reg.detail}",
                    violations=[reg.detail],
                    trial=trial.stage,
                    immune_id=self._immune(
                        "propose_skill", reason, reg.detail, source_id, subject=str(name)
                    ),
                )
            commit = self._commit(f"[evolve] skill {name}: {one_line(reason, 50)}")
        except Exception as exc:  # noqa: BLE001
            self._restore(path, previous, created=previous is None)
            return EvolutionReport(
                ok=False,
                op="propose_skill",
                target=str(path),
                message=f"进化失败：{type(exc).__name__}: {exc}",
                violations=[str(exc)],
                immune_id=self._immune(
                    "propose_skill", reason, str(exc), source_id, subject=str(name)
                ),
            )
        node = self.store.put(
            lod0=f"新技能：{name}",
            lod1=one_line(reason or code, 160),
            lod2=f"技能 {name} 源码：\n{code}",
            kind="skill",
            source_id=source_id,
            about_ids=[source_id],
            epistemic="direct",
            importance=0.7,
            tags=["skill", "evolve"],
            visibility=["owner"],
        )
        return EvolutionReport(
            ok=True,
            op="propose_skill",
            target=str(path),
            message=f"技能 {name} 已固化（#{node.id}）",
            commit=commit,
            trial="trial + regression passed",
        )

    # ------------------------------------------------------------------
    def handle_op(self, op: Mapping[str, Any], *, source_id: str, inner: str = "") -> EvolutionReport:
        kind = str(op.get("op", ""))
        reason = str(op.get("reason", ""))
        if kind == "teach_rule":
            return self.teach_rule(
                str(op.get("rule", "")), source_id=source_id, reason=reason, inner=inner
            )
        if kind == "propose_reflex":
            return self.propose_reflex(
                op.get("reflex") or {}, source_id=source_id, reason=reason, inner=inner
            )
        if kind == "propose_skill":
            return self.propose_skill(
                str(op.get("name", "")),
                str(op.get("code", "")),
                source_id=source_id,
                reason=reason,
                inner=inner,
            )
        return EvolutionReport(
            ok=False,
            op=kind or "unknown",
            target="",
            message=f"不认识的进化操作: {kind}",
            violations=[f"unknown op {kind}"],
        )
