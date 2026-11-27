"""受控「代码即思考」执行器。

两条通道同时存在，且共用同一套认知原语：

* **JSON 认知包**（默认、主力）：类型安全、零语法崩坏；
* **纯 Python 代码**（兼容旧范式 / 复杂编排）：``#`` 注释 = 内心 OS，
  语句调用原语（``say`` / ``recall`` / ``memorize`` / ``feel`` / ``set_focus``…）。

安全边界：
* AST 白名单审计：禁 import、禁 class、禁 while、禁 ``eval/exec/open`` 等；
* 行数预算（``settrace`` 计数）：即使写了 ``for i in range(10**9)`` 也会被掐断；
* 命名空间里只有内核注入的原语与安全内置函数，没有文件/网络/反射能力。
"""

from __future__ import annotations

import ast
import json
import sys
from dataclasses import dataclass, field
from typing import Any

from kernel.models import ThoughtPacket

import re

_FENCE = re.compile(r"```(?:python|py|json)?\s*(.*?)```", re.DOTALL)


class CodeRejected(PermissionError):
    """代码未通过静态审计。"""


class ExecutionBudgetExceeded(RuntimeError):
    """执行行数超出预算（防死循环）。"""


BANNED_NAMES = {
    "eval",
    "exec",
    "compile",
    "open",
    "input",
    "breakpoint",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
    "memoryview",
    "type",
    "object",
    "super",
    "property",
    "classmethod",
    "staticmethod",
    "exit",
    "quit",
    "help",
    "__import__",
    "__builtins__",
    "__loader__",
    "__spec__",
}

BANNED_NODES: tuple[type[ast.AST], ...] = (
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
    ast.ClassDef,
    ast.AsyncFunctionDef,
    ast.Await,
    ast.AsyncFor,
    ast.With,
    ast.AsyncWith,
    ast.Delete,
    ast.Yield,
    ast.YieldFrom,
    ast.While,
    ast.Raise,
    ast.TryStar,
)

SAFE_BUILTINS: dict[str, Any] = {
    "len": len,
    "min": min,
    "max": max,
    "sum": sum,
    "sorted": sorted,
    "abs": abs,
    "round": round,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "enumerate": enumerate,
    "range": range,
    "any": any,
    "all": all,
    "reversed": reversed,
    "zip": zip,
}

#: 允许调用的“字符串/容器方法”（其余属性访问一律拒绝）
SAFE_ATTRS = {
    "strip",
    "lstrip",
    "rstrip",
    "split",
    "rsplit",
    "join",
    "replace",
    "lower",
    "upper",
    "title",
    "startswith",
    "endswith",
    "find",
    "count",
    "get",
    "keys",
    "values",
    "items",
    "append",
    "extend",
    "insert",
    "pop",
    "sort",
    "copy",
}


class _Auditor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.violations: list[str] = []

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, BANNED_NODES):
            self.violations.append(
                f"禁止的语法 {type(node).__name__}（第 {getattr(node, 'lineno', '?')} 行）"
            )
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in BANNED_NAMES:
            self.violations.append(f"禁止的名字 {node.id}（第 {node.lineno} 行）")
        if node.id.startswith("__") and node.id.endswith("__"):
            self.violations.append(f"禁止的魔法名 {node.id}（第 {node.lineno} 行）")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in BANNED_NAMES or node.attr.startswith("_"):
            self.violations.append(f"禁止的属性访问 .{node.attr}（第 {node.lineno} 行）")
        elif node.attr not in SAFE_ATTRS:
            self.violations.append(
                f"属性 .{node.attr} 不在白名单内（第 {node.lineno} 行）"
            )
        self.generic_visit(node)


def audit_code(code: str) -> list[str]:
    """静态审计；返回违规列表（空 = 通过）。"""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"语法错误: {exc}"]
    auditor = _Auditor()
    auditor.visit(tree)
    return auditor.violations


@dataclass
class PacketBuilder:
    """把代码模式下的原语调用，累积成与 JSON 模式完全相同的认知包。"""

    speaker_id: str = "owner"
    store: Any = None
    relations: Any = None
    packet: ThoughtPacket = field(default_factory=ThoughtPacket)
    notes: list[str] = field(default_factory=list)

    # ---- 表达 ---------------------------------------------------------
    def say(self, text: Any, to: str | None = None) -> str:
        text = str(text)
        if to:
            self.packet.say = {"to": str(to), "text": text}
        else:
            self.packet.say = text
        return text

    def _append_op(self, op: dict[str, Any]) -> dict[str, Any]:
        self.packet.memory_ops.append(op)
        return op

    # ---- 记忆 ---------------------------------------------------------
    def recall(self, mem_id: str) -> dict[str, Any]:
        mid = str(mem_id)
        node = self.store.get(mid) if self.store is not None else None
        if node is None:
            self.notes.append(f"recall: 找不到记忆 {mid}")
            return {"id": mid, "missing": True}
        from memory.store import MemoryStore

        if not MemoryStore.is_visible(node, self.speaker_id):
            # 越权召回：连存在性都不告诉她
            self.notes.append(f"recall: {mid} 不存在")
            return {"id": mid, "missing": True}
        self._append_op({"op": "recall", "mem_id": mid})
        return {"id": mid, "lod0": node.lod0, "lod1": node.lod1, "status": "queued"}

    def recall_turn(self, turn_id: int) -> dict[str, Any]:
        tid = int(turn_id)
        turn = self.store.get_turn(tid) if self.store is not None else None
        if turn is None or not (
            "*" in turn.visibility or self.speaker_id in turn.visibility
        ):
            self.notes.append(f"recall_turn: {tid} 不可见")
            return {"turn_id": tid, "missing": True}
        self._append_op({"op": "recall_turn", "turn_id": tid})
        return {"turn_id": tid, "lod0": turn.lod0, "status": "queued"}

    def search_memory(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        if self.store is None:
            return []
        hits = self.store.search(self.speaker_id, str(query), int(limit))
        return [{"id": n.id, "lod0": n.lod0, "score": round(s, 3)} for n, s in hits]

    def memorize(
        self,
        lod0: str,
        lod1: str = "",
        lod2: str = "",
        kind: str = "episode",
        about: list[str] | None = None,
        visibility: list[str] | None = None,
        importance: float = 0.6,
        tags: list[str] | None = None,
        pinned: bool = False,
        epistemic: str | None = None,
    ) -> str:
        op = {
            "op": "memorize",
            "lod0": str(lod0),
            "lod1": str(lod1),
            "lod2": str(lod2),
            "kind": str(kind),
            "about_ids": list(about or []),
            "visibility": list(visibility or []),
            "importance": float(importance),
            "tags": list(tags or []),
            "pinned": bool(pinned),
        }
        if epistemic:
            op["epistemic"] = str(epistemic)
        self._append_op(op)
        return "queued"

    # ---- 注意力 / 情绪 / 社交 -----------------------------------------
    def set_focus(
        self,
        turn_id: int | None = None,
        decay_before: float | None = None,
        decay_after: float | None = None,
        pin: list[int] | None = None,
        unpin: list[int] | None = None,
    ) -> str:
        from kernel.models import FocusUpdate

        current = self.packet.focus_update or FocusUpdate()
        if turn_id is not None:
            current.turn_id = int(turn_id)
        if decay_before is not None:
            current.decay_before = float(decay_before)
        if decay_after is not None:
            current.decay_after = float(decay_after)
        if pin:
            current.pin.extend(int(x) for x in pin)
        if unpin:
            current.unpin.extend(int(x) for x in unpin)
        self.packet.focus_update = current
        return "ok"

    def feel(self, **deltas: Any) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, value in deltas.items():
            try:
                out[key] = float(value)
            except (TypeError, ValueError):
                continue
        self.packet.emotion_delta.update(out)
        return out

    def update_relation(self, person_id: str, **changes: Any) -> str:
        op = {"person_id": str(person_id)}
        op.update({k: v for k, v in changes.items()})
        self.packet.relation_ops.append(op)
        return "ok"

    # ---- 进化 ---------------------------------------------------------
    def teach_rule(self, rule: str) -> str:
        self.packet.evolve_ops.append({"op": "teach_rule", "rule": str(rule)})
        return "queued"

    def propose_reflex(self, reflex: dict[str, Any], reason: str = "") -> str:
        self.packet.evolve_ops.append(
            {"op": "propose_reflex", "reflex": dict(reflex), "reason": str(reason)}
        )
        return "queued"

    def propose_skill(self, name: str, code: str, reason: str = "") -> str:
        self.packet.evolve_ops.append(
            {
                "op": "propose_skill",
                "name": str(name),
                "code": str(code),
                "reason": str(reason),
            }
        )
        return "queued"

    # ---- 自主唤醒 -----------------------------------------------------
    def continue_think(self, reason: str = "") -> str:
        self.packet.wake.continue_think = {"reason": str(reason)}
        return "queued"

    def schedule_wake(
        self, after_sec: int, reason: str = "", target_speaker: str | None = None
    ) -> str:
        self.packet.wake.schedule_wake = {
            "after_sec": int(after_sec),
            "reason": str(reason),
            "target_speaker": target_speaker,
        }
        return "queued"


class _LineBudget:
    """用 settrace 限制执行行数，避免代码模式写出死循环。"""

    def __init__(self, max_lines: int = 200_000):
        self.max_lines = max_lines
        self.count = 0

    def __call__(self, frame, event, arg):  # noqa: ANN001
        if event == "line":
            self.count += 1
            if self.count > self.max_lines:
                raise ExecutionBudgetExceeded(
                    f"代码执行超过 {self.max_lines} 行预算，已强制中断"
                )
        return self


def _namespace(builder: PacketBuilder) -> dict[str, Any]:
    ns: dict[str, Any] = {"__builtins__": {}}
    ns.update(SAFE_BUILTINS)
    for name in (
        "say",
        "recall",
        "recall_turn",
        "search_memory",
        "memorize",
        "set_focus",
        "feel",
        "update_relation",
        "teach_rule",
        "propose_reflex",
        "propose_skill",
        "continue_think",
        "schedule_wake",
    ):
        ns[name] = getattr(builder, name)
    ns["remember"] = builder.memorize
    return ns


def extract_comments(code: str) -> str:
    """抽取整行注释作为第一人称内心 OS。"""
    lines: list[str] = []
    for raw in (code or "").splitlines():
        stripped = raw.strip()
        if stripped.startswith("#"):
            text = stripped.lstrip("#").strip()
            if text:
                lines.append(text)
    return "\n".join(lines)


def strip_fences(text: str) -> str:
    if not text:
        return ""
    m = _FENCE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def execute_python(
    code: str,
    builder: PacketBuilder,
    max_lines: int = 200_000,
) -> tuple[bool, str]:
    """在受限命名空间执行代码体；返回 ``(成功, 错误或输出摘要)``。"""
    violations = audit_code(code)
    if violations:
        return False, "静态审计拒绝: " + "; ".join(violations[:5])
    try:
        compiled = compile(code, "<xingli-thought>", "exec")
    except SyntaxError as exc:
        return False, f"编译失败: {exc}"
    budget = _LineBudget(max_lines=max_lines)
    old_trace = sys.gettrace()
    sys.settrace(budget)
    try:
        exec(compiled, _namespace(builder), {})  # noqa: S102 - 受限命名空间 + AST 审计
    except ExecutionBudgetExceeded as exc:
        return False, str(exc)
    except Exception as exc:  # noqa: BLE001 - 她写错了不该拖垮内核
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        sys.settrace(old_trace)
    return True, "ok"


def parse_thought_output(raw: str, speaker_id: str = "owner") -> tuple[ThoughtPacket, str]:
    """解析 Sys2 输出：优先 JSON 认知包，回退到纯 Python 代码载荷。

    返回 ``(packet, mode)``，mode ∈ {"json", "code", "empty"}。
    """
    text = strip_fences(raw or "")
    if not text:
        return ThoughtPacket(inner_monologue="", parse_warnings=["空输出"]), "empty"
    if text.lstrip().startswith("{") or text.lstrip().startswith("["):
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            # JSON 坏了：不猜测，交回错误让内核记录（绝不半执行）
            return ThoughtPacket(parse_warnings=[f"JSON 解析失败: {exc}"]), "empty"
        if not isinstance(data, dict):
            return ThoughtPacket(parse_warnings=["JSON 顶层不是对象"]), "empty"
        return ThoughtPacket.from_dict(data), "json"

    builder = PacketBuilder(speaker_id=speaker_id)
    ok, detail = execute_python(text, builder)
    monologue = extract_comments(text)
    packet = builder.packet
    packet.inner_monologue = monologue
    if not ok:
        packet.parse_warnings.append(detail)
    return packet, "code"


def apply_code_action(code: str, builder: PacketBuilder) -> dict[str, Any]:
    """执行 ``code_action``（与代码模式共用原语与安全审计）。"""
    ok, detail = execute_python(code, builder)
    return {"ok": ok, "detail": detail}
