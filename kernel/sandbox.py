"""进化安检门（SPEC 4.6 / 铁律 1）。

任何自我修改都必须依次通过：

1. **静态审计**：AST 白名单，拒绝 import / 文件 / 网络 / 反射 / eval 等能力；
2. **子进程试跑**：独立进程 + 超时 + 资源上限，跑坏了她也炸不到内核；
3. **回归探针**：用候选皮层唤醒临时工作区，核心不变量必须仍然成立。

只有三项全过，``evolution.Evolver`` 才会真正落盘并提交 Git。
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from kernel.paths import Workspace
from kernel.runtime_exec import BANNED_NAMES, SAFE_ATTRS, SAFE_BUILTINS
import ast

REPO_ROOT = Path(__file__).resolve().parent.parent

_ID_RE = re.compile(r"^[a-z][a-z0-9_]{1,39}$")
_ALLOWED_SENSOR_TYPES = {"noul", "choose", "score"}
_ALLOWED_ON_HIGH = {"emotion_delta", "inject_hint", "boost_tags"}
_ALLOWED_REFLEX_KEYS = {
    "id",
    "enabled",
    "sensor",
    "high_threshold",
    "low_threshold",
    "cooldown_sec",
    "decay",
    "max_delta",
    "hint_on_mid",
    "on_high",
    "created_by",
    "created_at",
    "provenance",
}
SKILL_AUDIT_EXTRA_BANNED = {
    "import",
    "from",
    "class",
    "while",
    "with",
    "open",
    "exec",
    "eval",
    "compile",
    "input",
    "os",
    "sys",
    "subprocess",
    "socket",
    "pathlib",
    "shutil",
    "requests",
    "urllib",
    "pickle",
    "marshal",
    "ctypes",
    "inspect",
}


class SkillAuditor(ast.NodeVisitor):
    """比“代码即思考”更严格的技能审计：技能是纯函数，不接触外部世界。"""

    def __init__(self) -> None:
        self.violations: list[str] = []

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
                ast.ClassDef,
                ast.While,
                ast.With,
                ast.AsyncWith,
                ast.AsyncFunctionDef,
                ast.Await,
                ast.Delete,
                ast.Global,
                ast.Nonlocal,
                ast.TryStar,
                ast.Yield,
                ast.YieldFrom,
            ),
        ):
            self.violations.append(
                f"禁止的语法 {type(node).__name__}（第 {getattr(node, 'lineno', '?')} 行）"
            )
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in BANNED_NAMES or node.id in SKILL_AUDIT_EXTRA_BANNED:
            self.violations.append(f"禁止的名字 {node.id}（第 {node.lineno} 行）")
        if node.id.startswith("__"):
            self.violations.append(f"禁止的魔法名 {node.id}（第 {node.lineno} 行）")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_") or node.attr in BANNED_NAMES:
            self.violations.append(f"禁止的属性 .{node.attr}（第 {node.lineno} 行）")
        elif node.attr not in SAFE_ATTRS:
            self.violations.append(f"属性 .{node.attr} 不在白名单（第 {node.lineno} 行）")
        self.generic_visit(node)


@dataclass
class TrialOutcome:
    ok: bool
    stage: str = ""
    detail: str = ""
    duration_ms: float = 0.0
    violations: list[str] = field(default_factory=list)


PROBE_TEMPLATE = r'''
import json, sys, builtins
SAFE = {name: getattr(builtins, name) for name in __SAFE_NAMES__}
path = sys.argv[1]
source = open(path, encoding="utf-8").read()
ns = {"__builtins__": SAFE}
try:
    exec(compile(source, path, "exec"), ns)
except Exception as exc:
    print(json.dumps({"ok": False, "detail": "exec: %s: %s" % (type(exc).__name__, exc)}))
    sys.exit(0)
fns = [v for k, v in ns.items() if callable(v) and not k.startswith("_") and getattr(v, "__module__", None) is None]
try:
    if "SELFTEST" in ns:
        ns["SELFTEST"]()
    elif fns:
        probe = fns[0]
        import inspect as _i
        if len(_i.signature(probe).parameters) == 0:
            probe()
except Exception as exc:
    print(json.dumps({"ok": False, "detail": "selftest: %s: %s" % (type(exc).__name__, exc)}))
    sys.exit(0)
print(json.dumps({"ok": True, "detail": "ok", "functions": len(fns)}))
'''


@dataclass
class Sandbox:
    ws: Workspace
    skill_timeout: float = 5.0
    regression_timeout: float = 45.0

    # ------------------------------------------------------------------
    def audit_skill(self, code: str) -> list[str]:
        try:
            tree = ast.parse(code or "")
        except SyntaxError as exc:
            return [f"语法错误: {exc}"]
        auditor = SkillAuditor()
        auditor.visit(tree)
        if not any(
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in tree.body
        ):
            auditor.violations.append("技能必须至少定义一个函数")
        return auditor.violations

    # ------------------------------------------------------------------
    def trial_skill(self, code: str, name: str = "candidate") -> TrialOutcome:
        start = time.monotonic()
        violations = self.audit_skill(code)
        if violations:
            return TrialOutcome(
                ok=False,
                stage="audit",
                detail="静态审计拒绝",
                violations=violations,
                duration_ms=(time.monotonic() - start) * 1000,
            )
        with tempfile.TemporaryDirectory(prefix="xingli-skill-") as tmp:
            path = Path(tmp) / f"{name}.py"
            path.write_text(code, encoding="utf-8")
            probe = Path(tmp) / "_probe.py"
            probe.write_text(
                PROBE_TEMPLATE.replace("__SAFE_NAMES__", repr(sorted(SAFE_BUILTINS))),
                encoding="utf-8",
            )
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", "-S", str(probe), str(path)],
                    capture_output=True,
                    text=True,
                    timeout=self.skill_timeout,
                    cwd=tmp,
                    preexec_fn=_limits,
                )
            except subprocess.TimeoutExpired:
                return TrialOutcome(
                    ok=False,
                    stage="trial",
                    detail=f"子进程超时（>{self.skill_timeout}s），疑似死循环",
                    duration_ms=(time.monotonic() - start) * 1000,
                )
        duration = (time.monotonic() - start) * 1000
        line = (proc.stdout or "").strip().splitlines()
        payload: dict[str, Any] = {}
        if line:
            try:
                payload = json.loads(line[-1])
            except json.JSONDecodeError:
                payload = {}
        if proc.returncode != 0 or not payload.get("ok"):
            detail = payload.get("detail") or (proc.stderr or "").strip()[-400:] or "未知失败"
            return TrialOutcome(
                ok=False, stage="trial", detail=detail, duration_ms=duration
            )
        return TrialOutcome(ok=True, stage="trial", detail="子进程试跑通过", duration_ms=duration)

    # ------------------------------------------------------------------
    def regression(self) -> TrialOutcome:
        """用候选皮层跑核心不变量探针（独立进程，带超时）。"""
        start = time.monotonic()
        env = {
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "XINGLI_PROBE_ROOT": str(self.ws.root),
            "PYTHONPATH": str(REPO_ROOT),
            "PYTHONIOENCODING": "utf-8",
            "HOME": str(self.ws.runtime),
        }
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "kernel.regression"],
                capture_output=True,
                text=True,
                timeout=self.regression_timeout,
                cwd=str(REPO_ROOT),
                env=env,
                preexec_fn=_limits,
            )
        except subprocess.TimeoutExpired:
            return TrialOutcome(
                ok=False,
                stage="regression",
                detail=f"回归探针超时（>{self.regression_timeout}s）",
                duration_ms=(time.monotonic() - start) * 1000,
            )
        duration = (time.monotonic() - start) * 1000
        payload: dict[str, Any] = {}
        for line in reversed((proc.stdout or "").strip().splitlines()):
            try:
                payload = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        if proc.returncode != 0 or not payload.get("ok"):
            detail = "; ".join(payload.get("errors") or []) or (proc.stderr or "").strip()[-400:]
            return TrialOutcome(
                ok=False,
                stage="regression",
                detail=detail or "回归未通过",
                violations=list(payload.get("errors") or []),
                duration_ms=duration,
            )
        return TrialOutcome(
            ok=True,
            stage="regression",
            detail=f"回归通过（{len(payload.get('checks', []))} 项）",
            duration_ms=duration,
        )

    # ------------------------------------------------------------------
    def validate_reflex(
        self, spec: Mapping[str, Any], existing_ids: Iterable[str] = ()
    ) -> list[str]:
        v: list[str] = []
        rid = spec.get("id")
        if not isinstance(rid, str) or not _ID_RE.match(rid):
            v.append("id 必须是 2~40 位小写字母/数字/下划线，且以字母开头")
        elif rid in set(existing_ids):
            v.append(f"反射 id 已存在: {rid}")
        extra = set(spec) - _ALLOWED_REFLEX_KEYS
        if extra:
            v.append(f"未知字段: {sorted(extra)}")
        sensor = spec.get("sensor")
        if not isinstance(sensor, Mapping):
            v.append("sensor 必须是对象")
        else:
            if sensor.get("type") not in _ALLOWED_SENSOR_TYPES:
                v.append("sensor.type 必须是 noul/choose/score")
            q = sensor.get("question")
            if not isinstance(q, str) or not q.strip() or len(q) > 300:
                v.append("sensor.question 必须是 1~300 字的字符串")
            if sensor.get("type") == "choose" and not sensor.get("options"):
                v.append("choose 类型必须给出 options")
            if sensor.get("type") == "score" and not sensor.get("levels"):
                v.append("score 类型必须给出 levels")
        try:
            high = float(spec.get("high_threshold", 0.8))
            low = float(spec.get("low_threshold", 0.25))
            if not (0.5 < high <= 1.0):
                v.append("high_threshold 必须落在 (0.5, 1.0]")
            if not (0.0 <= low < high):
                v.append("low_threshold 必须满足 0 ≤ low < high")
        except (TypeError, ValueError):
            v.append("阈值必须是数值")
        try:
            cooldown = float(spec.get("cooldown_sec", 0))
            if cooldown < 30:
                v.append("cooldown_sec 必须 ≥ 30（物理不应期，防自激）")
        except (TypeError, ValueError):
            v.append("cooldown_sec 必须是数值")
        try:
            max_delta = abs(float(spec.get("max_delta", 0.2)))
            if max_delta > 0.25:
                v.append("max_delta 不得超过 0.25（防情绪劫持）")
        except (TypeError, ValueError):
            v.append("max_delta 必须是数值")
        on_high = spec.get("on_high")
        if on_high is not None:
            if not isinstance(on_high, Mapping):
                v.append("on_high 必须是对象")
            else:
                extra_actions = set(on_high) - _ALLOWED_ON_HIGH
                if extra_actions:
                    v.append(f"on_high 只允许 {sorted(_ALLOWED_ON_HIGH)}")
                for key, delta in (on_high.get("emotion_delta") or {}).items():
                    try:
                        if abs(float(delta)) > abs(float(spec.get("max_delta", 0.2))):
                            v.append(f"emotion_delta.{key} 超过 max_delta")
                    except (TypeError, ValueError):
                        v.append(f"emotion_delta.{key} 必须是数值")
                if len(str(on_high.get("inject_hint", ""))) > 200:
                    v.append("inject_hint 不得超过 200 字")
                tags = on_high.get("boost_tags") or []
                if not isinstance(tags, list) or len(tags) > 5:
                    v.append("boost_tags 必须是最多 5 项的数组")
        if not isinstance(spec.get("enabled", True), bool):
            v.append("enabled 必须是布尔值")
        return v

    def validate_policy(self, rule: str) -> list[str]:
        v: list[str] = []
        if not isinstance(rule, str) or not rule.strip():
            v.append("准则内容为空")
        elif len(rule) > 300:
            v.append("准则不得超过 300 字（太长的准则无法被稳定执行）")
        if "\x00" in (rule or ""):
            v.append("准则包含非法字符")
        return v


def _limits() -> None:  # pragma: no cover - 依赖平台
    """子进程资源上限（Linux/macOS）；不可用时静默放弃。"""
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))
    except Exception:  # noqa: BLE001
        pass
