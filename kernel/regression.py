"""进化回归探针（在独立子进程里跑）。

任何一次 ``teach_rule`` / ``propose_reflex`` / ``propose_skill`` 都要先过这里：
用**候选皮层**（而不是仓库里的旧副本）唤醒一个临时工作区，验证核心不变量
仍然成立——焦点衰减数学、隐私物理过滤、三段门控、明文往返。

可以单独运行：

    XINGLI_PROBE_ROOT=/path/to/workspace python -m kernel.regression
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any


def _copy_cortex(src: Path, dst: Path) -> None:
    """只复制“代码/数据”型皮层文件；记忆数据不参与进化回归。"""
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("persona.py", "focus_policy.py", "reflexes.py", "policies.py", "relations.json"):
        p = src / name
        if p.is_file():
            shutil.copy2(p, dst / name)
    skills_dst = dst / "skills"
    skills_dst.mkdir(exist_ok=True)
    for path in (src / "skills").glob("*.py"):
        shutil.copy2(path, skills_dst / path.name)
    for name in ("episodes", "turns", "state"):
        (dst / name).mkdir(exist_ok=True)


def check(root: Path) -> dict[str, Any]:
    """在 ``root`` 的候选皮层上跑一组核心不变量检查。"""
    from kernel.config import Config
    from kernel.literal import literal_error
    from kernel.paths import Workspace
    from kernel.sys1_jev import LocalLexiconClient, QuestionSpec, ReflexEngine
    from memory.focal_context import clarity
    from memory.relations import RelationsStore
    from memory.store import MemoryStore

    checks: list[str] = []
    errors: list[str] = []
    workspace_root = Path(root).resolve()

    def run(name: str, fn) -> None:
        try:
            fn()
            checks.append(name)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {type(exc).__name__}: {exc}")

    with tempfile.TemporaryDirectory(prefix="xingli-probe-") as tmp:
        tmp_ws = Path(tmp) / "ws"
        _copy_cortex(workspace_root / "cortex", tmp_ws / "cortex")
        (tmp_ws / ".xingli").mkdir(parents=True, exist_ok=True)
        ws = Workspace(root=tmp_ws, config=Config.load(root=workspace_root))
        store = MemoryStore(ws)

        def focal_math() -> None:
            f, db, da = 10, 0.35, 0.2
            import math

            assert abs(clarity(10, f, db, da) - 1.0) < 1e-12
            assert abs(clarity(7, f, db, da) - math.exp(-db * 3)) < 1e-12
            assert abs(clarity(14, f, db, da) - math.exp(-da * 4)) < 1e-12
            # 非对称：同样距离，焦点之前衰减更快
            assert clarity(6, f, db, da) < clarity(14, f, db, da)

        def privacy() -> None:
            node = store.put(
                lod0="只有主人知道的秘密",
                lod1="秘密内容",
                lod2="秘密全文",
                source_id="owner",
                visibility=["owner"],
            )
            assert store.get(node.id) is not None
            assert [n.id for n in store.list_nodes("colleague_b")] == []
            assert [n.id for n in store.list_nodes("owner")] == [node.id]

        def plaintext_roundtrip() -> None:
            node = store.put(lod0="明文往返", lod1="内容", lod2="内容全文", source_id="owner")
            path = ws.episodes / f"{node.id}.json"
            assert path.is_file()
            data = json.loads(path.read_text(encoding="utf-8"))
            assert data["id"] == node.id and "pickle" not in path.name

        def gating() -> None:
            engine = ReflexEngine(ws, config=ws.config, client=LocalLexiconClient())
            specs = [
                QuestionSpec(
                    id="probe_high",
                    kind="noul",
                    question="命中？",
                    cues={"positive": ["熬夜"]},
                )
            ]
            result = engine.client.fan_out("他又在通宵熬夜赶工", specs)
            assert result["probe_high"].probability >= 0.8

        def literals() -> None:
            for name, var in (
                ("reflexes.py", "REFLEXES"),
                ("policies.py", "POLICIES"),
                ("focus_policy.py", "FOCUS_DEFAULTS"),
                ("persona.py", "EMOTION_BASELINE"),
            ):
                err = literal_error(ws.cortex / name, var)
                assert err is None, f"{name}: {err}"

        def relations_loaded() -> None:
            rel = RelationsStore(ws)
            assert rel.get("owner")["role"] == "owner"

        run("focal_decay_math", focal_math)
        run("privacy_physical_filter", privacy)
        run("plaintext_roundtrip", plaintext_roundtrip)
        run("sys1_gating", gating)
        run("cortex_literals", literals)
        run("relations", relations_loaded)

    return {"ok": not errors, "checks": checks, "errors": errors}


def main() -> int:
    root = os.environ.get("XINGLI_PROBE_ROOT") or os.getcwd()
    try:
        result = check(Path(root))
    except Exception:  # noqa: BLE001
        result = {
            "ok": False,
            "checks": [],
            "errors": [traceback.format_exc(limit=3)],
        }
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
