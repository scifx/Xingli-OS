"""把 `cortex/` 里的明文记忆文件变成人看得懂的变化流（CLI 专用）。

星璃的记忆不是藏在数据库里的黑箱：每一拍都会在 `cortex/` 里留下/更新明文
文件。这个模块只做三件事——**快照、求差、渲染**——让“她的记忆长出来的
样子”能直接出现在终端里。

不含任何认知逻辑，也不写盘。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

SKIP_DIRS = {"__pycache__", ".git", ".pytest_cache"}


@dataclass(frozen=True)
class FileState:
    size: int
    mtime_ns: int


@dataclass
class MemoryDiff:
    added: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.added or self.modified or self.removed)

    @property
    def count(self) -> int:
        return len(self.added) + len(self.modified) + len(self.removed)


def snapshot(root: Path) -> dict[str, FileState]:
    """记录目录下所有文件的相对路径、大小与 mtime。"""
    state: dict[str, FileState] = {}
    root = Path(root)
    if not root.exists():
        return state
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(root).parts
        if any(part in SKIP_DIRS for part in rel_parts):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        state[Path(*rel_parts).as_posix()] = FileState(stat.st_size, stat.st_mtime_ns)
    return state


def diff(before: Mapping[str, FileState], after: Mapping[str, FileState]) -> MemoryDiff:
    added = sorted(p for p in after if p not in before)
    removed = sorted(p for p in before if p not in after)
    modified = sorted(
        p
        for p in after
        if p in before and (after[p].size != before[p].size or after[p].mtime_ns != before[p].mtime_ns)
    )
    return MemoryDiff(added=added, modified=modified, removed=removed)


def human_size(size: int) -> str:
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f}KB"
    return f"{size / (1024 * 1024):.1f}MB"


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _node_preview(node: Any) -> str:
    """一行摘要：来源/认识论/重要度/可见范围 + LOD0。"""
    if node is None:
        return ""
    visibility = "/".join(getattr(node, "visibility", []) or [])
    return (
        f"〔{node.epistemic}·重要{node.importance:.2f}·可见 {visibility or '*'}"
        f"{'·钉选' if node.pinned else ''}〕\n        「{node.lod0}」"
    )


def describe_change(
    rel: str,
    root: Path,
    store: Any | None = None,
    *,
    preview: bool = True,
) -> str:
    """把单个文件变化渲染成一行（episodes 额外给 LOD0 预览）。"""
    path = root / rel
    size = human_size(path.stat().st_size) if path.exists() else "?"
    line = f"{rel}  {size}"
    if not preview or not rel.startswith("episodes/") or not rel.endswith(".json"):
        return line
    node_id = Path(rel).stem
    node = store.get(node_id) if store is not None else None
    if node is None:
        data = read_json(path) or {}
        return line + f"\n        「{data.get('lod0', '')}」"
    return line + "  " + _node_preview(node)


def _is_state(rel: str) -> bool:
    """``state/`` 里是情绪/焦点/计数器，每拍都会动；折叠成一行，别刷屏。"""
    return rel.startswith("state/")


def format_diff(
    changed: MemoryDiff,
    root: Path,
    store: Any | None = None,
    *,
    limit: int = 6,
) -> str:
    """渲染一次“记忆变化”区块；没有变化时返回空字符串。

    记忆文件（episodes / turns / 准则 / 技能）逐条列出；``state/`` 里的
    运行状态折叠成一行——它们会变，但那不是“她又记住了什么”。
    """
    if not changed.changed:
        return ""
    added = [rel for rel in changed.added if not _is_state(rel)]
    modified = [rel for rel in changed.modified if not _is_state(rel)]
    state_files = [
        Path(rel).name
        for rel in (*changed.added, *changed.modified)
        if _is_state(rel)
    ]
    if not added and not modified and not state_files and not changed.removed:
        return ""
    total = sum((root / rel).stat().st_size for rel in added if (root / rel).exists())
    head = f"记忆变化 · 新建 {len(added)} · 更新 {len(modified)}"
    if total:
        head += f" · +{human_size(total)}"
    lines = [head]
    shown = 0
    for rel in added:
        if shown >= limit:
            lines.append(f"  … 还有 {len(added) - shown} 个新文件")
            break
        lines.append(f"  + {describe_change(rel, root, store)}")
        shown += 1
    shown = 0
    for rel in modified:
        if shown >= limit:
            lines.append(f"  … 还有 {len(modified) - shown} 个更新")
            break
        lines.append(f"  ~ {describe_change(rel, root, store, preview=False)}")
        shown += 1
    for rel in changed.removed:
        lines.append(f"  - {rel}")
    if state_files:
        lines.append(f"  ~ 状态：{', '.join(sorted(set(state_files)))}")
    return "\n".join(lines)


def tree(root: Path, *, limit: int = 200) -> str:
    """`cortex/` 的目录树（含文件大小），按目录分组。"""
    root = Path(root)
    if not root.exists():
        return f"（{root} 还不存在）"
    rows: list[tuple[str, int]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(root).parts
        if any(part in SKIP_DIRS for part in rel_parts):
            continue
        rows.append((Path(*rel_parts).as_posix(), path.stat().st_size))
    if not rows:
        return f"（{root} 是空的）"
    total = sum(size for _, size in rows)
    out = [f"{root}  ·  {len(rows)} 个文件 / {human_size(total)}"]
    for rel, size in rows[:limit]:
        out.append(f"  {rel:<44} {human_size(size):>8}")
    if len(rows) > limit:
        out.append(f"  … 其余 {len(rows) - limit} 个文件省略")
    return "\n".join(out)


def list_memories(
    store: Any,
    speaker_id: str | None,
    episodes_dir: Path,
    *,
    limit: int = 12,
    include_archived: bool = False,
) -> str:
    """长期记忆清单：文件大小 + 认识论 + 重要度 + LOD0。"""
    nodes = store.list_nodes(
        speaker_id, include_archived=include_archived, limit=limit
    )
    turns = store.list_turns(speaker_id)
    if not nodes and not turns:
        return "（还没有长期记忆节点；聊几句、让她记住点什么，或者 /sleep）"
    sizes = {
        p.stem: p.stat().st_size
        for p in episodes_dir.glob("*.json")
        if p.is_file()
    } if episodes_dir.exists() else {}
    out = [
        f"长期记忆 {len(nodes)} 条 · 对话轮次 {len(turns)} 轮 · 记忆目录 {episodes_dir}"
    ]
    for node in nodes:
        size = human_size(sizes.get(node.id, 0)) if sizes else "-"
        out.append(
            f"  #{node.id}  {size:>7}  {node.created_at[:16]}  "
            f"{node.epistemic:<8} {node.importance:.2f}  {node.lod0}"
            + ("  [archived]" if node.archived else "")
        )
    return "\n".join(out)


def resolve_memory_path(
    arg: str, ws: Any, store: Any | None = None
) -> Path | None:
    """把 `/cat` 的参数解析成真实文件：id / episodes/xx.json / turns/3.json。"""
    arg = (arg or "").strip()
    if not arg:
        return None
    cortex = ws.cortex
    candidate = Path(arg)
    if candidate.is_absolute() and candidate.is_file():
        return candidate
    for base in (cortex, cortex / "episodes", cortex / "turns", cortex / "state", cortex / "skills"):
        path = base / arg
        if path.is_file():
            return path
    if "/" not in arg and not arg.endswith(".json"):
        for base in (cortex / "episodes", cortex / "turns"):
            path = base / f"{arg}.json"
            if path.is_file():
                return path
    if store is not None:
        turn = store.get_turn(int(arg)) if arg.isdigit() else None
        if turn is not None:
            path = cortex / "turns" / f"{int(arg):06d}.json"
            if path.is_file():
                return path
    return None


def iter_json_files(root: Path) -> Iterable[Path]:
    for path in sorted(Path(root).rglob("*.json")):
        if not any(part in SKIP_DIRS for part in path.parts):
            yield path
