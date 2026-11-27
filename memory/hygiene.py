"""存量记忆的卫生检查（``/tidy``）。

写入端的清洗只能保护新记忆；已经被污染的地貌（占位符、状态回显、同一件事的
七八种说法、LOD1/LOD2 被写成数字）需要一次**可审计、可回滚**的整理：

1. **垃圾归档**：占位符 / 纯数字 / 状态回显的节点 → ``archived``（不删除，可 ``/rollback``）；
2. **重复合并**：同一事实保留“钉选 > 重要度高 > 内容更全”的那条，其余归档；
3. **字段修复**：``lod1/lod2`` 是数字或占位符时，用 ``lod0`` 回填。

它不改任何事实内容，只决定“哪些不该再出现在地貌上”，所以是安全的；
所有动作都写成明文文件，Git 账本能追。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from kernel.memtext import is_richer, looks_like_placeholder, looks_like_state_echo, same_fact


@dataclass
class TidyReport:
    scanned: int = 0
    junk: list[tuple[str, str]] = field(default_factory=list)
    duplicates: list[tuple[str, str]] = field(default_factory=list)
    repaired: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def changed(self) -> int:
        return len(self.junk) + len(self.duplicates) + len(self.repaired)

    def summary(self) -> str:
        if not self.scanned:
            return "记忆卫生：没有可检查的节点"
        if not self.changed:
            return f"记忆卫生：检查 {self.scanned} 条，干净"
        return (
            f"记忆卫生：检查 {self.scanned} 条 · 归档垃圾 {len(self.junk)} 条 · "
            f"合并重复 {len(self.duplicates)} 条 · 修复字段 {len(self.repaired)} 条"
        )

    def lines(self, limit: int = 10) -> list[str]:
        out: list[str] = []
        for node_id, reason in self.junk[:limit]:
            out.append(f"归档 #{node_id}：{reason}")
        for node_id, kept in self.duplicates[:limit]:
            out.append(f"归档 #{node_id}：与 #{kept} 是同一事实")
        for node_id in self.repaired[:limit]:
            out.append(f"修复 #{node_id}：LOD1/LOD2 是数字或占位符，已用 LOD0 回填")
        extra = self.changed - len(out)
        if extra > 0:
            out.append(f"… 还有 {extra} 项")
        return out


def _quality(node: Any) -> tuple:
    """保留优先级：钉选 > 重要度 > 内容更全 > 更早创建（先说的算）。"""
    return (
        1 if getattr(node, "pinned", False) else 0,
        float(getattr(node, "importance", 0.0)),
        len(node.lod0 or ""),
    )


def tidy(
    store: Any,
    speaker_id: str | None = None,
    *,
    kinds: Sequence[str] = ("episode", "fact"),
    archive: bool = True,
    limit: int = 2000,
) -> TidyReport:
    """检查并（默认）归档脏记忆。``archive=False`` 时只报告不改动。"""
    report = TidyReport()
    try:
        nodes = store.list_nodes(
            speaker_id, kinds=kinds, include_archived=False, limit=limit
        )
    except Exception as exc:  # noqa: BLE001 - 卫生检查失败不该影响聊天
        report.errors.append(f"{type(exc).__name__}: {exc}")
        return report
    report.scanned = len(nodes)

    survivors: list[Any] = []
    for node in nodes:
        if looks_like_placeholder(node.lod0):
            report.junk.append((node.id, "LOD0 是占位符或数字"))
            if archive:
                store.archive(node.id)
            continue
        if looks_like_state_echo(node.lod0):
            report.junk.append((node.id, "LOD0 是社交状态回显，不是事件"))
            if archive:
                store.archive(node.id)
            continue
        repaired = False
        for field_name in ("lod1", "lod2"):
            value = str(getattr(node, field_name) or "")
            if value and looks_like_placeholder(value):
                setattr(node, field_name, node.lod0)
                repaired = True
        if repaired:
            report.repaired.append(node.id)
            if archive:
                store._write_node(node)  # noqa: SLF001 - 同包内的维护路径
        survivors.append(node)

    # 重复合并：先按质量排序，高质量的先“占位”，后来的同类归档
    ordered = sorted(survivors, key=_quality, reverse=True)
    kept: list[Any] = []
    for node in ordered:
        clash = next((k for k in kept if same_fact(k.lod0, node.lod0)), None)
        if clash is None:
            kept.append(node)
            continue
        # 更全的那条留下（即便它质量稍低）：信息不该因为排名而丢
        if is_richer(node.lod0, clash.lod0) and not clash.pinned:
            report.duplicates.append((clash.id, node.id))
            kept[kept.index(clash)] = node
            if archive:
                store.archive(clash.id)
        else:
            report.duplicates.append((node.id, clash.id))
            if archive:
                store.archive(node.id)

    if archive and report.changed:
        try:
            store.reindex()
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"索引重建失败: {exc}")
    return report
