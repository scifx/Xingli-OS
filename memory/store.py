"""混合持久化记忆引擎：内容寻址明文文件 + SQLite 索引。

铁律 2：核心记忆永远不落 pickle。每条记忆是一个人类/大模型都能读的 JSON
文件（``cortex/episodes/<id>.json``），SQLite 只是可随时重建的索引；即使在
索引损坏、类结构演化之后，把文件读回来就能恢复全部记忆。

铁律 3：隐私防火墙在这里开始——``visibility`` 被规范化进独立表，查询时
在 SQL 层就按 ``speaker_id`` 过滤，渲染器拿到的列表里根本不存在越权节点。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime
from typing import Any, Iterable, Mapping

from kernel.models import (
    MemoryNode,
    TurnRecord,
    dumps,
    iso,
    now_utc,
    parse_iso,
)
from kernel.paths import Workspace
from kernel.textutil import similarity

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    source_id TEXT NOT NULL,
    epistemic TEXT NOT NULL,
    importance REAL NOT NULL,
    pinned INTEGER NOT NULL,
    archived INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_accessed TEXT NOT NULL,
    access_count INTEGER NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS node_visibility (
    node_id TEXT NOT NULL,
    person_id TEXT NOT NULL,
    PRIMARY KEY (node_id, person_id)
);
CREATE INDEX IF NOT EXISTS idx_nodes_importance ON nodes(importance);
CREATE INDEX IF NOT EXISTS idx_nodes_kind ON nodes(kind);
CREATE TABLE IF NOT EXISTS turns (
    turn_id INTEGER PRIMARY KEY,
    speaker_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    archived INTEGER NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS turn_visibility (
    turn_id INTEGER NOT NULL,
    person_id TEXT NOT NULL,
    PRIMARY KEY (turn_id, person_id)
);
CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts
    USING fts5(id UNINDEXED, text, tokenize='trigram');
CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts
    USING fts5(turn_id UNINDEXED, text, tokenize='trigram');
"""

WILDCARD = "*"


class MemoryStore:
    """星璃的记忆库（长期节点 + 近期轮次）。"""

    def __init__(self, ws: Workspace):
        self.ws = ws
        self.ws.runtime.mkdir(parents=True, exist_ok=True)
        self.ws.episodes.mkdir(parents=True, exist_ok=True)
        self.ws.turns.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    # ------------------------------------------------------------------
    # 基础
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.ws.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    # ------------------------------------------------------------------
    # 长期记忆节点
    # ------------------------------------------------------------------
    @staticmethod
    def content_id(
        *,
        kind: str,
        lod0: str,
        lod1: str,
        lod2: str,
        source_id: str,
        about_ids: Iterable[str],
        epistemic: str,
        visibility: Iterable[str],
        created_at: str,
    ) -> str:
        """内容寻址：同样的内容 + 同样的时间 → 同样的哈希（12 位十六进制）。"""
        blob = dumps(
            {
                "kind": kind,
                "lod0": lod0,
                "lod1": lod1,
                "lod2": lod2,
                "source_id": source_id,
                "about_ids": sorted(set(about_ids)),
                "epistemic": epistemic,
                "visibility": sorted(set(visibility)),
                "created_at": created_at,
            }
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    def put(
        self,
        *,
        lod0: str,
        lod1: str = "",
        lod2: str = "",
        kind: str = "episode",
        source_id: str = "owner",
        about_ids: Iterable[str] | None = None,
        epistemic: str = "direct",
        visibility: Iterable[str] | None = None,
        importance: float = 0.6,
        pinned: bool = False,
        tags: Iterable[str] | None = None,
        emotion: Mapping[str, Any] | None = None,
        refs: Iterable[str] | None = None,
        created_at: str | None = None,
    ) -> MemoryNode:
        about = [str(x) for x in (about_ids or [source_id])]
        vis = [str(x) for x in (visibility or [])]
        if not vis:
            # 默认可见性：说话人本人 + 记忆所涉及的人（hearsay 反哺通道）
            vis = list(dict.fromkeys([source_id, *about]))
        ts = created_at or iso()
        node = MemoryNode(
            id=self.content_id(
                kind=kind,
                lod0=lod0,
                lod1=lod1,
                lod2=lod2,
                source_id=source_id,
                about_ids=about,
                epistemic=epistemic,
                visibility=vis,
                created_at=ts,
            ),
            kind=kind,
            lod0=lod0,
            lod1=lod1,
            lod2=lod2,
            source_id=source_id,
            about_ids=about,
            epistemic=epistemic,
            visibility=vis,
            importance=importance,
            pinned=pinned,
            tags=[str(t) for t in (tags or [])],
            emotion=dict(emotion or {}),
            refs=[str(r) for r in (refs or [])],
            created_at=ts,
        )
        existing = self.get(node.id)
        if existing is not None:
            # 幂等写入：同一内容不重复落盘，只强化权重与访问痕迹
            existing.importance = max(existing.importance, node.importance)
            existing.pinned = existing.pinned or node.pinned
            existing.tags = sorted(set(existing.tags) | set(node.tags))
            existing.refs = sorted(set(existing.refs) | set(node.refs))
            existing.last_accessed = iso()
            existing.access_count += 1
            existing.updated_at = iso()
            self._write_node(existing)
            return existing

        self._write_node(node)
        return node

    def _write_node(self, node: MemoryNode) -> None:
        path = self.ws.assert_writable(self.ws.episodes / f"{node.id}.json")
        path.write_text(dumps(node.to_dict()) + "\n", encoding="utf-8")
        payload = json.dumps(node.to_dict(), ensure_ascii=False)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO nodes(id, kind, source_id, epistemic, importance, pinned,
                                  archived, created_at, updated_at, last_accessed,
                                  access_count, payload)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    kind=excluded.kind, source_id=excluded.source_id,
                    epistemic=excluded.epistemic, importance=excluded.importance,
                    pinned=excluded.pinned, archived=excluded.archived,
                    updated_at=excluded.updated_at,
                    last_accessed=excluded.last_accessed,
                    access_count=excluded.access_count,
                    payload=excluded.payload
                """,
                (
                    node.id,
                    node.kind,
                    node.source_id,
                    node.epistemic,
                    node.importance,
                    int(node.pinned),
                    int(node.archived),
                    node.created_at,
                    node.updated_at,
                    node.last_accessed,
                    node.access_count,
                    payload,
                ),
            )
            conn.execute("DELETE FROM node_visibility WHERE node_id=?", (node.id,))
            for person in node.visibility:
                conn.execute(
                    "INSERT OR IGNORE INTO node_visibility(node_id, person_id) VALUES (?,?)",
                    (node.id, person),
                )
            conn.execute(
                "INSERT INTO nodes_fts(id, text) VALUES (?,?)",
                (node.id, self._fts_text(node)),
            )

    @staticmethod
    def _fts_text(node: MemoryNode) -> str:
        return "\n".join([node.lod0, node.lod1, node.lod2, " ".join(node.tags)])

    def get(self, node_id: str) -> MemoryNode | None:
        path = self.ws.episodes / f"{node_id}.json"
        if path.is_file():
            return MemoryNode.from_dict(json.loads(path.read_text(encoding="utf-8")))
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM nodes WHERE id=?", (node_id,)
            ).fetchone()
        return MemoryNode.from_dict(json.loads(row["payload"])) if row else None

    def exists(self, node_id: str) -> bool:
        return (self.ws.episodes / f"{node_id}.json").is_file()

    # ---- 可见性（物理防火墙） ----------------------------------------
    @staticmethod
    def is_visible(node: MemoryNode, speaker_id: str) -> bool:
        return WILDCARD in node.visibility or speaker_id in node.visibility

    def visible_ids(self, speaker_id: str) -> set[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT node_id FROM node_visibility
                WHERE person_id IN (?, ?)
                """,
                (speaker_id, WILDCARD),
            ).fetchall()
        return {r["node_id"] for r in rows}

    def list_nodes(
        self,
        speaker_id: str | None = None,
        *,
        kinds: Iterable[str] | None = None,
        include_archived: bool = False,
        limit: int | None = None,
    ) -> list[MemoryNode]:
        """按可见性硬过滤后的节点列表（speaker_id=None 表示内核视角）。"""
        sql = "SELECT payload FROM nodes"
        clauses: list[str] = []
        params: list[Any] = []
        if kinds:
            kinds = list(kinds)
            clauses.append(f"kind IN ({','.join('?' for _ in kinds)})")
            params.extend(kinds)
        if not include_archived:
            clauses.append("archived = 0")
        if speaker_id is not None:
            clauses.append(
                """id IN (SELECT node_id FROM node_visibility
                          WHERE person_id IN (?, ?))"""
            )
            params.extend([speaker_id, WILDCARD])
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY importance DESC, created_at DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        nodes = [MemoryNode.from_dict(json.loads(r["payload"])) for r in rows]
        if speaker_id is not None:
            # 双保险：即便 SQL 被演化改坏，Python 层再过滤一次
            nodes = [n for n in nodes if self.is_visible(n, speaker_id)]
        return nodes

    def search(
        self, speaker_id: str, query: str, limit: int = 5
    ) -> list[tuple[MemoryNode, float]]:
        """在当前说话人权限内检索；先 FTS5/子串粗筛，再按二元组相似度排序。"""
        query = (query or "").strip()
        if not query:
            return []
        candidates: list[MemoryNode] = []
        try:
            match = '"' + query.replace('"', " ") + '"'
            if len(query) >= 3:
                with self._connect() as conn:
                    rows = conn.execute(
                        """
                        SELECT n.payload FROM nodes_fts f
                        JOIN nodes n ON n.id = f.id
                        JOIN node_visibility v ON v.node_id = n.id
                        WHERE f.text MATCH ? AND v.person_id IN (?, ?)
                        LIMIT 80
                        """,
                        (match, speaker_id, WILDCARD),
                    ).fetchall()
                candidates = [
                    MemoryNode.from_dict(json.loads(r["payload"])) for r in rows
                ]
        except sqlite3.Error:
            candidates = []
        if not candidates:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT n.payload FROM nodes n
                    JOIN node_visibility v ON v.node_id = n.id
                    WHERE (n.payload LIKE ? OR n.payload LIKE ?)
                      AND v.person_id IN (?, ?)
                    LIMIT 80
                    """,
                    (f"%{query}%", f"%{query.lower()}%", speaker_id, WILDCARD),
                ).fetchall()
            candidates = [MemoryNode.from_dict(json.loads(r["payload"])) for r in rows]
        if not candidates:
            candidates = self.list_nodes(speaker_id, limit=200)

        scored = [
            (
                n,
                max(
                    0.6 * similarity(query, n.lod0 + n.lod1),
                    0.3 * similarity(query, n.lod2),
                    0.5 * (1.0 if query in (n.lod0 + n.lod1 + n.lod2) else 0.0),
                ),
            )
            for n in candidates
            if self.is_visible(n, speaker_id)
        ]
        scored.sort(key=lambda x: (-x[1], -x[0].importance))
        return scored[:limit]

    def touch(self, node_id: str) -> MemoryNode | None:
        node = self.get(node_id)
        if node is None:
            return None
        node.last_accessed = iso()
        node.access_count += 1
        # 间隔重复式强化：每次被成功召回，权重向上限靠拢
        node.importance = min(1.0, node.importance + 0.02)
        node.updated_at = iso()
        self._write_node(node)
        return node

    def archive(self, node_id: str, archived: bool = True) -> bool:
        """降权沉入冷库（永不物理删除）；索引仍在 SQLite 里可随时找回。"""
        node = self.get(node_id)
        if node is None:
            return False
        node.archived = archived
        node.updated_at = iso()
        self._write_node(node)
        return True

    def set_importance(self, node_id: str, value: float, pinned: bool | None = None) -> None:
        node = self.get(node_id)
        if node is None:
            return
        node.importance = max(0.0, min(1.0, value))
        if pinned is not None:
            node.pinned = pinned
        node.updated_at = iso()
        self._write_node(node)

    def decay(
        self,
        half_life_days: float,
        min_importance: float = 0.05,
        now: datetime | None = None,
    ) -> list[tuple[str, float, float]]:
        """艾宾浩斯式衰减：按距上次被访问的时间降权（永不物理删除）。

        返回 ``[(node_id, before, after), ...]``，仅包含真正变化的节点。
        """
        now = now or now_utc()
        changes: list[tuple[str, float, float]] = []
        for node in self.list_nodes(None, include_archived=False):
            if node.pinned or half_life_days <= 0:
                continue
            age_days = max(
                0.0, (now - parse_iso(node.last_accessed)).total_seconds() / 86400.0
            )
            factor = 0.5 ** (age_days / half_life_days)
            after = max(min_importance, node.importance * factor)
            if abs(after - node.importance) > 1e-4:
                changes.append((node.id, node.importance, after))
                node.importance = after
                node.updated_at = iso()
                self._write_node(node)
        return changes

    # ------------------------------------------------------------------
    # 对话轮次
    # ------------------------------------------------------------------
    def next_turn_id(self) -> int:
        counter = 0
        path = self.ws.turn_counter_file
        if path.is_file():
            try:
                counter = int(json.loads(path.read_text(encoding="utf-8")).get("value", 0))
            except (json.JSONDecodeError, ValueError):
                counter = 0
        with self._connect() as conn:
            row = conn.execute("SELECT MAX(turn_id) AS m FROM turns").fetchone()
        highest = max(counter, int(row["m"] or 0))
        nxt = highest + 1
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dumps({"value": nxt}) + "\n", encoding="utf-8")
        return nxt

    def add_turn(self, turn: TurnRecord) -> TurnRecord:
        path = self.ws.assert_writable(self.ws.turns / f"{turn.turn_id:06d}.json")
        path.write_text(dumps(turn.to_dict()) + "\n", encoding="utf-8")
        payload = json.dumps(turn.to_dict(), ensure_ascii=False)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO turns(turn_id, speaker_id, created_at, archived, payload)
                VALUES (?,?,?,?,?)
                ON CONFLICT(turn_id) DO UPDATE SET
                    speaker_id=excluded.speaker_id, created_at=excluded.created_at,
                    archived=excluded.archived, payload=excluded.payload
                """,
                (turn.turn_id, turn.speaker_id, turn.created_at, int(turn.archived), payload),
            )
            conn.execute("DELETE FROM turn_visibility WHERE turn_id=?", (turn.turn_id,))
            for person in turn.visibility:
                conn.execute(
                    "INSERT OR IGNORE INTO turn_visibility(turn_id, person_id) VALUES (?,?)",
                    (turn.turn_id, person),
                )
            conn.execute("DELETE FROM turns_fts WHERE turn_id=?", (str(turn.turn_id),))
            conn.execute(
                "INSERT INTO turns_fts(turn_id, text) VALUES (?,?)",
                (str(turn.turn_id), "\n".join([turn.lod0, turn.lod1, turn.lod2])),
            )
        return turn

    def get_turn(self, turn_id: int) -> TurnRecord | None:
        path = self.ws.turns / f"{int(turn_id):06d}.json"
        if path.is_file():
            return TurnRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM turns WHERE turn_id=?", (int(turn_id),)
            ).fetchone()
        return TurnRecord.from_dict(json.loads(row["payload"])) if row else None

    def list_turns(
        self,
        speaker_id: str | None = None,
        *,
        include_archived: bool = False,
        limit: int | None = None,
    ) -> list[TurnRecord]:
        sql = "SELECT payload FROM turns"
        clauses: list[str] = []
        params: list[Any] = []
        if not include_archived:
            clauses.append("archived = 0")
        if speaker_id is not None:
            clauses.append(
                """turn_id IN (SELECT turn_id FROM turn_visibility
                               WHERE person_id IN (?, ?))"""
            )
            params.extend([speaker_id, WILDCARD])
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY turn_id ASC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        turns = [TurnRecord.from_dict(json.loads(r["payload"])) for r in rows]
        if speaker_id is not None:
            turns = [
                t
                for t in turns
                if WILDCARD in t.visibility or speaker_id in t.visibility
            ]
        return turns

    def mark_turns_archived(self, turn_ids: Iterable[int], archived: bool = True) -> int:
        ids = [int(t) for t in turn_ids]
        if not ids:
            return 0
        for tid in ids:
            turn = self.get_turn(tid)
            if turn is None:
                continue
            turn.archived = archived
            self.add_turn(turn)
        return len(ids)

    # ------------------------------------------------------------------
    # 维护
    # ------------------------------------------------------------------
    def reindex(self) -> dict[str, int]:
        """从明文文件重建 SQLite 索引（类结构演化 / 索引损坏后的自救）。"""
        with self._connect() as conn:
            conn.executescript(
                """
                DROP TABLE IF EXISTS nodes;
                DROP TABLE IF EXISTS node_visibility;
                DROP TABLE IF EXISTS turns;
                DROP TABLE IF EXISTS turn_visibility;
                DROP TABLE IF EXISTS nodes_fts;
                DROP TABLE IF EXISTS turns_fts;
                """
            )
        self._ensure_schema()
        nodes = 0
        for path in sorted(self.ws.episodes.glob("*.json")):
            node = MemoryNode.from_dict(json.loads(path.read_text(encoding="utf-8")))
            self._write_node(node)
            nodes += 1
        turns = 0
        for path in sorted(self.ws.turns.glob("*.json")):
            turn = TurnRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
            self.add_turn(turn)
            turns += 1
        return {"nodes": nodes, "turns": turns}

    def stats(self, speaker_id: str | None = None) -> dict[str, Any]:
        nodes = self.list_nodes(speaker_id)
        turns = self.list_turns(speaker_id)
        with self._connect() as conn:
            archived = conn.execute(
                "SELECT COUNT(*) AS c FROM nodes WHERE archived=1"
            ).fetchone()["c"]
        return {
            "nodes": len(nodes),
            "turns": len(turns),
            "archived_nodes": int(archived),
            "pinned": sum(1 for n in nodes if n.pinned),
            "kinds": {
                k: sum(1 for n in nodes if n.kind == k)
                for k in sorted({n.kind for n in nodes})
            },
            "oldest": min((n.created_at for n in nodes), default=None),
            "newest": max((n.created_at for n in nodes), default=None),
        }
