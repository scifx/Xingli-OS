"""可插拔的外部世界传感器。

MVP 只采集“工作现场”这类无副作用信息；DDL 等日程从 ``cortex/world.json``
读取（属于她的可写皮层，可以被教导/更新）。未来接日历、IM、系统通知，
只需要 ``register(name, probe)``，认知帧会自动多出一行现实感。
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from kernel.paths import Workspace
from sensors.clock import ddl_line


@dataclass
class WorldSensor:
    ws: Workspace
    probes: dict[str, Callable[[Workspace], Any]] = field(default_factory=dict)

    def register(self, name: str, probe: Callable[[Workspace], Any]) -> None:
        self.probes[name] = probe

    @property
    def world_file(self) -> Path:
        return self.ws.cortex / "world.json"

    def _ddls(self) -> list[dict[str, Any]]:
        if not self.world_file.is_file():
            return []
        try:
            data = json.loads(self.world_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        ddls = data.get("ddls") if isinstance(data, dict) else None
        return [d for d in (ddls or []) if isinstance(d, dict)]

    def capture(self) -> dict[str, Any]:
        snapshot: dict[str, Any] = {
            "platform": platform.system().lower(),
            "workspace": self.ws.root.name,
            "cortex": str(self.ws.cortex.relative_to(self.ws.root)),
            "ddl_line": ddl_line(self._ddls()),
        }
        for name, probe in self.probes.items():
            try:
                snapshot[name] = probe(self.ws)
            except Exception as exc:  # 传感器失败绝不能拖垮一轮思考
                snapshot[name] = f"<{name} 传感器异常: {type(exc).__name__}>"
        return snapshot

    def render(self, snapshot: dict[str, Any]) -> str:
        parts = [f"环境: {snapshot.get('platform')} · 工作区 {snapshot.get('workspace')}"]
        if snapshot.get("ddl_line"):
            parts.append(f"DDL: {snapshot['ddl_line']}")
        for key, value in snapshot.items():
            if key in {"platform", "workspace", "cortex", "ddl_line"}:
                continue
            parts.append(f"{key}: {value}")
        return " ｜ ".join(parts)
