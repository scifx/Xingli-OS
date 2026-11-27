"""柔性人物关系字典（铁律 3/4 的社交侧）。

每个人的档案包含：角色、亲密度、全局信任、**领域细分信任**
（对同事 B 的 ``tech_opinion`` 与 ``personal_gossip`` 可以是两个值），
以及星璃自创的 ``custom`` 字段。

所有写入都限幅（单次 ±0.25），并且只写 ``cortex/``——可被 Git 账本追踪、
可被 ``/rollback`` 回滚。
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from kernel.models import clamp, dumps
from kernel.paths import Workspace


# ---------------------------------------------------------------------------
# 人物关系
# ---------------------------------------------------------------------------
DEFAULT_RELATIONS: dict[str, Any] = {
    "owner": {
        "role": "owner",
        "display_name": "主人",
        "intimacy": 0.85,
        "global_trust": 1.0,
        "domain_trust": {"tech_opinion": 1.0, "personal_gossip": 1.0},
        "custom": {"notes": "星璃的创造者与最重要的对话者"},
    },
    "colleague_b": {
        "role": "colleague",
        "display_name": "同事B",
        "intimacy": 0.35,
        "global_trust": 0.62,
        "domain_trust": {"tech_opinion": 0.72, "personal_gossip": 0.35},
        "custom": {},
    },
    "stranger": {
        "role": "stranger",
        "display_name": "陌生人",
        "intimacy": 0.05,
        "global_trust": 0.3,
        "domain_trust": {},
        "custom": {},
    },
}

_ROLE_DEFAULTS = {
    "owner": {"intimacy": 0.8, "global_trust": 1.0},
    "colleague": {"intimacy": 0.3, "global_trust": 0.55},
    "friend": {"intimacy": 0.6, "global_trust": 0.7},
    "stranger": {"intimacy": 0.05, "global_trust": 0.3},
}


class RelationsStore:
    """人物关系字典的读写。所有修改都经过限幅，防止一次对话就把信任拉满。"""

    def __init__(self, ws: Workspace):
        self.ws = ws
        self.path = ws.relations_file

    def _read(self) -> dict[str, Any]:
        if self.path.is_file():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, OSError):
                pass
        return {k: dict(v) for k, v in DEFAULT_RELATIONS.items()}

    def _write(self, data: Mapping[str, Any]) -> None:
        path = self.ws.assert_writable(self.path)
        path.write_text(dumps(data) + "\n", encoding="utf-8")

    # ---- 读 ----------------------------------------------------------
    def all(self) -> dict[str, Any]:
        return self._read()

    def get(self, person_id: str) -> dict[str, Any]:
        data = self._read()
        if person_id in data:
            profile = dict(data[person_id])
        else:
            profile = {
                "role": "stranger",
                "display_name": person_id,
                "intimacy": _ROLE_DEFAULTS["stranger"]["intimacy"],
                "global_trust": _ROLE_DEFAULTS["stranger"]["global_trust"],
                "domain_trust": {},
                "custom": {},
            }
        profile.setdefault("domain_trust", {})
        profile.setdefault("custom", {})
        return profile

    def trust_of(self, person_id: str, domain: str | None = None) -> float:
        profile = self.get(person_id)
        if domain:
            dt = profile.get("domain_trust") or {}
            if domain in dt:
                try:
                    return float(dt[domain])
                except (TypeError, ValueError):
                    pass
        try:
            return float(profile.get("global_trust", 0.5))
        except (TypeError, ValueError):
            return 0.5

    # ---- 写 ----------------------------------------------------------
    def ensure(self, person_id: str, role: str = "stranger") -> dict[str, Any]:
        data = self._read()
        if person_id not in data:
            base = {
                "role": role,
                "display_name": person_id,
                "intimacy": _ROLE_DEFAULTS.get(role, _ROLE_DEFAULTS["stranger"])["intimacy"],
                "global_trust": _ROLE_DEFAULTS.get(role, _ROLE_DEFAULTS["stranger"])[
                    "global_trust"
                ],
                "domain_trust": {},
                "custom": {},
            }
            data[person_id] = base
            self._write(data)
        return self.get(person_id)

    def update(self, person_id: str, ops: Mapping[str, Any]) -> dict[str, Any]:
        """应用一次关系更新。

        支持字段：``intimacy`` / ``global_trust`` / ``domain_trust``(dict) /
        ``custom``(dict) / ``role`` / ``display_name``；单次变化被限幅到 ±0.25。
        """
        data = self._read()
        if person_id not in data:
            data[person_id] = self.get(person_id)
        profile = data[person_id]
        profile.setdefault("domain_trust", {})
        profile.setdefault("custom", {})

        for key in ("intimacy", "global_trust"):
            if key in ops and ops[key] is not None:
                try:
                    before = float(profile.get(key, 0.5))
                    delta = clamp(float(ops[key]) - before, -0.25, 0.25)
                    profile[key] = round(clamp(before + delta, 0.0, 1.0), 4)
                except (TypeError, ValueError):
                    continue
        if isinstance(ops.get("domain_trust"), Mapping):
            for domain, value in ops["domain_trust"].items():
                try:
                    before = float(profile["domain_trust"].get(domain, profile.get("global_trust", 0.5)))
                    delta = clamp(float(value) - before, -0.25, 0.25)
                    profile["domain_trust"][str(domain)] = round(
                        clamp(before + delta, 0.0, 1.0), 4
                    )
                except (TypeError, ValueError):
                    continue
        if isinstance(ops.get("custom"), Mapping):
            profile["custom"].update(dict(ops["custom"]))
        for key in ("role", "display_name"):
            if isinstance(ops.get(key), str):
                profile[key] = ops[key]
        self._write(data)
        return profile
