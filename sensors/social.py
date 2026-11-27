"""社交现场：当前说话人是谁、我和他是什么关系、上次聊到哪、情绪余温。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memory.emotion import EmotionStore
from memory.relations import RelationsStore
from kernel.paths import Workspace


@dataclass
class SocialSensor:
    ws: Workspace
    relations: RelationsStore
    emotion: EmotionStore | None = None

    def capture(self, speaker_id: str) -> dict[str, Any]:
        profile = self.relations.get(speaker_id)
        domain_trust = profile.get("domain_trust") or {}
        snapshot: dict[str, Any] = {
            "speaker_id": speaker_id,
            "display_name": profile.get("display_name") or speaker_id,
            "role": profile.get("role", "stranger"),
            "intimacy": float(profile.get("intimacy", 0.05)),
            "global_trust": float(profile.get("global_trust", 0.3)),
            "domain_trust": {str(k): float(v) for k, v in domain_trust.items()},
            "custom": dict(profile.get("custom") or {}),
        }
        if self.emotion is not None:
            state = self.emotion.load()
            snapshot["emotion_residual"] = state.residual()
            snapshot["emotion_values"] = {k: round(v, 3) for k, v in state.values.items()}
        return snapshot

    def render(self, snapshot: dict[str, Any]) -> str:
        parts = [
            f"当前说话人: {snapshot['display_name']}({snapshot['speaker_id']})",
            f"关系: {snapshot['role']}",
            f"亲密度 {snapshot['intimacy']:.2f}",
            f"全局信任 {snapshot['global_trust']:.2f}",
        ]
        if snapshot.get("domain_trust"):
            domains = ", ".join(f"{k}:{v:.2f}" for k, v in snapshot["domain_trust"].items())
            parts.append(f"领域信任 {{{domains}}}")
        if snapshot.get("emotion_residual"):
            parts.append(f"情绪余温 {snapshot['emotion_residual']}")
        if snapshot.get("custom"):
            extra = ", ".join(f"{k}={v}" for k, v in snapshot["custom"].items() if k != "notes")
            if extra:
                parts.append(f"自定义 {extra}")
        return " ｜ ".join(parts)
