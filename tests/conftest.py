"""测试夹具：全部用临时工作区，绝不碰仓库里真实的星璃状态。"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernel.config import Config  # noqa: E402
from kernel.paths import Workspace  # noqa: E402
from kernel.sys2_llm import MockSys2Client  # noqa: E402


@dataclass
class FakeClock:
    """可快进的时间源（让衰减/调度测试不用真的 sleep）。"""

    current: datetime = field(
        default_factory=lambda: datetime(2026, 11, 26, 21, 30, tzinfo=timezone.utc)
    )

    def __call__(self) -> datetime:
        return self.current

    def advance(self, seconds: float) -> datetime:
        self.current = self.current + timedelta(seconds=seconds)
        return self.current


@pytest.fixture
def config() -> Config:
    return Config()


@pytest.fixture
def ws(tmp_path: Path, config: Config) -> Workspace:
    return Workspace.open(tmp_path / "xingli", config)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def hb(ws: Workspace, clock: FakeClock):
    from kernel.boot import Heartbeat

    heartbeat = Heartbeat.create(ws, ws.config, sys2=MockSys2Client(), now_fn=clock)
    heartbeat.on_event = lambda msg: None
    return heartbeat
