"""保证“理论模型验证脚本”本身长期可用（验证脚本也是一种契约）。"""

from __future__ import annotations

import demo


def test_demo_runs_green(tmp_path):
    assert demo.main(["--root", str(tmp_path / "verify")]) == 0
