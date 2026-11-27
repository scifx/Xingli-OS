"""CLI 的接线：初始化配置、探测端点、非交互对话、自检。

端点用本地假服务器，所以全部离线可跑。
"""

from __future__ import annotations

import json
from pathlib import Path

import cli
from fake_openai import FakeOpenAI, chat_reply, packet

from kernel.config import Config


def _memorize_reply():
    ops = [
        {
            "op": "memorize",
            "lod0": "下周三上午看牙医",
            "lod1": "主人说下周三上午去医院看牙，别排别的事。",
            "lod2": "主人：记住，下周三要去看牙医。",
            "kind": "episode",
            "about_ids": ["owner"],
            "visibility": ["owner"],
            "epistemic": "direct",
            "importance": 0.8,
        }
    ]
    return chat_reply(
        json.dumps(packet(say="记下了。", memory_ops=ops), ensure_ascii=False)
    )


def test_init_config_writes_loadable_template(tmp_path: Path):
    root = tmp_path / "ws"
    code = cli.main(
        [
            "--root",
            str(root),
            "--init-config",
            "--base-url",
            "http://127.0.0.1:8080/v1",
            "--model",
            "qwen2.5-7b",
        ]
    )
    assert code == 0
    path = root / "xingli.toml"
    assert path.is_file()
    cfg = Config.load(root=root)
    assert cfg.sys2.provider == "openai_compatible"
    assert cfg.sys2.base_url == "http://127.0.0.1:8080/v1"
    assert cfg.sys2.model == "qwen2.5-7b"
    assert cfg.sys1.provider == "local"
    # 二次生成会留备份，不静默覆盖
    cli.main(["--root", str(root), "--init-config", "--model", "other"])
    assert (root / "xingli.toml.bak").is_file()


def test_ping_reports_endpoint(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.delenv("XINGLI_CONFIG", raising=False)
    with FakeOpenAI() as server:
        root = tmp_path / "ws"
        cli.write_config(root, server.base_url, "fake-local")
        code = cli.main(["--root", str(root), "--ping"])
    out = capsys.readouterr().out
    assert code == 0
    assert "在线" in out
    assert "fake-local" in out


def test_say_mode_shows_memory_files(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.delenv("XINGLI_CONFIG", raising=False)
    with FakeOpenAI(lambda p, h: _memorize_reply()) as server:
        root = tmp_path / "ws"
        cli.write_config(root, server.base_url, "fake-local")
        code = cli.main(
            [
                "--root",
                str(root),
                "--say",
                "记住，我下周三要去看牙医",
                "--show-memory",
            ]
        )
    out = capsys.readouterr().out
    assert code == 0
    assert "星璃: 记下了。" in out
    assert "记忆变化" in out and "episodes/" in out
    episodes = list((root / "cortex" / "episodes").glob("*.json"))
    assert episodes and "看牙医" in episodes[0].read_text(encoding="utf-8")


def test_selftest_runs_and_prints_memory_file(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.delenv("XINGLI_CONFIG", raising=False)
    with FakeOpenAI(lambda p, h: _memorize_reply()) as server:
        root = tmp_path / "ws"
        cli.write_config(root, server.base_url, "fake-local")
        code = cli.main(["--root", str(root), "--selftest"])
    out = capsys.readouterr().out
    assert code == 0
    assert "端点返回" in out
    assert "新建的记忆文件" in out
    assert "lod0" in out
    selftest_episodes = list((root / ".xingli" / "selftest" / "cortex" / "episodes").glob("*.json"))
    assert selftest_episodes


def test_mock_mode_still_works_without_config(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.delenv("XINGLI_CONFIG", raising=False)
    code = cli.main(["--root", str(tmp_path / "ws"), "--mock", "--say", "你好"])
    out = capsys.readouterr().out
    assert code == 0
    assert "星璃:" in out


# ---------------------------------------------------------------------------
# 丘脑（System One）切换：本地决策模型接不上时，可见、可切、可持久化
# ---------------------------------------------------------------------------
def test_sys1_command_probes_and_persists(tmp_path: Path, monkeypatch, capsys):
    root = tmp_path / "ws"
    cli.write_config(root, "http://127.0.0.1:9/v1", "fake-local")

    class _FakeClient:
        name = "llm_fallback->local_lexicon"
        failures = 0
        last_error = None

        def fan_out(self, state, specs):
            from kernel.sys1_jev import DecisionResult

            return {
                s.id: DecisionResult(spec_id=s.id, kind=s.kind, value=1.0, confidence=0.8)
                for s in specs
            }

    monkeypatch.setattr(cli, "build_sys1_client", lambda cfg: _FakeClient())
    monkeypatch.delenv("XINGLI_CONFIG", raising=False)
    code = cli.main(["--root", str(root), "--mock", "--say", "你好"])
    assert code == 0
    capsys.readouterr()

    # 直接用 CLI 对象测指令（不启动交互循环）
    cfg = Config.load(root=root)
    from kernel.paths import Workspace

    ws = Workspace.open(root, cfg)
    from kernel.boot import Heartbeat
    from kernel.sys2_llm import MockSys2Client

    hb = Heartbeat.create(ws, cfg, sys2=MockSys2Client(), sys1_client=_FakeClient())
    chat = cli.ChatCLI(hb, ws, config_path=str(root / "xingli.toml"))
    import asyncio

    asyncio.run(chat._sys1_command("on Kev-4B-GGUF"))
    out = capsys.readouterr().out
    assert "活体探测：✓" in out
    assert "已写回" in out
    text = (root / "xingli.toml").read_text(encoding="utf-8")
    assert 'provider = "llm_fallback"' in text
    assert 'model = "Kev-4B-GGUF"' in text
    assert hb.config.sys1.provider == "llm_fallback"


def test_read_line_drains_queue_in_order(tmp_path: Path, monkeypatch, capsys):
    from kernel.boot import Heartbeat
    from kernel.paths import Workspace
    from kernel.sys2_llm import MockSys2Client

    cfg = Config()
    ws = Workspace.open(tmp_path / "ws", cfg)
    hb = Heartbeat.create(ws, cfg, sys2=MockSys2Client())
    chat = cli.ChatCLI(hb, ws)
    chat.queue.extend(["第一句", "/status", "第二句"])
    import asyncio

    async def drain():
        return [await chat._read_line() for _ in range(3)]

    assert asyncio.run(drain()) == ["第一句", "/status", "第二句"]
    assert chat._interrupt() is False
    chat.queue.append("插话")
    assert chat._interrupt() is True
