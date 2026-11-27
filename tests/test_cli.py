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
