"""Git 成长账本（海马体的索引层 / 时光机）。

``cortex/`` 是一个独立 Git 仓库（GIT_DIR 放在 ``.xingli/ledger.git``），
因此：

* 她的每一次进化都有 commit，commit message 就是那次进化的理由；
* ``git diff`` 能看清她“怎么长成今天这样”；
* ``/rollback <commit>`` 秒级回到任意历史健康快照。

用独立 GIT_DIR 而不是仓库根目录，是为了不污染用户自己的项目历史。
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

from kernel.paths import Workspace


class LedgerError(RuntimeError):
    pass


@dataclass
class CommitInfo:
    hash: str
    message: str
    date: str

    def to_dict(self) -> dict[str, str]:
        return {"hash": self.hash, "message": self.message, "date": self.date}


class GitLedger:
    def __init__(self, ws: Workspace):
        self.ws = ws
        self.git_dir = ws.ledger_dir
        self.work_tree = ws.cortex

    # ------------------------------------------------------------------
    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update(
            {
                "GIT_DIR": str(self.git_dir),
                "GIT_WORK_TREE": str(self.work_tree),
                "GIT_AUTHOR_NAME": "Xingli",
                "GIT_AUTHOR_EMAIL": "xingli@localhost",
                "GIT_COMMITTER_NAME": "Xingli",
                "GIT_COMMITTER_EMAIL": "xingli@localhost",
                "GIT_CONFIG_NOSYSTEM": "1",
            }
        )
        return env

    def _run(
        self, *args: str, check: bool = True, timeout: float = 20.0
    ) -> subprocess.CompletedProcess:
        proc = subprocess.run(
            ["git", *args],
            env=self._env(),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if check and proc.returncode != 0:
            raise LedgerError(
                f"git {' '.join(args)} 失败: {proc.stderr.strip() or proc.stdout.strip()}"
            )
        return proc

    @property
    def available(self) -> bool:
        return shutil.which("git") is not None

    @property
    def initialized(self) -> bool:
        return (self.git_dir / "HEAD").exists()

    def ensure(self) -> "GitLedger":
        if not self.available:
            raise LedgerError("系统里找不到 git，无法建立成长账本")
        if self.initialized:
            return self
        self.git_dir.parent.mkdir(parents=True, exist_ok=True)
        self.work_tree.mkdir(parents=True, exist_ok=True)
        # 独立 GIT_DIR + 指定 work tree
        self._run("init", "--quiet")
        self._run("config", "user.name", "Xingli")
        self._run("config", "user.email", "xingli@localhost")
        self._run("config", "core.quotepath", "false")
        return self

    # ------------------------------------------------------------------
    def has_commits(self) -> bool:
        if not self.initialized:
            return False
        proc = self._run("rev-parse", "--verify", "HEAD", check=False)
        return proc.returncode == 0

    def commit(self, message: str) -> str | None:
        """提交当前 cortex 快照；没有变化时返回 None。"""
        self.ensure()
        self._run("add", "-A")
        proc = self._run("commit", "-m", message, check=False)
        if proc.returncode != 0:
            if "nothing to commit" in (proc.stdout + proc.stderr):
                return None
            raise LedgerError(proc.stderr.strip() or proc.stdout.strip())
        head = self._run("rev-parse", "HEAD").stdout.strip()
        return head or None

    def log(self, limit: int = 10) -> list[CommitInfo]:
        if not self.has_commits():
            return []
        proc = self._run(
            "log",
            f"-{max(1, int(limit))}",
            "--pretty=format:%h%x1f%s%x1f%ad",
            "--date=iso",
            check=False,
        )
        out: list[CommitInfo] = []
        for line in proc.stdout.splitlines():
            parts = line.split("\x1f")
            if len(parts) == 3:
                out.append(CommitInfo(hash=parts[0], message=parts[1], date=parts[2]))
        return out

    def resolve(self, ref: str) -> str:
        if not self.has_commits():
            raise LedgerError("账本还是空的")
        proc = self._run("rev-parse", "--verify", f"{ref}^{{commit}}", check=False)
        if proc.returncode != 0:
            raise LedgerError(f"找不到提交: {ref}")
        return proc.stdout.strip()

    def show(self, ref: str, path: str | None = None) -> str:
        commit = self.resolve(ref)
        args = ["show", f"{commit}:{path}"] if path else ["show", "--stat", commit]
        return self._run(*args).stdout

    def rollback(self, ref: str) -> dict[str, Any]:
        """一键时光机：把 cortex/ 恢复到指定提交（含清理该提交后新增的文件）。"""
        commit = self.resolve(ref)
        before = self.log(1)
        files = self._run("ls-tree", "-r", "--name-only", commit).stdout.split()
        self._run("reset", "--hard", commit)
        # 清掉该快照里不存在的文件，避免“回滚了但残留坏文件”
        self._run("clean", "-fd")
        return {
            "commit": commit[:12],
            "files": len(files),
            "previous_head": before[0].hash if before else None,
        }

    def status(self) -> str:
        if not self.initialized:
            return "（账本尚未初始化）"
        return self._run("status", "--short").stdout.strip()
