"""工作区与路径守卫。

铁律 1（不可变脑干 / 可进化皮层）在这里落地为可执行的路径检查：

* 星璃（运行时代码、进化提案）只能写 ``cortex/`` 及其子目录；
* ``kernel/``、``memory/``、``sensors/``、``cli.py`` 对她是只读的；
* 运行时索引与 Git 账本放在 ``.xingli/``（不属于她的可写皮层）。

任何持久化入口都必须经过 :func:`assert_writable`，这样“安全边界不在她手里”
就不是一句口号，而是一个会在越界时抛异常的函数。
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from kernel.config import Config


class BoundaryViolation(PermissionError):
    """试图写越界路径（kernel / memory / sensors / 仓库根部）。"""


#: 仓库自带的种子皮层（用于在空工作区里唤醒一个初始的星璃）
SEED_ROOT = Path(__file__).resolve().parent.parent

#: 需要从种子复制的最小皮层文件
SEED_FILES = (
    "persona.py",
    "focus_policy.py",
    "reflexes.py",
    "policies.py",
    "relations.json",
    "world.json",
)
SEED_DIRS = ("skills", "episodes", "turns", "state")


@dataclass(frozen=True)
class Workspace:
    """一个星璃实例的根目录。

    ``root`` 之下：
      cortex/     她的活体动态权重（明文、可审计、Git 账本追踪）
      .xingli/    运行时数据库、Git 账本、日志（不属于她）
    """

    root: Path
    config: Config

    # ---- 构造 -----------------------------------------------------------
    @classmethod
    def open(
        cls,
        root: str | Path | None = None,
        config: Config | None = None,
        seed: bool = True,
    ) -> "Workspace":
        base = Path(root or Path.cwd()).resolve()
        cfg = config or Config.load(root=base)
        ws = cls(root=base, config=cfg)
        if seed:
            ws.ensure()
        return ws

    # ---- 目录 -----------------------------------------------------------
    @property
    def cortex(self) -> Path:
        return self.root / self.config.cortex_dir

    @property
    def runtime(self) -> Path:
        return self.root / self.config.runtime_dir

    @property
    def state(self) -> Path:
        return self.cortex / "state"

    @property
    def episodes(self) -> Path:
        return self.cortex / "episodes"

    @property
    def turns(self) -> Path:
        return self.cortex / "turns"

    @property
    def skills(self) -> Path:
        return self.cortex / "skills"

    @property
    def persona_file(self) -> Path:
        return self.cortex / "persona.py"

    @property
    def relations_file(self) -> Path:
        return self.cortex / "relations.json"

    @property
    def reflexes_file(self) -> Path:
        return self.cortex / "reflexes.py"

    @property
    def policies_file(self) -> Path:
        return self.cortex / "policies.py"

    @property
    def focus_policy_file(self) -> Path:
        return self.cortex / "focus_policy.py"

    @property
    def focus_state_file(self) -> Path:
        return self.state / "focus.json"

    @property
    def emotion_file(self) -> Path:
        return self.state / "emotion.json"

    @property
    def reflex_state_file(self) -> Path:
        return self.state / "reflex_state.json"

    @property
    def turn_counter_file(self) -> Path:
        return self.state / "turn_counter.json"

    @property
    def wake_queue_file(self) -> Path:
        return self.state / "wake_queue.json"

    @property
    def db_path(self) -> Path:
        return self.runtime / "xingli.db"

    @property
    def ledger_dir(self) -> Path:
        return self.runtime / "ledger.git"

    @property
    def sleep_reports(self) -> Path:
        return self.state / "sleep_reports"

    # ---- 初始化 ---------------------------------------------------------
    def ensure(self) -> "Workspace":
        """创建目录结构；若皮层是空的，则从仓库种子唤醒一份初始人格。"""
        for p in (
            self.cortex,
            self.state,
            self.episodes,
            self.turns,
            self.skills,
            self.runtime,
            self.sleep_reports,
        ):
            p.mkdir(parents=True, exist_ok=True)

        for name in SEED_FILES:
            target = self.cortex / name
            if not target.exists():
                src = SEED_ROOT / "cortex" / name
                if src.is_file() and src.resolve() != target.resolve():
                    shutil.copy2(src, target)
        init = self.skills / "__init__.py"
        if not init.exists():
            src = SEED_ROOT / "cortex" / "skills" / "__init__.py"
            if src.is_file() and src.resolve() != init.resolve():
                shutil.copy2(src, init)
            else:
                init.write_text('"""星璃自我进化产出的技能模块。"""\n', encoding="utf-8")
        return self

    # ---- 守卫 -----------------------------------------------------------
    def is_writable(self, path: str | Path) -> bool:
        try:
            Path(path).resolve().relative_to(self.cortex.resolve())
            return True
        except ValueError:
            return False

    def assert_writable(self, path: str | Path) -> Path:
        p = Path(path)
        if not self.is_writable(p):
            raise BoundaryViolation(
                f"越界写入被拒绝：{p} 不在可写皮层 {self.cortex} 之内（铁律 1）"
            )
        return p
