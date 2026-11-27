"""让 `pytest` 在仓库根目录直接运行（kernel/memory/sensors 是顶层包）。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
