"""读取/修改「数据型 .py 文件」的安全工具。

cortex/ 里的 reflexes.py、policies.py、focus_policy.py 都是**纯数据模块**
（只有 ``NAME = [...]`` 这样的字面量）。内核用 ``ast.literal_eval`` 读取它们，
因此即便这些文件被星璃进化坏了，也绝不会在内核进程里执行任意代码。

追加新条目时采用“文本级插入”（在列表右括号前插入），而不是整体重写——
这样人类留下的注释、排版与 git diff 都能保住，可审计性不打折。
"""

from __future__ import annotations

import ast
import pprint
from pathlib import Path
from typing import Any

_HEADER = '''"""{doc}"""\n\n'''


class LiteralModuleError(ValueError):
    """数据模块既不是合法 Python，也不含期望的字面量。"""


def _find_assign(tree: ast.Module, varname: str) -> ast.Assign:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == varname:
                    return node
        if isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == varname:
                raise LiteralModuleError(f"{varname} 是注解赋值，暂不支持自动追加")
    raise LiteralModuleError(f"找不到数据变量 {varname}")


def load_literal(path: str | Path, varname: str, default: Any = None) -> Any:
    """安全读取数据模块里的字面量；文件缺失或损坏时返回 default。"""
    p = Path(path)
    if not p.is_file():
        return default
    try:
        tree = ast.parse(p.read_text(encoding="utf-8"))
        node = _find_assign(tree, varname)
        return ast.literal_eval(node.value)
    except (SyntaxError, ValueError, LiteralModuleError):
        return default


def literal_error(path: str | Path, varname: str) -> str | None:
    """校验数据模块；返回错误描述或 None。"""
    p = Path(path)
    if not p.is_file():
        return f"{p} 不存在"
    try:
        tree = ast.parse(p.read_text(encoding="utf-8"))
        node = _find_assign(tree, varname)
        ast.literal_eval(node.value)
        return None
    except SyntaxError as exc:
        return f"语法错误: {exc}"
    except (ValueError, LiteralModuleError) as exc:
        return f"读取失败: {exc}"


def format_literal(item: Any, indent: int = 4) -> str:
    return pprint.pformat(item, width=96, sort_dicts=False, indent=1).replace(
        "\n", "\n" + " " * indent
    )


def append_list_item(path: str | Path, varname: str, item: Any, doc: str = "") -> str:
    """在 ``varname = [...]`` 的右括号前插入一个条目，保留原有注释与排版。"""
    p = Path(path)
    if not p.is_file():
        p.parent.mkdir(parents=True, exist_ok=True)
        text = _HEADER.format(doc=doc or f"{varname} data module") + (
            f"{varname} = [\n{format_literal(item)},\n]\n"
        )
        p.write_text(text, encoding="utf-8")
        return text

    source = p.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = _find_assign(tree, varname)
    if not isinstance(node.value, ast.List):
        raise LiteralModuleError(f"{varname} 不是列表字面量，拒绝自动追加")
    end_line = node.value.end_lineno
    end_col = node.value.end_col_offset
    if end_line is None or end_col is None:
        raise LiteralModuleError(f"无法定位 {varname} 的结束位置")

    lines = source.splitlines(keepends=True)
    target = lines[end_line - 1]
    # end_col 指向 ']' 之后；找到这一行里的 ']' 位置
    close_idx = end_col - 1
    if target[close_idx] != "]":
        close_idx = target.rfind("]", 0, end_col)
    if close_idx < 0:
        raise LiteralModuleError("无法定位列表的右括号")

    line_indent = target[: len(target) - len(target.lstrip())]
    body_indent = line_indent + "    "
    formatted = pprint.pformat(item, width=96, sort_dicts=False, indent=1)
    block = "".join(
        body_indent + line + "\n" for line in formatted.splitlines()
    )
    # 保底：条目前面要有逗号（原列表最后一项可能没有尾逗号）
    prefix = target[:close_idx]
    if prefix.strip() and not prefix.rstrip().endswith(","):
        block = ",\n" + block
    new_target = (
        target[:close_idx].rstrip() + "\n" + block + line_indent + target[close_idx:]
    )
    lines[end_line - 1] = new_target
    new_source = "".join(lines)

    # 写入前自检：必须仍能被 literal_eval 解析
    check = ast.parse(new_source)
    candidate = ast.literal_eval(_find_assign(check, varname).value)
    if not isinstance(candidate, list) or len(candidate) < 1:
        raise LiteralModuleError("追加后自检失败")
    p.write_text(new_source, encoding="utf-8")
    return new_source


def write_list_module(path: str | Path, varname: str, items: list[Any], doc: str = "") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = ",\n".join(pprint.pformat(i, width=96, sort_dicts=False) for i in items)
    p.write_text(
        _HEADER.format(doc=doc or varname) + f"{varname} = [\n{body}\n]\n",
        encoding="utf-8",
    )
