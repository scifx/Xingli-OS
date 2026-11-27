#!/usr/bin/env python3
"""把“为什么慢”量出来：帧有多大、一拍要发几次请求、一轮要几拍。

不连真实端点，也不改代码路径：用假端点（tests/fake_openai.py）复现
“思考型 4B”的行为，然后打印真实的 token / 请求数 / 拍数。

    python tools/measure.py

输出里的 Tokens 用 `kernel.textutil.estimate_tokens`（CJK 1 token/字，
其余 3.6 字符/token），和内核做预算时用的是同一把尺子。
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fake_openai import FakeOpenAI, chat_reply  # noqa: E402

from kernel.boot import Heartbeat  # noqa: E402
from kernel.config import Config  # noqa: E402
from kernel.models import TurnRecord  # noqa: E402
from kernel.paths import Workspace  # noqa: E402
from kernel.sys2_llm import SYSTEM_CONTRACT, SYSTEM_CONTRACT_COMPACT  # noqa: E402
from kernel.textutil import estimate_tokens  # noqa: E402
from memory.frame_compiler import render_identity  # noqa: E402

REASONING = "首先，用户要求只输出合法JSON。字段是 inner_monologue…我们需要分析一下他的话。"


def handler(payload, _headers):
    msgs = payload["messages"]
    if "记忆抽取器" in msgs[0]["content"]:
        return chat_reply(
            json.dumps(
                {"memory_ops": [{"op": "memorize", "lod0": "主人喜欢菠萝和芒果"}]},
                ensure_ascii=False,
            )
        )
    if msgs[-1]["role"] == "assistant":
        return chat_reply('好，我记着。"}')
    return chat_reply(REASONING)


def seed(hb: Heartbeat, memories: int, turns: int) -> None:
    for i in range(memories):
        hb.store.put(
            lod0=f"主人第{i}次提到工作与晚饭",
            lod1="他说项目紧，晚上想简单吃点。",
            lod2="他讲了很多实现细节，也说了很累。",
            importance=0.6,
            source_id="owner",
        )
    for i in range(turns):
        hb.store.add_turn(
            TurnRecord(
                turn_id=hb.store.next_turn_id(),
                speaker_id="owner",
                user_text=f"第{i}轮：今天加班到很晚，随便聊聊。",
                assistant_text="主人，我记着你爱吃菠萝和芒果。",
            )
        )


def main() -> int:
    print("== 固定开销（每一拍都要重新 prefill）==")
    print(f"  full 契约   {estimate_tokens(SYSTEM_CONTRACT):>5} tok")
    print(f"  compact 契约{estimate_tokens(SYSTEM_CONTRACT_COMPACT):>5} tok")
    print(f"  身份块      {estimate_tokens(render_identity({'NAME': '星璃'})):>5} tok（出厂 persona 约 247）")
    print()

    print("== 帧随历史增长（compact，--fast）==")
    print(f"  {'记忆':>4} {'轮次':>4}  帧tokens  分段")
    for memories, turns in ((0, 0), (4, 1), (10, 3), (22, 7), (40, 15), (80, 30)):
        with FakeOpenAI(handler) as server:
            cfg = Config().with_sys2(
                provider="openai_compatible",
                base_url=server.base_url,
                model="spark-4b",
                contract="compact",
                max_tokens=700,
            )
            ws = Workspace.open(Path(tempfile.mkdtemp()) / "ws", cfg)
            hb = Heartbeat.create(ws, cfg)
            seed(hb, memories, turns)
            hb.handle_message("owner", "我喜欢吃菠萝和芒果")
            frame = hb.last_frame
            sections = " ".join(f"{k}={v}" for k, v in frame.section_tokens.items())
            print(f"  {memories:>4} {turns:>4}  {frame.total_tokens:>8}  {sections}")
    print()

    print("== 一条用户消息要发几次请求 / 跑几拍（思考型模型 + 自动预填 + 记忆回填）==")
    with FakeOpenAI(handler) as server:
        cfg = Config().with_sys2(
            provider="openai_compatible",
            base_url=server.base_url,
            model="spark-4b",
            contract="compact",
            max_tokens=700,
        )
        ws = Workspace.open(Path(tempfile.mkdtemp()) / "ws", cfg)
        hb = Heartbeat.create(ws, cfg)
        seed(hb, 22, 7)
        result = hb.handle_message("owner", "我喜欢吃菠萝和芒果")
        print(f"  请求数 {server.calls} · 拍数 {len(result.ticks)} · 输出 {result.says}")
        print("  （请求数 3 拆解：原请求 + 检测到叙述后的预填重试 + 记忆抽取）")
    print()

    print("== 换算 ==  ")
    print("  你的机器实测：生成 53 tok/s、一拍 19~39s、一条消息最多 4 拍。")
    print("  1 拍 = 1~3 次请求 × (1.0~2.2k tok 的 prefill + 最多 700~1000 tok 生成)。")
    print("  4 拍 × 3 请求 = 最多 12 次请求 ≈ 一分钟；而 v1 的做法是 1 次请求、~360 tok。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
