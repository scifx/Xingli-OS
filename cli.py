#!/usr/bin/env python3
"""星璃最简 CLI 聊天测试台。

* 默认离线（mock 大脑 + 本地词法 Sys1），不需要任何 API Key；
* 配好 ``xingli.toml`` 后自动切到真实底座模型与 Jev；
* 在你不敲键盘时，她预约的 ``schedule_wake`` 也会真的在终端里主动开口。

常用指令（斜杠开头，不会当成聊天内容）：::

    /as <speaker_id>   切换当前说话人（测试隐私墙与社交反哺）
    /who               当前说话人与其关系档案
    /thought           开关内心 OS / 反射信号打印
    /frame             上一轮实际发给 Sys2 的认知帧与 Token 占用
    /focus             当前焦点、前后衰减与各轮 LOD 层级
    /wake              待触发的自主唤醒队列与配额
    /sleep             立刻跑一次睡眠整合
    /log               最近的 Git 进化提交
    /rollback <commit> 时光机回滚 cortex/
    /recall <id>       展开某条长期记忆
    /commit [msg]      给当前 cortex 打一个快照
    /stats             记忆与预算概览
    /help /quit
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from kernel.boot import Heartbeat
from kernel.config import Config
from kernel.git_ledger import LedgerError
from kernel.paths import Workspace
from kernel.sys2_llm import Sys2Error

BANNER = """\
┌─ 星璃 Xingli-OS ─────────────────────────────────────────────┐
│ 记忆不是仓库，是改变下一拍计算的差分算子。                    │
│ 输入 /help 看指令；直接说话即可。                              │
└──────────────────────────────────────────────────────────────┘"""


class ChatCLI:
    def __init__(self, hb: Heartbeat, ws: Workspace, speaker: str = "owner"):
        self.hb = hb
        self.ws = ws
        self.speaker = speaker
        self.show_thought = True
        self.running = True

    # ------------------------------------------------------------------
    def _print_result(self, result) -> None:
        for tick in result.ticks:
            if self.show_thought:
                sig = [
                    f"{s.reflex_id}:{s.gate}({s.probability:.2f})"
                    for s in tick.signals
                    if s.gate in {"high", "mid"}
                ]
                if sig:
                    print(f"  ⟨潜意识⟩ {' '.join(sig)}")
                if tick.packet and tick.packet.inner_monologue:
                    for line in tick.packet.inner_monologue.splitlines():
                        print(f"  # {line}")
                for warn in tick.actions.errors:
                    print(f"  ⟨内核⟩ {warn}")
                for evo in tick.actions.evolutions:
                    mark = "✓" if evo.get("ok") else "✗"
                    print(f"  ⟨进化{mark}⟩ {evo.get('message')}")
                if tick.actions.recalls:
                    print(f"  ⟨召回⟩ {len(tick.actions.recalls)} 条（注入下一拍）")
                if tick.actions.memorized:
                    print(f"  ⟨记住⟩ {', '.join(tick.actions.memorized)}")
        for say in result.says:
            print(f"星璃: {say}")
        if result.halted_reason:
            print(f"  ⟨熔断⟩ {result.halted_reason}")
        print()

    # ------------------------------------------------------------------
    async def _poll_wakes(self) -> None:
        while self.running:
            try:
                results = await asyncio.get_running_loop().run_in_executor(
                    None, self.hb.poll_wakes
                )
            except Exception as exc:  # noqa: BLE001
                print(f"⟨调度器异常⟩ {type(exc).__name__}: {exc}")
                results = []
            for result in results:
                reason = result.event.extra.get("reason", "")
                print(f"\n（星璃主动开口 · {reason or '自主唤醒'}）")
                self._print_result(result)
            await asyncio.sleep(0.5)

    async def _read(self, prompt: str) -> str | None:
        print(prompt, end="", flush=True)
        try:
            line = await asyncio.get_running_loop().run_in_executor(None, sys.stdin.readline)
        except Exception:  # noqa: BLE001
            return None
        if line == "":
            return None
        return line.rstrip("\n")

    async def run(self) -> None:
        print(BANNER)
        print(f"（System Two: {getattr(self.hb.sys2, 'name', '?')} ｜ "
              f"System One: {getattr(self.hb.reflex.client, 'name', '?')} ｜ "
              f"工作区: {self.ws.root}）\n")
        poller = asyncio.create_task(self._poll_wakes())
        try:
            while self.running:
                line = await self._read(f"{self.speaker}> ")
                if line is None:
                    print("\n（再见）")
                    break
                line = line.strip()
                if not line:
                    continue
                if line.startswith("/"):
                    await self._command(line)
                    continue
                result = await asyncio.get_running_loop().run_in_executor(
                    None, self.hb.handle_message, self.speaker, line
                )
                self._print_result(result)
        finally:
            self.running = False
            poller.cancel()

    # ------------------------------------------------------------------
    async def _command(self, line: str) -> None:
        cmd, _, arg = line.partition(" ")
        cmd = cmd.lower()
        arg = arg.strip()
        if cmd in {"/quit", "/exit", "/q"}:
            self.running = False
            print("（再见）")
        elif cmd == "/help":
            print(__doc__)
        elif cmd == "/as":
            if not arg:
                print(f"当前说话人: {self.speaker}")
            else:
                self.speaker = arg
                self.hb.relations.ensure(arg)
                profile = self.hb.relations.get(arg)
                print(
                    f"切换为 {arg}（{profile.get('role')}，"
                    f"亲密度 {profile.get('intimacy', 0):.2f}，"
                    f"信任 {profile.get('global_trust', 0):.2f}）"
                )
        elif cmd == "/who":
            print(json.dumps(self.hb.relations.get(self.speaker), ensure_ascii=False, indent=2))
        elif cmd == "/thought":
            self.show_thought = not self.show_thought
            print(f"内心 OS 打印：{'开' if self.show_thought else '关'}")
        elif cmd == "/frame":
            frame = self.hb.last_frame
            if not frame:
                print("（还没有产生认知帧）")
                return
            print(f"Token: {frame.total_tokens} / 预算 {self.hb.config.budget.frame_tokens}")
            for name, tokens in frame.section_tokens.items():
                print(f"  {name:<10} {tokens:>5}")
            print("\n" + frame.user)
        elif cmd == "/focus":
            snap = self.hb.focus.snapshot(self.speaker)
            print(
                f"焦点 #{snap['focus_turn_id']} ｜ 前衰减 {snap['decay_before']} ｜ "
                f"后衰减 {snap['decay_after']} ｜ 保底 {snap['recency_floor']} 轮 ｜ "
                f"钉选 {snap['pinned'] or '-'}"
            )
            for row in snap["levels"]:
                print(f"  #{row['turn_id']:<4} {row['lod']:<5} R={row['R']}")
        elif cmd == "/wake":
            tickets = self.hb.scheduler.pending()
            print(
                f"配额剩余 {self.hb.scheduler.quota_left()} ｜ 队列 {len(tickets)} 条"
            )
            for t in tickets:
                print(f"  {t.due_at}  {t.reason}  → {t.target_speaker}")
        elif cmd == "/sleep":
            print("（开始睡眠整合……）")
            report = await asyncio.get_running_loop().run_in_executor(None, self.hb.sleep)
            print(report.summary())
            for sample in report.samples:
                mark = "✓" if sample.passed else "✗"
                print(
                    f"  {mark} #{sample.node_id} 覆盖率 "
                    f"{sample.coverage_before:.2f}→{sample.coverage_after:.2f}"
                    + ("（已重写索引）" if sample.repaired else "")
                )
        elif cmd == "/log":
            commits = self.hb.ledger.log(10)
            if not commits:
                print("（账本还没有提交）")
            for c in commits:
                print(f"  {c.hash}  {c.date}  {c.message}")
        elif cmd == "/rollback":
            if not arg:
                print("用法：/rollback <commit>")
                return
            try:
                info = self.hb.ledger.rollback(arg)
            except LedgerError as exc:
                print(f"回滚失败：{exc}")
                return
            self.hb.store.reindex()
            print(f"已回滚到 {info['commit']}（{info['files']} 个文件）；索引已重建")
        elif cmd == "/recall":
            node = self.hb.store.get(arg) if arg else None
            if node is None or not self.hb.store.is_visible(node, self.speaker):
                print("（这条记忆在当前说话人面前不存在）")
                return
            print(f"#{node.id}  来源 {node.source_id}（{node.epistemic}）")
            print(node.lod2 or node.lod1)
        elif cmd == "/commit":
            commit = self.hb.ledger.commit(arg or "manual snapshot")
            print(f"已提交 {commit[:10]}" if commit else "（没有变化）")
        elif cmd == "/stats":
            print(json.dumps(self.hb.store.stats(), ensure_ascii=False, indent=2))
            print(f"配额剩余 {self.hb.scheduler.quota_left()}")
        elif cmd == "/init":
            print(f"工作区已初始化：{self.ws.root}")
        else:
            print(f"未知指令：{cmd}（/help 查看列表）")


def _prepare(args) -> tuple[Workspace, Config]:
    root = Path(args.root).resolve() if args.root else Path.cwd()
    cfg = Config.load(root=root)
    if args.mock:
        cfg = cfg.with_sys2(provider="mock")
    ws = Workspace.open(root, cfg)
    return ws, cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="星璃 CLI 聊天测试台")
    parser.add_argument("--root", help="工作区目录（默认当前目录）")
    parser.add_argument("--mock", action="store_true", help="强制使用离线 mock 大脑")
    parser.add_argument("--as", dest="speaker", default="owner", help="初始说话人 id")
    parser.add_argument("--say", help="非交互：只说一句话后退出")
    parser.add_argument("--sleep", action="store_true", help="非交互：跑一次睡眠整合")
    args = parser.parse_args(argv)

    ws, cfg = _prepare(args)
    try:
        hb = Heartbeat.create(ws, cfg)
    except Sys2Error as exc:
        print(f"System Two 初始化失败：{exc}", file=sys.stderr)
        print("提示：加 --mock 用离线大脑跑通流程；或在 xingli.toml 里配置 Key。", file=sys.stderr)
        return 2

    if args.sleep:
        print(hb.sleep().summary())
        return 0
    if args.say:
        result = hb.handle_message(args.speaker, args.say)
        for say in result.says:
            print(f"星璃: {say}")
        for tick in result.ticks:
            for warn in tick.actions.errors:
                print(f"⟨内核⟩ {warn}")
        return 0

    cli = ChatCLI(hb, ws, speaker=args.speaker)
    try:
        asyncio.run(cli.run())
    except KeyboardInterrupt:
        print("\n（中断）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
