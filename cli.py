#!/usr/bin/env python3
"""星璃最简 CLI 聊天测试台。

* 默认离线（mock 大脑 + 本地词法 Sys1），不需要任何 API Key；
* 配好 ``xingli.toml`` 后自动切到真实端点（llama.cpp / vLLM / 任何 OpenAI 兼容服务）；
* 每一轮结束都会打印 **记忆文件的变化**：新建了哪个 episode、LOD0 写了什么、
  哪些状态被更新——她的记忆是可以直接 `cat` 的明文。

常用指令（斜杠开头，不会当成聊天内容）：::

    /files [n]         长期记忆清单（文件大小 / 认识论 / 重要度 / LOD0）
    /cat <id|路径>     直接看某条记忆的原始 JSON（记忆文件长什么样）
    /tree              cortex/ 的目录树与占用
    /prefill on|off    预填 JSON 开头（思考型模型不肯按格式输出时的杀手锏）
    /sys1             丘脑（System One）状态 + 活体探测
    /sys1 on <模型>   用本地决策模型当丘脑（立刻生效，并写进 xingli.toml）
    /sys1 off         退回本地词法
    /forget <id>      作废一条记忆（只归档，文件与账本都留着）
    /tidy [dry]        存量记忆卫生：归档占位符/状态回显，合并同一事实的重复，修 LOD 字段
    /status            端点连通性、模型、Token 预算与记忆统计
    /as <speaker_id>   切换当前说话人（测试隐私墙与社交反哺）
    /who               当前说话人与其关系档案
    /thought           开关内心 OS / 反射信号打印
    /frame             上一轮实际发给 Sys2 的认知帧与 Token 占用
    /focus             当前焦点、前后衰减与各轮 LOD 层级
    /wake              待触发的自主唤醒队列与配额
    /sleep             立刻跑一次睡眠整合
    /log               最近的 Git 进化提交
    /rollback <commit> 时光机回滚 cortex/
    /commit [msg]      给当前 cortex 打一个快照
    /stats             记忆与预算概览
    /help /quit

非交互用法::

    python cli.py --init-config --base-url http://127.0.0.1:8080/v1 --model qwen2.5-7b-instruct
    python cli.py --ping                 # 只探测端点
    python cli.py --selftest             # 真跑一轮，打印解析结果与记忆文件
    python cli.py --say "你好" --show-memory
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import threading
import time
import traceback
from dataclasses import replace
from pathlib import Path

import viewer
from kernel.boot import Heartbeat
from kernel.config import Config
from kernel.git_ledger import LedgerError
from kernel.paths import Workspace
from kernel.sys1_jev import build_client as build_sys1_client
from kernel.sys2_llm import Sys2Error, probe_sys2

BANNER = """\
┌─ 星璃 Xingli-OS ─────────────────────────────────────────────┐
│ 记忆不是仓库，是改变下一拍计算的差分算子。                    │
│ 输入 /help 看指令；直接说话即可。                              │
└──────────────────────────────────────────────────────────────┘"""

THINKING_HINT = "  … 星璃在想（本地模型可能慢，稍等；这时你也可以直接打字，我收尾后立刻接）"


class _InputPump(threading.Thread):
    """后台读 stdin。

    这样“她在想”的那几十秒里你也能打字：主循环每 0.25s 收一次输入，
    既能把下一句排进队列，也能当成“别自转了，主人插话了”的刹车信号。
    """

    def __init__(self, stream=sys.stdin) -> None:
        super().__init__(daemon=True, name="xingli-stdin")
        self.stream = stream
        self._lines: list[str] = []
        self._lock = threading.Lock()
        self.closed = False

    def run(self) -> None:  # pragma: no cover - 交互线程
        try:
            for line in self.stream:
                with self._lock:
                    self._lines.append(line.rstrip("\r\n"))
        except Exception:  # noqa: BLE001 - 关窗 / Ctrl-C
            pass
        finally:
            self.closed = True

    def take(self) -> str | None:
        with self._lock:
            return self._lines.pop(0) if self._lines else None

    def pending(self) -> bool:
        with self._lock:
            return bool(self._lines)


class ChatCLI:
    def __init__(
        self,
        hb: Heartbeat,
        ws: Workspace,
        speaker: str = "owner",
        *,
        show_memory: bool = True,
        config_path: str = "",
    ):
        self.hb = hb
        self.ws = ws
        self.speaker = speaker
        self.show_thought = True
        self.show_memory = show_memory
        self.config_path = config_path
        self.mode_stats: dict[str, int] = {}
        self.running = True
        self.pump: _InputPump | None = None
        #: 她在想的时候你打的字：按顺序排队，收尾后照原顺序处理
        self.queue: list[str] = []
        self._prompt_shown = False
        self._busy = False
        # 聊天与自主唤醒串行执行（避免并发写记忆）；线程用 daemon，
        # 这样在等模型时按 Ctrl-C 能立刻退出，不会被 atexit join 卡住。
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def _serialized(self, fn, *args, **kwargs):
        with self._lock:
            return fn(*args, **kwargs)

    async def _call(self, fn, *args, **kwargs):
        """在守护线程里跑阻塞调用：按 Ctrl-C 时不必等 HTTP 请求收尾。"""
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        def runner() -> None:
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001 - 交回事件循环
                loop.call_soon_threadsafe(future.set_exception, exc)
            else:
                loop.call_soon_threadsafe(future.set_result, result)

        threading.Thread(target=runner, daemon=True, name="xingli-call").start()
        return await future

    def _gen_stats(self) -> tuple[float | None, int | None]:
        timings = getattr(self.hb.sys2, "last_timings", None) or {}
        usage = getattr(self.hb.sys2, "last_usage", None) or {}
        rate = timings.get("predicted_per_second")
        generated = timings.get("predicted_n") or usage.get("completion_tokens")
        return (
            float(rate) if isinstance(rate, (int, float)) else None,
            int(generated) if isinstance(generated, (int, float)) else None,
        )

    def _count_modes(self, result) -> None:
        for tick in result.ticks:
            if tick.mode:
                self.mode_stats[tick.mode] = self.mode_stats.get(tick.mode, 0) + 1

    def _memory_change(self, baseline, *, limit: int = 6) -> str:
        if not self.show_memory or baseline is None:
            return ""
        changed = viewer.diff(baseline, viewer.snapshot(self.ws.cortex))
        return viewer.format_diff(changed, self.ws.cortex, self.hb.store, limit=limit)

    def _print_result(self, result, baseline=None) -> None:
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
                seen: set[str] = set()
                for warn in tick.actions.errors:
                    if warn and warn not in seen:
                        seen.add(warn)
                        print(f"  ⟨内核⟩ {warn}")
                for note in getattr(tick.actions, "notes", []):
                    if note and note not in seen:
                        seen.add(note)
                        print(f"  ⟨记忆⟩ {note}")
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
        if self.show_thought and result.ticks:
            last = result.ticks[-1]
            total_ms = sum(t.duration_ms for t in result.ticks)
            rate, generated = self._gen_stats()
            extra = f" · 出 {generated} tok" if generated else ""
            extra += f" · {rate:.1f} tok/s" if rate else ""
            total_sys1 = sum(t.sys1_ms for t in result.ticks)
            print(
                f"  ⟨计时⟩ {total_ms / 1000:.1f}s · {len(result.ticks)} 拍 · "
                f"帧 {last.frame_tokens} tokens{extra}"
                + (f" · 丘脑 {total_sys1:.0f}ms" if total_sys1 >= 1 else "")
            )
            for tick in result.ticks:
                if tick.mode and tick.mode != "json":
                    print(
                        f"  ⟨降级⟩ 第 {tick.tick} 拍输出为「{tick.mode}」："
                        "本拍能说话，但记忆/进化动作被跳过"
                    )
        change = self._memory_change(baseline)
        if change:
            for line in change.splitlines():
                print(f"  │ {line}")
        print()

    # ------------------------------------------------------------------
    async def _poll_wakes(self) -> None:
        while self.running:
            if self._busy:
                # 正在聊就别插播主动唤醒：既避免抢锁，也避免对话被劈成两半
                await asyncio.sleep(0.5)
                continue
            baseline = viewer.snapshot(self.ws.cortex) if self.show_memory else None
            try:
                self._busy = True
                results = await self._run_call(
                    self.hb.poll_wakes, None, self._interrupt
                )
            except Exception as exc:  # noqa: BLE001
                print(f"⟨调度器异常⟩ {type(exc).__name__}: {exc}")
                results = []
            finally:
                self._busy = False
            for result in results:
                reason = result.event.extra.get("reason", "")
                print(f"\n（星璃主动开口 · {reason or '自主唤醒'}）")
                self._count_modes(result)
                self._print_result(result, baseline)
            await asyncio.sleep(0.5)

    # ------------------------------------------------------------------
    async def run(self) -> None:
        print(BANNER)
        for notice in SYS1_NOTICES:
            print(notice)
        print(self._banner_context())
        self.pump = _InputPump()
        self.pump.start()
        poller = asyncio.create_task(self._poll_wakes())
        try:
            while self.running:
                line = await self._read_line()
                if line is None:
                    print("\n（再见）")
                    break
                line = line.strip()
                if not line:
                    continue
                if line.startswith("/"):
                    await self._safe_command(line)
                    continue
                await self._turn(line)
        finally:
            self.running = False
            poller.cancel()

    async def _safe_command(self, line: str) -> None:
        try:
            await self._command(line)
        except Exception as exc:  # noqa: BLE001 - 指令写错了也不该掀翻聊天
            print(f"⟨指令出错⟩ {type(exc).__name__}: {exc}")
            if os.environ.get("XINGLI_TRACEBACK"):
                traceback.print_exc()

    # ------------------------------------------------------------------
    def _interrupt(self) -> bool:
        """给她一个刹车信号：主人已经在排队说下一句了。"""
        return bool(self.queue)

    def _absorb_input(self) -> None:
        """把后台读到的行搬进队列（等模型的间隙调用）。"""
        if self.pump is None:
            return
        while True:
            line = self.pump.take()
            if line is None:
                return
            line = line.strip()
            if not line:
                continue
            self.queue.append(line)
            print(f"\n  … 收下了（这一拍收尾就处理）：{line[:40]}")

    async def _read_line(self) -> str | None:
        if not self._prompt_shown:
            print(f"{self.speaker}> ", end="", flush=True)
            self._prompt_shown = True
        while self.running:
            if self.queue:
                print()
                self._prompt_shown = False
                return self.queue.pop(0)
            line = self.pump.take() if self.pump else None
            if line is not None:
                self._prompt_shown = False
                return line
            if self.pump is not None and self.pump.closed:
                return None
            await asyncio.sleep(0.05)
        return None

    async def _run_call(self, fn, *args, **kwargs):
        """跑一次阻塞调用，同时保持输入可读（并把插话变成刹车信号）。"""
        task = asyncio.create_task(self._call(self._serialized, fn, *args, **kwargs))
        while True:
            done, _ = await asyncio.wait({task}, timeout=0.25)
            self._absorb_input()
            if task.done():
                return task.result()

    async def _turn(self, line: str) -> None:
        baseline = viewer.snapshot(self.ws.cortex) if self.show_memory else None
        print(THINKING_HINT, end="", flush=True)
        self._busy = True
        try:
            result = await self._run_call(
                self.hb.handle_message, self.speaker, line, self._interrupt
            )
        except Sys2Error as exc:
            print(f"\n⟨端点错误⟩ {exc}")
            return
        except Exception as exc:  # noqa: BLE001 - 一拍崩了不该带走整个聊天
            print(f"\n⟨这一拍出了异常，聊天还在⟩ {type(exc).__name__}: {exc}")
            if os.environ.get("XINGLI_TRACEBACK"):
                traceback.print_exc()
            return
        finally:
            self._busy = False
            self._clear_hint()
        self._count_modes(result)
        self._print_result(result, baseline)

    @staticmethod
    def _clear_hint() -> None:
        print("\r" + " " * 48 + "\r", end="", flush=True)

    # ------------------------------------------------------------------
    # System One（丘脑）诊断
    # ------------------------------------------------------------------
    def _sys1_banner(self) -> str:
        cfg = self.hb.config
        client = self.hb.reflex.client
        name = getattr(client, "name", "?")
        body = f"System One：{name}"
        if cfg.sys1.provider == "local" or not cfg.sys1.base_url:
            body += "（本地词法；想用本地决策模型：/sys1 on Kev-4B-GGUF）"
        elif "->" not in name:
            reason = getattr(client, "build_error", "")
            body += "  ⚠ 配置了端点但没能接上，已退回本地词法"
            body += f"：{reason}" if reason else "（/sys1 看原因）"
        else:
            body += f"  ← {cfg.sys1.base_url}（{cfg.sys1.model or '?'}，{cfg.sys1.timeout_sec}s）"
        return body

    def _probe_sys1(self) -> tuple[bool, str]:
        """真发一次丘脑请求：接上了吗？多快？退回本地词法没有？"""
        from kernel.sys1_jev import BASE_PERCEPTIONS

        client = self.hb.reflex.client
        state = (
            "说话人: 主人(owner) 关系:owner 亲密度0.85\n"
            "时间: 现在 情绪余温: 平静\n内容: 你好，我叫 lucky，记住我喜欢菠萝。"
        )
        started = time.monotonic()
        try:
            results = client.fan_out(state, BASE_PERCEPTIONS[:2])
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"
        ms = (time.monotonic() - started) * 1000
        detail = ", ".join(
            f"{k}={v.probability:.2f}" for k, v in results.items()
        )
        failures = int(getattr(client, "failures", 0) or 0)
        if failures:
            last = getattr(client, "last_error", "") or ""
            return False, f"主适配器失败，已退回本地词法：{last}（{ms:.0f}ms）"
        return True, f"{detail or '（没有信号）'}（{ms:.0f}ms）"

    def _patch_config_sys1(self, provider: str, model: str, base_url: str) -> str:
        """把 sys1 的改动写回 xingli.toml（只动 [sys1] 段，别的不碰）。"""
        path = Path(self.config_path) if self.config_path else None
        if path is None or not path.is_file():
            return "（没有配置文件，跳过持久化）"
        values = {"provider": provider, "base_url": base_url, "model": model}
        text = path.read_text(encoding="utf-8")
        if not re.search(r"^\[sys1\]", text, re.MULTILINE):
            block = "\n[sys1]\n" + "".join(
                f'{key} = "{value}"\n' for key, value in values.items()
            ) + "timeout_sec = 8\n"
            path.write_text(text.rstrip() + "\n" + block, encoding="utf-8")
            return f"已追加 [sys1] 到 {path}"
        out: list[str] = []
        written: set[str] = set()
        inside = False
        for line in text.splitlines():
            if re.match(r"^\s*\[", line):
                if inside:
                    out.extend(
                        f'{key} = "{value}"'
                        for key, value in values.items()
                        if key not in written
                    )
                inside = line.strip().lower().startswith("[sys1")
                written.clear() if inside else None
                out.append(line)
                continue
            if inside:
                match = re.match(r"^\s*(provider|base_url|model)\s*=", line)
                if match:
                    key = match.group(1)
                    if key in written:
                        continue  # 重复键：丢掉旧的
                    out.append(f'{key} = "{values[key]}"')
                    written.add(key)
                    continue
            out.append(line)
        if inside:
            out.extend(
                f'{key} = "{value}"'
                for key, value in values.items()
                if key not in written
            )
        path.write_text("\n".join(out) + "\n", encoding="utf-8")
        return f"已写回 {path}"

    async def _sys1_command(self, arg: str) -> None:
        cfg = self.hb.config
        parts = arg.split()
        sub = (parts[0] if parts else "").lower()
        if sub in {"", "status", "probe"}:
            print(self._sys1_banner())
            print(
                f"  配置：provider={cfg.sys1.provider} · model={cfg.sys1.model or '(空)'} · "
                f"base_url={cfg.sys1.base_url or '(空)'} · timeout={cfg.sys1.timeout_sec}s"
            )
            ok, detail = await self._call(self._probe_sys1)
            print(f"  活体探测：{'✓' if ok else '✗'} {detail}")
            if not ok and cfg.sys1.provider != "local":
                print("  常见原因：模型名写错 / 端点只跑了一个模型（llama.cpp 一次一个）/ 超时太短")
                print("  试用：/sys1 on Kev-4B-GGUF  或  /sys1 on Qwen3.5-2B-GGUF")
            return
        if sub in {"off", "local", "none"}:
            self.hb.config = replace(
                self.hb.config, sys1=replace(cfg.sys1, provider="local")
            )
            self.hb.reflex.config = self.hb.config
            self.hb.reflex.client = build_sys1_client(self.hb.config)
            print(f"System One 已切回本地词法：{getattr(self.hb.reflex.client, 'name', '?')}")
            print(self._patch_config_sys1("local", cfg.sys1.model, cfg.sys1.base_url))
            return
        if sub in {"on", "llm", "llm_fallback", "jev"}:
            model = " ".join(parts[1:]).strip() or cfg.sys1.model
            base_url = (cfg.sys1.base_url or cfg.sys2.base_url).rstrip("/")
            provider = "jev" if sub == "jev" else "llm_fallback"
            new_sys1 = replace(
                cfg.sys1,
                provider=provider,
                model=model,
                base_url=base_url,
                timeout_sec=min(cfg.sys1.timeout_sec, 15) or 8,
            )
            self.hb.config = replace(self.hb.config, sys1=new_sys1)
            self.hb.reflex.config = self.hb.config
            self.hb.reflex.client = build_sys1_client(self.hb.config)
            print(f"System One：{getattr(self.hb.reflex.client, 'name', '?')}（{model or '默认模型'}）")
            ok, detail = await self._call(self._probe_sys1)
            print(f"  活体探测：{'✓' if ok else '✗'} {detail}")
            if not ok:
                print("  ⚠ 没接上：仍然会正常工作，但丘脑退回本地词法（不会卡住聊天）")
            print(self._patch_config_sys1(provider, model, base_url))
            return
        print("用法：/sys1 [status] ｜ /sys1 on <模型名> ｜ /sys1 off")

    # ------------------------------------------------------------------
    def _banner_context(self) -> str:
        hb = self.hb
        cfg = self.hb.config
        episodes = len(list(self.ws.episodes.glob("*.json"))) if self.ws.episodes.exists() else 0
        turns = len(list(self.ws.turns.glob("*.json"))) if self.ws.turns.exists() else 0
        lines = [
            f"System Two：{getattr(hb.sys2, 'name', '?')}"
            + (
                f"  ← {cfg.sys2.base_url}"
                if cfg.sys2.provider != "mock" and cfg.sys2.base_url
                else ""
            ),
            self._sys1_banner(),
            f"工作区：{self.ws.root}"
            + (f"  ·  配置：{self.config_path}" if self.config_path else "  ·  配置：默认值"),
            f"记忆文件：cortex/episodes {episodes} 个 · cortex/turns {turns} 个"
            f"（每轮结束会显示变化；/tree 看目录）",
        ]
        if cfg.sys2.provider != "mock":
            probe = probe_sys2(cfg, timeout=2.0)
            if probe["ok"]:
                models = probe["models"]
                hint = ""
                if models and cfg.sys2.model and cfg.sys2.model not in models:
                    hint = f"  ⚠ 配置的模型 {cfg.sys2.model} 不在列表里"
                elif models and not cfg.sys2.model:
                    hint = f"  ⚠ 未配置 model，可填 {models[0]}"
                lines.append(
                    f"端点探测：在线 {probe['latency_ms']}ms · 模型 {models or '(未列出)'}{hint}"
                )
            else:
                lines.append(
                    f"端点探测：✗ {probe['detail']}（先启动 llama.cpp，或 /status 重试）"
                )
        return "\n".join(lines) + "\n"

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
        elif cmd == "/files":
            limit = int(arg) if arg.isdigit() else 12
            print(
                viewer.list_memories(
                    self.hb.store, self.speaker, self.ws.episodes, limit=limit
                )
            )
        elif cmd == "/cat":
            path = viewer.resolve_memory_path(arg, self.ws, self.hb.store)
            if path is None:
                print("用法：/cat <记忆id | episodes/xxx.json | 轮次号>")
                return
            node_id = path.stem
            if path.parent.name == "episodes":
                node = self.hb.store.get(node_id)
                if node is not None and not self.hb.store.is_visible(node, self.speaker):
                    print("（这条记忆在当前说话人面前不存在）")
                    return
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                print(f"# {path}")
                print(json.dumps(raw, ensure_ascii=False, indent=2))
            except (OSError, json.JSONDecodeError) as exc:
                print(f"读取失败：{exc}")
        elif cmd == "/tree":
            print(viewer.tree(self.ws.cortex))
        elif cmd == "/status":
            cfg = self.hb.config
            probe = probe_sys2(cfg, timeout=3.0) if cfg.sys2.provider != "mock" else None
            print(f"System Two：{getattr(self.hb.sys2, 'name', '?')}")
            if probe is not None:
                print(f"  端点：{cfg.sys2.base_url}")
                print(
                    f"  探测：{'在线' if probe['ok'] else '离线'} "
                    f"{probe['latency_ms']}ms · {probe['detail']}"
                )
                print(f"  模型：{probe['models'] or '(未列出)'}")
                notes = getattr(self.hb.sys2, "notes", [])
                for note in notes[-3:]:
                    print(f"  适配：{note}")
            print(f"System One：{getattr(self.hb.reflex.client, 'name', '?')}")
            print(f"  配置：provider={cfg.sys1.provider} · model={cfg.sys1.model or '(空)'}")
            last_frame_tick = self.hb.last_tick
            if last_frame_tick is not None:
                print(f"  上一拍丘脑耗时：{last_frame_tick.sys1_ms:.0f}ms")
            client = self.hb.reflex.client
            if getattr(client, "circuit_open", False):
                print(f"  ⚠ 断路器打开：{client.last_error}")
            print(
                f"预算：帧 {cfg.budget.frame_tokens} · 连环 {cfg.budget.max_chain_ticks} 拍 · "
                f"契约 {cfg.sys2.contract} · 结构化输出 {cfg.sys2.structured_output}"
            )
            print(
                f"降级开关：预填 {getattr(self.hb.sys2, 'prefill', 'n/a')}"
                f"（自动 {cfg.sys2.auto_prefill}）· 记忆回填 {cfg.sys2.memory_pass}"
            )
            frame = self.hb.last_frame
            if frame:
                from kernel.textutil import estimate_tokens

                print(
                    f"系统提示：{estimate_tokens(frame.system)} tokens"
                    f"（身份 {estimate_tokens(frame.identity)}）"
                    f" · 帧末尾格式提醒 {cfg.sys2.contract_in_user}"
                )
            if self.mode_stats:
                total = sum(self.mode_stats.values())
                parts = "／".join(f"{k} {v}" for k, v in sorted(self.mode_stats.items()))
                print(f"本会话输出形态：{parts}（共 {total} 拍）")
                if self.mode_stats.get("json", 0) == 0 and total:
                    print(
                        "  ⚠ 还没有一次按 JSON 契约输出：先 /selftest 看建议，"
                        "或换更守 JSON 的 instruct 模型"
                    )
            print(json.dumps(self.hb.store.stats(self.speaker), ensure_ascii=False))
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
        elif cmd == "/prefill":
            client = self.hb.sys2
            if not hasattr(client, "prefill"):
                print("（当前 System Two 不是 HTTP 端点，没有预填）")
                return
            mode = (arg or "").strip().lower()
            if mode in {"", "on", "say"}:
                client.prefill = "say"
                print("预填已开：让模型接着 `{\"say\": \"` 往下写（本次会话有效）")
                print('  想持久化：在 xingli.toml 的 [sys2] 加 prefill = "say"')
            elif mode in {"off", "none"}:
                client.prefill = "off"
                print("预填已关")
            else:
                print("用法：/prefill on|off")
        elif cmd == "/sys1":
            await self._sys1_command(arg)
        elif cmd == "/tidy":
            from memory.hygiene import tidy

            dry = arg.strip().lower() in {"dry", "check", "preview"}
            baseline = viewer.snapshot(self.ws.cortex) if self.show_memory else None
            report = await self._call(
                self._serialized, tidy, self.hb.store, self.speaker, archive=not dry
            )
            print(("（预演，不改动）" if dry else "") + report.summary())
            for line in report.lines():
                print(f"  {line}")
            if not dry and report.changed:
                change = self._memory_change(baseline)
                if change:
                    print()
                    for line in change.splitlines():
                        print(f"  │ {line}")
        elif cmd == "/forget":
            node = self.hb.store.get(arg) if arg else None
            if node is None or not self.hb.store.is_visible(node, self.speaker):
                print("（这条记忆在当前说话人面前不存在）")
                return
            if self.hb.store.archive(node.id):
                print(f"已作废 #{node.id}「{node.lod0[:40]}」（文件与 Git 账本都还在，可 /rollback）")
            else:
                print(f"#{node.id} 早已作废")
        elif cmd == "/thought":
            self.show_thought = not self.show_thought
            print(f"内心 OS 打印：{'开' if self.show_thought else '关'}")
        elif cmd == "/frame":
            frame = self.hb.last_frame
            if not frame:
                print("（还没有产生认知帧）")
                return
            from kernel.textutil import estimate_tokens

            print(f"Token: {frame.total_tokens} / 预算 {self.hb.config.budget.frame_tokens}")
            print(
                f"  系统（身份 {estimate_tokens(frame.identity)} + 契约 "
                f"{estimate_tokens(frame.system) - estimate_tokens(frame.identity)}）"
            )
            for name, tokens in frame.section_tokens.items():
                print(f"  {name:<10} {tokens:>5}")
            print("\n【身份块】\n" + (frame.identity or "（无）"))
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
            print(f"配额剩余 {self.hb.scheduler.quota_left()} ｜ 队列 {len(tickets)} 条")
            for t in tickets:
                print(f"  {t.due_at}  {t.reason}  → {t.target_speaker}")
        elif cmd == "/sleep":
            baseline = viewer.snapshot(self.ws.cortex) if self.show_memory else None
            print("（开始睡眠整合……）")
            report = await self._call(self._serialized, self.hb.sleep)
            print(report.summary())
            for sample in report.samples:
                mark = "✓" if sample.passed else "✗"
                print(
                    f"  {mark} #{sample.node_id} 覆盖率 "
                    f"{sample.coverage_before:.2f}→{sample.coverage_after:.2f}"
                    + ("（已重写索引）" if sample.repaired else "")
                )
            change = self._memory_change(baseline)
            if change:
                print()
                print(change)
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
        elif cmd == "/commit":
            commit = self.hb.ledger.commit(arg or "manual snapshot")
            print(f"已提交 {commit[:10]}" if commit else "（没有变化）")
        elif cmd == "/recall":
            node = self.hb.store.get(arg) if arg else None
            if node is None or not self.hb.store.is_visible(node, self.speaker):
                print("（这条记忆在当前说话人面前不存在）")
                return
            print(f"#{node.id}  来源 {node.source_id}（{node.epistemic}）")
            print(node.lod2 or node.lod1)
        elif cmd == "/stats":
            print(json.dumps(self.hb.store.stats(), ensure_ascii=False, indent=2))
            print(f"配额剩余 {self.hb.scheduler.quota_left()}")
        elif cmd == "/init":
            print(f"工作区已初始化：{self.ws.root}")
        else:
            print(f"未知指令：{cmd}（/help 查看列表）")


# ---------------------------------------------------------------------------
# 配置与启动
# ---------------------------------------------------------------------------
CONFIG_TEMPLATE = """\
# 星璃配置（cli.py --init-config 生成）
# 本地 llama.cpp：先运行
#   llama-server -m model.gguf -c 8192 --port 8080
# 再按实际模型名改 [sys2] model（可先跑 python cli.py --ping 看列表）。

[sys2]
provider = "openai_compatible"
base_url = "{base_url}"
model = "{model}"
api_key = ""                  # 本地服务通常不校验；云端点请改用 api_key_env
json_mode = true              # 端点不支持 response_format 时会自动降级
repair = true                 # 输出不是 JSON 时追加一次“只要 JSON”的请求
retries = 1
timeout_sec = 300             # CPU 推理一帧可能要几十秒
temperature = 0.7
max_tokens = {max_tokens}
contract = "{contract}"      # full（默认）| compact（小模型 / 上下文紧张）
structured_output = "auto"    # auto: json_schema → json_object → 关；llama.cpp 会据此做语法约束
prefill = "{prefill}"            # off | say：预填 JSON 开头，逼思考型模型直接说人话
# contract_in_user = true     # 某些模型的 chat template 会吞掉 system：在帧末尾再贴一次短提醒

[sys1]
provider = "{sys1_provider}"  # local（最快）| llm_fallback | jev | openai_decisions
base_url = "{base_url}"
model = "{sys1_model}"
timeout_sec = 15
# 连续失败两次后自动退回本地词法，避免每一拍都白等超时
# （断路器逻辑在 kernel/sys1_jev.py::ChainClient）
# 没配好也没关系：聊天里 /sys1 on Kev-4B-GGUF 立即接管并写回这里；/sys1 看活体探测。

[budget]
frame_tokens = 6000
max_chain_ticks = 4
max_chain_seconds = 60         # 一轮里连环思考最多花多久（本地 4B 一拍可能 15~40s）
min_wake_sec = 30              # 自主唤醒最短预约间隔，防止自激循环
max_proactive_per_hour = 6

# 若你的模型把预算全花在“思考”上、content 为空：
# [sys2.extra_body]
# chat_template_kwargs = {{ enable_thinking = false }}
"""


def write_config(
    root: Path,
    base_url: str,
    model: str,
    sys1_provider: str = "local",
    sys1_model: str = "",
    contract: str = "full",
    max_tokens: int = 1000,
    prefill: str = "off",
    path: Path | None = None,
) -> Path:
    target = path or (root / "xingli.toml")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        backup = target.with_suffix(".toml.bak")
        backup.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
    target.write_text(
        CONFIG_TEMPLATE.format(
            base_url=base_url.rstrip("/"),
            model=model,
            sys1_provider=sys1_provider,
            sys1_model=sys1_model or model,
            contract=contract,
            max_tokens=max_tokens,
            prefill=prefill,
        ),
        encoding="utf-8",
    )
    return target


#: 一次性提示（在启动时打印，告诉你 sys1 为什么没生效）
SYS1_NOTICES: list[str] = []


def _maybe_promote_sys1(cfg: Config) -> Config:
    """`[sys1] model` 填了、provider 却还是 local —— 这是最常见的“我明明配了”的场景。

    这种配置没有歧义：写了模型名就是想让它当丘脑。直接升格成 llm_fallback 并明说，
    接不上时 ChainClient 会自己退回本地词法，不会把聊天卡死。
    """
    model = (cfg.sys1.model or "").strip()
    if cfg.sys1.provider != "local" or not model:
        return cfg
    base = (cfg.sys1.base_url or cfg.sys2.base_url or "").strip()
    if not base:
        SYS1_NOTICES.append(
            f"⚠ 你填了 [sys1] model = {model!r}，但没有 base_url，丘脑只能用本地词法"
        )
        return cfg
    SYS1_NOTICES.append(
        f"System One 自动接管：provider=local 但 model={model!r} 已填 → 按 llm_fallback 使用"
        f"（接不上会自动退回本地词法；想关掉就把 model 清空）"
    )
    return cfg.with_sys1(provider="llm_fallback", base_url=base)


def _prepare(args) -> tuple[Workspace, Config, str]:
    root = Path(args.root).resolve() if args.root else Path.cwd()
    cfg = Config.load(path=args.config, root=root)
    if args.mock:
        cfg = cfg.with_sys2(provider="mock")
    else:
        cfg = _maybe_promote_sys1(cfg)
    ws = Workspace.open(root, cfg)
    used = args.config or os.environ.get("XINGLI_CONFIG")
    if not used:
        for name in ("xingli.toml", "config.toml"):
            if (root / name).is_file():
                used = str(root / name)
                break
    return ws, cfg, used or ""


def _print_probe(cfg: Config) -> bool:
    probe = probe_sys2(cfg, timeout=3.0)
    print(f"端点：{probe['base_url']}")
    print(f"耗时：{probe['latency_ms']}ms")
    print(f"状态：{'在线' if probe['ok'] else '离线'} · {probe['detail']}")
    if probe["models"]:
        print(f"模型：{', '.join(probe['models'])}")
        if cfg.sys2.model and cfg.sys2.model not in probe["models"]:
            print(f"⚠ 配置里的 model=「{cfg.sys2.model}」不在列表中，聊天可能报“模型不存在”")
        elif not cfg.sys2.model:
            print(f"提示：把 [sys2] model 设成 {probe['models'][0]}")
    if not probe["ok"]:
        print("检查：llama-server 是否已启动、端口是否一致、base_url 是否带 /v1")
    return bool(probe["ok"])


def _run_once(
    hb: Heartbeat,
    ws: Workspace,
    speaker: str,
    text: str,
    show_memory: bool,
    show_thought: bool = False,
) -> int:
    baseline = viewer.snapshot(ws.cortex) if show_memory else None
    started = time.monotonic()
    try:
        result = hb.handle_message(speaker, text)
    except Sys2Error as exc:
        print(f"⟨端点错误⟩ {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - 只跑一句，也不该甩一整页 traceback
        print(f"⟨这一拍出了异常⟩ {type(exc).__name__}: {exc}", file=sys.stderr)
        if os.environ.get("XINGLI_TRACEBACK"):
            traceback.print_exc()
        return 1
    elapsed = time.monotonic() - started
    for tick in result.ticks:
        if show_thought and tick.packet and tick.packet.inner_monologue:
            for line in tick.packet.inner_monologue.splitlines():
                print(f"  # {line}")
        for warn in tick.actions.errors:
            print(f"⟨内核⟩ {warn}")
    for say in result.says:
        print(f"星璃: {say}")
    if not result.says:
        print("（这一轮她选择沉默；加 --show-thought 看内心 OS）")
    last = result.ticks[-1] if result.ticks else None
    tokens = last.frame_tokens if last else 0
    rate = None
    timings = getattr(hb.sys2, "last_timings", None) or {}
    if isinstance(timings.get("predicted_per_second"), (int, float)):
        rate = float(timings["predicted_per_second"])
    print(
        f"⟨计时⟩ {elapsed:.1f}s · {len(result.ticks)} 拍 · 帧 {tokens} tokens"
        + (f" · {rate:.1f} tok/s" if rate else "")
    )
    for tick in result.ticks:
        if tick.mode and tick.mode != "json":
            print(f"⟨降级⟩ 输出为「{tick.mode}」：能说话，但本拍没有记忆/进化动作")
    if show_memory and baseline is not None:
        change = viewer.format_diff(
            viewer.diff(baseline, viewer.snapshot(ws.cortex)), ws.cortex, hb.store
        )
        if change:
            print()
            print(change)
    return 0


def _selftest(root: Path, cfg: Config) -> int:
    """真跑一轮：验证端点、认知包解析、记忆落盘三件事。"""
    target = root / cfg.runtime_dir / "selftest"
    shutil.rmtree(target, ignore_errors=True)
    ws = Workspace.open(target, cfg)
    print("== 星璃自检 ==")
    print(f"工作区：{target}（每次重建）")
    client = build_sys1_client(cfg)
    print(f"System One：{getattr(client, 'name', '?')}（provider={cfg.sys1.provider}）")
    if cfg.sys2.provider == "mock":
        print("⚠ 当前是 mock 大脑；要测真实端点请先去掉 --mock 并配好 xingli.toml")
    if cfg.sys2.provider != "mock":
        if not _print_probe(cfg):
            return 2
    try:
        hb = Heartbeat.create(ws, cfg)
    except Sys2Error as exc:
        print(f"初始化失败：{exc}")
        return 2
    baseline = viewer.snapshot(ws.cortex)
    print("\n发送：你好，我叫林深。记住：我下周三要去看牙医，别安排别的事。")
    started = time.monotonic()
    try:
        result = hb.handle_message(
            "owner", "你好，我叫林深。记住：我下周三要去看牙医，别安排别的事。"
        )
    except Sys2Error as exc:
        print(f"\n✗ 调用失败：{exc}")
        print("  排查：端点是否在线 / model 名称是否正确 / max_tokens 是否太小 / 上下文长度是否够")
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"\n✗ 这一拍出了异常：{type(exc).__name__}: {exc}")
        if os.environ.get("XINGLI_TRACEBACK"):
            traceback.print_exc()
        return 1
    elapsed = time.monotonic() - started
    modes = [t.mode or "空包" for t in result.ticks]
    print(f"\n✓ 端点返回，用时 {elapsed:.1f}s，{len(result.ticks)} 拍（{' / '.join(modes)}）")
    last = result.ticks[-1]
    timings = getattr(hb.sys2, "last_timings", None) or {}
    rate = timings.get("predicted_per_second")
    rate_text = f" · 生成 {rate:.1f} tok/s" if isinstance(rate, (int, float)) else ""
    print(f"  认知帧：{last.frame_tokens} tokens（预算 {cfg.budget.frame_tokens}）{rate_text}")
    print(
        "  解析："
        + ("JSON 契约 ✓" if last.mode == "json" else f"{last.mode}（降级）")
        + f" · 系统提示含身份块 {len(hb.last_frame.identity)} 字"
        if hb.last_frame
        else ""
    )
    for tick in result.ticks:
        if tick.packet and tick.packet.inner_monologue:
            print(f"  内心 OS：{tick.packet.inner_monologue.splitlines()[0][:120]}")
        seen: set[str] = set()
        for warn in tick.actions.errors:
            if warn not in seen:
                seen.add(warn)
                print(f"  ⚠ {warn}")
        for warn in tick.packet.parse_warnings if tick.packet else []:
            if warn not in seen:
                seen.add(warn)
                print(f"  ⚠ 解析：{warn}")
    for say in result.says:
        print(f"  她说：{say[:160]}")
    if last.mode != "json":
        print(
            "  建议：① 在聊天里发 /prefill on 再试（思考型模型的杀手锏，立即生效）；"
            "② --fast 重跑（compact 契约 + 短输出）；"
            "③ 若模板会吞 system，加 contract_in_user = true；"
            "④ 换更守 JSON 的 instruct 模型"
        )
    if not result.says and not result.ticks[-1].actions.errors:
        print("  （没有 say：模型可能只写了 memory_ops，或返回了空字段）")
    change = viewer.format_diff(
        viewer.diff(baseline, viewer.snapshot(ws.cortex)), ws.cortex, hb.store
    )
    print("\n" + (change or "（这一轮没有写记忆文件；可以试试明确说“记住……”）"))
    print("\n" + viewer.tree(ws.cortex, limit=20))
    episodes = sorted((ws.episodes).glob("*.json")) if ws.episodes.exists() else []
    if episodes:
        print("\n--- 新建的记忆文件（原样）---")
        print(episodes[-1].read_text(encoding="utf-8"))
    notes = getattr(hb.sys2, "notes", [])
    if notes:
        print("\n适配记录：" + "；".join(notes[-3:]))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="星璃 CLI 聊天测试台")
    parser.add_argument("--root", help="工作区目录（默认当前目录）")
    parser.add_argument("--config", help="配置文件路径（默认找 ./xingli.toml、./config.toml）")
    parser.add_argument("--mock", action="store_true", help="强制使用离线 mock 大脑")
    parser.add_argument("--as", dest="speaker", default="owner", help="初始说话人 id")
    parser.add_argument("--say", help="非交互：只说一句话后退出")
    parser.add_argument("--sleep", action="store_true", help="非交互：跑一次睡眠整合")
    parser.add_argument(
        "--tidy",
        action="store_true",
        help="非交互：整理存量记忆（归档占位符/状态回显，合并重复），打印报告后退出",
    )
    parser.add_argument("--show-memory", action="store_true", help="非交互时也打印记忆变化")
    parser.add_argument("--show-thought", action="store_true", help="非交互时打印内心 OS")
    parser.add_argument("--ping", action="store_true", help="只探测 System Two 端点后退出")
    parser.add_argument("--selftest", action="store_true", help="真跑一轮并打印解析与记忆文件")
    parser.add_argument("--init-config", action="store_true", help="生成 xingli.toml 模板")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1", help="llama.cpp 等端点")
    parser.add_argument("--model", default="", help="模型名（llama.cpp 的 --alias 或路径）")
    parser.add_argument("--sys1", dest="sys1_provider", default="local", help="sys1 provider")
    parser.add_argument("--sys1-model", default="", help="sys1 模型名（默认同 sys2）")
    parser.add_argument(
        "--contract",
        default="full",
        choices=["full", "compact"],
        help="行动契约版本：compact 更适合 7B 级别 / 上下文紧张的本地模型",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="本地小模型省时档：compact 契约 + max_tokens 700（--init-config 时生效）",
    )
    parser.add_argument(
        "--prefill",
        action="store_true",
        help="让模型接着 `{\"inner_monologue\": \"\", \"say\": \"` 往下写（思考型模型适用）",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve() if args.root else Path.cwd()

    if args.init_config:
        path = write_config(
            root,
            args.base_url,
            args.model or "local-model",
            sys1_provider=args.sys1_provider,
            sys1_model=args.sys1_model,
            contract="compact" if args.fast else args.contract,
            max_tokens=700 if args.fast else 1000,
            prefill="say" if args.prefill else "off",
            path=Path(args.config) if args.config else None,
        )
        print(f"已写入配置：{path}")
        print("下一步：python cli.py --ping，然后 python cli.py")
        return 0

    ws, cfg, used = _prepare(args)

    if args.config and not Path(args.config).is_file():
        print(f"配置文件不存在：{args.config}", file=sys.stderr)
        return 2

    if args.ping:
        if cfg.sys2.provider == "mock":
            print("当前是 mock 大脑（--mock 或未配置）。--ping 需要一个真实端点。")
            return 1
        return 0 if _print_probe(cfg) else 2

    if args.selftest:
        return _selftest(root, cfg)

    try:
        hb = Heartbeat.create(ws, cfg)
    except Sys2Error as exc:
        print(f"System Two 初始化失败：{exc}", file=sys.stderr)
        print("提示：加 --mock 用离线大脑跑通流程；或运行 python cli.py --init-config。", file=sys.stderr)
        return 2

    if args.tidy:
        from memory.hygiene import tidy

        report = tidy(hb.store, args.speaker)
        print(report.summary())
        for line in report.lines():
            print(f"  {line}")
        return 0 if not report.errors else 1
    if args.sleep:
        print(hb.sleep().summary())
        return 0
    if args.say:
        return _run_once(
            hb,
            ws,
            args.speaker,
            args.say,
            show_memory=args.show_memory,
            show_thought=args.show_thought,
        )

    cli = ChatCLI(hb, ws, speaker=args.speaker, show_memory=True, config_path=used)
    try:
        asyncio.run(cli.run())
    except KeyboardInterrupt:
        print("\n（中断）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
