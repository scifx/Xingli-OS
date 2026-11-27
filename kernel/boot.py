"""主心跳引擎：现场捕获 → Sys1 扫描 → 认知帧编译 → Sys2 思考 → 事务提交。

一次外部事件（有人说话 / 闹钟响了）会引发 1~``MAX_CHAIN_TICKS`` 拍思考：
每一拍都可能查证记忆、写记忆、改关系、调焦、预约下一次唤醒，或者保持沉默。

这也是把 README 的核心公式落到代码的地方::

    (y_t, W_t, M_{t+1}) = M_t(x_t, E_t; Sys1_φ, Sys2_θ)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Callable, Sequence

from kernel.claims import Claim, extract_claims, is_correction
from kernel.config import Config
from kernel.evolution import Evolver
from kernel.git_ledger import GitLedger
from kernel.memtext import (
    is_richer,
    looks_like_placeholder,
    looks_like_state_echo,
    mem_key,
    same_fact,
)
from kernel.models import (
    AppliedActions,
    FocusUpdate,
    Event,
    HeartbeatResult,
    ThoughtPacket,
    TickReport,
    TurnRecord,
    now_utc,
)
from kernel.paths import Workspace
from kernel.runtime_exec import PacketBuilder, apply_code_action, parse_thought_output
from kernel.sandbox import Sandbox
from kernel.scheduler import ProactiveBudgetExceeded, Scheduler
from kernel.sys1_jev import ReflexEngine, build_client
from kernel.sys2_llm import Sys2Client, Sys2Error, build_sys2
from kernel.textutil import extractive_summary, one_line, similarity
from memory.emotion import EmotionStore
from memory.focal_context import FocalContext
from memory.frame_compiler import Frame, FrameCompiler
from memory.lod_renderer import LodRenderer
from memory.relations import RelationsStore
from memory.store import MemoryStore
from sensors.clock import ClockSensor
from sensors.social import SocialSensor
from sensors.world import WorldSensor


#: 值得长期记住的“话头”（弱模型自己不会写 memory_ops，内核据此决定要不要补抽）
MEMORY_CUES = re.compile(
    "记住|记下|别忘|不要忘|喜欢|讨厌|最爱|生日|答应|承诺|以后|我叫|我的名字|"
    "偏好|过敏|习惯|提醒|私密|别告诉|只告诉|约定"
)


@dataclass
class Heartbeat:
    ws: Workspace
    sys2: Sys2Client
    config: Config
    store: MemoryStore
    relations: RelationsStore
    emotion: EmotionStore
    focus: FocalContext
    landscape: LodRenderer
    compiler: FrameCompiler
    reflex: ReflexEngine
    scheduler: Scheduler
    ledger: GitLedger
    sandbox: Sandbox
    evolver: Evolver
    now_fn: Callable[[], datetime] = now_utc
    last_frame: Frame | None = None
    last_packet: ThoughtPacket | None = None
    last_tick: TickReport | None = None
    on_event: Callable[[str], None] | None = None

    # ------------------------------------------------------------------
    @classmethod
    def create(
        cls,
        ws: Workspace | None = None,
        config: Config | None = None,
        sys2: Sys2Client | None = None,
        sys1_client: Any | None = None,
        now_fn: Callable[[], datetime] = now_utc,
    ) -> "Heartbeat":
        ws = ws or Workspace.open()
        cfg = config or ws.config
        store = MemoryStore(ws)
        relations = RelationsStore(ws)
        emotion = EmotionStore(ws)
        focus = FocalContext(ws, store, cfg)
        landscape = LodRenderer(ws, store, cfg, trust_lookup=relations.trust_of)
        clock = ClockSensor(ws, store, now_fn=now_fn)
        social = SocialSensor(ws, relations, emotion)
        world = WorldSensor(ws)
        compiler = FrameCompiler(
            ws, store, cfg, focal=focus, landscape=landscape, clock=clock, social=social, world=world
        )
        reflex = ReflexEngine(ws, cfg, client=sys1_client or build_client(cfg), now_fn=now_fn)
        scheduler = Scheduler(ws, cfg, now_fn=now_fn)
        ledger = GitLedger(ws)
        sandbox = Sandbox(ws)
        evolver = Evolver(ws, store, cfg, ledger=ledger, sandbox=sandbox)
        return cls(
            ws=ws,
            sys2=sys2 or build_sys2(cfg),
            config=cfg,
            store=store,
            relations=relations,
            emotion=emotion,
            focus=focus,
            landscape=landscape,
            compiler=compiler,
            reflex=reflex,
            scheduler=scheduler,
            ledger=ledger,
            sandbox=sandbox,
            evolver=evolver,
            now_fn=now_fn,
        )

    # ------------------------------------------------------------------
    @property
    def max_chain_ticks(self) -> int:
        return max(1, int(self.config.budget.max_chain_ticks))

    def _emit(self, text: str) -> None:
        if self.on_event:
            self.on_event(text)

    # ------------------------------------------------------------------
    # 对外入口
    # ------------------------------------------------------------------
    def handle_message(
        self, speaker_id: str, text: str, should_stop: Callable[[], bool] | None = None
    ) -> HeartbeatResult:
        return self.handle_event(
            Event(kind="message", speaker_id=speaker_id or "owner", text=text or ""),
            should_stop=should_stop,
        )

    def handle_event(
        self, event: Event, should_stop: Callable[[], bool] | None = None
    ) -> HeartbeatResult:
        """处理一次外部事件；``should_stop`` 让调用方（比如 CLI）随时收尾。

        连环思考有三个刹车：重复输出、单轮时间预算、以及调用方的“有人插话了”。
        没有新信息就不再自转——自转是手段，不是目的。
        """
        result = HeartbeatResult(event=event)
        carried: list[dict[str, str]] = []
        says: list[str] = []
        monologues: list[str] = []
        said_fingerprints: list[str] = []
        tick = 1
        halted = ""
        started = time.monotonic()
        budget_cfg = getattr(self.config.budget, "max_chain_seconds", 60)
        budget_sec = None if budget_cfg is None else float(budget_cfg)
        while True:
            report = self._tick(event, tick, carried)
            # 同一轮里重复的话只留一次：自言自语不该变成刷屏
            fresh: list[str] = []
            for say in report.actions.says:
                if any(similarity(say, prev) >= 0.9 for prev in said_fingerprints):
                    continue
                said_fingerprints.append(say)
                fresh.append(say)
            repeated = len(report.actions.says) - len(fresh)
            report.actions.says = fresh
            result.ticks.append(report)
            packet = report.packet
            if packet is None:
                halted = halted or "本轮没有可用的认知包"
                break
            if packet.inner_monologue:
                monologues.append(packet.inner_monologue)
            says.extend(fresh)
            actions = report.actions
            progress = bool(
                fresh
                or actions.recalls
                or actions.memorized
                or actions.relation_updates
                or actions.focus_changed
                or actions.scheduled
                or actions.notes
            )
            next_carried = self._carried_from(actions)
            wants_continue = packet.wake.continue_think is not None
            if repeated and not fresh:
                halted = "模型在重复上一拍的话，内核停止自转"
                break
            if should_stop is not None and should_stop():
                halted = "主人插话了，这一轮先收尾"
                break
            auto_continue = bool(next_carried) and not fresh and not wants_continue
            if not progress or not (wants_continue or auto_continue):
                break
            tick_cap = self.max_chain_ticks
            cap_note = f"MAX_CHAIN_TICKS={tick_cap}"
            if event.kind == "wake":
                # 自主唤醒没人等着听：连锁思考只在她真有事跟进时才有价值
                wake_cap = int(
                    getattr(self.config.budget, "max_wake_chain_ticks", 1) or 1
                )
                if wake_cap < tick_cap:
                    tick_cap = wake_cap
                    cap_note = f"MAX_WAKE_CHAIN_TICKS={wake_cap}（自主唤醒不自转）"
            if tick >= tick_cap:
                halted = f"达到 {cap_note} 硬熔断"
                break
            if budget_sec is not None and (time.monotonic() - started) >= budget_sec:
                halted = f"单轮思考超过 {budget_sec:.0f}s 预算，先收尾（可改 [budget] max_chain_seconds）"
                break
            if auto_continue:
                self._emit(
                    f"（内核补一拍：已召回 {len(next_carried)} 条内容，等她消化后再开口）"
                )
            carried = next_carried
            tick += 1

        # 轮次 = 一次交流。自主唤醒如果只是脑子里过了一遍（没有 say），
        # 就不该在账本里留下“等待下一拍”的空壳——它会污染近况，还会被反复召回。
        if says or (event.kind != "wake" and monologues):
            self._record_turn(event, says, monologues)
        result.halted_reason = halted
        return result

    def poll_wakes(
        self,
        now: datetime | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> list[HeartbeatResult]:
        """触发所有到期的自主唤醒（受每小时配额限制）。"""
        now = now or self.now_fn()
        results: list[HeartbeatResult] = []
        for ticket in self.scheduler.due(now):
            ok, why = self.scheduler.can_wake(now, check_queue=False)
            if not ok:
                self._emit(f"（自主唤醒被熔断：{why}）")
                break
            self.scheduler.mark_fired(ticket, now)
            event = Event(
                kind="wake",
                speaker_id=ticket.target_speaker,
                text="",
                extra={"reason": ticket.reason, "ticket": ticket.id},
            )
            results.append(self.handle_event(event, should_stop=should_stop))
        self.scheduler.prune()
        return results

    # ------------------------------------------------------------------
    # 单拍
    # ------------------------------------------------------------------
    def _tick(
        self, event: Event, tick: int, carried: Sequence[dict[str, str]]
    ) -> TickReport:
        start = time.monotonic()
        now = self.now_fn()

        # 1) 情绪先向基线回弹（防止自激漂移）
        state = self.emotion.load()
        state.tick_decay()

        # 2) Sys1：一次并行扇出 + 三段门控
        state_text = self._sys1_state(event, now)
        scan_started = time.monotonic()
        scan = self.reflex.scan(state_text, event.speaker_id, now)
        sys1_ms = (time.monotonic() - scan_started) * 1000
        if scan.emotion:
            state.apply(scan.emotion)
        self.emotion.save(state)

        # 3) 认知帧编译
        frame = self.compiler.compile(
            event, scan.signals, speaker_id=event.speaker_id, carried=carried, now=now
        )
        self.last_frame = frame

        # 4) Sys2：一帧进，一包出
        report = TickReport(
            tick=tick,
            event=event,
            frame_text=frame.user,
            frame_tokens=frame.total_tokens,
            section_tokens=frame.section_tokens,
            signals=list(scan.signals),
            env=frame.env,
            sys1_ms=sys1_ms,
        )
        try:
            raw = self.sys2.complete(frame.system, frame.user)
        except Sys2Error as exc:
            report.actions.errors.append(f"Sys2 调用失败: {exc}")
            report.duration_ms = (time.monotonic() - start) * 1000
            return report
        packet, mode = parse_thought_output(raw, event.speaker_id)
        packet.parse_warnings.extend(scan.errors)
        report.packet = packet
        report.mode = mode
        self.last_packet = packet
        parse_errors = list(packet.parse_warnings) if mode in {"empty", "prose"} else []

        # 5) 应用（声明式字段 + 可选代码载荷）
        #    单个畸形认知包不该让整轮聊天崩掉：出错就只丢这一拍的动作。
        try:
            report.actions = self._apply(packet, event, scan)
        except Exception as exc:  # noqa: BLE001
            report.actions = AppliedActions()
            report.actions.errors.append(
                f"应用认知包失败（已跳过本拍动作）: {type(exc).__name__}: {exc}"
            )
        report.actions.errors.extend(parse_errors)
        if tick == 1:
            self._memory_pass(event, packet, report.actions)
            self._apply_claims(event, report.actions)
        report.duration_ms = (time.monotonic() - start) * 1000
        self.last_tick = report
        return report

    def _memory_pass(self, event: Event, packet: ThoughtPacket, actions: AppliedActions) -> None:
        """弱模型补洞：模型没写记忆动作时，用一次极短请求抽取值得记住的事。

        这是显式的降级路径：抽取结果同样走 `_op_memorize`，隐私/信源铁律一条不少。
        """
        if not getattr(self.config.sys2, "memory_pass", True):
            return
        if event.kind != "message" or packet.memory_ops:
            return
        if not MEMORY_CUES.search(event.text or ""):
            return
        extractor = getattr(self.sys2, "extract_memory_ops", None)
        if extractor is None:
            return
        try:
            ops = extractor(event.text, "\n".join(actions.says))
        except Exception as exc:  # noqa: BLE001 - 回填失败绝不毁掉这一拍
            actions.errors.append(f"记忆回填失败：{type(exc).__name__}: {exc}")
            return
        if not ops:
            return
        shim = SimpleNamespace(importance=lambda default=0.6: default)
        before = len(actions.memorized)
        for op in ops[:3]:
            self._op_memorize(op, event, shim, actions)
        added = len(actions.memorized) - before
        if added:
            actions.errors.append(
                f"记忆回填：模型没写 memory_ops，内核补抽了 {added} 条（弱模型降级路径）"
            )

    def _sys1_state(self, event: Event, now: datetime) -> str:
        clock = self.compiler.clock.capture(event.speaker_id, now=now)
        social = self.compiler.social.capture(event.speaker_id)
        if event.kind == "message":
            utterance = event.text
        elif event.kind == "wake":
            utterance = f"（自主唤醒）{event.extra.get('reason', '')}"
        else:
            utterance = event.text
        return "\n".join(
            [
                f"说话人: {social['display_name']}({event.speaker_id}) 关系:{social['role']}",
                f"时间: {clock['local_time']} {clock['period']} 距上次:{clock['since_last_human']}",
                f"情绪余温: {social.get('emotion_residual', '')}",
                f"内容: {utterance}",
            ]
        )

    # ------------------------------------------------------------------
    # 应用认知包
    # ------------------------------------------------------------------
    def _apply(self, packet: ThoughtPacket, event: Event, scan) -> AppliedActions:
        actions = AppliedActions()
        builder = PacketBuilder(event.speaker_id, store=self.store, relations=self.relations)
        self._normalize_ops(packet, actions)

        if packet.code_action:
            outcome = apply_code_action(packet.code_action, builder)
            if not outcome.get("ok"):
                actions.errors.append(f"code_action 未执行: {outcome.get('detail')}")
            extra = builder.packet
            if extra.memory_ops:
                packet.memory_ops = list(extra.memory_ops) + list(packet.memory_ops)
            if extra.relation_ops:
                packet.relation_ops = list(extra.relation_ops) + list(packet.relation_ops)
            if extra.evolve_ops:
                packet.evolve_ops = list(extra.evolve_ops) + list(packet.evolve_ops)
            if extra.focus_update and not packet.focus_update:
                packet.focus_update = extra.focus_update
            for k, v in extra.emotion_delta.items():
                packet.emotion_delta.setdefault(k, v)
            if packet.say is None and extra.say is not None:
                packet.say = extra.say
            if extra.wake.continue_think and not packet.wake.continue_think:
                packet.wake.continue_think = extra.wake.continue_think
            if extra.wake.schedule_wake and not packet.wake.schedule_wake:
                packet.wake.schedule_wake = extra.wake.schedule_wake
            if extra.inner_monologue:
                packet.inner_monologue = (
                    packet.inner_monologue + "\n" + extra.inner_monologue
                ).strip()
            actions.code_output = "; ".join(builder.notes)

        # ---- 注意力调焦 ------------------------------------------------
        if packet.focus_update:
            visible = [t.turn_id for t in self.store.list_turns(event.speaker_id)]
            _, warnings = self.focus.apply_update(packet.focus_update, visible)
            actions.focus_changed = True
            actions.errors.extend(warnings)

        # ---- 情绪 ------------------------------------------------------
        if packet.emotion_delta:
            state = self.emotion.load()
            cap = min(state.max_delta_per_tick, 0.2)
            applied = state.apply(packet.emotion_delta, max_delta=cap)
            self.emotion.save(state)
            actions.emotion = applied

        # ---- 记忆 ------------------------------------------------------
        for op in packet.memory_ops:
            kind = str(op.get("op", "")).lower()
            try:
                if kind in {"archive", "forget", "retract", "supersede"}:
                    self._op_archive(op, event, actions)
                elif kind == "recall":
                    self._op_recall(op, event, actions)
                elif kind == "recall_turn":
                    self._op_recall_turn(op, event, actions)
                elif kind == "search_memory":
                    self._op_search(op, event, actions)
                elif kind in {"memorize", "remember"}:
                    self._op_memorize(op, event, scan, actions)
                else:
                    actions.errors.append(f"未知 memory_op: {kind}")
            except Exception as exc:  # noqa: BLE001
                actions.errors.append(f"memory_op {kind} 失败: {type(exc).__name__}: {exc}")

        # ---- 社交 ------------------------------------------------------
        for op in packet.relation_ops:
            self._op_relation(op, event, actions)

        # ---- 进化 ------------------------------------------------------
        for op in packet.evolve_ops:
            report = self.evolver.handle_op(
                op, source_id=event.speaker_id, inner=packet.inner_monologue
            )
            actions.evolutions.append(report.to_dict())
            if not report.ok:
                actions.errors.append(f"进化被拒绝: {report.message}")

        # ---- 表达 ------------------------------------------------------
        text = packet.say_text()
        if text:
            target = packet.say_target()
            if target and target != event.speaker_id:
                actions.says.append(f"（致 {target}）{text}")
            else:
                actions.says.append(text)

        # ---- 自主唤醒 --------------------------------------------------
        spec = packet.wake.schedule_wake
        if spec:
            try:
                after_sec = spec.get("after_sec", 60)
                try:
                    after_sec = float(after_sec)
                except (TypeError, ValueError):
                    after_sec = 60.0
                floor = float(getattr(self.config.budget, "min_wake_sec", 30) or 0)
                if after_sec < floor:
                    actions.notes.append(
                        f"预约唤醒间隔 {after_sec:.0f}s 太短，按最短 {floor:.0f}s 排"
                    )
                    after_sec = floor
                ticket = self.scheduler.schedule(
                    after_sec,
                    reason=str(spec.get("reason", "")),
                    target_speaker=spec.get("target_speaker"),
                    now=self.now_fn(),
                )
                actions.scheduled.append(ticket.to_dict())
            except ProactiveBudgetExceeded as exc:
                actions.errors.append(str(exc))
            except Exception as exc:  # noqa: BLE001
                actions.errors.append(f"预约唤醒失败: {exc}")

        return actions

    # ------------------------------------------------------------------
    # 认知包里的“放错抽屉”的字段
    # ------------------------------------------------------------------
    def _normalize_ops(self, packet: ThoughtPacket, actions: AppliedActions) -> None:
        """把模型塞错位置的动作挪回它该在的抽屉。

        实测里最常见的三种：``focus_update`` 写进 ``memory_ops``、
        ``emotion_delta`` / ``update_affinity`` 写进 ``evolve_ops``。
        它们不是“不认识的进化操作”，只是放错了地方——静默归位，并留一条痕迹。
        """
        moved: list[str] = []
        for op in list(packet.memory_ops):
            kind = str(op.get("op") or op.get("kind") or "").lower()
            if kind in {"focus", "focus_update", "attention"}:
                packet.memory_ops.remove(op)
                packet.focus_update = packet.focus_update or FocusUpdate.from_dict(
                    {k: v for k, v in op.items() if k not in {"op", "kind"}}
                )
                moved.append("focus_update")
            elif kind in {"emotion", "emotion_delta", "feel"}:
                packet.memory_ops.remove(op)
                for key, value in op.items():
                    if key not in {"op", "kind"}:
                        packet.emotion_delta.setdefault(key, value)
                moved.append("emotion_delta")
            elif kind in {"relation", "update_affinity", "relation_update"}:
                packet.memory_ops.remove(op)
                packet.relation_ops.append(
                    {k: v for k, v in op.items() if k not in {"op", "kind"}}
                )
                moved.append("relation_op")

        kept: list[dict[str, Any]] = []
        for op in packet.evolve_ops:
            kind = str(op.get("op") or op.get("kind") or "").lower()
            body = {k: v for k, v in op.items() if k not in {"op", "kind"}}
            if kind in {"focus", "focus_update", "attention"}:
                packet.focus_update = packet.focus_update or FocusUpdate.from_dict(body)
                moved.append("focus_update")
            elif kind in {"emotion", "emotion_delta", "feel"}:
                for key, value in body.items():
                    packet.emotion_delta.setdefault(key, value)
                moved.append("emotion_delta")
            elif kind in {"memorize", "remember", "memory", "memory_op"}:
                packet.memory_ops.append({"op": "memorize", **body})
                moved.append("memorize")
            elif kind in {"relation", "update_affinity", "affinity", "relation_update"}:
                packet.relation_ops.append(body)
                moved.append("relation_op")
            else:
                kept.append(op)
        if len(kept) != len(packet.evolve_ops):
            packet.evolve_ops = kept
        if moved:
            actions.notes.append(
                "字段归位：" + "、".join(sorted(set(moved))) + "（模型放错了抽屉，内核已接住）"
            )

    # ------------------------------------------------------------------
    def _op_recall(self, op: dict[str, Any], event: Event, actions: AppliedActions) -> None:
        mem_id = str(op.get("mem_id") or op.get("id") or "")
        node = self.store.get(mem_id) if mem_id else None
        if node is None or not MemoryStore.is_visible(node, event.speaker_id):
            # 越权/不存在：不区分，避免用“存在性”当侧信道
            actions.errors.append(f"recall：记忆 {mem_id or '(空)'} 不存在")
            return
        self.store.touch(node.id)
        actions.recalls.append(
            {
                "kind": "memory",
                "id": node.id,
                "lod0": node.lod0,
                "lod1": node.lod1,
                "lod2": node.lod2,
                "source_id": node.source_id,
                "epistemic": node.epistemic,
            }
        )

    def _op_recall_turn(self, op: dict[str, Any], event: Event, actions: AppliedActions) -> None:
        try:
            tid = int(op.get("turn_id"))
        except (TypeError, ValueError):
            actions.errors.append("recall_turn：turn_id 非法")
            return
        turn = self.store.get_turn(tid)
        if turn is None or not (
            "*" in turn.visibility or event.speaker_id in turn.visibility
        ):
            actions.errors.append(f"recall_turn：轮次 {tid} 不存在")
            return
        actions.recalls.append(
            {
                "kind": "turn",
                "turn_id": tid,
                "lod0": turn.lod0,
                "lod1": turn.lod1,
                "lod2": turn.lod2,
            }
        )

    def _op_search(self, op: dict[str, Any], event: Event, actions: AppliedActions) -> None:
        query = str(op.get("query") or "")
        try:
            limit = int(op.get("limit", 5))
        except (TypeError, ValueError):
            limit = 5
        hits = self.store.search(event.speaker_id, query, limit=max(1, min(limit, 20)))
        if not hits:
            actions.recalls.append({"kind": "search", "query": query, "results": []})
            return
        for node, score in hits:
            self.store.touch(node.id)
        actions.recalls.append(
            {
                "kind": "search",
                "query": query,
                "results": [
                    {"id": n.id, "lod1": n.lod1, "score": round(s, 3)} for n, s in hits
                ],
            }
        )

    def _op_archive(self, op: dict[str, Any], event: Event, actions: AppliedActions) -> None:
        """作废一条记忆（``forget``）：只标记归档，文件与账本都留着。"""
        mem_id = str(op.get("mem_id") or op.get("id") or "")
        node = self.store.get(mem_id) if mem_id else None
        if node is None or not MemoryStore.is_visible(node, event.speaker_id):
            actions.errors.append(f"forget：记忆 {mem_id or '(空)'} 不存在")
            return
        if node.source_id != event.speaker_id and event.speaker_id != "owner":
            actions.errors.append("forget：只有记忆的来源者本人或主人可以作废它")
            return
        if self.store.archive(node.id):
            actions.notes.append(f"作废记忆 #{node.id}「{one_line(node.lod0, 28)}」")
        elif node.archived:
            actions.notes.append(f"记忆 #{node.id} 早已作废")

    @staticmethod
    def _mem_key(text: str) -> str:
        """历史调用点；判据统一在 ``kernel.memtext.mem_key``。"""
        return mem_key(text)

    def _near_duplicate(self, event: Event, lod0: str, about: Sequence[str]):
        """找一条“几乎一样”的旧记忆：同一件事不该每次都长出一个新文件。"""
        try:
            recent = self.store.list_nodes(event.speaker_id, limit=150)
        except Exception:  # noqa: BLE001 - 去重是增强，不是必需
            return None
        for node in recent:
            if node.lod0 == lod0:
                return node
            if about and not (set(node.about_ids) & set(about)):
                continue
            if same_fact(node.lod0, lod0):
                return node
        return None

    def _op_memorize(
        self, op: dict[str, Any], event: Event, scan, actions: AppliedActions
    ) -> None:
        lod0 = str(op.get("lod0") or op.get("text") or "").strip()
        if not lod0:
            actions.errors.append("memorize：lod0 不能为空")
            return
        if looks_like_placeholder(lod0):
            actions.notes.append(
                f"记忆被拒：lod0 是占位符或数字（{lod0[:12]!r}），不是事实"
            )
            return
        if looks_like_state_echo(lod0):
            actions.notes.append(f"记忆被拒：这是社交状态回显，不是事件（{lod0[:16]}…）")
            return
        for field_name in ("lod1", "lod2"):
            raw_field = str(op.get(field_name) or "").strip()
            if raw_field and looks_like_placeholder(raw_field):
                op[field_name] = lod0
                actions.notes.append(
                    f"记忆清洗：{field_name} 是数字/占位符（{raw_field[:8]!r}），已用 lod0 替代"
                )
        about = [str(x) for x in (op.get("about_ids") or op.get("about") or [])] or [
            event.speaker_id
        ]
        visibility = [str(x) for x in (op.get("visibility") or [])]
        epistemic = op.get("epistemic")
        if epistemic not in {"direct", "hearsay", "inferred"}:
            epistemic = None
        if epistemic is None:
            # 铁律 4：关于第三方的陈述永远是转述，不可能“亲历”
            third_party = [a for a in about if a != event.speaker_id]
            epistemic = "hearsay" if third_party else "direct"
        elif epistemic == "direct":
            third_party = [a for a in about if a != event.speaker_id]
            if third_party:
                epistemic = "hearsay"
                actions.errors.append(
                    "memorize：关于第三方的陈述被强制标记为 hearsay（铁律 4）"
                )
        visibility = visibility or list(dict.fromkeys([event.speaker_id, *about]))

        importance = op.get("importance")
        try:
            importance = float(importance) if importance is not None else scan.importance()
        except (TypeError, ValueError):
            importance = 0.6

        duplicate = self._near_duplicate(event, lod0, about)
        if duplicate is not None:
            if is_richer(lod0, duplicate.lod0):
                # 同一件事写得更全：新版本留下，旧版本归档（文件与账本都还在）
                self.store.archive(duplicate.id)
                actions.notes.append(
                    f"记忆升级：「{one_line(duplicate.lod0, 20)}」→"
                    f"「{one_line(lod0, 24)}」（旧版已归档）"
                )
            else:
                self.store.touch(duplicate.id)
                actions.notes.append(
                    f"记忆去重：已经记过「{one_line(lod0, 24)}」（#{duplicate.id}），没有重复新建"
                )
                return

        node = self.store.put(
            lod0=lod0,
            lod1=str(op.get("lod1") or lod0),
            lod2=str(op.get("lod2") or op.get("lod1") or lod0),
            kind=str(op.get("kind") or "episode"),
            source_id=event.speaker_id,
            about_ids=about,
            epistemic=epistemic,
            visibility=visibility,
            importance=importance,
            pinned=bool(op.get("pinned", False)),
            tags=[str(t) for t in (op.get("tags") or [])],
            refs=[str(r) for r in (op.get("refs") or [])],
        )
        actions.memorized.append(node.id)

    def _op_relation(self, op: dict[str, Any], event: Event, actions: AppliedActions) -> None:
        person = str(op.get("person_id") or "")
        if not person:
            actions.errors.append("relation_op 缺少 person_id")
            return
        changes = {k: v for k, v in op.items() if k != "person_id"}
        if event.speaker_id != "owner" and person != event.speaker_id:
            actions.errors.append(
                f"relation_op 被拒绝：{event.speaker_id} 无权修改 {person} 的关系档案"
            )
            return
        if event.speaker_id != "owner" and changes.get("role") == "owner":
            actions.errors.append("relation_op 被拒绝：不能把自己升格为 owner")
            return
        self.relations.update(person, changes)
        actions.relation_updates.append(person)

    # ------------------------------------------------------------------
    # 显式事实校正：她必须能改口，也必须分得清“谁叫什么”
    # ------------------------------------------------------------------
    def _apply_claims(self, event: Event, actions: AppliedActions) -> None:
        """从主人原话里读事实（称呼 / 偏好 / 生日），并让它**覆盖**旧记忆。

        模型把“我的名字是 lucky”读成“主人叫星璃”是常态，所以她需要一条不依赖模型的
        校正路径：改关系档案里的称呼、作废互相矛盾的旧称呼记忆、写一条钉选事实。
        """
        if event.kind != "message" or not (event.text or "").strip():
            return
        claims = extract_claims(event.text)
        if not claims:
            return
        persona_name = str(self.compiler.persona().get("NAME") or "星璃")
        correction = is_correction(event.text)
        for claim in claims:
            if claim.kind == "name":
                self._claim_name(event, claim, correction, actions)
            elif claim.kind == "self_name":
                if claim.value and claim.value != persona_name:
                    actions.notes.append(
                        f"主人把她叫成「{claim.value}」，但她记得自己叫「{persona_name}」"
                        "（改名属于人格进化，不擅自采纳）"
                    )
            elif claim.kind in {"like", "dislike", "allergy", "birthday"}:
                self._claim_fact(event, claim, actions)

    def _claim_name(
        self, event: Event, claim: Claim, correction: bool, actions: AppliedActions
    ) -> None:
        speaker = event.speaker_id
        value = claim.value
        profile = self.relations.get(speaker)
        old = str(profile.get("display_name") or speaker)
        if old == value:
            return
        if correction:
            actions.notes.append("（听出这是纠错语气，旧称呼一并作废）")
        self.relations.update(speaker, {"display_name": value})
        if speaker not in actions.relation_updates:
            actions.relation_updates.append(speaker)
        actions.notes.append(f"称呼更正：{old} → {value}（这是对方的名字）")
        removed = self._retract_name_memories(speaker, value, actions)
        if removed:
            actions.notes.append(f"作废 {removed} 条旧称呼记忆，防止她再叫错")
        node = self.store.put(
            lod0=f"称呼：{value}（{speaker} 亲口说的自己的名字）",
            lod1=f"{speaker} 的名字是 {value}；她的名字是 "
            f"{self.compiler.persona().get('NAME') or '星璃'}，两者不能混。",
            lod2=claim.raw or f"{speaker} 说：我叫 {value}",
            kind="fact",
            source_id=speaker,
            about_ids=[speaker],
            epistemic="direct",
            visibility=[speaker],
            importance=0.9,
            pinned=True,
            tags=["称呼"],
        )
        actions.memorized.append(node.id)

    def _claim_fact(self, event: Event, claim: Claim, actions: AppliedActions) -> None:
        speaker = event.speaker_id
        templates = {
            "like": ("主人喜欢{value}", 0.7, ["偏好"]),
            "dislike": ("主人不喜欢{value}", 0.7, ["偏好"]),
            "allergy": ("主人对{value}过敏", 0.9, ["禁忌"]),
            "birthday": ("主人生日是{value}", 0.85, ["生日"]),
        }
        template, importance, tags = templates[claim.kind]
        lod0 = template.format(value=claim.value)
        if claim.kind in {"like", "dislike"}:
            # “我改口味了”：同一件东西的相反偏好作废，否则她会同时相信两种
            removed = self._retract_opposite(claim.kind, claim.value, speaker)
            if removed:
                actions.notes.append(
                    f"作废 {removed} 条相反的旧偏好（现在按「{lod0}」记）"
                )
        duplicate = self._near_duplicate(event, lod0, [speaker])
        if duplicate is not None:
            self.store.touch(duplicate.id)
            return
        node = self.store.put(
            lod0=lod0,
            lod1=f"{lod0}（{speaker} 在 {event.text[:40]} 里说过）",
            lod2=event.text or lod0,
            kind="fact",
            source_id=speaker,
            about_ids=[speaker],
            epistemic="direct",
            visibility=[speaker],
            importance=importance,
            tags=tags,
        )
        actions.memorized.append(node.id)
        actions.notes.append(f"事实入账：{lod0}")

    _LIKE_WORDS = ("喜欢", "偏爱", "偏好", "钟爱", "爱吃", "爱")
    _DISLIKE_WORDS = ("不喜欢", "讨厌", "受不了", "不爱", "不吃")

    def _retract_opposite(self, kind: str, value: str, speaker: str) -> int:
        item = self._mem_key(value)
        if len(item) < 2:
            return 0
        opposite = self._DISLIKE_WORDS if kind == "like" else self._LIKE_WORDS
        removed = 0
        try:
            nodes = self.store.list_nodes(speaker, limit=500)
        except Exception:  # noqa: BLE001
            return 0
        item_chars = set(item)
        for node in nodes:
            text = node.lod0
            if not any(word in text for word in opposite):
                continue
            if kind == "dislike" and any(word in text for word in self._DISLIKE_WORDS):
                continue  # 已经是“不喜欢”，不用再作废
            # 只作废“同一件东西”的偏好：物品词必须落在旧记忆里
            key = set(self._mem_key(text))
            if len(item_chars & key) / max(1, len(item_chars)) < 0.8:
                continue
            if self.store.archive(node.id):
                removed += 1
        return removed

    def _retract_name_memories(
        self, speaker: str, keep: str, actions: AppliedActions
    ) -> int:
        hints = ("名字", "称呼", "自称", "叫作", "叫我", "唤我")
        removed = 0
        try:
            nodes = self.store.list_nodes(speaker, limit=500)
        except Exception:  # noqa: BLE001
            return 0
        for node in nodes:
            text = node.lod0
            if keep and keep in text:
                continue
            if not any(hint in text for hint in hints):
                continue
            if speaker not in (node.about_ids or [speaker]):
                continue
            if self.store.archive(node.id):
                removed += 1
        return removed

    # ------------------------------------------------------------------
    def _carried_from(self, actions: AppliedActions) -> list[dict[str, str]]:
        carried: list[dict[str, str]] = []
        for recall in actions.recalls:
            if recall["kind"] == "memory":
                label = f"召回结果 #{recall['id']}｜来源:{recall.get('source_id')}｜{recall.get('epistemic')}"
                body = recall.get("lod2") or recall.get("lod1") or ""
                carried.append({"label": label, "text": body})
            elif recall["kind"] == "turn":
                carried.append(
                    {
                        "label": f"召回结果 · 第 {recall['turn_id']} 轮全文",
                        "text": recall.get("lod2", ""),
                    }
                )
            elif recall["kind"] == "search":
                if not recall.get("results"):
                    carried.append(
                        {
                            "label": f"检索结果「{recall['query']}」",
                            "text": "（权限内没有找到相关记忆）",
                        }
                    )
                else:
                    lines = [
                        f"#{r['id']}（相关{r['score']}）{r['lod1']}"
                        for r in recall["results"]
                    ]
                    carried.append(
                        {
                            "label": f"检索结果「{recall['query']}」",
                            "text": "\n".join(lines),
                        }
                    )
        return carried

    # ------------------------------------------------------------------
    def _record_turn(
        self, event: Event, says: Sequence[str], monologues: Sequence[str]
    ) -> TurnRecord | None:
        try:
            user_text = event.text if event.kind == "message" else (
                f"（自主唤醒）{event.extra.get('reason', '')}"
            )
            turn = TurnRecord(
                turn_id=self.store.next_turn_id(),
                speaker_id=event.speaker_id,
                user_text=user_text,
                assistant_text="\n".join(says),
                inner_monologue="\n".join(m for m in monologues if m),
                tags=[],
                visibility=[event.speaker_id],
            )
            turn.lod2 = turn.render_full()
            turn.lod1 = extractive_summary(turn.lod2, self.config.sleep.lod1_chars)
            turn.lod0 = one_line(turn.lod1 or turn.lod2, self.config.sleep.lod0_chars)
            self.store.add_turn(turn)
            return turn
        except Exception as exc:  # noqa: BLE001
            self._emit(f"（写入本轮记忆失败: {type(exc).__name__}: {exc}）")
            return None

    # ------------------------------------------------------------------
    # 便捷 / 诊断
    # ------------------------------------------------------------------
    def sleep(self):
        from memory.sleep_cycle import SleepCycle

        return SleepCycle(self.ws, config=self.config, store=self.store).run()

    def status(self) -> dict[str, Any]:
        frame = self.last_frame
        return {
            "speaker_turns": self.store.stats(),
            "focus": self.focus.snapshot("owner"),
            "emotion": self.emotion.load().to_dict(),
            "wakes": [t.to_dict() for t in self.scheduler.pending()],
            "last_frame_tokens": frame.total_tokens if frame else None,
            "last_frame_sections": frame.section_tokens if frame else None,
            "ledger": [c.to_dict() for c in self.ledger.log(5)],
            "sys2": getattr(self.sys2, "name", type(self.sys2).__name__),
            "sys1": getattr(self.reflex.client, "name", "unknown"),
        }

    def frame_report(self) -> dict[str, Any] | None:
        if self.last_frame is None:
            return None
        f = self.last_frame
        return {
            "tokens": f.total_tokens,
            "sections": f.section_tokens,
            "focus": f.focal.state.to_dict() if f.focal else None,
            "frame": f.user,
            "speaker_id": f.speaker_id,
        }
