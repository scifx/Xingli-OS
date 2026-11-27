"""主心跳引擎：现场捕获 → Sys1 扫描 → 认知帧编译 → Sys2 思考 → 事务提交。

一次外部事件（有人说话 / 闹钟响了）会引发 1~``MAX_CHAIN_TICKS`` 拍思考：
每一拍都可能查证记忆、写记忆、改关系、调焦、预约下一次唤醒，或者保持沉默。

这也是把 README 的核心公式落到代码的地方::

    (y_t, W_t, M_{t+1}) = M_t(x_t, E_t; Sys1_φ, Sys2_θ)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Sequence

from kernel.config import Config
from kernel.evolution import Evolver
from kernel.git_ledger import GitLedger
from kernel.models import (
    AppliedActions,
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
from kernel.textutil import extractive_summary, one_line
from memory.emotion import EmotionStore
from memory.focal_context import FocalContext
from memory.frame_compiler import Frame, FrameCompiler
from memory.lod_renderer import LodRenderer
from memory.relations import RelationsStore
from memory.store import MemoryStore
from sensors.clock import ClockSensor
from sensors.social import SocialSensor
from sensors.world import WorldSensor


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
    def handle_message(self, speaker_id: str, text: str) -> HeartbeatResult:
        return self.handle_event(
            Event(kind="message", speaker_id=speaker_id or "owner", text=text or "")
        )

    def handle_event(self, event: Event) -> HeartbeatResult:
        result = HeartbeatResult(event=event)
        carried: list[dict[str, str]] = []
        says: list[str] = []
        monologues: list[str] = []
        tick = 1
        halted = ""
        while True:
            report = self._tick(event, tick, carried)
            result.ticks.append(report)
            packet = report.packet
            if packet is None:
                halted = halted or "本轮没有可用的认知包"
                break
            if packet.inner_monologue:
                monologues.append(packet.inner_monologue)
            if report.actions.says:
                says.extend(report.actions.says)
            next_carried = self._carried_from(report.actions)
            wants_continue = packet.wake.continue_think is not None
            auto_continue = bool(next_carried) and not report.actions.says and not wants_continue
            if not (wants_continue or auto_continue):
                break
            if tick >= self.max_chain_ticks:
                halted = f"达到 MAX_CHAIN_TICKS={self.max_chain_ticks} 硬熔断"
                break
            if auto_continue:
                self._emit(
                    f"（内核补一拍：已召回 {len(next_carried)} 条内容，等她消化后再开口）"
                )
            carried = next_carried
            tick += 1

        if event.kind == "wake" or says or monologues:
            self._record_turn(event, says, monologues)
        result.halted_reason = halted
        return result

    def poll_wakes(self, now: datetime | None = None) -> list[HeartbeatResult]:
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
            results.append(self.handle_event(event))
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
        scan = self.reflex.scan(state_text, event.speaker_id, now)
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
        self.last_packet = packet
        if mode == "empty" and packet.parse_warnings:
            report.actions.errors.extend(packet.parse_warnings)

        # 5) 应用（声明式字段 + 可选代码载荷）
        report.actions = self._apply(packet, event, scan)
        report.duration_ms = (time.monotonic() - start) * 1000
        return report

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
            actions.code_output = extra.notes and "; ".join(extra.notes) or ""

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
                if kind == "recall":
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
                ticket = self.scheduler.schedule(
                    spec.get("after_sec", 60),
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

    def _op_memorize(
        self, op: dict[str, Any], event: Event, scan, actions: AppliedActions
    ) -> None:
        lod0 = str(op.get("lod0") or op.get("text") or "").strip()
        if not lod0:
            actions.errors.append("memorize：lod0 不能为空")
            return
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
