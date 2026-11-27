# 实现说明：把「记忆 = 差分算子」跑起来

> 这份文档写给两类人：想跑起来看的人，和想继续改的人。
> 它对 `README.md`（思想与方向）和 `XINGLI_OS_SPEC.md`（契约与验收）**不做任何方向性修改**，
> 只记录“规格落到代码时，哪些地方需要更精确的定义”。

## 0. 一句话

纯标准库 Python 实现了 M1–M4 的核心闭环：**认知帧 + 焦点双向 LOD + 结构化认知包（可选代码载荷）
+ System One 三段门控 + 物理隐私墙/信源溯源 + 受控自进化（AST 审计/子进程试跑/回归/Git 账本）
+ 睡眠整合（蒸馏/衰减/反失忆抽查）**，并配有 57 个 pytest 与一个 50 项机制的离线验证脚本。

它**没有**验证的部分也写在明处：真实底座模型的表达质量、真实 Jev 端点的语义准确率、
任何 benchmark 成绩。离线 `mock` 大脑是**确定性脚本**，只负责让管线可复现。

---

## 1. 跑起来

```bash
# 1) 机制验证（离线，约 2 秒，50 项检查，临时工作区，不碰仓库状态）
python demo.py

# 2) 自动化测试
python -m pytest -q            # 57 passed

# 3) CLI 聊天台（离线）
python cli.py --mock
#   进去后：/help  /as colleague_b  /frame  /focus  /sleep  /log  /rollback <commit>

# 4) 接真实模型
cp config.example.toml xingli.toml
#   把 [sys2] provider 改成 openai_compatible，并 export XINGLI_SYS2_KEY=...
#   有 Jev Key 时把 [sys1] provider 改成 jev 并填 base_url
```

运行时产生的记忆、账本、状态都在 `.xingli/` 与 `cortex/{episodes,turns,state}/`（已 gitignore）；
它们由 `cortex/` 自己的 Git 账本（`.xingli/ledger.git`）追踪，不会污染主仓库历史。

---

## 2. 目录与 SPEC 的对应

| SPEC | 本仓库 | 说明 |
|---|---|---|
| `kernel/models.py` | ✅ | `TurnRecord` / `MemoryNode` / `FocusState` / `ThoughtPacket` / `Event` |
| `kernel/boot.py` | ✅ | 心跳：现场 → Sys1 → 帧 → Sys2 → 应用 → 记轮次 |
| `kernel/runtime_exec.py` | ✅ | JSON 认知包解析 + 代码即思考（AST 白名单 + 行预算 + 受限命名空间） |
| `kernel/sandbox.py` | ✅ | 技能 AST 审计 + 子进程试跑（超时/资源上限）+ 反射/准则校验 |
| `kernel/git_ledger.py` | ✅ | 独立 `GIT_DIR` 账本：commit / log / rollback |
| `kernel/scheduler.py` | ✅ | `schedule_wake` 队列 + 每小时双闸门配额 |
| `kernel/sys1_jev.py` | ✅ | `noul/choose/score/fan_out` + Jev / Decisions / LLM / 本地降级 |
| `kernel/sys2_llm.py` | ✅ | OpenAI 兼容客户端 + 恒定系统契约 + 离线 mock |
| `sensors/clock.py` | ✅ | 时段、距上次间隔、DDL 距离 |
| `sensors/social.py` | ✅ | 说话人身份、关系档案、情绪余温 |
| `sensors/world.py` | ✅ | 可插拔外部环境（`register(name, probe)`） |
| `memory/store.py` | ✅ | 内容寻址明文 JSON + SQLite（FTS5 trigram）+ 可见性表 |
| `memory/focal_context.py` | ✅ | 焦点式双向 LOD 衰减 + 预算降级 |
| `memory/lod_renderer.py` | ✅ | 长期地貌打分/渲染 + 物理 ACL + 准则/技能目录 |
| `memory/frame_compiler.py` | ✅ | 五段认知帧 + 总预算收敛 |
| `memory/sleep_cycle.py` | ✅ | 蒸馏 / 衰减 / 反失忆抽查 / 反射噪声统计 / 快照 |
| `cortex/*` | ✅ | 人格、焦点策略、关系、反射、准则、技能、记忆与轮次 |

新增（SPEC 未列出，但落地需要）：

| 文件 | 为什么需要 |
|---|---|
| `kernel/config.py` | TOML 配置与默认值（预算、权重、阈值一处可调） |
| `kernel/paths.py` | `Workspace` + `assert_writable`：铁律 1 变成会抛异常的函数 |
| `kernel/textutil.py` | Token 估算、CJK 二元组相似度、抽取式摘要、事实覆盖率 |
| `kernel/literal.py` | 数据型 `.py` 的安全读取与“保留注释的追加” |
| `kernel/evolution.py` | 把“提案 → 审计 → 试跑 → 回归 → 提交/回滚 + 免疫记忆”串成一条管线 |
| `kernel/regression.py` | 进化回归探针（在候选皮层上跑核心不变量） |
| `memory/relations.py` / `memory/emotion.py` | 关系字典与情绪状态机（限幅 + 基线回弹） |
| `demo.py` | 理论模型验证脚本（离线、可复现、带数字） |

---

## 3. 一拍心跳的数据流

```
外部事件 (speaker_id, 文本 | 闹钟)
   │
   ├─ ① 情绪向基线回弹         memory/emotion.py: tick_decay()
   ├─ ② 现场采样               sensors/clock+social+world → CPU 级“现实感”
   ├─ ③ Sys1 一次并行扇出      kernel/sys1_jev.py: fan_out(基础感知 + 活跃反射)
   │      └ 三段门控：high→生效 / mid→只提示 / low→静默；冷却期内不进请求
   ├─ ④ 编译认知帧            memory/frame_compiler.py（五段，锁预算）
   ├─ ⑤ Sys2 一帧进一包出      kernel/sys2_llm.py + runtime_exec.parse_thought_output
   ├─ ⑥ 应用认知包            kernel/boot.py: _apply()
   │      ├ focus_update → memory/focal_context.py（越界/不可见会被裁剪）
   │      ├ emotion_delta → 双重限幅（反射 max_delta ∩ 每拍上限）
   │      ├ memory_ops    → recall / recall_turn / search_memory / memorize
   │      ├ relation_ops  → 非 owner 不能改他人档案、不能自我升格；单次 ±0.25
   │      ├ evolve_ops    → kernel/evolution.py（安检门 + Git）
   │      └ wake          → continue_think（本事件内）/ schedule_wake（配额）
   ├─ ⑦ 需要时进入下一拍      召回结果注入 [CURRENT_EVENT]，最多 MAX_CHAIN_TICKS 拍
   └─ ⑧ 写回本轮记忆           TurnRecord（明文 JSON + 索引 + 可见性）
```

`Heartbeat.handle_event()` 返回 `HeartbeatResult`（含每一拍的帧、Token、反射信号、动作、错误），
CLI 与 `demo.py` 都只消费这个结构——没有隐藏状态。

---

## 4. 数据契约（落盘形态）

**长期记忆节点** `cortex/episodes/<id>.json`（id = 内容哈希前 12 位）：

```json
{
  "id": "a3f9c1d0e8b2",
  "kind": "episode",
  "lod0": "同事B提到主人在架构X上可能低估了并发瓶颈",
  "lod1": "……（结构化要点）",
  "lod2": "……（完整对话片段 + 当时的内心 OS）",
  "source_id": "colleague_b",
  "about_ids": ["owner"],
  "epistemic": "hearsay",
  "visibility": ["colleague_b", "owner"],
  "importance": 0.72,
  "pinned": false,
  "tags": ["sleep"],
  "emotion": {"valence": 0.3, "arousal": 0.7},
  "refs": ["b81e04…"],
  "created_at": "2026-11-26T23:40:00+00:00",
  "last_accessed": "2026-11-27T09:12:00+00:00",
  "access_count": 2,
  "archived": false,
  "schema_version": 1
}
```

**对话轮次** `cortex/turns/000012.json`：`turn_id / speaker_id / user_text / assistant_text /
inner_monologue / lod0 / lod1 / lod2 / tags / visibility / epistemic / created_at / archived`。

**焦点状态** `cortex/state/focus.json`：`focus_turn_id / decay_before / decay_after /
recency_floor / pinned_turn_ids`（默认值来自 `cortex/focus_policy.py`）。

**情绪状态** `cortex/state/emotion.json`：`values / baseline / decay_rate / max_delta_per_tick`
（基线来自 `cortex/persona.py`）。

**结构化认知包**（Sys2 输出，字段顺序即 Schema 顺序）：

```json
{
  "inner_monologue": "……先想清楚",
  "focus_update": {"turn_id": 12, "decay_before": 0.5, "decay_after": 0.15, "pin": [12], "unpin": []},
  "emotion_delta": {"caring": 0.1},
  "memory_ops": [{"op": "recall", "mem_id": "a3f9c1d0e8b2"}],
  "relation_ops": [],
  "say": null,
  "wake": {"continue_think": {"reason": "…"}, "schedule_wake": null},
  "code_action": null,
  "evolve_ops": []
}
```

代码通道同样可用（兼容 v1 范式），`#` 注释自动成为 `inner_monologue`：

```python
# 先确认细节，别急着猜。
detail = recall("a3f9c1d0e8b2")
feel(caring=0.1)
say("你之前说想让我记住这件事，我一直留着。")
continue_think("再消化一下这段")
```

两条通道共用同一套原语，最终都归约成同一个 `ThoughtPacket`，走同一条应用路径。

---

## 5. 公式与阈值（实现口径）

**焦点清晰度**（`memory/focal_context.py::clarity`）：

```
d = i − f
R(i) = exp(−decay_before·|d|)   d < 0
R(i) = 1.0                      d = 0
R(i) = exp(−decay_after·|d|)    d > 0
```

**LOD 映射**：`pinned` 或「最近 `recency_floor` 轮」或 `R ≥ θ2` → LOD2；
`θ1 ≤ R < θ2` → LOD1；`θ0 ≤ R < θ1` → LOD0；否则冷区（折叠成带 `turn_id` 的统计行）。

**预算不足时的降级顺序**（这是 SPEC 没写死、但必须定下来的部分）：

1. 牺牲“最远/分数最低”的**非保底**轮次：LOD2→LOD1→LOD0→冷区；
2. 仍然超支 → 缩短单条全文截断长度（700→350→175→…→30 字）；
3. 还装不下 → 才动「最近 K 轮保底」；
4. `pinned` 永不因预算降级（那是她显式给出的承诺；`/focus` 里可见）。

**长期地貌打分**（`memory/lod_renderer.py::score`）：

```
score = 0.40·语义相似 + 0.20·时间近度(半衰 7 天) + 0.25·重要度
      + 0.10·社交关联 + 0.05·反射加权   (+0.08 若 pinned)
```

权重全部在 `[render]` 里可调；语义相似用字符二元组余弦（CJK 友好、零依赖），
接口保持可替换（换成嵌入模型只需改 `similarity`）。

**情绪物理稳态**：每条反射自带 `max_delta`（≤0.25，进化时强校验），
状态机再施加每拍总上限（默认 0.2）与每拍向基线回弹 8%——双保险，防自激。

**反失忆覆盖率**：`coverage(LOD0+LOD1, LOD2) = |关键事实标记 ∩ 视图| / |关键事实标记|`，
关键事实标记 = 数字 / 拉丁词 / 中文双字片段。低于 0.6 视为“当前视距下已经失真”，
睡眠期重写索引描述。

---

## 6. 与 SPEC 的偏差 / 修正（都不改变方向，只把话说死）

| 位置 | SPEC 原话 | 实现 | 为什么 |
|---|---|---|---|
| 焦点公式 | `d < 0` 用 `decay_before`、`d > 0` 用 `decay_after` | 完全一致 | 与 README“刚发生的事更清楚”自洽（`decay_after < decay_before`） |
| `decay` 字段 | 反射字段里带 `decay` | 由 `EmotionState.tick_decay()` 统一执行 | 两套衰减会互相打架；保底用状态机，反射只保留 `max_delta` |
| 只读边界 | `kernel/` 只读 | `kernel/` + `memory/` + `sensors/` + `cli.py` 只读 | 隐私墙、预算、账本代码若可被她改，铁律 3/6 就形同虚设 |
| 准则（policy） | 未提可见性 | 每条准则带 `visibility`，渲染前过滤 | 否则“私下教的规矩”会通过准则目录泄漏给另一个人 |
| 唤醒配额 | 每小时最多 6 次 | “已触发/小时” + “待触发队列长度”双闸门 | 只限已发生次数会被“攒一长串预约”绕过 |
| 记忆修复 | “自动重写索引描述” vs “永不原地改写” | 生成**新节点**（`refs` 指向旧节点），旧节点 `archived=true` 降权沉入冷库 | 同时满足两条：历史可回溯、索引可变清晰 |
| Git 账本 | `cortex/` 纳入 Git 追踪 | 独立 `GIT_DIR=.xingli/ledger.git`，work-tree=`cortex/` | 不污染用户自己的仓库历史；rollback 用 `reset --hard` + `clean -fd` |
| 代码输出 | 默认 JSON，保留代码模式开关 | 两种都实现，共用原语与审计 | 兼容 README §八 的 v1 脉络 |
| 记忆落盘 | JSON/.py 明文 + SQLite | JSON 是正典；SQLite 可 `reindex()` 从文件重建 | 索引坏了不丢记忆 |
| 睡眠“习惯固化” | 高频套路蒸馏成反射 | MVP **只统计并报告候选**（标签频次、高噪反射） | 自动化反射需要外部选择压力与证据，宁可先报告，不擅自长新本能 |

---

## 7. 六条铁律 → 代码落点 → 守护测试

| 铁律 | 代码 | 测试 |
|---|---|---|
| 1 只读脑干 / 可写皮层 | `kernel/paths.py::assert_writable`、`sandbox.SkillAuditor` | `test_evolution_sandbox.py::test_cortex_boundary_guard` |
| 2 禁 pickle，明文 + 索引 + Git | `memory/store.py`、`kernel/git_ledger.py` | `test_store_sleep.py::test_memory_is_plaintext_and_reindexable` |
| 3 物理隐私墙 | `MemoryStore.list_nodes/visible_ids`、`LodRenderer.render` | `test_privacy_acl.py`（6 项） |
| 4 hearsay 不覆写 direct | `boot._op_memorize` 强制降级 + `_annotation` 标注 | `test_privacy_acl.py::test_hearsay_cannot_overwrite_direct` |
| 5 三段门控 + 冷却/衰减/限幅 | `sys1_jev.ReflexEngine.scan`、`memory/emotion.py` | `test_sys1_gate.py`（8 项） |
| 6 自主唤醒硬熔断 | `boot.handle_event`、`scheduler.Scheduler` | `test_code_thought.py`（连环 3 拍 / 配额 / 队列） |

`python demo.py` 是这些测试的“人读版”：每一项都打印实测数字（帧 Token、公式误差、
覆盖率、拦截结果、Git 提交），而不是只给一个绿勾。

---

## 8. 已知边界（诚实清单）

1. **没有真实模型参与**：`mock` 大脑是确定性脚本；真实表达质量取决于你在 `[sys2]` 里配的底座。
2. **Jev 端点未联网验证**：`HttpDecisionClient` 的解析对 `results/answers/…` 做了宽容处理，
   但不同供应商字段仍需按文档微调（`_parse_results` 一处即可）。
3. **语义检索是词面级**：字符二元组 + FTS5 trigram，不是嵌入。规模上万条后建议换向量索引，
   接口已在 `textutil.similarity` 与 `MemoryStore.search` 收口。
4. **技能沙盒是进程级**：AST 白名单 + 资源上限 + 超时，能挡住误写与常见逃逸，
   但不是容器/系统级隔离；高危场景请再套一层 OS 沙盒。
5. **“习惯固化”只做候选报告**：没有把统计结果直接变成新反射（避免无证据的自激增长）。
6. **睡眠蒸馏用抽取式摘要**：不依赖 LLM，因此可离线复现；接底座模型后可换成生成式摘要
   （`sleep_cycle._distill` 一处）。
7. **没有多通道 / Web UI**：CLI 是 SPEC 指定的最小测试台。

---

## 9. 继续改的三个入口

* **加一条反射**：在 `cortex/reflexes.py` 追加（或让她自己 `propose_reflex`）。
  阈值/冷却/`max_delta` 会先过 `sandbox.validate_reflex`。
* **换一个 Sys1 后端**：实现 `fan_out(state, specs)` 返回 `{id: DecisionResult}`，
  在 `build_client` 里注册；一次请求必须并行评估完所有问题。
* **加一个认知原语**：在 `kernel/runtime_exec.py::PacketBuilder` 增加方法，
  并在 `boot._apply` 里消费它——JSON 与代码两种通道会自动同时获得这个能力。
