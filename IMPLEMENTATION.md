# 实现说明：把「记忆 = 差分算子」跑起来

> 这份文档写给两类人：想跑起来看的人，和想继续改的人。
> 它对 `README.md`（思想与方向）和 `XINGLI_OS_SPEC.md`（契约与验收）**不做任何方向性修改**，
> 只记录“规格落到代码时，哪些地方需要更精确的定义”。

## 0. 一句话

纯标准库 Python 实现了 M1–M4 的核心闭环：**认知帧 + 焦点双向 LOD + 结构化认知包（可选代码载荷）
+ System One 三段门控 + 物理隐私墙/信源溯源 + 受控自进化（AST 审计/子进程试跑/回归/Git 账本）
+ 睡眠整合（蒸馏/衰减/反失忆抽查）**，并配有 140 个 pytest（其中 43 个打在本地假端点，验证真实协议路径）与一个 50 项机制的离线验证脚本。

它**没有**验证的部分也写在明处：真实底座模型的表达质量、真实 Jev 端点的语义准确率、
任何 benchmark 成绩。离线 `mock` 大脑是**确定性脚本**，只负责让管线可复现。

---

## 1. 跑起来

```bash
# 1) 机制验证（离线，约 2 秒，50 项检查，临时工作区，不碰仓库状态）
python demo.py

# 2) 自动化测试
python -m pytest -q            # 140 passed

# 3) CLI 聊天台（离线）
python cli.py --mock
#   进去后：/help  /files  /cat <id>  /tree  /status  /as  /frame  /focus  /sleep  /log  /rollback

# 4) 接本地 llama.cpp（或任何 OpenAI 兼容端点）
llama-server -m model.gguf -c 8192 --port 8080
python cli.py --init-config --base-url http://127.0.0.1:8080/v1 --model <模型名>
python cli.py --ping        # /v1/models 探测
python cli.py --selftest    # 真跑一轮 + 打印新建记忆文件
python cli.py               # 开聊（每轮打印记忆变化）
```

CLI 的阻塞调用跑在守护线程里：等模型时按 Ctrl-C 会立刻退出，
不会在 Windows 上抛 `Exception ignored on threading shutdown`。

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
| `kernel/jsonx.py` | 模型输出的宽容解析（围栏、废话、尾逗号、reasoning 通道） |
| `viewer.py` | 把 `cortex/` 的明文变化渲染成终端里的“记忆变化”（CLI 专用） |
| `demo.py` | 理论模型验证脚本（离线、可复现、带数字） |
| `tests/fake_openai.py` | 本地假端点：离线验证真实协议路径（降级/围栏/401/重试） |

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

## 3.5 接真实端点（llama.cpp / 任何 OpenAI 兼容 / Jev）

星璃只有两个外部依赖，都是 HTTP，都是可选的：

| 回路 | 配置 | 请求形状 |
|---|---|---|
| System Two（前额叶） | `[sys2] provider = "openai_compatible"` | `POST {base_url}/chat/completions`，恒定为 `system` + `user` **两条消息** |
| System One（丘脑） | `[sys1] provider = "llm_fallback"` / `"jev"` / `"openai_decisions"` | 一次请求并行评估所有感知器；失败自动退回本地词法 |

### 本地 llama.cpp 的四步

```bash
llama-server -m model.gguf -c 8192 --port 8080            # 1) 起服务
python cli.py --init-config --base-url http://127.0.0.1:8080/v1 --model <模型名>   # 2) 写配置
python cli.py --ping                                      # 3) /v1/models 探测
python cli.py --selftest                                  # 4) 真跑一轮 + 打印记忆文件
```

`--selftest` 会：探测端点 → 建临时工作区（`.xingli/selftest`，每次重建）→ 发一句
“记住……” → 打印**解析模式、帧 Token、内心 OS、她说的话、记忆变化、新记忆文件原文**。
它把“端点通不通”和“她有没有真的长记忆”一次验证完。

**上下文长度**：一次请求 = 系统契约（683 / compact 327）+ 认知帧（≤ `budget.frame_tokens`，默认 6000）
+ 生成（`max_tokens`，默认 1400）。所以 `llama-server -c` 至少 8192，建议 16384；
只有 8K 上下文时把 `[budget] frame_tokens` 降到 4000 即可，LOD 会自动按预算降级。

### 结构化输出：先语法约束，再退让

`[sys2] structured_output`（默认 `auto`）是一条**降级阶梯**，每一级都实测过：

```
json_schema（llama.cpp 会转成 GBNF，语法上强制出合法 JSON）
   └─ 端点 400 → json_object（只要求“是 JSON”）
        └─ 还是 400 → off（纯提示词，再不行就降级成纯发言）
```

认知包的 Schema 在 `kernel/sys2_llm.py::THOUGHT_SCHEMA`，只要求 `inner_monologue` 与 `say`
两个必填字段，其余可省——对 4B 级别的模型来说，“被语法逼着写合法 JSON”比“被请求写 JSON”
可靠得多。

**她是谁**：`cortex/persona.py` 的 `SELF_CONCEPT / VALUES / TRAITS` 会被压成一段恒定的
「身份块」放在 system 提示最前面（约 250 tokens，不随对话增长）。这是实现时补的——
SPEC 没写，但没有它，底座模型根本不知道自己叫星璃，更不会用第一人称讲自己。

### 思考型 / 极弱模型的三道保险

真实测试里最常见的失败不是“不会说话”，而是**模型把思考过程写进正文**（`首先，用户要求只输出合法JSON…`），
然后被输出上限截断，既没有 JSON 也没有一句人话。为此加了三条，全部是显式降级、可关闭：

| 保险 | 触发 | 行为 | 开关 |
|---|---|---|---|
| 预填（prefill） | 正文以“首先/用户要求/我来分析…”开头，或手动开启 | 追加一条 assistant 消息 `{"inner_monologue": "", "say": "`，让模型**接着说**；截断时用 `salvage_say` 抢救已写的句子 | `prefill = "say"` / `auto_prefill` / `/prefill on` |
| 记忆回填（memory pass） | 这句话含“记住/喜欢/约定/…”且模型没写 `memory_ops` | 再发一次极短抽取请求，结果仍走 `_op_memorize`（隐私与 hearsay 铁律一条不少） | `memory_pass = true` |
| 帧内格式提醒 | `chat template` 吞掉 system 消息 | 在帧末尾追加一行 `[OUTPUT_FORMAT]` 短提醒 | `contract_in_user = true` |

预填的代价要说明白：被预填的那一拍通常只产出 `say`（模型把话说完了就收尾），
所以**长期记忆要靠“记忆回填”这一趟**才能长出来；记忆回填失败就是没有记忆，绝不编。
结构化 JSON 通道仍然是主路径——这三条只在模型不守契约时兜底，且全程在终端里标注。

### 自转的三道刹车 + 一个输入窗口

本地 4B 一拍 15~40s，模型又爱把 `wake.continue_think` 写成客套话，于是“连环思考”很容易变成
“她没有输入窗口、你只能干等”。现在 `Heartbeat.handle_event()` 每拍结束时检查：

| 刹车 | 判据 | 结果 |
|---|---|---|
| 复读 | 本拍的 `say` 与之前某拍相似度 ≥ 0.9 | 丢弃重复，`模型在重复上一拍的话，内核停止自转` |
| 无进展 | 既没有说话、也没有记忆/召回/调焦/预约 | 静默收尾（自转是手段不是目的） |
| 时间预算 | `[budget] max_chain_seconds`（默认 60s） | `单轮思考超过 Ns 预算，先收尾` |
| 主人插话 | CLI 传入的 `should_stop()` | `主人插话了，这一轮先收尾` |

CLI 侧把 stdin 交给一个**后台读线程**（`_InputPump`），主循环每 0.25s 吸一次输入：
你在她思考时打的字会排队（`… 收下了（这一拍收尾就处理）`），并立刻成为刹车信号；
`/quit`、`/status` 这类指令也按原顺序排在后面处理。另外 `schedule_wake` 的 `after_sec`
被 `[budget] min_wake_sec`（默认 30s）钳住——`after_sec: 0` 不会再把自己叫醒成自激循环。

### 事实校正：她必须能改口，也必须分得清“谁叫什么”

4B 级模型会把“我的名字是 lucky”读成“主人叫星璃”，而且旧记忆只增不改，于是越聊越错。
这里走的是**内核确定性解析**（`kernel/claims.py`），不依赖模型理解：

1. 陈述句里抽出 `我叫 X / 我的名字是 X / 主人叫 X / 我喜欢 X / 我讨厌 X / 我对 X 过敏 / 我的生日是 X`
   （问句不算：`你的名字是什么？`「我叫什么来着」都不产生事实）；
2. 称呼写进关系档案 `display_name`，旧的、互相矛盾的称呼记忆**归档作废**（不删除，可 `/rollback`），
   同时写一条钉选事实 `称呼：lucky（owner 亲口说的自己的名字）`；
3. 偏好/禁忌写成 `主人喜欢菠萝和芒果` 这类可检索的事实；
4. 认知帧的环境段固定加一行：`称呼：对方叫「lucky」；你叫「星璃」。两个名字分清楚。`
5. 同一条事实第二次出现时按骨架相似度（去掉标点与“吃得了的着了”）去重，只 touch 不新建——
   实测里 19 条记忆有 11 条是同一句名字的变体，这条规则把记忆增长从线性噪声变成事件驱动。

她也因此真正“能修改记忆”：认知包里可以发 `{"op": "forget", "mem_id": "..."}`（只能作废自己来源或主人视角的），
CLI 里则是 `/forget <id>`。

### 记忆病历：真实数据里发现的五类错误

把线上 `xingli.db` 摊开逐条看，暴露的不是“模型不够聪明”，而是**内核没有把好门**：

1. **占位符入账**：契约里的示例值 `lod0="一句话"` / `lod1="两三句要点"` 被模型照抄成记忆。
   → 契约改成“尖括号说明类型”，抽取提示明确“必须是具体内容”；写入端按 `kernel/memtext.py`
   的判据拒收占位符与纯数字。
2. **LOD 被当成分数**：`lod1="0.85"`。→ 写入时用 LOD0 回填，并在报告里标注清洗过。
3. **状态回显当作记忆**：`lod0="主人亲密度0.85，信任值1.00"`。→ 拒收（`STATE_ECHO`）。
4. **同一事实七八个变体**：`主人偏好：喜欢菠萝和芒果` / `owner喜欢食用菠萝和芒果…`。
   → `mem_key()` 抹掉时间前缀、谓词、连词、同义词后做骨架比对；同事实只留一条，
   更全的版本**升级**并把旧版归档；相反极性（喜欢/不喜欢）永不合并。
5. **重要性被召回次数顶满**：一条闲话被召回 11 次后 importance=0.97。
   → 强化只在 `base_importance + 0.12` 以内生效，`access_count` 单独记录。

存量用 `/tidy [dry]` 一次整理（归档而非删除，Git 账本可回滚）。另外两处“空转”也顺手修了：
自主唤醒的连环思考上限为 1 拍（`max_wake_chain_ticks`），且**没有 say 的唤醒不落轮次**——
线上 7 轮“对话”里有 4 轮是这种空壳。

**降级但绝不装死**：如果模型输出既不是 JSON 也不是代码，内核按「纯发言」处理——
`say` 照样上屏、轮次照样落盘，但这一拍没有记忆/进化动作，并在 `actions.errors` 与 CLI
里显式标注 `⟨降级⟩`。修复请求只在“像写坏的 JSON”时发（纯散文不重发），省掉一整趟生成。

### 真实端点常见的毛病，客户端已经处理

| 毛病 | 处理方式 | 配置 |
|---|---|---|
| 本地服务不校验 Key | 没有 Key 就不发 `Authorization`；401 时提示“本地通常不需要 Key” | `api_key = ""` |
| 不支持 `response_format`（老版 llama.cpp / 某些代理） | 400 且错误里含 `response_format` → 自动去掉重发 | `json_mode = true` |
| 只认 `max_completion_tokens` | 400 且错误里含 `max_tokens` → 自动换字段重发 | 自动 |
| 输出带 ```json 围栏 / 前后有人话 / 尾随逗号 | 围栏剥离 + 括号配平截取 + 尾逗号容错 | `kernel/jsonx.py` |
| 把 JSON 塞进 `reasoning_content`，`content` 为空 | 依次尝试 content / reasoning / text 通道 | 自动 |
| 完全不肯输出 JSON | 追加一次“只输出 JSON、不要解释”的修复请求 | `repair = true` |
| 连接抖动 | 重试一次（0.6s 退避） | `retries = 1` |
| 干脆只说人话（4B 常见） | 降级为「纯发言」：照样说话/记轮次，但没有记忆与进化动作，显式标注 | 自动 + `⟨降级⟩` |
| chat template 吞掉 system 消息 | 在帧末尾再贴一次 `[OUTPUT_FORMAT]` 短提醒 | `contract_in_user = true` |
| 模型把预算花在思考上 | 用额外字段关掉思考模板 | `[sys2.extra_body] chat_template_kwargs = { enable_thinking = false }` |
| 不按契约命名（`reply`/`monologue`/`memories`），或把包在外壳里 | 别名表 + 单层解包 + 单对象包成数组，并在 `parse_warnings` 里留下“降级记录” | `kernel/models.py::PACKET_ALIASES` |
| 小模型跟不上长契约 | 换紧凑契约（683 → 327 tokens，字段与例子都更短） | `[sys2] contract = "compact"` |

小模型建议 `--init-config --fast`：compact 契约（327 vs 683 tokens）+ 输出上限 700。
只改提示词措辞与生成长度，认知帧、记忆格式、安全边界完全不变——这是“模型能力适配”，
不是“设计降级”。

**速度**：每轮结束都会打印 `⟨计时⟩ 54.9s · 1 拍 · 帧 815 tokens · 出 512 tok · 6.4 tok/s`。
`出 N tok` 是模型实际生成的长度——如果它远大于 100，说明模型在“自由发挥”而不是按需作答
（语法约束 + compact 契约能把它压下来）。生成速率由 llama.cpp 的 `timings` 直接读回。

`--status` 会显示最近三条“适配记录”（例如“端点不支持 response_format，已改用提示词约束 JSON”），
让你知道刚才悄悄降级了什么，而不是假装一切正常。

### System One 的三档接法

* `local`（默认）：离线词法近似，零延迟。**先用它把聊天跑顺**；
* `llm_fallback`：任何 OpenAI 兼容 `/chat/completions`，一次请求拿到全部感知器的概率；
* `jev` / `openai_decisions`：真正实现了 `/v1/systemone` 或 `/v1/decisions` 的专用端点。

后两者都包在 `ChainClient` 里：**连续失败 2 次后打开 60s 断路器**，期间直接走本地词法，
避免“端点挂了 → 每一拍都白等一次超时”。`/status` 里能看到断路器状态。

### 记忆文件长什么样

一轮对话后，`cortex/` 里至少会多两个明文文件：

```
cortex/episodes/f9b67852df8a.json   长期记忆节点（LOD0/1/2 + 来源 + 认识论 + 可见性）
cortex/turns/000002.json            这一轮的完整轮次（含内心 OS）
cortex/state/{emotion,focus,reflex_state}.json  运行时状态（每拍都会动）
```

`episodes/<id>.json` 的字段按 **可读顺序** 落盘（不是字母序）：

```json
{
  "id": "f9b67852df8a",
  "kind": "episode",
  "lod0": "下周三上午看牙医（别排别的事）",
  "lod1": "主人说下周三上午要去医院看牙，当天不要安排其它事情。",
  "lod2": "主人对星璃说：记住，我下周三要去看牙医。",
  "source_id": "owner",
  "about_ids": ["owner"],
  "epistemic": "direct",
  "visibility": ["owner"],
  "importance": 0.85,
  "pinned": false,
  "tags": ["承诺"],
  "emotion": {},
  "refs": [],
  "created_at": "2026-11-27T11:08:07+00:00",
  "updated_at": "2026-11-27T11:08:07+00:00",
  "last_accessed": "2026-11-27T11:08:07+00:00",
  "access_count": 0,
  "archived": false,
  "schema_version": 1
}
```

`id` 是内容哈希：同一件事重复说不会长出重复节点（`access_count` 增加）。
`/cat <id>` 直接打印这个文件；`/files` 列出全部；`/tree` 看占用。

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
  "updated_at": "2026-11-27T09:12:00+00:00",
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

**真实端点路径**另有两组测试，全部打在本地假端点上（离线、确定性）：

| 覆盖点 | 测试 |
|---|---|
| 无 Key 本地调用 / JSON 模式降级 / max_tokens 降级 / 围栏清洗 / reasoning 通道 / 修复重试 / 401 提示 / 探测 | `tests/test_openai_compat.py`（13 项） |
| 真跑一轮后记忆文件落盘、且下一帧里出现该记忆的句柄 | `tests/test_openai_compat.py::test_heartbeat_writes_plaintext_memory_that_reaches_next_frame` |
| Sys1 `llm_fallback` 解析 / 降级 / 断路器打开与恢复 | `tests/test_sys1_http.py`（6 项） |
| `--init-config` / `--ping` / `--say --show-memory` / `--selftest` | `tests/test_cli.py`（5 项） |
| 紧凑契约真的更短、未被契约的字段别名被接住 | `tests/test_openai_compat.py` / `tests/test_packet_aliases.py`（5 项） |
| 思考型模型自动预填、截断抢救、记忆回填、开关生效 | `tests/test_identity_and_fallback.py`（7 项） |
| code_action 回执不再崩、重复/插话/超时刹车、错位字段归位、称呼纠正与作废、去重、唤醒钳间隔 | `tests/test_robustness_and_corrections.py`（11 项） |
| `/sys1` 切换与活体探测、输入队列顺序 | `tests/test_cli.py`（2 项） |
| 占位符/数字/状态回显拒收、LOD 字段修复、同一事实去重与升级、`/tidy`、强化封顶 | `tests/test_memory_hygiene.py`（9 项） |
| 唤醒不自转、空唤醒不落轮次、sys1 升格与错误可见 | `tests/test_robustness_and_corrections.py`（+5 项） |
| 身份块来自 cortex/persona.py 且可被改写、纯发言降级、schema→object→off 阶梯、帧末尾格式提醒 | `tests/test_identity_and_fallback.py`（7 项） |
| 记忆快照求差、`/cat` 路径解析、`/files` 渲染 | `tests/test_viewer.py`（7 项） |

---

## 8. 已知边界（诚实清单）

1. **没有真实模型参与**：`mock` 大脑是确定性脚本；真实表达质量取决于你在 `[sys2]` 里配的底座。
2. **没有真实 Jev / GPT 账号参与验证**：所有端点路径都在本地假服务器上测过
   （含 400 降级、围栏输出、reasoning 通道、401、重试），但真实供应商的字段细节
   仍可能需要微调（`_parse_results` / `_post` 各一处）。
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
* **换一个底座模型**：只改 `xingli.toml` 的 `[sys2]`；若模型有特殊请求字段（关思考、
  温度曲线、`n_predict` 等），塞进 `[sys2.extra_body]`，不需要动代码。
