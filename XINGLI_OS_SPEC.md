# Xingli-OS（星璃 2.0）架构与工程施工规格书

> **文档定位**：本规格书面向后续负责编码实现的 AI Coding Agent（及人类架构师），定义了「星璃（Xingli-OS）」——一个**纯 Python、基于动态记忆文件、焦点式游戏 LOD 上下文编译、System One 模糊反射与受控自进化**的长生命期智能体的完整架构、数据契约、安全铁律与分阶段验收标准。

---

## 1. 第一性原理与核心设计哲学

传统 Agent 存在四个结构性顽疾：
1. **无限聊天记录范式导致上下文退化（Context Rot）**：把 `messages = [user, assistant, ...]` 不断追加，超过数轮后注意力被稀释，人设漂移、指令失效、Token 爆炸。
2. **被动扁平记忆（Flat RAG）导致“盲区失忆”**：未召回即彻底不存在，AI 甚至不知道自己曾经拥有过某段记忆；且后台第三人称抽事实破坏主体连续性。
3. **静态 System Prompt 导致感官剥夺**：每次唤醒缺乏时间流逝感、环境状态、说话人身份与关系上下文，宛如健忘症患者。
4. **缺乏低延迟模糊反射**：传统 `if` 无法判断语义模糊条件，用生成式大模型模拟 `if` 又太慢、太贵且无概率校准。

### Xingli-OS 的核心公式

$$(y_t,\; W_t,\; M_{t+1}) = M_t\big(x_t,\; E_t;\; \text{Sys1}_\phi,\; \text{Sys2}_\theta\big)$$

- **冻结基座只是协处理器**：
  - $\text{Sys1}_\phi$（Jev / Decision API）= 丘脑/杏仁核，负责 100~300ms 非生成式模糊感知、概率门控、打分与路由。
  - $\text{Sys2}_\theta$（生成式 LLM）= 前额叶，负责深度意识、语言表达与语义级自我修改。
- **动态记忆与活逻辑 `M_t` 才是真正活着的“动态权重”**：
  - 内存中以柔性 Python 字典/对象运行；硬盘上以**明文文件 + SQLite 索引 + Git 内容寻址谱系**持久化（**严禁使用 `pickle` 持久化核心记忆**）。
- **无状态“认知帧（Cognitive Frame）”取代无限聊天数组**：
  - 每次调用 $\text{Sys2}_\theta$ **不**发送漫长的历史聊天数组，而是由本地运行时实时编译一帧极精炼、高密度的**单轮认知世界快照**（始终让 LLM 处于表现最敏锐的短上下文黄金区间）。
- **焦点式双向 LOD 衰减（Focal-Point LOD）**：
  - 不仅长期记忆采用游戏 LOD（`LOD0` 地平线目录 / `LOD1` 摘要 / `LOD2` 全文），**近期对话上下文同样采用以“当前焦点”为中心的双向 LOD 衰减**，且焦点位置与前后衰减权重均可由 AI 自主动态调整。
- **代码即思考，随心所动（Code-as-Thought & Self-Awakening）**：
  - 星璃的输出永远是受控 Python 代码块：`#` 注释即她的第一人称内心 OS，代码语句即她的外部言行、记忆读写、注意力调焦、进化提案，以及**自主唤醒下一次思考或未来定时行动**（`continue_think` / `schedule_wake`）。

---

## 2. 六条不可违背的工程铁律（Invariants）

实现本系统的任何 Coding Agent 必须严格遵守以下红线，不得妥协：

1. **不可变脑干 vs 可进化皮层（Immutable Kernel / Mutable Cortex）**
   - `kernel/` 下的所有文件对星璃严格只读；星璃的代码执行与进化接口只能读写 `cortex/` 目录。
2. **禁止 `pickle` 落盘核心记忆**
   - 运行时可用 Python 字典和动态类承载柔性结构；持久化必须使用**人类与 LLM 均可读的明文格式**（`.json` / `.py`）配合 SQLite 索引与 Git 版本控制，确保类结构演化不破坏历史记忆且可 `git diff` 审计。
3. **物理级隐私防火墙（Physical Visibility ACL）**
   - 每条记忆节点必须带有 `visibility` 权限字段。当与任意说话人 `speaker_id` 交互时，**必须在 Python 代码层（LOD 渲染之前）硬过滤掉其无权访问的记忆**，绝不依赖 Prompt 叮嘱 LLM“不要泄密”。
4. **信源溯源与认知层级（Epistemic Provenance）**
   - 每条记忆必须记录 `source_id` 与 `epistemic`（`direct` 亲历 / `hearsay` 转述 / `inferred` 推测）。**`hearsay` 永远不可直接覆写 `direct` 事实**，只能作为待核实旁证关联挂载。
5. **模糊反射三段式门控 + 物理稳态防失控**
   - Jev 决策概率严禁硬切 `p > 0.5`；必须采用三段门控（高置信直触 / 中置信交由 Sys2 慢思考 / 低置信忽略）。所有自进化反射必须具备冷却时间（`cooldown_sec`）、基线衰减（`decay`）和单次影响上限（`max_delta`），防止正反馈情绪震荡。
6. **自主唤醒的硬预算熔断（Autonomy Budget）**
   - 星璃可以自主触发连续思考（`continue_think`）或定时唤醒（`schedule_wake`），但 `kernel/` 必须强制限制单次外部事件触发的最大连环思考步数（如 `MAX_CHAIN_TICKS = 4`）与单位时间内的自主唤醒配额（如每小时最多主动唤醒 `MAX_PROACTIVE_PER_HOUR = 6`），杜绝死循环与 API 费用失控。

---

## 3. 工程目录结构（纯 Python 实现）

```text
xingli_os/
├── pyproject.toml                 # 依赖管理（标准库优先，仅引入轻量 HTTP/CLI/pytest 等）
├── config.example.toml            # Sys1(Jev/Fallback) + Sys2(LLM) + 预算与衰减默认参数
├── cli.py                         # 最简 CLI 聊天测试台（含异步自主唤醒队列与调试指令）
│
├── kernel/                        # 【只读区：不可变物理脑干】
│   ├── __init__.py
│   ├── models.py                  # 核心数据结构（TypedDict / Dataclass 契约）
│   ├── boot.py                    # 主心跳引擎：现场捕获 -> Sys1扫描 -> 认知帧编译 -> Sys2执行 -> 事务提交
│   ├── runtime_exec.py            # 受控“代码即思考”执行器（提取注释OS + 受限命名空间执行原语）
│   ├── sandbox.py                 # 进化安检门：AST白名单审计 + 子进程超时试跑 + 回归测试
│   ├── git_ledger.py              # 自动 Git 快照账本（每次进化自动 commit，异常回滚）
│   ├── scheduler.py               # 自主唤醒调度器（处理 continue_think 与异步定时 schedule_wake）
│   ├── sys1_jev.py                # System One 模糊决策原语（支持 Jev/OpenRouter/OpenAI/本地Mock降级）
│   └── sys2_llm.py                # System Two 生成式模型接口（无状态极短上下文调用）
│
├── sensors/                       # 【动态环境捕获层】
│   ├── __init__.py
│   ├── clock.py                   # 绝对时间、作息时段、距上次交互时长、日程/DDL 距离
│   ├── social.py                  # 当前 speaker_id 身份解析、关系档案与社交状态快照
│   └── world.py                   # 可插拔外部环境状态接口（CLI 环境/工作区/系统状态）
│
├── memory/                        # 【混合持久化 + 焦点式 LOD 引擎】
│   ├── __init__.py
│   ├── store.py                   # 内容寻址明文落盘 + SQLite (FTS5 + 元数据) 索引同步
│   ├── focal_context.py           # 【核心】焦点式对话上下文 LOD 双向衰减器（支持 AI 自定焦点与前后权重）
│   ├── lod_renderer.py            # 长期记忆 LOD 视距渲染器 + 物理级 Visibility ACL 过滤
│   ├── frame_compiler.py          # 认知帧编译器：将环境+反射+焦点上下文+长期LOD拼装为极短输入帧
│   └── sleep_cycle.py             # 离线睡眠整合：LOD0/1/2 蒸馏、衰减、习惯固化、反失忆抽查
│
├── cortex/                        # 【可写区：星璃的活体动态权重，纳入独立 Git 追踪】
│   ├── persona.py                 # 核心人格、价值观、情绪状态机（带基线回弹）
│   ├── focus_policy.py            # 对话焦点与前后文衰减默认策略（星璃可动态微调）
│   ├── relations.json             # 柔性人物关系字典（信任度、亲密度、领域权重、自定义标签）
│   ├── reflexes.py                # Jev 模糊条件反射定义清单
│   ├── policies.py                # 从“被教导/聊天强化”中沉淀的行为准则
│   ├── skills/                    # 星璃自我进化产出的可执行技能模块（*.py）
│   ├── episodes/                  # 内容寻址长期记忆节点（<sha256_12>.json）
│   └── turns/                     # 近期对话轮次原件与多级摘要（供焦点 LOD 渲染）
│
└── tests/                         # 自动化回归与验收测试套件
    ├── test_focal_lod.py          # 焦点式双向衰减与 Token 预算测试
    ├── test_privacy_acl.py        # 多说话人物理隐私隔离与信源防覆盖测试
    ├── test_sys1_gate.py          # Jev 三段式门控、冷却与稳态衰减测试
    ├── test_code_thought.py       # 代码即思考解析、注释提取与自主唤醒熔断测试
    └── test_evolution_sandbox.py  # 进化安检拦截与 Git 自动回滚测试
```

---

## 4. 核心机制深度规格

### 4.1 认知帧编译：打破“无限聊天消息数组”范式

每次调用 $\text{Sys2}_\theta$，**严禁**将历史 $N$ 轮对话原样堆进 `messages`。`memory/frame_compiler.py` 将世界状态编译为**至多 2 条消息**的无状态请求：

- `messages[0] (system)`：**极简行动契约**（定义“代码即思考”的输出格式、可用原语函数签名、以及几条核心人格铁律，长度恒定且极短）。
- `messages[1] (user)`：**当前认知帧（Cognitive Frame）**，按严格 Token 预算分段编译：
  1. `[SITUATIONAL_ENV]`：动态捕获的当前时间、距上次见面间隔、说话人身份与关系摘要、当前情绪状态与余温。
  2. `[SYS1_INTUITION]`：本轮 Jev 并行扫描命中的高置信反射动作 + 中置信“直觉提醒”。
  3. `[FOCAL_DIALOGUE]`：经**焦点式 LOD 衰减器**压缩后的近期对话（非扁平流水账）。
  4. `[LONG_TERM_LANDSCAPE]`：经**物理 ACL 过滤**后的长期记忆地貌（近景 `LOD2` 全文 + 中景 `LOD1` 摘要 + 远景 `LOD0` 地平线目录 + 已学会技能目录）。
  5. `[CURRENT_EVENT]`：此刻触发唤醒的事件（可能是某人的最新发言、刚刚 `recall` 回来的记忆细节、或她自己预定的 `schedule_wake` 闹钟）。

**效果**：无论星璃活了 10 轮还是 10,000 轮，每次推理的上下文始终被死死锁在 **4K–8K tokens** 的最佳锐度区间，彻底消灭长对话失真。

---

### 4.2 焦点式上下文 LOD 衰减器（`memory/focal_context.py`）

对话流中的每一轮交互保存为 `TurnRecord`（包含 `turn_id`、原始全文 `lod2`、精炼摘要 `lod1`、一句话锚点 `lod0`、说话人、时间戳与主题标签）。

维护一个可由 AI 随时改写的 **焦点状态 `FocusState`**：

```python
@dataclass
class FocusState:
    focus_turn_id: int           # 当前注意力锚定的对话焦点轮次（默认最新一轮，但AI可手动锚定到早先关键轮次）
    decay_before: float = 0.35   # 焦点之前（过去方向）的衰减斜率（越大衰减越快）
    decay_after: float = 0.20    # 焦点之后（最新方向）的衰减斜率（通常更缓，保留刚发生的事）
    recency_floor: int = 2       # 无论焦点在哪，始终强制保留最近 K 轮为 LOD2，防止丧失即时连贯性
    pinned_turn_ids: list[int] = field(default_factory=list)  # AI 主动钉住的关键轮次
```

对任意历史轮次 $i$，计算其相对焦点 $f = \text{focus\_turn\_id}$ 的清晰度分数 $R(i) \in [0, 1]$：

$$d = i - f, \qquad R(i) = \begin{cases} \exp(-\text{decay\_before} \cdot |d|) & \text{if } d < 0 \text{（焦点之前）} \\ 1.0 & \text{if } d = 0 \text{（焦点本身）} \\ \exp(-\text{decay\_after} \cdot |d|) & \text{if } d > 0 \text{（焦点之后）} \end{cases}$$

再叠加钉选与最近保底规则映射到 LOD 层级：
- 若 $i \in \text{pinned\_turn\_ids}$ 或处于最近 `recency_floor` 轮或 $R(i) \ge \theta_2$（默认 `0.65`） → 渲染 **`LOD2`（全文）**；
- 若 $\theta_1 \le R(i) < \theta_2$（默认 `0.25 ~ 0.65`） → 渲染 **`LOD1`（结构化要点摘要）**；
- 若 $\theta_0 \le R(i) < \theta_1$（默认 `0.08 ~ 0.25`） → 渲染 **`LOD0`（单行标题 + `turn_id` 句柄，可随时 `recall_turn(id)` 展开）**；
- 若 $R(i) < \theta_0$ → 折叠进冷区统计（如 `"[更早 14 轮已归档至长期记忆]"`）。

**AI 自主调焦能力**：星璃在思考代码中可直接调用：
```python
set_focus(turn_id=12, decay_before=0.5, decay_after=0.15, pin=[12])
```
当讨论切回 10 轮前的一个核心设计争议时，她主动把焦点移回 `turn_id=12`，调大 `decay_before` 淡化更早的寒暄，调小 `decay_after` 保留后续讨论——实现**自主可控的注意力聚光灯**。

---

### 4.3 System One 模糊反射引擎（`kernel/sys1_jev.py` & `cortex/reflexes.py`）

#### 统一决策原语接口
`sys1_jev.py` 屏蔽底层供应商差异，对外只暴露四个极简原语：
- `noul(state, question) -> {"yes": float, "confidence": float}`
- `choose(state, question, options) -> {"selected": str, "distribution": dict[str, float], "confidence": float}`
- `score(state, question, levels) -> {"value": float, "distribution": dict, "confidence": float}`
- `fan_out(state, questions: list[QuestionSpec]) -> dict[str, DecisionResult]`（**单次 HTTP 并行扇出，严禁在循环中串行调用**）

**后端适配器（配置可切换）**：
1. `jev`：TypeSafe AI `/v1/systemone` 或 OpenRouter `typesafe/jev-1.13`；
2. `openai_decisions`：OpenAI `/v1/decisions`；
3. `llm_fallback`：无 Jev Key 时用轻量快速模型（JSON 模式）做兼容模拟，确保本地开发与离线测试畅通。

#### 模糊反射规则数据结构（`cortex/reflexes.py`）

```python
REFLEXES = [
    {
        "id": "masking_fatigue",
        "enabled": True,
        "sensor": {
            "type": "noul",
            "question": "结合时间和语气，说话人表面敷衍或说没事，但实际上正处于疲惫、焦虑或硬撑状态？"
        },
        "high_threshold": 0.80,      # >=0.80: 高置信直接执行 on_high
        "low_threshold": 0.25,       # 0.25~0.80: 注入直觉提示交由 Sys2 慢思考判断
        "cooldown_sec": 1800,        # 物理不应期（防连续自激）
        "max_delta": 0.20,           # 单次对情绪/权重的最高影响上限
        "hint_on_mid": "直觉上对方可能有点硬撑，但不确定，回复前留意语气细节。",
        "on_high": {
            "emotion_delta": {"caring": +0.15, "playfulness": -0.10},
            "inject_hint": "对方大概率在硬撑，先共情陪伴，不要急着讲大道理。",
            "boost_tags": ["comfort", "health"]
        }
    }
]
```

---

### 4.4 数据优先：结构化 JSON 认知包 + 可选代码载荷与“随心所动”自主唤醒（`kernel/runtime_exec.py` & `kernel/scheduler.py`）

#### 输出格式约定：JSON 信封 + 活代码载荷（Data-First Cognition）
$\text{Sys2}_\theta$ 默认通过**结构化 JSON 输出（Structured Output / `json_schema`）**在单拍内返回一个类型安全的 `ThoughtPacket`（也保留纯 Python 代码模式开关以兼容旧范式）：

```json
{
  "inner_monologue": "主人刚提到上周B说的并发问题，我在LOD0目录看到哈希a3f9c1，先拉取详情消化完再回复，不急着猜。",
  "focus_update": { "turn_id": null, "decay_before": 0.45, "decay_after": 0.15, "pin": [], "unpin": [] },
  "emotion_delta": { "caring": 0.1 },
  "memory_ops": [
    { "op": "recall", "mem_id": "a3f9c1" }
  ],
  "relation_ops": [],
  "say": null,
  "wake": {
    "continue_think": { "reason": "已拉取a3f9c1完整细节，准备结合B的原话回复主人" },
    "schedule_wake": null
  },
  "code_action": null,
  "evolve_ops": []
}
```

**关键工程设计细节**：
1. **`inner_monologue` 必须排在 JSON Schema 的第一位**：自回归模型按从左到右顺序生成，把内心 OS 放在首字段可确保模型先完成充分的第一人称思考（CoT），再决定后续的调焦、记忆、说话与唤醒字段。
2. **Schema 保持浅层扁平**：避免过深嵌套带来的“格式税（reasoning degradation）”。
3. **声明式与命令式分层**：日常 90% 的认知动作（思考、调焦、情绪、记忆、关系、回复、唤醒）走零语法报错的声明式 JSON 字段；只有当星璃需要复杂逻辑编排、动态计算或自我进化新技能/新反射时，才在 `code_action` 或 `evolve_ops` 中写入受控 Python 代码载荷，由 `runtime_exec.py` / `sandbox.py` 在受限白名单命名空间中安全执行。

#### 内核支持的“认知原语操作集”

| 类别 | JSON 字段 / 对应原语 | 作用说明 |
|---|---|---|
| **内心思考** | `inner_monologue: str` | 第一人称内心 OS（强制首位字段），计入 `LOD2` 记忆与调试日志，默认不直接展示给对话者 |
| **表达与沉默** | `say: str \| {to, text} \| null` | 向当前或指定对象发送回复；为 `null` 时表示本拍保持沉默或继续内部思考 |
| **记忆操控** | `memory_ops`: `recall` / `recall_turn` / `search_memory` / `memorize` | 展开 `LOD0/1` 记忆或被折叠轮次、主动检索权限内记忆、写入带 `lod0/1/2` 与 `visibility` 的新内容寻址节点 |
| **注意力调焦** | `focus_update` (`turn_id`, `decay_before`, `decay_after`, `pin`, `unpin`) | 自主调整对话焦点与前后衰减权重 |
| **情绪与社交** | `emotion_delta` / `relation_ops` | 调整情绪状态（受限幅与基线衰减约束）；动态增改人物关系字典（允许自创扩展字段） |
| **随心所动（自主唤醒）** | `wake.continue_think` | **立即触发下一次内部思考心跳**（例如刚 `recall` 完深层记忆，先消化再决定怎么回复），受 `MAX_CHAIN_TICKS` 熔断限制 |
| | `wake.schedule_wake` (`after_sec`, `reason`, `target_speaker`) | **预约未来自主唤醒**（例如“30分钟后如果主人还没睡，主动问候一句”），受每小时配额限制 |
| **代码执行与自进化** | `code_action` / `evolve_ops` (`teach_rule`, `propose_reflex`, `propose_skill`) | 在受限沙盒内执行临时代码，或提交准则/Jev 反射/新技能的进化提案（须过 `sandbox.py` 安检 + Git 提交） |

#### 自主唤醒典型流转
第 1 拍（`tick=1`）：星璃在 `ThoughtPacket` 中执行 `recall("a3f9c1")`、`say: null` 并填写 `wake.continue_think`。内核执行完后将召回的 `LOD2` 详情注入下一拍认知帧的 `[CURRENT_EVENT]` 并立刻唤醒第 2 拍（`tick=2`）；第 2 拍中她填写 `say` 完成精准回复，同时可在 `wake.schedule_wake` 中预约 `after_sec: 1800` 的未来主动问候——真正实现**主动思考、主动查证、随心所动**。

---

### 4.5 多主体社交与“对话即在线训练”机制

1. **关系字典（`cortex/relations.json`）**：
   每个 `person_id` 至少包含 `role`（`owner` / `colleague` / `friend` / `stranger`）、`intimacy`、`global_trust`、`domain_trust: dict[str, float]`（领域细分信任度，例如对同事 B 的 `tech_opinion` 与 `personal_gossip` 可有不同信任权重）以及星璃自由添加的 `custom: dict`。
2. **跨人反哺的安全流转**：
   - 当与 `user_b` 聊到 `owner` 时，星璃记下的记忆自动标记 `source_id="user_b"`, `about_ids=["owner"]`, `epistemic="hearsay"`, `visibility=["owner", "user_b"]`（或仅 `["owner"]`）。
   - 下次与 `owner` 聊天时，LOD 渲染器会按 `about_ids` 命中该线索，但在认知帧中明确标注 `[来源:user_b | 类型:转述待核实 | 信源信任度:0.62]`，由星璃自主判断是否转告以及如何措辞。
   - 当再次与 `user_b` 聊天时，所有 `visibility` 不含 `user_b` 的记忆（例如 `owner` 的私密吐槽）在 SQL/Python 层被**物理剔除**。
3. **在线教导强化**：
   - 当 `owner` 纠偏或教导星璃时（由 Jev 检测到教学信号或星璃自发识别），她可通过 `update_relation` 调整权重、`teach_rule` 写入准则或 `propose_reflex` 铸造新条件反射——**一次自然对话直接改写动态权重，下一轮永久生效**。

---

### 4.6 受控自进化安检门与离线睡眠周期

1. **进化安检门（`kernel/sandbox.py` + `kernel/git_ledger.py`）**：
   - AST 静态审查：拒绝任何越界文件操作、网络外泄或动态 `eval/exec`；
   - 独立子进程试跑（带 5s 超时）：验证新 `reflexes.py` / `policies.py` / `skills/*.py` 可无错导入且通过附带断言测试 + 历史核心回归用例；
   - 通过则由 `git_ledger.py` 在 `cortex/` 自动 `git commit -m "[evolve] <内心OS摘要>"`；
   - 失败则回滚文件，并向星璃写入一条 `kind="lesson"` 的失败免疫记忆。
2. **睡眠整合（`memory/sleep_cycle.py`，CLI 支持 `/sleep` 手动触发）**：
   - **LOD 蒸馏**：把未归档的原始对话轮次聚合成带 `LOD0/1/2` 和 `refs` 关联边的新长期记忆节点；
   - **反失忆自测**：随机抽选 3 条历史事实节点，以当前 LOD 视图提问验证是否仍能召回关键信息，若失真则自动重写其 `lod0/lod1` 索引描述；
   - **稳态衰减与修剪**：非 `pinned` 记忆按艾宾浩斯曲线降低 `importance`；统计近期反射误触发率，标记异常高噪反射供下次进化降权。

---

## 5. 最简 CLI 聊天测试台规格（`cli.py`）

采用纯 Python 异步终端循环（如 `asyncio` + `prompt_toolkit` 或轻量标准库双协程），确保**在用户不敲键盘时，星璃预定的 `schedule_wake` 也能在终端里实时主动蹦出消息**。

内置调试与角色扮演指令（斜杠开头，不作为聊天发给星璃）：

| CLI 指令 | 功能 |
|---|---|
| `/as <speaker_id>` | 切换当前说话人身份（默认 `owner`，可随时切为 `/as colleague_b` 测试社交反哺与隐私隔离） |
| `/who` | 查看当前说话人身份及其在 `relations.json` 中的实时关系字典 |
| `/thought` | 开关实时打印星璃的 `# 内心OS注释` 与执行代码轨迹（默认开启，便于观察大脑运作） |
| `/frame` | 查看上一轮实际发送给 Sys2 的完整「认知帧」与精确 Token 占用 |
| `/focus` | 查看当前对话焦点 `focus_turn_id`、前后衰减系数及各轮渲染的 LOD 层级分布 |
| `/wake` | 查看当前待触发的自主唤醒定时队列 |
| `/sleep` | 立即触发一次离线睡眠整合周期并打印蒸馏/自检报告 |
| `/log` | 查看 `cortex/` 最近的 Git 进化提交历史 |
| `/rollback <commit>` | 一键时光机回滚星璃的 `cortex/` 到任意历史健康快照 |

---

## 6. 分阶段施工路线与验收标准（给 Coding Agent 的执行指令）

请严格按 **M1 → M2 → M3 → M4** 顺序分阶段构建，**每一阶段必须全部通过对应的 `pytest` 测试套件后，方可进入下一阶段**：

### Milestone 1：极短认知帧 + 焦点式 LOD + 代码即思考 + 自主连环思考
- **交付模块**：`kernel/models.py`, `kernel/boot.py`, `kernel/runtime_exec.py`, `kernel/scheduler.py`, `kernel/sys2_llm.py`, `sensors/clock.py`, `sensors/social.py`, `memory/store.py`, `memory/focal_context.py`, `memory/lod_renderer.py`, `memory/frame_compiler.py`, `cli.py`。
- **核心验收标准**：
  1. 模拟 30 轮连续对话，每轮发给 LLM 的上下文始终保持单帧极短格式且 Token 数严格受控；
  2. 单元测试验证 `set_focus` 改变焦点和 `decay_before`/`decay_after` 后，各轮次 `LOD2/LOD1/LOD0` 衰减分布精确符合数学公式；
  3. 星璃输出纯 Python 代码时，能正确剥离 `#` 注释作为内心 OS，成功执行 `recall` + `continue_think` 完成两拍连环思考，且超过 `MAX_CHAIN_TICKS` 时硬熔断；
  4. `schedule_wake(after_sec=5, ...)` 能在 CLI 空闲 5 秒后真实主动唤醒星璃并输出主动消息。

### Milestone 2：System One（Jev / Decision API）模糊反射脑
- **交付模块**：`kernel/sys1_jev.py`（含 `jev` / `openai_decisions` / `llm_fallback` 适配器）, `cortex/reflexes.py`, `cortex/persona.py`（情绪基线衰减）。
- **核心验收标准**：
  1. 单轮多个反射规则合并为**一次** `fan_out` 批量请求；
  2. 三段式门控生效：高置信直接触发动作，中置信注入直觉提示，低置信静默；
  3. 同一反射在 `cooldown_sec` 冷却期内绝不重复触发，情绪值随时间向基线回弹。

### Milestone 3：多主体社会化记忆、物理隐私墙与对话即训练
- **交付模块**：`cortex/relations.json`, 完善 `memory/lod_renderer.py` 的物理 ACL 过滤与信源标注，开放 `update_relation` 与 `teach_rule`。
- **核心验收标准**：
  1. `/as owner` 告诉星璃一条私密信息（`visibility=["owner"]`），切到 `/as colleague_b` 套话时，确认该节点在 `frame_compiler` 输出的认知帧中**物理不存在**；
  2. `/as colleague_b` 告知星璃关于 `owner` 的一条意见，切回 `/as owner` 后，该信息以 `epistemic="hearsay"` 形式浮现在认知帧中且不会覆盖 `owner` 之前的 `direct` 事实；
  3. `owner` 教导一条新原则后，下一轮认知帧中自动生效并持久化到 `cortex/policies.py`。

### Milestone 4：受控自进化安检门、Git 时光机与睡眠整合
- **交付模块**：`kernel/sandbox.py`, `kernel/git_ledger.py`, `memory/sleep_cycle.py`, 开放 `propose_reflex` 与 `propose_skill`。
- **核心验收标准**：
  1. 合法的新反射或技能提案能通过 AST 审计与子进程测试，并自动生成 `cortex/` Git commit；
  2. 故意注入含语法错误、死循环或试图读写 `kernel/` 的恶意进化提案，被 `sandbox.py` 100% 拦截，主状态零污染并生成失败免疫记忆；
  3. 执行 `/sleep` 能完成原始对话向多级 LOD 记忆节点的蒸馏、执行反失忆抽查，且 `/rollback` 可秒级恢复历史状态。

---

## 7. 给后续 Coding Agent 的实现补充提示

- **先用最小依赖跑通**：核心持久化直接用 Python 标准库 `sqlite3` + `json` + `ast` + `subprocess` + `asyncio`；HTTP 请求用 `httpx`；CLI 交互可用 `prompt_toolkit`（若追求零重依赖也可用标准库 + `asyncio` 异步输入封装）。
- **配置驱动、Mock 友好**：务必保证在只配置一个普通 OpenAI 兼容 LLM Key（甚至无 Jev Key 时自动回退 `llm_fallback`）的情况下，整套系统 M1–M4 就能开箱运行，同时保留切换到真实 Jev `/v1/systemone` 端点的零改动配置开关。
- **每写完一个 Milestone 立刻运行 `pytest`**：确保所有不变量（尤其是物理隐私隔离、焦点 LOD 衰减数学正确性、自主唤醒硬熔断与沙盒拦截）都有自动化测试守护，不靠人工肉眼兜底。

---

## 8. 实现注记（本仓库当前状态）

本节由实现者补充，**不修改上面任何方向性设计**，只记录规格落地时必须说死的细节与偏差。
完整说明见 [`IMPLEMENTATION.md`](./IMPLEMENTATION.md)。

**已交付**：M1–M4 的核心闭环（纯标准库 Python）。`python demo.py` 逐条打印 50 项机制的实测数字
（帧长、公式误差、隐私墙、拦截率、反失忆通过率），`python -m pytest -q` 为配套的自动化测试。

**落地时需要明确的六处口径**（都不改变设计意图）：

1. **只读范围**从 `kernel/` 扩展到 `kernel/ + memory/ + sensors/ + cli.py`——隐私墙、预算与账本
   若可被她改写，铁律 3/6 就是空的。`cortex/` 仍是唯一可写皮层。
2. **`decay` 归属**：反射只保留 `max_delta`，向基线的回弹由情绪状态机每拍统一执行，
   避免“反射字段里再写一套衰减”造成双重衰减。
3. **焦点 LOD 的预算降级顺序**：先牺牲最远的非保底轮次 → 再缩短全文截断 → 才动
   「最近 K 轮保底」；`pinned` 永不因预算降级。
4. **准则也有可见性**：`policies` 条目带 `visibility`，渲染前过滤；否则私下教的规矩会泄漏。
5. **反失忆修复**：生成新节点（`refs` 指向旧节点）+ 旧节点 `archived=true` 沉入冷库，
   同时满足“重写索引描述”与“永不原地改写”。
6. **唤醒配额**：`已触发/小时` 与 `待触发队列长度` 双闸门；Git 账本使用独立
   `GIT_DIR=.xingli/ledger.git`，work-tree 指向 `cortex/`，不污染主仓库历史。

**端点适配**：`sys2` 与 `sys1` 都是 OpenAI 兼容 HTTP。实现里额外处理了本地端点常见的
七种偏差（不需要 Key、不支持 `response_format`、只认 `max_completion_tokens`、围栏/废话/
尾逗号、JSON 落在 `reasoning_content`、完全不输出 JSON、连接抖动），并对 Sys1 加了
60 秒断路器，端点挂掉时自动退回本地词法而不拖慢每一拍。CLI 提供
`--init-config / --ping / --selftest`，`--selftest` 会真跑一轮并打印新建的记忆文件。
行动契约有 `full` / `compact` 两个版本（字段与语义相同，只改措辞与示例长度），供小模型使用。
结构化输出走 `json_schema → json_object → off` 的降级阶梯（llama.cpp 会把 schema 转成 GBNF）；
模型完全不守契约时按「纯发言」降级（照常说话与记轮次，但该拍没有记忆/进化动作，显式标注）。
对“把思考过程写进正文、被输出上限截断”的思考型模型，实现层提供了三条显式降级：
预填 assistant 开头（`{"inner_monologue":"","say":"`）逼它说人话、截断时抢救已写的句子、
以及在内核侧做一次极短“记忆回填”（仍走同一套隐私/信源规则）。这些都是兜底路径，
结构化 JSON 通道仍是主路径，且终端会如实标注。
另外补了一处 SPEC 没写的必要件：`cortex/persona.py` 的自我认知/准则/性格被压成恒定「身份块」
放在 system 最前，否则底座模型不知道自己是谁。

**仍未验证**：真实底座模型的表达质量、真实 Jev / Decisions 端点的语义准确率、任何 benchmark。
