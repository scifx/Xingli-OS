"""Jev 模糊条件反射清单（纯数据模块，星璃可经安检门新增）。

每个反射都是一条「模糊感知器 → 概率 → 三段门控 → 物理限幅动作」的弧。

字段说明：
* ``sensor``        : 交给 System One 的模糊问题（noul 是/否概率、choose 多路、score 打分）
* ``high_threshold``: ≥ 该值 → 高置信，直接执行 on_high
* ``low_threshold`` : 低于该值 → 静默忽略；两者之间 → 只注入“直觉提示”交给 Sys2
* ``cooldown_sec``  : 物理不应期（防同一条反射连续自激）
* ``max_delta``     : 单次对情绪/权重的最大影响（防某条反射劫持人格）
* ``cues``          : 给离线词法适配器用的线索；接真实 Jev 时它会被忽略

注意：``decay`` 由情绪状态机统一执行（每拍向基线回弹），因此这里不再
重复声明，避免两套衰减互相打架。
"""

REFLEXES = [
    {
        "id": "masking_fatigue",
        "enabled": True,
        "sensor": {
            "type": "noul",
            "question": "结合时间和语气，说话人表面敷衍或说没事，但实际上正处于疲惫、焦虑或硬撑状态？",
            "cues": {
                "positive": ["没事", "还行", "随便", "累", "困", "熬夜", "通宵", "压力", "算了", "无所谓"],
                "negative": ["开心", "睡了", "休息", "搞定了"],
            },
        },
        "high_threshold": 0.8,
        "low_threshold": 0.25,
        "cooldown_sec": 1800,
        "max_delta": 0.2,
        "hint_on_mid": "直觉上对方可能有点硬撑，但不确定，回复前留意语气细节。",
        "on_high": {
            "emotion_delta": {"caring": 0.15, "playfulness": -0.1},
            "inject_hint": "对方大概率在硬撑，先共情陪伴，不要急着讲大道理或给方案。",
            "boost_tags": ["comfort", "health"],
        },
        "provenance": {"source": "seed", "note": "出厂的守护反射"},
    },
    {
        "id": "teaching_signal",
        "enabled": True,
        "sensor": {
            "type": "noul",
            "question": "说话人是否正在教导、纠偏，或给我立一条以后要长期遵守的规则？",
            "cues": {
                "positive": ["记住", "以后", "不要", "别再", "不准", "规矩", "应该"],
                "negative": ["开玩笑", "随便说"],
            },
        },
        "high_threshold": 0.8,
        "low_threshold": 0.3,
        "cooldown_sec": 300,
        "max_delta": 0.15,
        "hint_on_mid": "对方像是在给一条规则，但还不确定是不是长期有效，先别急着写成准则。",
        "on_high": {
            "emotion_delta": {"focus": 0.12},
            "inject_hint": "这是一次教导：把它整理成一条可长期执行的准则（teach_rule），并说明你会怎么照做。",
            "boost_tags": ["policy", "commitment"],
        },
        "provenance": {"source": "seed", "note": "对话即在线训练的入口"},
    },
    {
        "id": "urgency_deadline",
        "enabled": True,
        "sensor": {
            "type": "noul",
            "question": "这句话里是否出现了时间压力（临近的截止、来不及、必须今天完成）？",
            "cues": {
                "positive": ["今天", "明天", "截止", "ddl", "deadline", "赶紧", "来不及", "催"],
                "negative": ["不急", "有空再说"],
            },
        },
        "high_threshold": 0.82,
        "low_threshold": 0.3,
        "cooldown_sec": 900,
        "max_delta": 0.15,
        "hint_on_mid": "好像有时间压力，但不确定急到什么程度。",
        "on_high": {
            "emotion_delta": {"focus": 0.15, "energy": 0.1},
            "inject_hint": "对方在赶时间：回答要短、要能立刻执行，别铺陈。",
            "boost_tags": ["deadline", "work"],
        },
        "provenance": {"source": "seed", "note": "时间感优先"},
    },
    {
        "id": "affection_moment",
        "enabled": True,
        "sensor": {
            "type": "noul",
            "question": "说话人此刻是在表达亲近、信任或感谢（而不是客套）？",
            "cues": {
                "positive": ["喜欢你", "谢谢", "爱你", "想你", "辛苦", "陪你", "信任"],
                "negative": ["客气", "不用了"],
            },
        },
        "high_threshold": 0.8,
        "low_threshold": 0.25,
        "cooldown_sec": 600,
        "max_delta": 0.18,
        "hint_on_mid": "对方语气里有点亲近的意味，但不确定是不是客套。",
        "on_high": {
            "emotion_delta": {"warmth": 0.15, "playfulness": 0.08},
            "inject_hint": "这是亲近的信号：自然接住，不要用客套把它推回去。",
            "boost_tags": ["affection"],
        },
        "provenance": {"source": "seed", "note": "关系权重正向反馈"},
    },
]
