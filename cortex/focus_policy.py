"""焦点与衰减的默认策略（纯数据模块）。

星璃可以在思考里调用 ``set_focus(...)`` 临时改写这些参数；这里保存的是
“出厂设置”和她的长期偏好。她也可以在进化中修改本文件（走安检门）。
"""

FOCUS_DEFAULTS = {
    "focus_turn_id": None,
    "decay_before": 0.35,
    "decay_after": 0.20,
    "recency_floor": 2,
    "pinned_turn_ids": [],
}

FOCUS_RULES = [
    "话题绕回旧争议时：把焦点锚到那一轮，调大 decay_before 压暗更早的寒暄，调小 decay_after 保留后续讨论。",
    "不要为了显得记得而钉选一切；钉选是承诺，不是收藏。",
]
