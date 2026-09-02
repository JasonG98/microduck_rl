"""Microduck VelStand 环境: 行走 + 摔倒恢复, 单 policy.

REBASED (2026-07, 审计后续) 在 velocity 配方 — 已验证的 walker — 上, 而非
旧 velstand 使用的废弃旧配方. 2026-07 审计发现旧设计饿死了行走: 仅 ~25%
经验是干净指令行走 (2/3 俯卧重置 + 摔倒环境在完整 20 s episode 中刷恢复
奖励), 恢复奖励对步态征税 (常开的姿态双重计数, 来自 com_upward_velocity
低于行走高度的弹跳激励), 且俯卧初始化从 0.20–0.25 m 处丢下机器人 (函数
默认值 — 一个暴力失控冲击开启大多数 episode).

现在设计:
  - 行走层  = make_microduck_velocity_env_cfg, 逐字. 好 walker 的一切
    (跟踪权重, air_time, 原地转体桶, 固定指令范围, DR/noise/obs) 按构造
    流入.
  - 机器人  = 全碰撞 standup XML (body 可物理躺下).
  - 恢复    = 小奖励层, 门控在确实摔倒 (躯干 z < 0.10 m 或 倾斜 > 40°):
    干净行走时贡献恰好零, 仅在倒下时引导. upright_linear 给出处处
    朝向梯度; com_upward_velocity 为上升支付. (旧 com_height_recovery
    被丢弃: 在其带内平坦/无梯度, 与上面两项冗余 — 审计发现 3.)
  - 冲击惩罚 (躯干/头) 阻止硬着陆, 未门控.
  - joint_torque_rate_l2 (standup 验证的抗抖动) 用于转移平滑 — 惩罚力矩
    变化, 从不阻止恢复翻转.

Run-5 教训 (crouch 端点): 恢复走得很好但停在 40° 门控之后的深蹲 — 每个密集
恢复项在那里停止支付, recovery_success 悬赏要求 z > 0.105, 高于 policy
真实站立包络 (0.084–0.096), 所以它从不触发. 修复: (1) 共享 "恢复完成" 定义
(倾斜 < 25° 且 z > 0.09 — 可达) 用于悬赏, (2) fallen_tax 滞回在摔倒后继续
征税直到该定义满足, (3) height_progress — 基于势的 Δz 项, 给 crouch→stand
最后一英里其他项都不提供的密集梯度.

Run-6 教训 (4k 时仍停车): 修复经济学不够 — 悬赏触发了 (上升的 recovery_success
曲线) 但保持探索稀少, 因为最后一英里几乎得不到 on-policy DATA: 一个俯卧
episode 把大部分 5 s 摔倒预算花在到达 crouch, 然后 fallen_too_long 在前沿
回收它. 旧 velstand 之所以学恢复快, 正因为 2/3 俯卧重置 + 20 s episode
使摔倒状态数据充足 (代价是饿死行走). Run-6 不饿死地恢复了数据密度: (1)
crouch_prob 反向课程切片 — 直接重置到随机中间恢复蹲姿, 从 step 0 起密集
最后一英里数据; (2) 摔倒超时 5 → 8 s; (3) 经济学在 800 (行走 ~750 稳定) 且
整个俯卧渐升提前 ~500 iter.

Run-7 教训 (run 6 vs run 5 @4k 的无头评估, 2026-07-21): crouch 切片起作用了 —
run 6 真正垂直站立 (倾斜 ≈1°, z ≈0.117) 且从蹲姿恢复 94–97% — 但俯卧恢复
塌缩到 0% (run 5: 从俯卧起来但停在 ~30°). 原因: run 6 在 iter 800 同时开启
税 + 悬赏 + 俯卧 + crouch, 删除了无税自然摔倒窗口 (run 5 的 500→1200), 那里
俯卧翻转探索便宜且密集进度项单独教会了它 — run 5 的 recovery_success 在其
权重 1200 开启的那一刻就触发了. 税从 800 活跃且无望的俯卧 episode 以
-0.5/step 流失完整 8 s 超时, run-3 的回避/冻结机制在俯卧状态重现, 而 PPO
容量流向了容易的 crouch 切片奖励. Run-7: 保留 crouch 切片 (已验证) + 8 s
超时, 经济学恢复到 1200, 俯卧恢复到 run-5 渐升 (1500+), crouch 切片单独
从 800 (预经济学无害: 它只加站立数据).

阶段 (如前, 但带恢复后盾):
  阶段 1 (0 → 500 iter): `fell_over` 终止活跃 (70°) → 先干净行走.
  阶段 2 (500+): fell_over 禁用 (限位 → π) 使摔倒成为恢复机会 — 但
    `fallen_too_long` (持续倒下 5 s) 回收失败恢复而非让它刷完整 20 s episode.
  阶段 3 (1500+): 俯卧初始化渐升: 先面朝下 (更易), 后混入面朝上, 上限
    45% 俯卧使行走数据份额保持 ≥ ~55% (原 2/3 俯卧 → ~25% 行走份额).
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
)

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

# 阶段边界 (PPO 迭代; 环境步计数器按 num_steps_per_env=24 缩放)
FELL_OVER_DISABLE_ITER = 500
NUM_STEPS_PER_ENV = 24

# 摔倒门控. 教训 (首次 rebase 训练运行): 恢复奖励必须仅门控在倾斜上. 同时
# 门控在低高度上使坐姿 (z≈0.07, 躯干直立) 开启门控 → policy 学会坐着刷
# upright_linear 同时为 com_upward_velocity 上下弹跳并通过 air_time 窗口抖腿.
# 门控正奖励在坏状态上奖励进入该状态. 倾斜 > 40° 无法从舒适姿态刷 — 你真的
# 翻倒了. 终止保留 z 条件, 使坐者和卡低环境被回收 (终止) 而非被支付.
REWARD_GATE_TILT_DEG = 40.0  # 恢复奖励: 摔倒 = 倾斜 > 40° 仅
# 终止 z 门控 0.08, 非 0.10 (run-3 教训): 正常摇摆直立机器人下到 z=0.084-0.096
# — 0.10 在早期学习包络内, 每 5 s 回收蹲行探索者. 0.08 仍捕获坐姿 (z≈0.07)
# 和俯卧 (z≈0.05).
TERM_GATE_Z = 0.08  # fallen_too_long: z < 0.08 或 倾斜 > 40°
TERM_GATE_TILT_DEG = 40.0

# "恢复完成" 定义 — recovery_success 悬赏和 fallen_tax 释放共享 (run-5
# crouch 端点教训). z 阈值必须在 policy 真实站立包络内: run 3 实测正常摇摆
# 直立机器人 z ≈ 0.084–0.096, 完整 STAND 关键帧稳定在 ≈ 0.117. 旧 up_z=0.105
# 要求站得比 policy 实际更高 → 悬赏从不触发 → 恢复收敛到 40° 门控之后的深蹲
# (每个密集恢复项停止支付处) 而非完成站立. 0.09 每次站立可达, 仍比坐 (z≈0.07)
# 高 2 cm, 比俯卧 (z≈0.05) 高 4 cm.
RECOVERED_UP_TILT_DEG = 25.0
RECOVERED_UP_Z = 0.09

# 税和悬赏为恢复阶段而存在. Run-3 教训: fallen_tax 从 step 0 活跃 (密集,
# -0.5) 在 ~25 iter 内教会 "不惜代价避免倾斜" → 行走引导前的蹲冻局部最优
# (ep_len 钉在 5 s 回收, air_time 从不增长). Run-6 试 800 ("行走 ~750 稳定")
# 俯卧恢复从不引导 — 1200 从来不是关于行走; 它买了一个无税窗口 (fell_over
# 500 关 → 经济学 1200 开), 那里自然摔倒起立尝试零成本, 密集进度项单独教会
# 它们. Run-7 恢复它.
RECOVERY_ECON_KICKIN_ITER = 1200

# 失败恢复后盾: 持续倒下此时长 → 终止/重置. Run-6: 5 s → 8 s. 5 s 时面朝下
# 恢复把大部分预算花在到达深蹲, 在前沿被回收 — 蹲→站最后一英里几乎没有
# on-policy 数据.
FALLEN_TIMEOUT_S = 8.0

# 俯卧 + 蹲姿初始化渐升 (阶段 3). 俯卧上限 45% (原 2/3 — 饿死行走); 先面朝下
# (更易恢复), 后混入面朝上. Run-6: crouch_prob 加反向课程切片 — 环境直接重置
# 到随机中间恢复蹲姿 (见 set_random_crouch_state), 使最后一英里获得密集数据
# 而非仅在罕见好 rollout 尾部到达. Run-7: 回到 run-5 俯卧计划 (俯卧在经济学
# 之后, 经济学在无税自然摔倒窗口之后 — 见上方经济学注); run 6 在 800 同时开
# 俯卧+经济学, 俯卧恢复从不引导. crouch 切片单独从 800 起: 近直立状态, 经济学
# 前无税, 它兼作完整站立姿态数据 (run 6 真正垂直站立).
PRONE_RAMP_STAGES = [
    {
        "step": 0,
        "params": {"prone_prob": 0.00, "face_down_prob": 1.0, "crouch_prob": 0.00},
    },
    {
        "step": 800 * NUM_STEPS_PER_ENV,
        "params": {"prone_prob": 0.00, "face_down_prob": 1.0, "crouch_prob": 0.15},
    },
    {
        "step": 1500 * NUM_STEPS_PER_ENV,
        "params": {"prone_prob": 0.15, "face_down_prob": 0.80, "crouch_prob": 0.15},
    },
    {
        "step": 2000 * NUM_STEPS_PER_ENV,
        "params": {"prone_prob": 0.30, "face_down_prob": 0.65, "crouch_prob": 0.15},
    },
    {
        "step": 2500 * NUM_STEPS_PER_ENV,
        "params": {"prone_prob": 0.45, "face_down_prob": 0.50, "crouch_prob": 0.15},
    },
]


def make_microduck_velstand_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg:
    """构建 VelStand 环境 cfg (velocity 配方 + 摔倒恢复 + 身体姿态)."""
    # 行走层: 已验证的 velocity 配方, 逐字.
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)

    # play 模式下课程不运行, 所以下方摔倒终止禁用从不触发 — 直接删除终止.
    if play:
        cfg.terminations.pop("fell_over", None)

    # 全碰撞 standup XML: 躯干/头壳保留接触, 使机器人可物理躺在地上并推地.
    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}

    # velocity 环境的 head_pose_bias 未门控流入 (仅行走环境上没问题 — fell_over
    # 在那里终止摔倒 episode). Velstand episode 在摔倒中存活, 所以未门控 EMA 会在
    # 整个地面阶段对头部 "下垂" 收费 — 一个对摔倒的平坦税, 恢复经济学 (run 1-7)
    # 从未计入. 加直立门控: 误差在 z=0.09 以下 / 40° 倾斜之外 (匹配
    # REWARD_GATE_TILT_DEG) 停止喂入 EMA, 使该项恰好定价 velocity 环境中它定价
    # 的 — 实际站立/行走时的持续下垂 — 恢复期间不计.
    cfg.rewards["head_pose_bias"].params.update(
        {
            "gate_height_low": 0.09,
            "gate_height_high": 0.11,
            "gate_tilt_full_deg": 20.0,
            "gate_tilt_zero_deg": REWARD_GATE_TILT_DEG,
        }
    )

    # ── 恢复奖励层 ─────────────────────────────────────────────────────────
    # 教训 (run 1/2/4 — 坐, 躺, 头三脚架): 任何对处于摔倒状态的正奖励都会从某
    # 舒适姿态刷. 朝向奖励因此基于势 (Δcos 倾斜): 上升支付, 下降收费, 保持任何
    # 支付零. 不可刷, 未门控, 也奖励行走中接住踉跄. (Run 4 具体: 移除头部冲击
    # 惩罚解锁了 ~55° 的头三脚架刷门控的 +2·cos(tilt) — run 2 仅靠该惩罚保护.)
    cfg.rewards["upright_progress"] = RewardTermCfg(
        func=microduck_mdp.upright_progress,
        weight=5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    # upright_progress 的 z 轴伙伴 (run-5 crouch 端点教训): 蹲→站最后一英里
    # 主要是适度倾斜下的高度变化 — 那里 Δcos(倾斜) 微小且高斯 upright/pose
    # 奖励平坦. 同样基于势构造: 不可刷 (保持/弹跳净零), 未门控, 对称收费摔倒.
    # 完整俯卧→站立升起 (0.05 → 0.115 m) 收集 Δ≈+0.065 × 30 ≈ +2; 蹲→站里程 ≈ +1.
    cfg.rewards["height_progress"] = RewardTermCfg(
        func=microduck_mdp.height_progress,
        weight=30.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            "ceiling": 0.115,
        },
    )
    cfg.rewards["com_upward_velocity"] = RewardTermCfg(
        func=microduck_mdp.com_upward_velocity,
        weight=0.0,  # 恢复项 — 在 RECOVERY_ECON_KICKIN_ITER 渐入
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            # 高度门控略高于站立 (standup 用 0.125), 使上升奖励直到完全站起仍
            # 支付; 摔倒门控阻止步态弹跳刷, 不是这个上限.
            "max_height": 0.125,
            # 仅倾斜门控: z=0.0 从不触发 (见上方教训)
            "gate_z_below": 0.0,
            "gate_tilt_above_deg": REWARD_GATE_TILT_DEG,
        },
    )
    # 无冲击惩罚 (首次运行教训 #2): standup 专家没有 — 鸭的恢复用头/躯干推地,
    # 头部惩罚 (-1.0 @ 2 N) 恰好对该策略征税. 摔倒比起来更便宜. 下方
    # joint_torque_rate_l2 覆盖着陆粗暴. standup 验证的抗抖动项: 惩罚力矩变化
    # (非幅度或旋转) → 平滑转移而不阻止恢复翻转.
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torque_rate_l2,
        weight=-2e-3,
    )

    # ── 恢复经济学 (首次运行教训 #3-#5) ──────────────────────────────────
    # air_time 在摔倒时归零: 躺在躯干上的机器人可节奏性地在摆动窗口中点脚 —
    # 观察到的 "抖腿" 刷.
    at = cfg.rewards["air_time"]
    at_params = dict(at.params)
    cfg.rewards["air_time"] = RewardTermCfg(
        func=microduck_mdp.feet_air_time_upright,
        weight=at.weight,
        params={**at_params, "gate_tilt_above_deg": REWARD_GATE_TILT_DEG},
    )
    # 摔倒时平坦税: 躺着不动必须比尝试严格更糟. (没有它, 等 5 s 的
    # fallen_too_long 回收是理性的 — 恢复尝试花 action-rate/torque 惩罚,
    # 等待花 0.)
    cfg.rewards["fallen_tax"] = RewardTermCfg(
        func=microduck_mdp.fallen_state_penalty,
        weight=0.0,  # 在 RECOVERY_ECON_KICKIN_ITER 渐升到 -0.5 (见课程)
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            "gate_tilt_above_deg": REWARD_GATE_TILT_DEG,
            # 滞回 (run-5 crouch 端点教训): 恢复停在 40° 门控之下的深蹲 — 越过
            # 每个恢复项的门控, 但未达站立. 释放条件匹配 recovery_success 悬赏
            # (下方), 摔倒持续征税直到站立真正完成; 40° 以下的蹲不再是零成本
            # 休息状态. 仅在倾斜 > 40° 时生效, 所以正常步态从不被税.
            "release_tilt_below_deg": RECOVERED_UP_TILT_DEG,
            "release_z_above": RECOVERED_UP_Z,
        },
    )
    # 完成恢复的一次性悬赏 (摔倒 ≥0.5 s → 真正起来), 带滞回使门控振荡零支付.
    # 密集门控项缺乏的强端点信号.
    cfg.rewards["recovery_success"] = RewardTermCfg(
        func=microduck_mdp.recovery_success,
        weight=0.0,  # 在 RECOVERY_ECON_KICKIN_ITER 渐升到 +10 (见课程)
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            "fallen_tilt_deg": REWARD_GATE_TILT_DEG,
            "min_fallen_s": 0.5,
            "up_tilt_deg": RECOVERED_UP_TILT_DEG,
            "up_z": RECOVERED_UP_Z,  # 曾 0.105 — 不可达, 见常量注
        },
    )

    # ── 事件: 俯卧初始化 ────────────────────────────────────────────────
    # z 修正 (审计 BUG): 函数默认 0.20–0.25 m — 每个 俯卧 episode 开启 15–20 cm
    # 自由落体. 面朝下躯干静止在 ~0.044 m; 改为在地面稍上方生成.
    cfg.events["random_prone_init"] = EventTermCfg(
        func=microduck_mdp.maybe_set_random_prone_orientation,
        mode="reset",
        params={
            "prone_prob": 0.0,  # 由 prone_init_prob 课程渐升
            "face_down_prob": 1.0,
            "prone_z_min": 0.05,
            "prone_z_max": 0.09,
            "crouch_prob": 0.0,  # 由 prone_init_prob 课程渐升
        },
    )

    # ── 终止 ──────────────────────────────────────────────────────────
    # 失败恢复后盾 (见模块 docstring, 阶段 2).
    cfg.terminations["fallen_too_long"] = TerminationTermCfg(
        func=microduck_mdp.fallen_too_long,
        time_out=False,
        params={
            "gate_z_below": TERM_GATE_Z,
            "gate_tilt_above_deg": TERM_GATE_TILT_DEG,
            "max_duration_s": FALLEN_TIMEOUT_S,
        },
    )

    # ── 课程 ─────────────────────────────────────────────────────────
    # 阶段 1 → 2: iter 500 禁用 fell_over (限位 70° → 180°), 使摔倒成为
    # 恢复训练而非 episode 结束.
    if not play:
        cfg.curriculum["fell_over_disable"] = CurriculumTermCfg(
            func=microduck_mdp.termination_param_curriculum,
            params={
                "term_name": "fell_over",
                "param_stages": [
                    {"step": 0, "params": {"limit_angle": math.radians(70.0)}},
                    {
                        "step": FELL_OVER_DISABLE_ITER * NUM_STEPS_PER_ENV,
                        "params": {"limit_angle": math.pi},
                    },
                ],
            },
        )

    # 阶段 3: 俯卧初始化渐升 (先面朝下, 后面朝上, 上限 45%).
    cfg.curriculum["prone_init_prob"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_curriculum,
        params={
            "event_name": "random_prone_init",
            "param_stages": PRONE_RAMP_STAGES,
        },
    )

    # 恢复经济学渐升: 税 + 悬赏在行走建立前关闭 (见 RECOVERY_ECON_KICKIN_ITER
    # 注 — run-3 蹲冻教训).
    cfg.curriculum["fallen_tax_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "fallen_tax",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": RECOVERY_ECON_KICKIN_ITER * NUM_STEPS_PER_ENV, "weight": -0.5},
            ],
        },
    )
    cfg.curriculum["recovery_success_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "recovery_success",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": RECOVERY_ECON_KICKIN_ITER * NUM_STEPS_PER_ENV, "weight": 10.0},
            ],
        },
    )
    cfg.curriculum["com_upward_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "com_upward_velocity",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": RECOVERY_ECON_KICKIN_ITER * NUM_STEPS_PER_ENV, "weight": 2.0},
            ],
        },
    )

    return cfg


MicroduckVelStandRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="velstand",
    run_name="velstand",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=20_000,
)
