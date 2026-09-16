"""Microduck VelStand 环境: 行走 + 摔倒保护 + 恢复, 单 policy.

PROTECTIVE-FALL 重建 (2026-09, 分支 protective_fall). 动机: 真实机器人
不断损坏 XL330 齿轮箱. daemon 的摔倒检测跛行 (kp→50) 有帮助但不完美,
且跛行→站立交接产生 "抽搐". 目标: 一个既能行走、又能以保护舵机的方式
摔倒、并温和起身的 policy — 这样最终可以关掉 daemon 跛行.

相对 2026-07 设计 (保留在下方, 仍然有效) 的变化:
  - 从已部署 walk 热启动 (WARM START) (wandb 441tzs6d @ model_3750 —
    alpha_walking.onnx 背后的确切 checkpoint). 相同的 61D obs / 512-256-128
    actor / 76D critic, 所以 `--agent.resume True --wandb-run-path ...
    --wandb-checkpoint-name model_3750.pt` 加载一切. 结果: walk 在 iter 0
    就存在, 因此每个 velocity 配方课程 (action_rate 斜坡, standing-env
    比例, head/body 指令范围, CoM DR, head_pose_bias 权重) 都被折叠到
    最终阶段 (WARM_START) — 从阶段 0 重跑会在更宽松条件下重新教 walk 并
    使其漂移. 只有 velstand 自己的 phase 会斜坡, 且比从头开始的调度短 ~2×.
    LAUNCH:  MICRODUCK_WARM_START=1 uv run train Mjlab-VelStand-Flat-MicroDuck \
               --env.scene.num-envs 4096 --agent.resume True \
               --wandb-run-path pollen-robotics/mjlab_microduck/441tzs6d \
               --wandb-checkpoint-name model_3750.pt
    这个环境变量很重要: mjlab 的 runner 会从 checkpoint 恢复
    common_step_counter (90 000) 和 iteration (3750), 这会让下面每个基于
    step 的课程在 iteration 1 跳到最终阶段 (第一次 smoke test 中观察到).
    MICRODUCK_WARM_START=1 (tasks/mdp.py Patch 5) 在加载权重/normalizer/
    optimizer 后将两者都重置为 0, 因此下面的课程相对本次运行的起点.
    不要从 stand expert (69u48n8l) 热启动: 它的 obs normalizer 的 twist
    std 是 0.005 — 把行走指令送进去是 40σ 输入.
  - ROBOT = robot_allcollisions.xml (真正的全碰撞导出; 与 groundcontact 的
    body/质量/行程相同, 70 个碰撞 geom vs 11), 15 个 XL330 外壳 geom 被
    命名 (name_servo_collision_geoms), 使接触传感器能区分 "舵机撞地" 与
    "外壳撞地".
  - SERVO-PROTECTION 代价, 全部 mjlab 风格 (>= 0, 负权重), 干净行走时全部
    ≈ 0, 所以只对摔倒和坏的恢复计价:
      servo_impact    — 外壳接触力 > 2 N (剥齿轮的事件: 舵机外壳承担落地)
      head_impact     — 头部子树对地力 > 15 N (≈2× 体重: 撑头起身是准静态
                        ~7 N 且保持 FREE — 2026-07 的教训: 未门控的头惩罚
                        会征税恢复策略; 脸着地是 60 N 尖峰, 实测)
      trunk_impact    — 躯干外壳对地力 > 20 N (同理)
      servo_acc_spike — 舵机关节 |accel| > 300 rad/s² (冲击反驱通过反射转子
                        惯量加载齿轮系; CPU 摔倒无论跛行策略如何峰值都是
                        450–550 — policy 的杠杆是它如何落地)
      servo_stall     — 舵机在 < 0.5 rad/s 时推 > 0.4 N·m (肢体被压在身体
                        下而舵机与之对抗: 抽搐 / 剥齿 / 过热的情形)
      gentle_rise     — 躯干 |a_z| (自取负, 正权重)
    在恢复经济学生效后 25% → 100% 斜坡, 这样恢复发现窗口不会被它自己
    失败的尝试征税.
  - TOPPLE 推挤: 第二个推挤事件 (topple_push), 速度踢远超行走 DR
    (斜坡 ±0.6 → ±1.2 m/s; 1 m/s Δv 在 CPU 基线中 ~100% 推倒该机器人),
    所以摔倒是一线 on-policy 数据, 而非罕见事故.
  - Backlash 孪生: robot_allcollisions_backlash.xml (add_backlash.py, 总 2°)
    → Mjlab-VelStand-*-Backlash-MicroDuck 按要求镜像基础模型.
  - 有意不建模 (用户决定 2026-09-09): daemon 跛行. 如果 policy 用完全
    权限找到更温和的策略, 不如让它去.

Run-1 教训 (wandb 4otqmkb4, @1098 终止, ckpt 750 的 headless eval):
  walk 迁移成功 (iter 100 = 源运行的 tracking/upright) 且 topple 斜坡产生
  了摔倒, 但从来没有恢复: 73% 的 eval env-steps 处于摔倒, 摔倒动作率是
  直立时的 1/5, 力矩 1/3 → policy 学会了躺着不动. 两个原因, 都已修复:
  (1) ATTEMPT TAX. 热启动折叠把 action_rate_l2 从 step 0 钉在 -1.0
      (-1.5..-2/step, 栈中最大的代价). 加上平坦的 -0.5 摔倒税, 乱动比等待
      更贵 → "什么都不做" 获胜 (AGENTS.md 的 smoothness-after-discovery
      规则, 从另一侧命中). 修复: 倾斜 > 40° 时 action_rate / torque_rate
      ×FALLEN_SMOOTHNESS_SCALE (0.1) (action_rate_l2_fallen_scaled) —
      walk 保留全额税.
  (2) SPAWN HOLE. 79% 的摔倒落在侧面, 20% 面朝上, 0% 面朝下; prone init
      只知道 face-down/face-up. 修复: prone 切片中的 side_prob.
  另外: 恢复斜坡提前 (walk 不需要 bootstrap 窗口), crouch 切片扩大.

Run-2 教训 (wandb 1bqctpkq, @1085 终止, ckpt 1000 的 per-spawn battery):
  standing 98% / crouch→stand 97% / face-down, face-up, side 0% — 仍然躺平
  (摔倒 |Δa| 只有直立的四分之一). 减轻尝试税还不够: 热启动 walk 的 action
  std ≈ 0.2, 躺着随机乱挥永远不会产生部分起身, 所以基于 potential 的项
  永远不付 → 完全没有梯度 (从头开始的 velstand 运行以 std 1.0 乱挥发现
  恢复, 代价是 walk). 已部署的 stand expert (69u48n8l@9750) 跑同一 battery:
  100/100/95% (side 从未训练!) 在 ~1 s 内. 修复: PpoWithExpertBc — 每次
  PPO 更新后向那个冻结 expert 的摔倒门控行为克隆 (distill.py). RL 继续
  塑造摔倒本身和最后一英里.

Run-3 教训 (wandb 4lflk7ii, @1060): 它从正面和背面都站起来了 — 但 walk
  在第一次 BC 传递就死了 (iteration 8: 在几百帧摔倒帧上 1e-3 的 20 个 Adam
  step; iteration 12 时 fall 终止 2 → 72, 然后 falls→fallen-frames→更多
  BC 的循环; yaw 跟踪误差 0.9 → 3.4 rad/s). 没有任何东西约束直立帧上的
  共享权重. 修复 (distill.py): 热启动 WALK checkpoint 是第二个冻结 teacher,
  锚定 tilt < 25° 的帧; tilt > 35° 用 stand expert; 中间和 reward 叠加的
  一切由 PPO 单独负责. BC lr 3e-4, min mini-batch 512. 64 envs 下 40-iter
  检查, 50% prone spawns + 1 m/s 推挤: fallen frac 0.44 → 0.33, 同时直立
  lin-vel 误差保持 0.22 m/s (run 3 的配方在更温和条件下 12 个 iteration
  内毁掉 walk). 冲击代价被确认真实但微弱 (脸着地尖峰 129 N, 0.5% 的步数
  ≈ 每次摔倒 -0.3, 对比 ~7/step 行走) — 一旦存在恢复再 ×5 提高, 而不是
  在此之前.

Run-4 (wandb 6op8a8u8) WORKS, 机器上也行 → 已发布参考. 相对已部署的
  walk→limp→stand 管线的仿真基准 (3 seeds): 恢复 88-95% vs 83-86% 的推倒,
  中位数 2.7 s vs 3.3 s, 舵机外壳接触 0 vs 每次摔倒 50 ms, 躯干冲击 0 vs
  14 N, 卡死减半; 前进速度与 walk expert 持平; 头部冲击相等 (~32 N) —
  开放目标. 原地转体 yaw 速率训练中漂移 0.34 → 0.12 rad/s (停住, 不关键).
  机器人反馈: (a) 真实摔倒后从背部起身时抽搐上升 — 仿真中推倒后的面朝上
  落地恢复 58-77% vs 面朝上 SPAWN 的 100%; stand expert 有同样的洞 (71%):
  真实摔倒后腿在任意位置, HOME 姿态生成永远不会展示这点. (b) 偶尔前向
  起身后向后过冲 — 仿真中未复现 (0/192 次再摔倒).

Runs 5-6 (axr82ws4, padquu42; 从 5999 恢复) — 2026-09-13 回滚:
  尝试类摔倒 prone 生成 (randomize_servo_joints_uniform, 保留在 mdp.py,
  此处 OFF), servo_stall ×3, 然后 fallen smoothness 0.4 / gentle_rise ×4
  / BC coef 0.3. 教训: servo_stall ×3 使起身变得弹道式 ("低速高扭矩" 就是
  缓慢小心的俯卧撑) — 机器人过冲 + 再摔倒; stand expert 在随机关节面朝上
  生成上只有 83%, 学生在那里平台化 (teacher 天花板); run 6 给了轻微的
  推倒恢复增益但面朝上仍停在 85%, 前向起身再摔倒 9/190 (5999: 0/192),
  原因未知. 机器上不优于 5999 → 回到这个配方.

Rough+backlash run 1 (wy8gcaus) — 灾难性, 根因已在 mdp.py 修复:
  每个摔倒生成函数都写绝对躯干 z (0.05-0.09 m); rough env 的 origin 范围
  0-0.21 m (斜坡金字塔) → 从 iter 700 (prone spawns 开启) 起机器人在地形
  内部生成 → value loss 0.6 → 42, entropy 5 → 33. 生成 z 现在相对 origin
  (_env_origin_z). 相同常数的平坦运行全程稳定. 生产配方 = 本文件在
  Mjlab-VelStand-Rough-Backlash-MicroDuck 上, 从 6op8a8u8@5999 热启动.

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

from mjlab.envs.mdp import push_by_setting_velocity
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_ALLCOLLISIONS_ROBOT_CFG,
    SERVO_GEOM_SUFFIX,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    HEAD_BODY_NAMES,
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.distill import PpoWithExpertBcCfg, default_bc_cfg
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

NUM_STEPS_PER_ENV = 24

# 从已部署 walk 热启动 (见模块 docstring). 把每个 velocity 配方课程折叠到其
# 最终阶段, 使加载的 walk 在其训练条件下的同一条件下训练. 设 False 则从头
# 训练, 使用原始从头调度 (下面的阶段常量随之缩放).
WARM_START = True
WARM_START_RUN = "pollen-robotics/mjlab_microduck/441tzs6d"   # alpha_walking.onnx
WARM_START_CHECKPOINT = "model_3750.pt"

# 阶段边界 (PPO 迭代; 环境步计数器按 num_steps_per_env=24 缩放).
# 热启动: walk 在 iter 0 就存在, 所以 fell_over 只需要一个短的适应窗口
# 来适应全碰撞模型, 然后摔倒才成为数据.
FELL_OVER_DISABLE_ITER = 100 if WARM_START else 500

# 摔倒门控. 教训 (首次 rebase 训练运行): 恢复奖励必须仅门控在倾斜上. 同时
# 门控在低高度上使坐姿 (z≈0.07, 躯干直立) 开启门控 → policy 学会坐着刷
# upright_linear 同时为 com_upward_velocity 上下弹跳并通过 air_time 窗口抖腿.
# 门控正奖励在坏状态上奖励进入该状态. 倾斜 > 40° 无法从舒适姿态刷 — 你真的
# 翻倒了. 终止保留 z 条件, 使坐者和卡低环境被回收 (终止) 而非被支付.
REWARD_GATE_TILT_DEG = 40.0   # 恢复奖励: 摔倒 = 倾斜 > 40° 仅
# 终止 z 门控 0.08, 非 0.10 (run-3 教训): 正常摇摆直立机器人下到 z=0.084-0.096
# — 0.10 在早期学习包络内, 每 5 s 回收蹲行探索者. 0.08 仍捕获坐姿 (z≈0.07)
# 和俯卧 (z≈0.05).
TERM_GATE_Z = 0.08            # fallen_too_long: z < 0.08 或 倾斜 > 40°
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
RECOVERY_ECON_KICKIN_ITER = 600 if WARM_START else 1200

# Run-2 修复 (1bqctpkq, @1085 零恢复即使尝试税已减轻): 把已部署的 stand
# expert 蒸馏进摔倒帧 (见 distill.py). 该 expert 在此模型上 ~1 s 内恢复
# 100% 面朝下/上 和 95% 侧面.
ENABLE_EXPERT_BC = True
EXPERT_BC_COEF = 1.0
EXPERT_BC_GATE_TILT_DEG = 35.0

# Run-1 修复 (1): 摔倒时平滑税按比例缩小, 使起身尝试负担得起; 直立时全额
# (walk 的平滑不受影响).
FALLEN_SMOOTHNESS_SCALE = 0.1

# 舵机保护代价斜坡 (见 docstring). 从 step 0 起 25% 保持梯度存活; 恢复经济学
# 就位后全权重.
PROTECT_FULL_ITER = RECOVERY_ECON_KICKIN_ITER + 400
PROTECT_STAGE0_FRAC = 0.25
SERVO_IMPACT_WEIGHT = -0.02     # 每 N 超过 2 N 的舵机外壳, 每 step
HEAD_IMPACT_WEIGHT = -0.01      # 每 N 超过 15 N 的头部子树
TRUNK_IMPACT_WEIGHT = -0.01     # 每 N 超过 20 N 的躯干外壳
SERVO_ACC_SPIKE_WEIGHT = -1e-3  # 每 rad/s² 超过 300 (各舵机求和)
SERVO_STALL_WEIGHT = -0.05      # 每个卡死舵机每 step
GENTLE_RISE_WEIGHT = 0.005      # POSITIVE: trunk_vertical_accel_penalty 自取负
SERVO_IMPACT_THRESH_N = 2.0
HEAD_IMPACT_THRESH_N = 15.0
TRUNK_IMPACT_THRESH_N = 20.0
SERVO_ACC_THRESH = 300.0
SERVO_STALL_TORQUE = 0.4
SERVO_STALL_VEL = 0.5

# Topple 推挤: 足以推倒机器人的速度踢, 使摔倒 (以及保护性落地 + 恢复) 成为
# 密集的 on-policy 数据. 斜坡在 fell_over 关闭后开始 (之前的 topple 只是
# 一次 reset).
TOPPLE_PUSH_INTERVAL_S = (4.0, 8.0)
TOPPLE_PUSH_STAGES = [
    {"step": 0,                                          "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    {"step": FELL_OVER_DISABLE_ITER * NUM_STEPS_PER_ENV, "velocity_range": {"x": (-0.6, 0.6), "y": (-0.6, 0.6)}},
    {"step": 400 * NUM_STEPS_PER_ENV,                    "velocity_range": {"x": (-0.9, 0.9), "y": (-0.9, 0.9)}},
    {"step": 800 * NUM_STEPS_PER_ENV,                    "velocity_range": {"x": (-1.2, 1.2), "y": (-1.2, 1.2)}},
]

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
# 热启动: 同样的形状, 更早 (walk 不需要 bootstrap 窗口). Run-1 修复 (2):
# side_prob 把该比例放到侧面 (自然摔倒的占优终点状态, eval 中 79%); 其余按
# face_down_prob 划分. crouch 切片 0.15 → 0.20 且从 iter 150 起: 兼作站立
# 姿态数据.
_PRONE_ITERS = (150, 700, 1000, 1400) if WARM_START else (800, 1500, 2000, 2500)
PRONE_SIDE_PROB = 0.5
PRONE_RAMP_STAGES = [
    {"step": 0,                                   "params": {"prone_prob": 0.00, "face_down_prob": 1.0,  "side_prob": PRONE_SIDE_PROB, "crouch_prob": 0.00}},
    {"step": _PRONE_ITERS[0] * NUM_STEPS_PER_ENV, "params": {"prone_prob": 0.00, "face_down_prob": 1.0,  "side_prob": PRONE_SIDE_PROB, "crouch_prob": 0.20}},
    {"step": _PRONE_ITERS[1] * NUM_STEPS_PER_ENV, "params": {"prone_prob": 0.15, "face_down_prob": 0.80, "side_prob": PRONE_SIDE_PROB, "crouch_prob": 0.20}},
    {"step": _PRONE_ITERS[2] * NUM_STEPS_PER_ENV, "params": {"prone_prob": 0.30, "face_down_prob": 0.65, "side_prob": PRONE_SIDE_PROB, "crouch_prob": 0.20}},
    {"step": _PRONE_ITERS[3] * NUM_STEPS_PER_ENV, "params": {"prone_prob": 0.45, "face_down_prob": 0.50, "side_prob": PRONE_SIDE_PROB, "crouch_prob": 0.20}},
]


def _collapse_curricula_to_final(cfg: ManagerBasedRlEnvCfg) -> None:
    """Warm start: pin every inherited velocity curriculum at its final stage.

    Every velocity-recipe curriculum is a list of ``{"step": ..., ...}`` stage
    dicts under some ``*_stages`` param (weight_stages / standing_stages /
    range_stages / param_stages / push_stages). Keep only the last stage, at
    step 0, and — for reward_weight terms — also write the final weight into
    the reward cfg so step 0 itself already runs at the final value.
    """
    for name, term in cfg.curriculum.items():
        for key, val in list(term.params.items()):
            if isinstance(val, list) and val and all(isinstance(v, dict) and "step" in v for v in val):
                final = {**val[-1], "step": 0}
                term.params[key] = [final]
                if key == "weight_stages" and term.params.get("reward_name") in cfg.rewards:
                    cfg.rewards[term.params["reward_name"]].weight = final["weight"]


def make_microduck_velstand_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg:
    """构建 VelStand 环境 cfg (velocity 配方 + 摔倒恢复 + 身体姿态)."""
    # 行走层: 已验证的 velocity 配方, 逐字.
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)

    # play 模式下课程不运行, 所以下方摔倒终止禁用从不触发 — 直接删除终止.
    if play:
        cfg.terminations.pop("fell_over", None)

    # 热启动: 在添加 velstand 自己的课程 (仍须渐升) 之前, 把继承的 velocity
    # 课程钉在最终阶段.
    if WARM_START and not play:
        _collapse_curricula_to_final(cfg)

    # 真正的全碰撞模型: 机器人可以躺在/推开任何部件, 且舵机外壳已命名, 使
    # 下方的冲击传感器能单独识别它们. 70 个碰撞 geom (vs 11) → 堆叠时有更多
    # 同时接触; rough 地形的 nconmax (200) 在平坦上也是合适的预算.
    cfg.scene.entities = {"robot": MICRODUCK_ALLCOLLISIONS_ROBOT_CFG}
    cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 200)

    servo_ground_cfg = ContactSensorCfg(
        name="servo_ground_contact",
        primary=ContactMatch(mode="geom", pattern=rf"^.*{SERVO_GEOM_SUFFIX}$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("force",),
        reduce="netforce",
        num_slots=1,
    )
    head_ground_cfg = ContactSensorCfg(
        name="head_ground_contact",
        primary=ContactMatch(mode="body", pattern=rf"^({'|'.join(HEAD_BODY_NAMES)})$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("force",),
        reduce="netforce",
        num_slots=1,
    )
    trunk_ground_cfg = ContactSensorCfg(
        name="trunk_ground_contact",
        primary=ContactMatch(mode="body", pattern="^trunk_base$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("force",),
        reduce="netforce",
        num_slots=1,
    )
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (servo_ground_cfg, head_ground_cfg, trunk_ground_cfg)

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
    # standup 验证的抗抖动项: 惩罚力矩变化 (非幅度或旋转) → 平滑转移而不阻止
    # 恢复翻转. Run-1 修复 (1): 摔倒时两个平滑税 ×FALLEN_SMOOTHNESS_SCALE.
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torque_rate_l2_fallen_scaled,
        weight=-2e-3,
        params={"fallen_scale": FALLEN_SMOOTHNESS_SCALE, "gate_tilt_above_deg": REWARD_GATE_TILT_DEG},
    )
    ar = cfg.rewards["action_rate_l2"]
    cfg.rewards["action_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.action_rate_l2_fallen_scaled,
        weight=ar.weight,
        params={"fallen_scale": FALLEN_SMOOTHNESS_SCALE, "gate_tilt_above_deg": REWARD_GATE_TILT_DEG},
    )

    # ── Servo-protection costs (see module docstring) ─────────────────────────
    # 2026-07 lesson kept: the head penalty at -1.0 @ 2 N taxed the push-off-
    # with-the-head recovery. These thresholds sit ABOVE quasi-static push-off
    # loads (≈ body weight, 7 N) so only impact spikes are billed. All start at
    # PROTECT_STAGE0_FRAC and ramp to full at PROTECT_FULL_ITER.
    cfg.rewards["servo_impact"] = RewardTermCfg(
        func=microduck_mdp.body_impact_cost,
        weight=SERVO_IMPACT_WEIGHT * PROTECT_STAGE0_FRAC,
        params={"sensor_name": servo_ground_cfg.name, "threshold": SERVO_IMPACT_THRESH_N},
    )
    cfg.rewards["head_impact"] = RewardTermCfg(
        func=microduck_mdp.body_impact_cost,
        weight=HEAD_IMPACT_WEIGHT * PROTECT_STAGE0_FRAC,
        params={"sensor_name": head_ground_cfg.name, "threshold": HEAD_IMPACT_THRESH_N},
    )
    cfg.rewards["trunk_impact"] = RewardTermCfg(
        func=microduck_mdp.body_impact_cost,
        weight=TRUNK_IMPACT_WEIGHT * PROTECT_STAGE0_FRAC,
        params={"sensor_name": trunk_ground_cfg.name, "threshold": TRUNK_IMPACT_THRESH_N},
    )
    cfg.rewards["servo_acc_spike"] = RewardTermCfg(
        func=microduck_mdp.servo_acc_spike_penalty,
        weight=SERVO_ACC_SPIKE_WEIGHT * PROTECT_STAGE0_FRAC,
        params={"acc_thresh": SERVO_ACC_THRESH},
    )
    cfg.rewards["servo_stall"] = RewardTermCfg(
        func=microduck_mdp.servo_stall_penalty,
        weight=SERVO_STALL_WEIGHT * PROTECT_STAGE0_FRAC,
        params={"torque_thresh": SERVO_STALL_TORQUE, "vel_thresh": SERVO_STALL_VEL},
    )
    # ⚠️ POSITIVE weight: trunk_vertical_accel_penalty returns -|a_z| already.
    cfg.rewards["gentle_rise"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=GENTLE_RISE_WEIGHT * PROTECT_STAGE0_FRAC,
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
            "side_prob": PRONE_SIDE_PROB,
            "prone_z_min": 0.05,
            "prone_z_max": 0.09,
            "crouch_prob": 0.0,  # 由 prone_init_prob 课程渐升
        },
    )

    # Topple 推挤 (docstring): velocity 环境的 push_robot (±0.3 m/s) 训练踉跄
    # 恢复; 这一个训练摔倒. 范围由 topple_push_range 渐升.
    cfg.events["topple_push"] = EventTermCfg(
        func=push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 2.0) if play else TOPPLE_PUSH_INTERVAL_S,
        params={
            "velocity_range": TOPPLE_PUSH_STAGES[-1 if play else 0]["velocity_range"],
            "asset_cfg": SceneEntityCfg("robot"),
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
    if not play:
        cfg.curriculum["topple_push_range"] = CurriculumTermCfg(
            func=microduck_mdp.push_curriculum,
            params={"event_name": "topple_push", "push_stages": TOPPLE_PUSH_STAGES},
        )
    for name, full in (
        ("servo_impact", SERVO_IMPACT_WEIGHT),
        ("head_impact", HEAD_IMPACT_WEIGHT),
        ("trunk_impact", TRUNK_IMPACT_WEIGHT),
        ("servo_acc_spike", SERVO_ACC_SPIKE_WEIGHT),
        ("servo_stall", SERVO_STALL_WEIGHT),
        ("gentle_rise", GENTLE_RISE_WEIGHT),
    ):
        cfg.curriculum[f"{name}_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": name,
                "weight_stages": [
                    {"step": 0, "weight": full * PROTECT_STAGE0_FRAC},
                    {"step": PROTECT_FULL_ITER * NUM_STEPS_PER_ENV, "weight": full},
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
    algorithm=PpoWithExpertBcCfg(
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
        bc_cfg={**default_bc_cfg(), "coef": EXPERT_BC_COEF, "gate_tilt_deg": EXPERT_BC_GATE_TILT_DEG} if ENABLE_EXPERT_BC else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="velstand",
    run_name="velstand",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=6_000,
)
