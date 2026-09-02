"""Microduck 滚轮起立 — 在滚轮上自行起立.

专用 episodic policy: 机器人从地面启动 (趴卧 / 仰卧) 或已站立, 需在滚轮上
自行起立并保持站立. 将 `standup` (步履鸭) 的配方移植到 rollers 模型.

派生自 roller env (`make_microduck_velocity_rollers_env_cfg`) → 原样继承滚轮
机器人, 传感器, 全部 DR 和 61D 观测, 因此可在 runtime 互换 (--new-cmd-obs).
与 roller_slope 的模式一致.

与 `standup` 的两个结构性差异:
  - 被动轮子在关节顺序中是 INTERLEAVED → 索引需重映射
    (下方 _LEG_JOINTS), 由 tests/test_roller_standup_cfg.py 锁定;
  - 无 head_pose 指令: head/body 槽位保持零填充 (滚轮家族的约定), 头部
    由 neck_joint_pos_l2 按名解析保持正立.

新增的核心是滚动摩擦课程, 反向 (轮子制动 → 自由): 轮子会滚, 因此无任何
纵向抓地力可推地面. 用近乎锁死的轮子做引导, 然后逐步 ramp 到真实滚动
物理. 若 `standing_composite` 在某个阶段崩溃, "脚踏式" 动作无法迁移到
自由轮, 需要引入滑冰者技巧 (膝撑, 一次一只滑板).

目标部署: 在 `--standing` 时面对 `--walking` 的 roller policy, 由速度指令
幅值自动切换 (infer_policy.py:262, 阈值 0.05); twist 槽位保持为零
(infer_policy.py:239).
"""

import math
import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    RewardTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_rollers_env_cfg import (
    make_microduck_velocity_rollers_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

# ── trunk 高度 (m) ─────────────────────────────────────────────────────
# 通过精确运动学测量 (碰撞 geom 网格顶点的最小值, STAND 姿态, trunk 贴地) 在
# scene_rollers.xml 上测得: 站立 0.1407, 趴卧 0.0752, 仰卧 0.0475.
# 对照: 无轮子模型运动学测得 0.1172, 而 standup 负载下测得 STAND_Z=0.115 →
# ~2 mm 下沉, 此处同样适用.
# 0.138 落在 roller env 已使用的 reset_base z 范围 (0.1335–0.1435) 内.
ROLLER_STAND_Z = 0.138
ROLLER_PRONE_Z = 0.075

EPISODE_LENGTH_S = 6.0  # 起立 + 稳定, 同 standup
NUM_STEPS_PER_ENV = 24

# ── play 覆盖: 强制仰卧启动比例 ───────────────────────────────────────
# play 时 env 重新构建: common_step_counter 归零, ground_state_mix 课程应用
# 阶段 0, 其中 face_up_prob = 0. 因此 play 时从不出现仰卧启动 — 而这恰恰是
# 最难的情形, 也是想人工检查的. 该变量强制其出现.
#   STANDUP_PLAY_FACE_UP=1.0  -> 100% 仰卧启动
#   STANDUP_PLAY_FACE_UP=0.4  -> 课程最后阶段的混合比例
#   未定义 / "none" / "random" -> 默认行为 (阶段 0)
# 仅在 play=True 时生效. 与 roller_slope 的 SLOPE_PLAY_DIFFICULTY 同理.
PLAY_FACE_UP = None
# 课程最后阶段的 趴卧:站立 比例 (0.40 / 0.20 = 2:1). 余下 (1 - face_up) 按
# 此比例分配, 使 0.4 可精确复现训练末期的混合.
_PLAY_FACE_DOWN_SHARE = 2.0 / 3.0


def _resolve_play_face_up():
    """play 时仰卧启动比例: 优先读 STANDUP_PLAY_FACE_UP 环境变量, 否则用常量."""
    raw = os.environ.get("STANDUP_PLAY_FACE_UP")
    if raw is None:
        return PLAY_FACE_UP
    raw = raw.strip().lower()
    if raw in ("", "none", "random"):
        return None
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        print(f"[roller_standup] STANDUP_PLAY_FACE_UP='{raw}' 无效 -> 默认 {PLAY_FACE_UP}")
        return PLAY_FACE_UP


# ── 关节索引 — 被动轮子是 INTERLEAVED ───────────────────────────────
# rollers 模型的真实顺序 (free-joint 后 18 个关节), 在 MuJoCo 中通过
# get_walk_rollers_spec().compile() 验证:
#   0-4   left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle
#   5-6   passive_LF_wheel, passive_LR_wheel
#   7-10  neck_pitch, head_pitch, head_yaw, head_roll
#   11-15 right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee, right_ankle
#   16-17 passive_RF_wheel, passive_RR_wheel
# standup 使用 [0-4, 9-13] / [5-8]: 这些是无轮子模型的索引, 在此处无效.
# 由 tests/test_roller_standup_cfg.py 锁定.
#
# 仅 _LEG_JOINTS 被消费 (姿态奖励使用). _NECK_JOINTS 和 _WHEEL_JOINTS 用于
# 文档与索引测试: 颈部按名解析 (neck_joint_pos_l2 每步调用
# find_joints(r".*(neck|head).*"), 轮子用 ^passive_.* 正则).
_LEG_JOINTS = [0, 1, 2, 3, 4, 11, 12, 13, 14, 15]
_NECK_JOINTS = [7, 8, 9, 10]
_WHEEL_JOINTS = [5, 6, 16, 17]

# roller env 的滑行奖励: 在地上时无意义.
# feet_flat: 起立期间刀片并非平放 → 会对抗动作.
# hip_roll_neutral: 起立需要张开双腿.
# pose / com_height_target: 由起立的姿态/高度目标替代.
# upright (基础高斯): 由 upright_linear + upright_sharp 替代.
_SKATING_REWARDS = (
    "wheel_speed",
    "braking",
    "skating_air_time",
    "glide",
    "single_support",
    "gait_symmetry",
    "forward_lean",
    "heading_hold",
    "feet_flat",
    "hip_roll_neutral",
    "pose",
    "com_height_target",
    "upright",
)


def make_microduck_roller_standup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """滚轮上的起立 env: 地面启动, 目标 = 在轮子上站立."""
    cfg = make_microduck_velocity_rollers_env_cfg(play=play)

    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── 移除滑行奖励 ─────────────────────────────────────────────────
    for name in _SKATING_REWARDS:
        cfg.rewards.pop(name, None)

    # ── 指令: twist 槽位中和 (≈ 0) ───────────────────────────────────
    # roller env 装配 RelativeHeadingVelocityCommandCfg (cmd[2] = 内部计算的
    # 航向误差). 此处无需操控: 回到 standup 那样的 command-only 中和. head_pose
    # (4) 和 body_pose (6) 槽位保持零填充 → 61D obs 对等性保留.
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.heading_command = False
    command.ranges.heading = None
    command.resampling_time_range = (EPISODE_LENGTH_S, EPISODE_LENGTH_S * 2)
    command.debug_vis = False
    command.ranges.lin_vel_x = (-0.01, 0.01)
    command.ranges.lin_vel_y = (-0.01, 0.01)
    command.ranges.ang_vel_z = (-0.05, 0.05)
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))

    # ── 数值鲁棒性 (同 roller_slope 的选择) ───────────────────────────
    # 罕见接触 (~1/25M 步) 会使 free-joint 发散为 NaN: 清理 obs (→ 0) 避免终止
    # 训练, 故障 env 在下一步 reset.
    for grp in ("actor", "critic"):
        cfg.observations[grp].nan_policy = "sanitize"

    # ── 起立奖励 — 从 standup 移植, 重映射 ──────────────────────────
    # 权重来自 microduck_standup_env_cfg.py 中记录的迭代过程: 非有理由不调整.
    # 此处仅关节索引和两个高度值不同.
    # 注意: 每个术语用 NEW SceneEntityCfg — mjlab 会就地解析并修改, 共享对象
    # 会给出过时索引.

    # 目标姿态 = HOME (target_overrides=None), 仅腿部: 颈和头由继承的
    # neck_joint_pos_l2 按名解析保持.
    cfg.rewards["pose_stand_legs"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match,
        weight=8.0,
        params={
            "std": 0.5,
            "joint_indices": _LEG_JOINTS,
            "target_overrides": None,
        },
    )
    # Bootstrap L1: 远离 HOME 时高斯饱和, 提供恒定梯度.
    cfg.rewards["pose_stand_l1"] = RewardTermCfg(
        func=microduck_mdp.pose_l1_penalty,
        weight=5.0,
        params={
            "joint_indices": _LEG_JOINTS,
            "target_overrides": None,
        },
    )

    # 高度分三层: 宽高斯 (从地面拉起), 窄高斯 (在宽高斯饱和时强迫最后几厘米),
    # 以及强 L1 使 "留在地面" 净 NÉGATIF — 没有它, policy 会满足于 "地面静止"
    # 的懒惰最优.
    cfg.rewards["height_stand"] = RewardTermCfg(
        func=microduck_mdp.height_target_gaussian,
        weight=4.0,
        params={
            "std": 0.04,
            "target_height": ROLLER_STAND_Z,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["height_stand_sharp"] = RewardTermCfg(
        func=microduck_mdp.height_target_gaussian,
        weight=4.0,
        params={
            "std": 0.015,
            "target_height": ROLLER_STAND_Z,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["height_stand_l1"] = RewardTermCfg(
        func=microduck_mdp.height_l1_penalty,
        weight=30.0,
        params={
            "target_height": ROLLER_STAND_Z,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # 奖励上升的 MOVEMENT, 不只是终点: 没有它, "坐着收部分姿态奖励" 占优.
    # 截断高度在目标之上 10 mm, 否则 policy 会停在截断高度, 不完成起立.
    cfg.rewards["com_upward_velocity"] = RewardTermCfg(
        func=microduck_mdp.com_upward_velocity,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            "max_height": ROLLER_STAND_Z + 0.010,
        },
    )
    # 柔和上升: 惩罚 |a_z|. 与 com_upward_velocity 兼容 — 恒定垂直速度同时
    # 拿这一项且 a_z = 0 → 两项压力共同选择匀速平滑上升.
    #
    # ⚠️ 权重为正, 不是笔误. mdp.py 混用两种符号约定:
    # trunk_vertical_accel_penalty 已返回 -|a_z| (mdp.py:2171), 同
    # height_l1_penalty 和 pose_l1_penalty — 后者此处也以 +30 和 +5 的正权重
    # 使用. 继承自 standup 的 -0.02 因此是双重负号, 实际 RÉCOMPENSait 垂直
    # 加速度: 在 vweolw91 run 上测得 Episode_Reward/gentle_rise = +0.0118
    # (唯一日志为正的惩罚项). 这是 "非常暴力" 的原因, 也解释了 standup 中
    # 记录的那些失败阻尼尝试 — 它们在对抗一个主动反向推动的项.
    #
    # 幅度保持 0.02 (原本就是想要的值) 故意取小: 从仰卧翻转时 |a_z| 必然高,
    # 此处大权重会成为动作阻断. 真正的阻尼由 joint_torque_rate_l2 承担, 它
    # 惩罚扭矩的 VARIATION 而非动作本身.
    cfg.rewards["gentle_rise"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=+0.02,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    # trunk 垂直分两层: 躺平时 cos(tilt) 梯度大但接近垂直时变弱; 高度门控的
    # 窄高斯接管, 抑制后倾 (standup 的失败模式: 伸直腿向后翻).
    cfg.rewards["upright_linear"] = RewardTermCfg(
        func=microduck_mdp.body_upright_linear,
        weight=6.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    cfg.rewards["upright_sharp"] = RewardTermCfg(
        func=microduck_mdp.upright_gaussian_at_height,
        weight=6.0,
        params={
            "std": 0.3,
            "height_low": ROLLER_PRONE_Z,
            "height_high": ROLLER_STAND_Z,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # 乘性分数 高度 × 垂直度 × 姿态: 因子相乘, 三项中只做好两项无收益 →
    # 打破加性奖励允许的 "正确高度但倾斜" 妥协. std 故意取 LARGES 使起立
    # 期间可见 (过窄的 std 给出 ~5e-5 分, 即零梯度).
    cfg.rewards["standing_composite"] = RewardTermCfg(
        func=microduck_mdp.standing_composite_score,
        weight=15.0,
        params={
            "target_height": ROLLER_STAND_Z,
            "height_std": 0.04,
            "upright_std": 0.40,
            "pose_std": 0.40,
            "joint_indices": _LEG_JOINTS,
            "target_overrides": None,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # 抗抖动: 惩罚扭矩 VARIATION, 而非其幅度或 trunk 旋转 → 阻尼抖动而不
    # 阻断翻转. standup 确认这是唯一不杀仰卧起立的阻尼项, 因此是可放心调高的
    # 杠杆.
    #
    # -2e-3 (standup 继承值) 在饱和 95-99% 任务奖励 ~+41.6 面前仅贡献
    # -0.0002/步 — 即毫无作用. 所有阻尼合计与任务之比约 35:1, 完全没理由
    # 温和. 在 vweolw91 run 第 7500 iter 测得.
    #
    # 重标定: 收敛时 |Δτ|² 原始值 ~0.1, 故贡献 ≈ 0.1 × |权重|. 在权重 -2.0
    # (run d8rnko6p) 时测得 -0.255/步 — 因此并非冻结的元凶, 但仍下调至 -0.2
    # 以腾出阻尼预算, 单独隔离符号 bug 的影响. 若仍过暴力, 调高此项 (按上
    # 述公式) 而非 body_ang_vel 或 action_rate, 它们是动作阻断, 会冻结仰卧
    # 起立.
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torque_rate_l2,
        weight=-0.2,
    )

    # 不加头部冲击惩罚. 用 velstand 的值尝试过 (body_impact_cost, `neck`
    # 子树, 权重 -1.0, 阈值 2.0): policy 收敛到躺平不动. 实测 (run d8rnko6p):
    # head_impact_penalty -1.01/步, 表中最大负项, 同时 standing_composite
    # 从 +14.3 崩到 +3.3.
    #
    # 推理错误在于认为 "定向" 惩罚不会限制动作. 此处错误: 从仰卧起立时,
    # 此机器人以头和肩为支点 PIVOTE. 头是翻转的支点, 不是附带伤害 — 惩罚它
    # 即阻断唯一可用机制, 而仰卧本就是失败的 case.
    #
    # 正在测试的假设: 撞头是暴力的 SYMPTÔME (gentle_rise 的符号 bug 在奖励
    # 暴力, 暴力起立会以头着地结束), 而非独立缺陷. 若符号修正后撞头重现,
    # 应使用高度门控的惩罚 — 如 upright_sharp 那样 — 以放过地面翻转阶段.
    #
    # ⚠️ 注意使此冻结成为可能的懒惰最优: 机器人躺平时 pose_stand_legs
    # 仍达 +7.72/8 (躺姿下腿在 HOME 位置 → 几乎免费拿到的奖励). 应由
    # height_stand_l1 (权重 +30) 让 "留在地面" 净负.

    # ── 地面启动: 趴卧 / 仰卧 / 已站立 ─────────────────────────────
    # 在 cfg.events 中最后添加: 执行顺序按插入顺序, 此项需覆盖 reset_base /
    # reset_robot_joints 设定的姿态.
    # "已站立" 桶不是装饰: 没它 policy 学会起立但不会 TENIR, 起立后立即摔倒.
    # 无 "坐着" 桶 → 无需重映射 sitting_joint_overrides (standup 的是无轮子
    # 模型的索引).
    # 下方概率 = ground_state_mix 课程阶段 0.
    cfg.events["set_ground_state"] = EventTermCfg(
        func=microduck_mdp.set_random_ground_state,
        mode="reset",
        params={
            "face_down_prob": 0.50,  # 趴卧 (+90° pitch)
            "face_up_prob": 0.00,  # 仰卧 — 最难, 后期引入
            "sitting_prob": 0.00,
            "standing_prob": 0.50,
            "sitting_joint_overrides": None,
            # 两个起始姿态 (趴卧/仰卧) 共享单一 z 范围, 而它们的接触毫无共同:
            # 趴卧从 0.0752 才离地, 仰卧静息于 0.0475. 单一下限无法对两者都
            # 理想. 取 0.076 以消除趴卧侧的穿透 (测得: 0.05 时入地 +25 mm),
            # 代价是仰卧起始高出静息 28-42 mm — 比接触推压柔和得多的伪影.
            "prone_z_min": 0.076,
            "prone_z_max": 0.09,
            # 轮上站立: ROLLER_STAND_Z = 0.138 (无轮时 0.11-0.12).
            "standing_z_min": 0.134,
            "standing_z_max": 0.144,
            # 起始 pitch/roll 噪声. 注意: set_random_ground_state 中 "站立" 桶
            # 复用 "坐着" 桶的四元数, 故该噪声也作用于站立启动 — 这是刻意的
            # (避免对完美正立过拟合).
            "sitting_tilt_max": math.radians(10),
        },
    )

    # 机器人从摔倒启动 → 倾斜终止在此无意义 (会在第一步就结束 episode).
    # 继承的 nan_state 保留.
    cfg.terminations.pop("fell_over", None)

    # 起始姿态课程, 易 → 难. 一开始就混合会导致 policy 优化多数简单 case,
    # 仰卧欠训练 (standup 的教训: 它在该姿态上冻结为 "什么都不做"). 故先引入
    # 站立+趴卧, 仰卧后引入, 末期偏向难姿态使其获得最多训练.
    cfg.curriculum["ground_state_mix"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_curriculum,
        params={
            "event_name": "set_ground_state",
            "param_stages": [
                {
                    "step": 0,
                    "params": {
                        "standing_prob": 0.50,
                        "sitting_prob": 0.00,
                        "face_down_prob": 0.50,
                        "face_up_prob": 0.00,
                    },
                },
                {
                    "step": 600 * NUM_STEPS_PER_ENV,
                    "params": {
                        "standing_prob": 0.35,
                        "sitting_prob": 0.00,
                        "face_down_prob": 0.45,
                        "face_up_prob": 0.20,
                    },
                },
                {
                    "step": 1500 * NUM_STEPS_PER_ENV,
                    "params": {
                        "standing_prob": 0.25,
                        "sitting_prob": 0.00,
                        "face_down_prob": 0.40,
                        "face_up_prob": 0.35,
                    },
                },
                {
                    "step": 2500 * NUM_STEPS_PER_ENV,
                    "params": {
                        "standing_prob": 0.20,
                        "sitting_prob": 0.00,
                        "face_down_prob": 0.40,
                        "face_up_prob": 0.40,
                    },
                },
            ],
        },
    )

    # play 覆盖: 强制仰卧启动以便检查. 在事件中写概率, 并删除课程 — 否则
    # event_param_curriculum (在 reset 事件之前运行) 会在第一次 reset 就用
    # 阶段 0 改写. 仅 play 时生效, 因此训练及其 易→难 课程不受影响.
    if play:
        play_face_up = _resolve_play_face_up()
        if play_face_up is not None:
            remainder = 1.0 - play_face_up
            cfg.events["set_ground_state"].params.update(
                {
                    "face_up_prob": play_face_up,
                    "face_down_prob": remainder * _PLAY_FACE_DOWN_SHARE,
                    "standing_prob": remainder * (1.0 - _PLAY_FACE_DOWN_SHARE),
                    "sitting_prob": 0.00,
                }
            )
            del cfg.curriculum["ground_state_mix"]

    # ── 滚动摩擦反向: 制动 → 自由 ───────────────────────────────────
    # 这是本 env 真正新增的部分, 也是难点的核心: 轮子会滚, 因此纵向无任何
    # 抓地力推地面. roller env 让此摩擦 MONTER (0 → 0.0015); 此处让其
    # DESCENDRE, 以在简单问题 (近乎锁死的轮子 ≈ 脚) 上 bootstrap 动作,
    # 再施加真实滚动物理.
    #
    # 待观察诊断: 若 Episode_Reward/standing_composite 在某阶段崩塌, "脚踏式"
    # 动作无法迁移到自由轮 → 需引入滑冰者技巧 (中间膝撑, 一次一只滑板).
    # 这是可用的结果, 不是失败.
    #
    # 注意 sim2real: 仅最后阶段 (iter 4000+) 之后的 checkpoint 才是部署候选.
    # 在此之前, policy 依赖真机上不存在的滚动摩擦.
    _WHEEL_FRICTION_STAGE0 = (0.0500, 0.0500)
    cfg.curriculum["wheel_friction"] = CurriculumTermCfg(
        func=microduck_mdp.wheel_friction_curriculum,
        params={
            "event_name": "randomize_wheel_friction",
            "ranges_stages": [
                {"step": 0, "ranges": _WHEEL_FRICTION_STAGE0},
                {"step": 1000 * NUM_STEPS_PER_ENV, "ranges": (0.0200, 0.0200)},
                {"step": 2000 * NUM_STEPS_PER_ENV, "ranges": (0.0080, 0.0080)},
                {"step": 3000 * NUM_STEPS_PER_ENV, "ranges": (0.0030, 0.0030)},
                {"step": 4000 * NUM_STEPS_PER_ENV, "ranges": (0.0015, 0.0015)},
            ],
        },
    )
    # 防御性冗余: 课程管理器在每次 reset (含第一次) 时先于 reset 事件运行,
    # 且 wheel_friction_curriculum 自身默认到阶段 0 — 实际上这行从不必要.
    # 仅保持事件 PAR DÉFAUT 值与课程阶段 0 一致, 以防日后有人删除课程但
    # 保留事件.
    cfg.events["randomize_wheel_friction"].params["ranges"] = _WHEEL_FRICTION_STAGE0

    # ── action_rate: standup 的 ramp, 而非 roller 的 ─────────────────
    # roller env 升到 -2.0 以求步态平稳. 这是动作阻断: 减慢仰卧起立所需的
    # 快速动作 (standup 记录过强 action_rate 杀死此恢复). 此处的柔和由
    # joint_torque_rate_l2 承担.
    cfg.rewards["action_rate_l2"].weight = -0.6
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.4},
                {"step": 250 * NUM_STEPS_PER_ENV, "weight": -0.8},
                {"step": 500 * NUM_STEPS_PER_ENV, "weight": -1.0},
            ],
        },
    )

    # ── ramp 推力 ───────────────────────────────────────────────────
    # push_robot 继承自 roller env (±0.2 m/s, 每 3-6 s) 但无课程. 第 0 步就
    # 推一下会干扰起立 bootstrap: 像 standup 那样逐步引入.
    cfg.curriculum["push_magnitude"] = CurriculumTermCfg(
        func=microduck_mdp.push_curriculum,
        params={
            "event_name": "push_robot",
            "push_stages": [
                {"step": 0, "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
                {
                    "step": 500 * NUM_STEPS_PER_ENV,
                    "velocity_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08)},
                },
                {
                    "step": 1000 * NUM_STEPS_PER_ENV,
                    "velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                },
            ],
        },
    )

    return cfg


# ── RL runner 配置 — 与 standup 相同 ──────────────────────────────────
MicroduckRollerStandUpRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # normaliseur 必须由 export.py 烘焙进 ONNX
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
        # 对称性 OFF: SYMMETRY_CFG 为旧的 51D 布局硬编码, 在 61D 上会破裂
        # (与所有 v1.5+ env 相同情况).
        symmetry_cfg=None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="roller_standup",
    run_name="roller_standup",
    save_interval=250,
    num_steps_per_env=NUM_STEPS_PER_ENV,
    max_iterations=15_000,
)
