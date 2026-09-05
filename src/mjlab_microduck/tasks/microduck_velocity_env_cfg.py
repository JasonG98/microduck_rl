"""Microduck velocity (行走) 环境.

主行走任务: 速度指令跟踪 + 头部姿态指令.
奖励/正则化方案以行走为核心 (前倾跟踪 + 步态/足部项, 课程式渐升的 action-rate 平滑), 包含:

  - foot_slip 保持 -0.1 (刻意偏弱 — 更强对该机器人以转体为主的转弯过于严苛)
  - 固定、适中的指令范围 (ang ±1.0 让转弯可学) 替代超出机器人能力的渐扩课程
  - turn-in-place: 15% 的环境使用 lin=0 + |ang| ∈ [0.4, 1.0] (2026-07 审计:
    独立均匀采样使原地旋转仅占数据 ~2% → 学不到)
  - head_pose_tracking 作为主目标, 加上基于 EMA 的 head_pose_bias 惩罚,
    仅对可逃避的 DC 头部下垂计价 (见下文)
  - body_pose 跟踪基础设施保留但禁用 (权重 0), 以保持 obs 槽位对使用它的环境活跃
"""

import math
from copy import deepcopy

NUM_STEPS_PER_ENV = 24

# 被指令原地旋转的环境比例 (lin=0, |ang| ∈ [0.4·max, max]).
TURN_IN_PLACE_FRACTION = 0.15

# 对称性
ENABLE_SYMMETRY = False

# Domain randomization 开关
ENABLE_COM_RANDOMIZATION = True
ENABLE_HEAD_COM_RANDOMIZATION = True  # 随机化头部组件各 body 的 CoM
ENABLE_KP_RANDOMIZATION = False  # 曾为 True
ENABLE_KD_RANDOMIZATION = False  # 曾为 True
ENABLE_MASS_INERTIA_RANDOMIZATION = True  # 行走稳定后可启用
ENABLE_JOINT_FRICTION_RANDOMIZATION = True  # 通过 FrictionDRBamActuator.friction_scale 按环境缩放 BAM 的摩擦预算
ENABLE_JOINT_DAMPING_RANDOMIZATION = False
ENABLE_ARMATURE_RANDOMIZATION = True  # 反射转子惯量 (microban 风格). 对 BAM 有效 (armature 被设置, 未被清零).
ENABLE_VELOCITY_PUSHES = True  # 基于速度的推力, 用于鲁棒性训练
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True  # 模拟安装误差
ENABLE_ENCODER_BIAS = True  # 每环境关节编码器标定偏移 (actor obs 看到 joint_pos + bias)
ENABLE_BASE_ORIENTATION_RANDOMIZATION = False  # 随机化初始倾斜以强制反应式行为

# 头部/身体姿态指令跟踪 (替代旧的颈部偏移扰动方案).
# 头部姿态: neck/head 关节相对 HOME 的 4D 增量; vel 环境将其作为主目标跟踪.
# 身体姿态: [x, y, z, roll, pitch, yaw] 的 6D 增量; vel 环境采样小范围 + 微小奖励权重,
# 以保持输入神经元活跃但跟踪不是重点 (standup 环境会提高权重).
HEAD_POSE_CMD_RESAMPLE_S = (2.0, 5.0)
BODY_POSE_CMD_RESAMPLE_S = (2.0, 5.0)

# 观测配置
USE_PROJECTED_GRAVITY = True  # 为 True 时使用 projected gravity 而非原始加速度计

# Domain randomization 范围 (按需调整)
# 已证明稳定的保守范围 - 如需要可逐渐加大
COM_RANDOMIZATION_RANGE = 0.003  # ±3mm 初始, 通过课程渐升到 ±8mm
# 头部 CoM 随机化: 每个 episode 应用于头部组件的每个 body
# (neck → neck_pitch → yaw_roll_motion → head-roll body). 与上面的躯干 CoM 随机化
# 使用相同的非累积机制. head-roll body 在 walk 模型中名为 bottom_head_shell,
# 在 2026-07 roller 模型中名为 jaw_soft, 因此这里交替使用. 注意: bearing_roll
# 不是头部 body — 在两个模型中它都是右髋 yaw 连杆 (trunk_base 的子级); 此前列在此处
# 纯属失误, 仅保留以维持现有 DR 行为.
HEAD_COM_RANDOMIZATION_RANGE = 0.003  # ±3mm 初始, 通过课程渐升
HEAD_BODY_NAMES = (
    "neck",
    "neck_pitch",
    "yaw_roll_motion",
    "(bottom_head_shell|jaw_soft)",
    "bearing_roll",
)
MASS_INERTIA_RANDOMIZATION_RANGE = (
    0.95,
    1.05,
)  # ±5% 同时应用于质量和惯量.
KP_RANDOMIZATION_RANGE = (0.85, 1.15)  # ±15%
KD_RANDOMIZATION_RANGE = (0.9, 1.1)  # ±10% (可加大到 0.8-1.2)
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.9, 1.1)
JOINT_DAMPING_RANDOMIZATION_RANGE = (0.9, 1.1)
ARMATURE_RANDOMIZATION_RANGE = (
    0.9,
    1.1,
)  # ±10% 反射转子惯量 (microban: dr.joint_armature, 同范围)
VELOCITY_PUSH_INTERVAL_S = (3.0, 6.0)  # 每 3-6 秒施加一次推力
VELOCITY_PUSH_RANGE = (-0.3, 0.3)  # 速度变化范围 (m/s). 曾为 ±0.5 — 每 3-6 秒一次
# 大于最大行走速度 (0.4) 的 ADDITIVE 冲击会训练出永久紧张的摔倒恢复步态
# (2026-07 审计). ±0.3 保留推力鲁棒性同时让更平稳的步态成为最优.
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0  # 最高 6° 随机轴 IMU 安装误差. 注意: 零中心 (随机轴) — 训练对失准 *幅度* 的容忍, 而非俯仰偏置. 真实电路板的系统性 ~5° 俯仰偏置在 runtime 源头修正 (imu-pitch-offset), 不在此处.
ENCODER_BIAS_RANGE = (
    -0.015,
    0.015,
)  # ±0.86° 每关节编码器偏移 (每环境常量)
BASE_ORIENTATION_MAX_PITCH_DEG = 10.0  # episode 开始时 ±10° 前后倾斜
BASE_ORIENTATION_MAX_ROLL_DEG = 5.0  # episode 开始时 ±5° 左右倾斜

import mjlab.terrains as terrain_gen
import mujoco as _mujoco
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
)
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.symmetry import SYMMETRY_CFG, PpoWithSymmetryCfg

# Microduck 专用粗糙地形: 比默认 ROUGH_TERRAINS_CFG 平缓得多.
# 机器人抬脚仅 ~1-2 cm, 台阶上限 1.5 cm.
MICRODUCK_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    sub_terrains={
        "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.25),
        "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.0, 0.015),  # 上限 1.5 cm (默认 10 cm)
            step_width=0.15,
            platform_width=2.0,
            border_width=1.0,
        ),
        # 注意: BoxInvertedPyramidStairsTerrainCfg 已移除 — 它将 env_origin_z 设为坑
        # 底 (负值), 导致 root_z = 0.12 + env_origin_z ≈ −0.10 m 的重置, 把机器人放
        # 在坑底下方, 使其穿地掉落.
        # 不平整鹅卵石状地面: 每格随机高度偏移.
        # grid_width=0.12 在 8m 块上 = 66×66 = 4 356 个 box/块 → 总 ~261 K → OOM.
        # 0.45 m 给 17×17 = 289 个 box/块 → 总 ~17 K (border = 0.35 m ✓).
        # 不能整除地形尺寸 (8.0 m): 0.45 × 17 = 7.65 ✓
        "random_grid": terrain_gen.BoxRandomGridTerrainCfg(
            proportion=0.30,
            grid_width=0.45,
            grid_height_range=(0.0, 0.010),  # 上限 1 cm
            platform_width=1.5,
        ),
        # 平缓坡道 (高场金字塔, 平台在顶 — 机器人生于平平台上, 随指令重采样
        # 沿坡道下/上/横走). slope_range 为 rise/run: 0.03→0.10 ≈ 1.7°→5.7° 按
        # 难度 — 小机器人, 小坡度. 非倒置 (见上 inverted-pyramid env_origin 注 —
        # 同类坑生风险).
        # vertical_scale=0.001 将量化步长保持 1 mm, 使平缓坡道平滑而非 5 mm 阶梯.
        "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.20,
            slope_range=(0.03, 0.10),
            platform_width=2.0,
            vertical_scale=0.001,
        ),
    },
    add_lights=False,
)


def _soften_terrain_contacts(spec: _mujoco.MjSpec) -> None:
    """软化地形 box geom 接触, 以降低边接触 NaN 不稳定性.

    Box 地形将相邻 geom 放在不同高度. 高度变化处的硬边在脚落在其上时引起
    接触法线不稳定性, 可在 MuJoCo 求解器中产生冲击式 NaN 力.

    将 solref 时间常数加倍 (0.02 → 0.04 s) 使接触弹簧变软 2 倍 — 足以阻尼不稳定,
    同时不明显改变宏观行走物理. 应用于 "terrain" body 中的所有 geom,
    该 body 包含 TerrainGenerator 生成的每一个 box.
    """
    body = spec.body("terrain")
    count = 0
    for geom in body.geoms:
        geom.solref = [0.04, 1.0]  # 时间常数软 2 倍 (默认: 0.02)
        geom.solimp = [0.85, 0.95, 0.001, 0.5, 2.0]  # 阻抗略软
        count += 1
    print(f"[rough terrain] spec_fn: softened {count} terrain geoms (solref=0.04)")


def make_microduck_velocity_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """创建 Microduck 速度跟踪环境配置."""
    std_standing = {
        # 下肢 — 更紧以让机器人站立时保持 home 姿态
        r".*hip_yaw.*": 0.1,
        r".*hip_roll.*": 0.05,  # 0.1→0.06→0.05 — 保持 5° 向内站姿 (脚底平放), 防止腿外撇
        r".*hip_pitch.*": 0.15,
        r".*knee.*": 0.15,
        r".*ankle.*": 0.1,
    }

    std_walking = {
        # 下肢
        r".*hip_yaw.*": 0.3,
        r".*hip_roll.*": 0.05,  # 0.1→0.06→0.05 — 保持 5° 向内站姿, 防止腿撇到垂直
        r".*hip_pitch.*": 0.4,
        r".*knee.*": 0.4,
        r".*ankle.*": 0.25,  # 曾为 0.15
    }

    site_names = ("left_foot", "right_foot")

    # 足部接触传感器 - 左, 右顺序
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",  # 左脚在前, 右脚在后
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    # mjlab 1.3.0: foot_height obs + foot_clearance/foot_swing_height 奖励现在
    # 由每只脚的地形高度射线传感器驱动 (原为 site_pos 基).
    # 镜像 microban 的 foot_height_scan.
    foot_height_scan_cfg = TerrainHeightSensorCfg(
        name="foot_height_scan",
        frame=tuple(ObjRef(type="site", name=s, entity="robot") for s in site_names),
        pattern=RingPatternCfg.single_ring(radius=0.04, num_samples=2),
        ray_alignment="yaw",
        max_distance=1.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
        debug_vis=False,
    )

    foot_frictions_geom_names = (
        "left_foot_collision",
        "right_foot_collision",
    )

    # 基础配置
    cfg = make_velocity_env_cfg()

    # 机器人设置
    cfg.scene.entities = {"robot": MICRODUCK_WALK_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg, foot_height_scan_cfg)
    cfg.viewer.body_name = "trunk_base"

    # 动作配置
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # === 奖励 ===
    # 姿态奖励配置
    cfg.rewards["pose"].params["std_standing"] = std_standing  # command=0 时更紧
    cfg.rewards["pose"].params["std_walking"] = std_walking
    cfg.rewards["pose"].params["std_running"] = std_walking
    # 姿态奖励仅作用于腿部关节. 头/颈由指令驱动 (head_pose_tracking) — 如果
    # 它们也在这个奖励里, 会把它们拉向 HOME, 而 head_pose_tracking 把它们拉向
    # 指令, policy 会收敛到 "忽略指令", 因为一旦 head_pose_tracking 的梯度在大
    # 指令下衰减, 姿态奖励就占主导.
    cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=(r"^(?!passive_|.*neck.*|.*head.*).*",)
    )
    cfg.rewards["pose"].params["walking_threshold"] = 0.01
    cfg.rewards["pose"].weight = 1.0

    # Body 专属奖励配置
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    # upright: 刻意强 (2.0 / std²=0.05, 曾为 1.0 / std²=0.1).
    # 2026-07 俯仰-速度评估: policy 行走时带 +2-4° 稳态前倾 (p90 ~6-8°),
    # 速度下 ~2/3 的推力摔倒是向前. 权重 1.0 / std²=0.1 时 4° 前倾成本 ~0.05/步 —
    # 几乎免费. 2.0 / std²=0.05 时成本 ~0.19/步: 足够的梯度让稳态步态保持躯干
    # 水平, 同时瞬态前倾 (推力恢复, 加速) 仍可负担.
    cfg.rewards["upright"].weight = 2.0
    cfg.rewards["upright"].params["std"] = math.sqrt(0.05)

    # 足部专属配置. mjlab 1.3.0 中 foot_swing_height 完全由传感器驱动 (无
    # asset_cfg); 只有 foot_clearance/foot_slip 仍带 asset_cfg, 其 site_names
    # 选中足部.
    for reward_name in ["foot_clearance", "foot_slip"]:
        cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

    # Body 专属配置
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)

    # foot_slip 刻意偏弱 (-0.1, 非 -1.0): -1.0 对该机器人以转体为主的转弯太严苛.
    cfg.rewards["foot_slip"].weight = -0.1
    cfg.rewards["foot_slip"].params["command_threshold"] = 0.01

    cfg.rewards.pop("soft_landing", None)

    # 自碰撞惩罚: 阻止腿撞向躯干电池座 (leg, leg_2, battery_holder 上
    # self_collision_only 类的 geom). 有关节范围限制时 policy 实际够不到躯干,
    # 但这里的正信号让它保持远离.
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    # air_time 窗口 [0.125, 0.300] s. 注意: 零指令下静止站立由 standing_envs
    # 课程教授 (到 ~iter 2000 时 →25% 站立环境), 而非显式的静止/不迈步项.
    cfg.rewards["air_time"].weight = 3.0
    cfg.rewards["air_time"].params["command_threshold"] = 0.01
    cfg.rewards["air_time"].params["threshold_min"] = 0.125
    cfg.rewards["air_time"].params["threshold_max"] = 0.300

    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02

    # 速度跟踪奖励
    cfg.rewards["track_linear_velocity"].weight = 2.0
    cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.1)
    cfg.rewards["track_angular_velocity"].weight = 2.0
    cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.5)

    # 动作平滑: 阶段 0 值; 下面的 action_rate_weight 课程将其渐升
    # -0.1 → -1.0 (到 iter 1500).
    cfg.rewards["action_rate_l2"].weight = -0.1

    cfg.rewards["foot_clearance"].params["command_threshold"] = 0.01
    cfg.rewards["foot_clearance"].params["target_height"] = 0.02  # 从 0.01 调高以惩罚拖地

    cfg.rewards["foot_swing_height"].params["command_threshold"] = 0.01
    cfg.rewards["foot_swing_height"].params["target_height"] = 0.02  # 从 0.01 调高以强制抬脚

    # 注意: 没有 neck-only 的 action-rate 项 — 共享的 action_rate_l2 求和所有
    # action 维度 (含 neck), 且下面的 head_pose_tracking 给 4 个 neck/head DOF
    # 一个位置目标, 所以 neck 已被充分塑造.

    # 事件
    # BAM (mjlab_frictionloss 分支) 每步写入每环境的 dof_frictionloss/dof_damping;
    # 这个 no-op 事件为按世界展开注册这些字段.
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )

    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (
        0.7,
        1.3,
    )  # 更粘的脚垫 — 从 (0.3, 1.2) 收窄
    # 终止数值不稳定 (NaN 物理) 的环境.
    # MuJoCo 在极端接触冲击下可能产生 NaN 关节位置.
    # 立即终止会重置到有效状态, 防止 NaN 传播到观测缓冲并污染网络权重.
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": (feet_ground_cfg.name,)},
    )

    cfg.events["reset_base"].params["pose_range"]["z"] = (0.12, 0.13)

    # 基于速度的推力, 用于鲁棒性训练
    if ENABLE_VELOCITY_PUSHES:
        # play 模式下使用更短间隔以更好可见
        interval = (0.5, 1.0) if play else VELOCITY_PUSH_INTERVAL_S

        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=interval,
            params={
                "velocity_range": {
                    "x": VELOCITY_PUSH_RANGE,
                    "y": VELOCITY_PUSH_RANGE,
                },
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

    # Domain randomization — 重置时每个 episode 重采样. mjlab 1.3.0 中
    # 原生 dr.* 操作 (operation="add"/"scale") 每次重置从编译时默认字段读取
    # (Operation.uses_defaults=True), 因此它们原生 NON-accumulating — 这一上游
    # 行为替代了 microduck 旧的绕过累积陷阱的自定义 restore-then-add 函数.
    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        # 随机化头部组件各 body 的 CoM (每次重置每 body 全新偏移).
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "operation": "add",
                "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_KP_RANDOMIZATION or ENABLE_KD_RANDOMIZATION:
        # 随机化电机 PD 增益
        # 使用处理 DelayedActuator 的自定义函数
        kp_range = KP_RANDOMIZATION_RANGE if ENABLE_KP_RANDOMIZATION else (1.0, 1.0)
        kd_range = KD_RANDOMIZATION_RANGE if ENABLE_KD_RANDOMIZATION else (1.0, 1.0)
        cfg.events["randomize_motor_gains"] = EventTermCfg(
            func=microduck_mdp.randomize_delayed_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "operation": "scale",
                "kp_range": kp_range,
                "kd_range": kd_range,
            },
        )

    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        # 物理一致的质量 + 惯量随机化, 通过 mjlab 的 pseudo_inertia:
        # alpha 以 e^(2*alpha) 同时缩放质量和惯量, CoM 不变 (因此与
        # randomize_com 不冲突). alpha_range 由 ±5% 质量缩放范围派生:
        # e^(2*alpha) ∈ [0.95, 1.05].
        # 替代旧的自定义 randomize_mass_and_inertia, 后者在 mjlab 1.3.0 下是
        # 静默 no-op (直接按环境写 body_mass/body_inertia 不会被展开, 塌缩为
        # 单一共享值). Startup 模式 = 每环境整次运行固定 (质量 DR 标准; 无累积).
        _mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(_mi_lo) / 2.0, math.log(_mi_hi) / 2.0),
            },
        )

    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        # BAM 下的关节摩擦 DR: 通过 FrictionDRBamActuator 的 friction_scale 钩子
        # 按环境缩放 BAM 的速度无关摩擦预算 (Coulomb + Stribeck + load).
        # MuJoCo 的 dof_frictionloss 在 BAM 下被清零, 所以原生 dr.dof_frictionloss
        # 是 no-op — 这是 BAM 原生路径.
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_JOINT_DAMPING_RANDOMIZATION:
        # 随机化关节阻尼 (润滑, 温度效应).
        # 自定义非累积缩放器. 注意: BAM 下为 no-op (dof_damping 在 edit_spec 中
        # 被清零); 仅影响 XML 位置执行器.
        cfg.events["randomize_joint_damping"] = EventTermCfg(
            func=microduck_mdp.randomize_dof_field_scaled,
            mode="reset",
            domain_randomization=True,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "field": "dof_damping",  # domain_randomization=True 必需
                "scale_range": JOINT_DAMPING_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_ARMATURE_RANDOMIZATION:
        # 随机化反射转子惯量 (armature), 与 microban 完全一致
        # (dr.joint_armature, scale, ±10%). 非累积 (uses_defaults). 对 BAM
        # 执行器有效 — BAM 设置 dof_armature (~0.0018), 未被清零.
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )

    # IMU 朝向随机化 (安装误差) 在下面的 OBSERVATION 层应用 (每环境常量旋转
    # projected_gravity + base_ang_vel). 旧的事件式 randomize_imu_orientation
    # 写入 site_quat, 在 mjlab 1.3.0 下既不按环境展开, 也不被这些 obs 读取 —
    # 一个 no-op.

    # 基座朝向随机化 (强制反应式行为)
    if ENABLE_BASE_ORIENTATION_RANDOMIZATION:
        cfg.events["randomize_base_orientation"] = EventTermCfg(
            func=microduck_mdp.randomize_base_orientation,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "max_pitch_deg": BASE_ORIENTATION_MAX_PITCH_DEG,
                "max_roll_deg": BASE_ORIENTATION_MAX_ROLL_DEG,
            },
        )

    # 观测
    del cfg.observations["actor"].terms["base_lin_vel"]
    # mjlab 1.3.0 默认向两组添加 height_scan 项 (地形射线扫描). microduck
    # 没有这样的机身安装地形传感器供 policy 使用, 所以从两组移除 (镜像 microban).
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    # 仅向 critic 添加 base_lin_vel (特权信息)
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel,
        scale=1.0,
    )

    # 根据开关确定重力/加速度计项名称
    gravity_term_name = "projected_gravity" if USE_PROJECTED_GRAVITY else "raw_accelerometer"

    # 若开关为 False, 用 raw_accelerometer 替换 projected_gravity
    if not USE_PROJECTED_GRAVITY:
        # 移除 projected_gravity 并添加 raw_accelerometer
        del cfg.observations["actor"].terms["projected_gravity"]
        cfg.observations["actor"].terms["raw_accelerometer"] = ObservationTermCfg(
            func=microduck_mdp.raw_accelerometer,
            scale=1.0,
        )

    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(cfg.observations["actor"].terms[gravity_term_name])
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(cfg.observations["actor"].terms["base_ang_vel"])

    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms[
        "base_ang_vel"
    ].delay_max_lag = 1  # 曾为 3 (=60 ms 最坏); 真实 dxl IMU 路径快 — ±20 ms 包络 (2026-07 审计)
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64

    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[
        gravity_term_name
    ].delay_max_lag = 1  # 曾为 3 (=60 ms 最坏); 真实 dxl IMU 路径快 — ±20 ms 包络 (2026-07 审计)
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    # critic 的传感器派生项是 `nan_state` 无法保护的唯一 obs 路径
    # (它检查关节 + 根状态; 这些读取射线/接触传感器数据, MuJoCo 可在状态仍
    # 干净时返回非有限值). 这里一个 NaN 会通过 rsl_rl 的 check_nan 杀掉整次
    # 运行 — 即 2026-08-21 Velocity2-Rough-Backlash 崩溃. 仅 critic, 所以
    # 清洗对 policy 无成本.
    for _term, _safe in (
        ("foot_contact_forces", microduck_mdp.foot_contact_forces_safe),
        ("foot_height", microduck_mdp.foot_height_safe),
        ("foot_air_time", microduck_mdp.foot_air_time_safe),
    ):
        if _term in cfg.observations["critic"].terms:
            cfg.observations["critic"].terms[_term].func = _safe

    # 观测噪声配置 (按需修改这些值)
    cfg.observations["actor"].terms["base_ang_vel"].noise = Unoise(n_min=-0.03, n_max=0.03)  # 曾为 0.2
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)  # 曾为 0.15
    cfg.observations["actor"].terms["joint_pos"].noise = Unoise(n_min=-0.001, n_max=0.001)  # 曾为 0.05
    cfg.observations["actor"].terms["joint_vel"].noise = Unoise(n_min=-0.25, n_max=0.25)  # 曾为 2.0

    # IMU 安装失准 DR (对 IMU 派生观测的每环境常量旋转). 仅应用于 ACTOR
    # (policy 看到略微旋转的 IMU 帧, 如真实安装误差); critic 保留真值.
    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        if USE_PROJECTED_GRAVITY:
            g = cfg.observations["actor"].terms[gravity_term_name]
            g.func = microduck_mdp.projected_gravity_imu_misaligned
            g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    # joint_vel 上 1-ctrl-step 滞后: Dynamixel 固件通过前一个位置采样窗口的
    # 滑动平均计算 present_velocity, 所以 policy 实际读到的值约 1 个控制周期
    # 之前. 匹配真实并阻止 policy 依赖瞬时 qdot 反馈.
    cfg.observations["actor"].terms["joint_vel"] = deepcopy(cfg.observations["actor"].terms["joint_vel"])
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    # 从 joint_pos/vel obs 中排除 passive_* 关节 (颌连接), 使观测维度
    # 匹配动作维度 (14), 而非原始铰接 (16).
    # 先深拷贝每个 joint_pos/joint_vel 项 — actor 和 critic 从基础模板共享
    # 相同的 term 对象/params 字典, 修改一个会泄漏到另一个 (如下面的
    # encoder-bias `biased` 标志).
    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    # Encoder-bias DR: 基础模板采样每环境常量关节编码器偏移 (startup 事件
    # "encoder_bias"), 但 joint_pos_rel 在 biased=True 前会忽略它. 仅向 ACTOR
    # 喂入带偏移的关节位置 (真实编码器上报的内容); critic 保留真值 (特权).
    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    # 指令 — 深拷贝以避免其他 env cfg 的共享状态污染
    # (make_velocity_env_cfg() 返回的对象带共享可变引用; standup/ground_pick
    # 环境会就地修改 commands["twist"], 将范围清零)
    command: UniformVelocityCommandCfg = deepcopy(cfg.commands["twist"])
    cfg.commands["twist"] = command
    command.rel_standing_envs = 0.02  # 从一开始小但非零, 由课程渐升
    command.rel_heading_envs = 0.0
    # 适中、FIXED 指令范围 (无渐扩课程): 渐升到 lin ±0.4 / ang ±2.0 超出
    # 机器人能力, 并对应 iter-1000 后的奖励/episode-长度下降. ang ±1.0 是
    # 关键变化 — 它让转弯可学.
    command.ranges.lin_vel_x = (-0.4, 0.4)
    command.ranges.lin_vel_y = (-0.3, 0.3)
    command.ranges.ang_vel_z = (-1.0, 1.0)
    command.viz.z_offset = 0.5
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))
    # 显式 turn-in-place 桶 (见上面的 TURN_IN_PLACE_FRACTION).
    cfg.commands["twist"].rel_turn_in_place_envs = TURN_IN_PLACE_FRACTION

    # 头部姿态指令 (相对 HOME 的 4D 增量, 关节顺序:
    #   neck_pitch, head_pitch, head_yaw, head_roll). 作为主奖励跟踪 —
    # 见下方添加的 "head_pose_tracking". 初始范围小且非零, 使输入神经元
    # 从 step 0 起活跃; 课程会逐步加宽.
    # 每关节最终上限反映各关节从 HOME 机械可达的增量 (XML 极限减去 HOME
    # 偏移, 留 ~10% 安全余量):
    #   neck_pitch / head_pitch: ±1.10 rad (极限 ±π/2, HOME=±20°)
    #   head_yaw                : ±1.40 rad (极限 ±π/2, HOME=0)
    #   head_roll               : ±0.31 rad (极限 ±20°)
    # 初始范围小且非零, 使输入神经元从 step 0 起活跃.
    cfg.commands["head_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=HEAD_POSE_CMD_RESAMPLE_S,
        ranges=(
            (-0.05, 0.05),  # neck_pitch
            (-0.05, 0.05),  # head_pitch
            (-0.07, 0.07),  # head_yaw
            (-0.015, 0.015),  # head_roll (更紧 — 机械范围小得多)
        ),
    )
    # 身体姿态指令 (相对标称站立的 6D 增量: [x, y, z, roll, pitch, yaw]).
    # vel 环境携带此槽位以保持 runtime obs 形状对齐; 以微小权重跟踪以
    # 保持输入神经元活跃但不引导 policy. standup 环境会提高权重并加宽范围.
    cfg.commands["body_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=BODY_POSE_CMD_RESAMPLE_S,
        ranges=(
            (-0.005, 0.005),  # x (m)
            (-0.005, 0.005),  # y (m)
            (-0.005, 0.005),  # z (m)
            (-0.05, 0.05),  # roll
            (-0.05, 0.05),  # pitch
            (-0.05, 0.05),  # yaw
        ),
    )

    # 向 policy 和 critic 两组追加 head + body 指令 obs 项.
    # 顺序对 runtime obs 布局重要: [twist(3), head_pose(4), body_pose(6)].
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "head_pose"},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "body_pose"},
        )

    # === 姿态跟踪奖励 ===
    # head_pose: vel 环境的主目标 — 整个重写的意义所在.
    # std=0.5 配每关节高斯 (见 mdp.py 的 head_pose_tracking): 在全 ±1.0 rad
    # 指令下, 不跟踪的 policy 仍看到每关节奖励 exp(-(1/0.5)²)=exp(-4)≈0.018 —
    # 一个小但非零的梯度 — 所以课程加宽不会杀掉信号. 最终奖励是 4 个关节
    # 的均值, 所以部分跟踪即部分奖励 (非全有或全无).
    cfg.rewards["head_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.head_pose_tracking,
        weight=2.0,
        params={"command_name": "head_pose", "std": 0.5},
    )
    # body_pose: 基础设施保留但禁用 (权重 0) — obs 槽位和指令保持活跃,
    # 供提高权重的环境使用 (standup).
    cfg.rewards["body_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.body_pose_tracking_6d,
        weight=0.0,
        params={
            "command_name": "body_pose",
            "nominal_height": 0.095,
            "xy_std": 0.05,
            "z_std": 0.02,
            "angle_std": math.radians(15),
        },
    )

    # 头部下垂修正 (2026-08-20). 头部行走时俯仰 ~15° 向下 (实测: 运行
    # ww1g2198 head_pose_tracking 1.544/2.0 → 14.6° 平均关节误差).
    # 不要通过收紧 head_pose_tracking 的 std 来修正: 运行 5yay13u4 试过
    # fine_std=0.1, policy 在 iter 300 前完全停止行走 (air_time 1.01 → 0.02,
    # 峰值抬脚高度 15 mm → 2 mm, 熵塌缩 10.9 → 1.9).
    # 瞬时紧容差每步对行走计税 0.77 — 占整个 air_time 奖励的 76% — 且
    # UNESCAPABLE, 因为 280 g 的头 (机器人质量的 38%) 在迈步时必须振荡.
    # 站着不动得分更高, 所以它就站着不动.
    # DC 偏置与振荡不同, 是可逃避的 (向上偏置 neck 指令以抵消重力下垂),
    # 所以只对它计价: 对误差的 1 s EMA 取 L1. 在最优处这对行走 policy 无成本.
    cfg.rewards["head_pose_bias"] = RewardTermCfg(
        func=microduck_mdp.head_pose_bias_penalty,
        weight=0.0,  # 由下面的 head_pose_bias_weight 课程渐升
        params={"command_name": "head_pose", "tau_s": 1.0},
    )

    # 地形
    if not rough:
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None
    else:
        cfg.scene.terrain.terrain_type = "generator"
        cfg.scene.terrain.terrain_generator = MICRODUCK_ROUGH_TERRAINS_CFG

        # 软化地形 box 接触: 相邻 box 不同高度产生硬边, 会破坏接触求解器
        # 稳定并产生 NaN 力.
        cfg.scene.spec_fn = _soften_terrain_contacts

        # velocity 环境默认 nconmax=35 对粗糙地形偏紧: 当机器人摔倒且多个
        # body 连杆同时撞击多个 box 时, 接触溢出 → 部分被静默丢弃 → 突然
        # 解压 → NaN.
        cfg.sim.nconmax = 200  # 曾为 35

        # velocity 环境只用 10 次求解器迭代 (对比默认 100), 对粗糙 box 地形
        # 上的边接触求解太少. 三倍迭代显著减少接触求解失败, GPU 上计算
        # 成本适中 (MJWarp 跨环境并行).
        cfg.sim.mujoco.iterations = 30  # 曾为 10
        cfg.sim.mujoco.ls_iterations = 50  # 曾为 20

        if play:
            cfg.scene.terrain.terrain_generator.curriculum = False
            cfg.scene.terrain.terrain_generator.num_cols = 5
            cfg.scene.terrain.terrain_generator.num_rows = 5

    # action_rate 权重渐升: 步态引导期间温和平滑, 之后渐紧到 -1.0 (iter 1500).
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.1},
                {"step": 500 * NUM_STEPS_PER_ENV, "weight": -0.2},
                {"step": 750 * NUM_STEPS_PER_ENV, "weight": -0.4},
                {"step": 1000 * NUM_STEPS_PER_ENV, "weight": -0.6},
                {"step": 1250 * NUM_STEPS_PER_ENV, "weight": -0.8},
                {"step": 1500 * NUM_STEPS_PER_ENV, "weight": -1.0},
            ],
        },
    )

    # 行走建立后逐渐加大站立环境比例
    cfg.curriculum["standing_envs"] = CurriculumTermCfg(
        func=microduck_mdp.standing_envs_curriculum,
        params={
            "command_name": "twist",
            "standing_stages": [
                {"step": 0, "rel_standing_envs": 0.02},
                {"step": 500 * 24, "rel_standing_envs": 0.05},
                {"step": 750 * 24, "rel_standing_envs": 0.1},
                {"step": 1000 * 24, "rel_standing_envs": 0.15},
                {"step": 1500 * 24, "rel_standing_envs": 0.2},
                {"step": 2000 * 24, "rel_standing_envs": 0.25},
            ],
        },
    )

    # 注意: 无速度指令范围课程 — 范围固定 (见上面的指令部分).

    # 头部姿态指令范围课程 — 每关节, 按各关节从 HOME 可达增量缩放
    # (距 XML 极限 ~10% 余量). 与之前相同的 5 阶段形状 (每关节最终上限的
    # 5% → 15% → 35% → 65% → 100%).
    # neck/head pitch 最终 ±1.10 rad, head_yaw ±1.40, head_roll ±0.31.
    cfg.curriculum["head_pose_range"] = CurriculumTermCfg(
        func=microduck_mdp.pose_command_range_curriculum,
        params={
            "command_name": "head_pose",
            "range_stages": [
                # step,                ranges = ((neck_pitch), (head_pitch), (head_yaw),  (head_roll))
                {
                    "step": 0,
                    "ranges": (
                        (-0.05, 0.05),
                        (-0.05, 0.05),
                        (-0.07, 0.07),
                        (-0.015, 0.015),
                    ),
                },
                {
                    "step": 500 * 24,
                    "ranges": (
                        (-0.17, 0.17),
                        (-0.17, 0.17),
                        (-0.21, 0.21),
                        (-0.047, 0.047),
                    ),
                },
                {
                    "step": 1000 * 24,
                    "ranges": (
                        (-0.39, 0.39),
                        (-0.39, 0.39),
                        (-0.49, 0.49),
                        (-0.11, 0.11),
                    ),
                },
                {
                    "step": 1500 * 24,
                    "ranges": (
                        (-0.72, 0.72),
                        (-0.72, 0.72),
                        (-0.91, 0.91),
                        (-0.20, 0.20),
                    ),
                },
                {
                    "step": 2000 * 24,
                    "ranges": (
                        (-1.10, 1.10),
                        (-1.10, 1.10),
                        (-1.40, 1.40),
                        (-0.31, 0.31),
                    ),
                },
            ],
        },
    )

    # 身体姿态指令范围课程: 在 vel 环境保持小. standup 环境以宽范围 +
    # 重奖励权重覆盖此课程.
    cfg.curriculum["body_pose_range"] = CurriculumTermCfg(
        func=microduck_mdp.pose_command_range_curriculum,
        params={
            "command_name": "body_pose",
            "range_stages": [
                {
                    "step": 0,
                    "ranges": (
                        (-0.005, 0.005),  # x (m)
                        (-0.005, 0.005),  # y (m)
                        (-0.005, 0.005),  # z (m)
                        (-0.05, 0.05),  # roll
                        (-0.05, 0.05),  # pitch
                        (-0.05, 0.05),  # yaw
                    ),
                },
            ],
        },
    )

    # CoM 随机化范围课程 - 从小开始, 渐升
    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    # 上限 ±15 mm (2026-07 审计): 之前渐升到 ±30 mm 超出足部
                    # 支撑多边形 (脚跟仅在踝后 20 mm) — 随机化后的 CoM 可完全
                    # 在支撑外, 强制宽/快超反应步态, 使向后平衡不可训练.
                    # 回归时间线与渐升对应: 0.015 → 0.02 → 0.03 时 policy 变差.
                    {"step": 0, "range": 0.003},
                    {"step": 500 * 24, "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                    {"step": 1500 * 24, "range": 0.015},
                ],
            },
        )

    # 头部 CoM 随机化范围课程 - 从小开始, 渐升
    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    # 上限 ±10 mm (2026-07 审计 — 与躯干 CoM 同样的过度保守
                    # 顾虑; 头部是一个长杠杆臂).
                    {"step": 0, "range": 0.003},
                    {"step": 500 * 24, "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                ],
            },
        )

    # 禁用默认课程
    if not rough:
        del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    # head_pose_bias 渐升: iter 600 前关闭, 之后 1.0 → 3.0 (到 iter 1500).
    # 早期保持 0, 因为步态存在之前姿态精度项是干扰. 权重 3.0 时 15° 残余
    # 偏置成本 0.79/步, 2° 偏置成本 0.10/步.
    cfg.curriculum["head_pose_bias_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "head_pose_bias",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 600 * NUM_STEPS_PER_ENV, "weight": 1.0},
                {"step": 1000 * NUM_STEPS_PER_ENV, "weight": 2.0},
                {"step": 1500 * NUM_STEPS_PER_ENV, "weight": 3.0},
            ],
        },
    )

    return cfg


MicroduckRlCfg = RslRlOnPolicyRunnerCfg(
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
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="velocity",  # 目录名
    run_name="velocity",  # 追加到 wandb 中的 datetime 后: <datetime>_velocity
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=50_000,
)
