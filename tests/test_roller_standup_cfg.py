import pytest

from mjlab_microduck.tasks.microduck_roller_standup_env_cfg import (
    EPISODE_LENGTH_S,
    make_microduck_roller_standup_env_cfg,
)
from mjlab_microduck.tasks.microduck_velocity_rollers_env_cfg import make_microduck_velocity_rollers_env_cfg

# 滑行 reward: 它们不该在一个起身 env 里残留.
SKATING_REWARDS = (
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


def test_env_builds_train_and_play():
    assert make_microduck_roller_standup_env_cfg() is not None
    assert make_microduck_roller_standup_env_cfg(play=True) is not None


def test_episode_is_short():
    # 短 episode: 起身后稳定, 像 standup (6 s).
    cfg = make_microduck_roller_standup_env_cfg()
    assert cfg.episode_length_s == EPISODE_LENGTH_S == 6.0


def test_no_skating_rewards_survive():
    cfg = make_microduck_roller_standup_env_cfg()
    for name in SKATING_REWARDS:
        assert name not in cfg.rewards, f"残留的滑行 reward: {name}"


def test_smoothness_regularisers_kept():
    # 从 roller 继承保留: 起身需要 sim2real 的柔和, 但 body_ang_vel
    # 必须保持轻量 (standup 文档说在 -0.15 它会冻结).
    cfg = make_microduck_roller_standup_env_cfg()
    for name in (
        "action_over_limit",
        "self_collisions",
        "body_ang_vel",
        "angular_momentum",
        "action_rate_l2",
        "neck_action_rate_l2",
        "neck_joint_pos_l2",
        "joint_torques_l2",
    ):
        assert name in cfg.rewards, f"丢失的正则项: {name}"
    assert cfg.rewards["body_ang_vel"].weight == -0.05


def test_twist_command_is_neutralised():
    # 无驾驶: policy 部署在 --standing, runtime 让 twist slot 保持为零
    # (见 infer_policy.py:239).
    cfg = make_microduck_roller_standup_env_cfg()
    cmd = cfg.commands["twist"]
    assert cmd.ranges.lin_vel_x == (-0.01, 0.01)
    assert cmd.ranges.lin_vel_y == (-0.01, 0.01)
    assert cmd.ranges.ang_vel_z == (-0.05, 0.05)
    assert cmd.heading_command is False
    assert cmd.ranges.heading is None
    assert cmd.rel_standing_envs == 0.0


def test_twist_command_is_not_heading_relative():
    # roller env 装的是 RelativeHeadingVelocityCommandCfg (cmd[2] = 航向
    # 误差, 内部计算).这里 cmd[2] 必须是一个真正的带噪零.
    from mjlab_microduck.tasks import mdp as microduck_mdp

    cfg = make_microduck_roller_standup_env_cfg()
    cmd = cfg.commands["twist"]
    assert isinstance(cmd, microduck_mdp.VelocityCommandCommandOnlyCfg)
    assert not isinstance(cmd, microduck_mdp.RelativeHeadingVelocityCommandCfg)


def test_obs_nan_policy_sanitize():
    # 罕见 contact 让 free-joint 在 NaN 上发散: 我们 sanitize obs 而不是
    # 杀训练 (与 roller_slope 一致).
    cfg = make_microduck_roller_standup_env_cfg()
    assert cfg.observations["actor"].nan_policy == "sanitize"
    assert cfg.observations["critic"].nan_policy == "sanitize"


def test_obs_parity_with_roller_env():
    # 61D 对齐是硬性要求: 否则 ONNX 不能加载进 runtime slot.
    standup = make_microduck_roller_standup_env_cfg()
    roller = make_microduck_velocity_rollers_env_cfg()
    for grp in ("actor", "critic"):
        assert list(standup.observations[grp].terms.keys()) == list(roller.observations[grp].terms.keys()), (
            f"observation layout 在 {grp} 组上分叉了"
        )


def test_terrain_is_plain_plane():
    # 继承自 roller env: 平地, 没有生成器.本 v1 不提供 rough 变体.
    cfg = make_microduck_roller_standup_env_cfg()
    assert cfg.scene.terrain.terrain_type == "plane"
    assert cfg.scene.terrain.terrain_generator is None


def test_task_is_registered():
    from mjlab.tasks.registry import list_tasks

    import mjlab_microduck.tasks  # noqa: F401  (import 触发注册)

    assert "Mjlab-RollerStandUp-Flat-MicroDuck" in list_tasks()


def test_joint_indices_match_actual_roller_model():
    """锁: 被动轮子在 joint 顺序里是交错的.

    复用 standup 的索引 ([0-4, 9-13]) 会把 reward 指到轮子上.本测试
    编译真实的 rollers MjSpec 并校验所用索引上的名字.纯 CPU, 无 sim.
    """
    import mujoco

    from mjlab_microduck.robot.microduck_constants import get_walk_rollers_spec
    from mjlab_microduck.tasks.microduck_roller_standup_env_cfg import (
        _LEG_JOINTS,
        _NECK_JOINTS,
        _WHEEL_JOINTS,
    )

    model = get_walk_rollers_spec().compile()
    articulated = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in range(model.njnt)
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
    ]

    assert [articulated[i] for i in _LEG_JOINTS] == [
        "left_hip_yaw",
        "left_hip_roll",
        "left_hip_pitch",
        "left_knee",
        "left_ankle",
        "right_hip_yaw",
        "right_hip_roll",
        "right_hip_pitch",
        "right_knee",
        "right_ankle",
    ]
    assert [articulated[i] for i in _NECK_JOINTS] == [
        "neck_pitch",
        "head_pitch",
        "head_yaw",
        "head_roll",
    ]
    assert [articulated[i] for i in _WHEEL_JOINTS] == [
        "passive_LF_wheel",
        "passive_LR_wheel",
        "passive_RF_wheel",
        "passive_RR_wheel",
    ]
    # 无重叠, 三份清单覆盖所有关节.
    assert len(set(_LEG_JOINTS) | set(_NECK_JOINTS) | set(_WHEEL_JOINTS)) == len(articulated)


def test_recovery_rewards_present_with_expected_weights():
    cfg = make_microduck_roller_standup_env_cfg()
    expected = {
        "pose_stand_legs": 8.0,
        "pose_stand_l1": 5.0,
        "height_stand": 4.0,
        "height_stand_sharp": 4.0,
        "height_stand_l1": 30.0,
        "com_upward_velocity": 3.0,
        # gentle_rise: 权重为正.trunk_vertical_accel_penalty 已经
        # 返回 -|a_z|, 所以一个负权重会把它变成对暴力的奖励
        # (实测 bug: Episode_Reward/gentle_rise 记到 +0.0118).
        "gentle_rise": +0.02,
        "upright_linear": 6.0,
        "upright_sharp": 6.0,
        "standing_composite": 15.0,
        # -2e-3 在 ~+41.6 的任务 reward 面前只贡献 -0.0002/步: 等于零.
        # -2.0 实测 -0.255/步 (run d8rnko6p) —— 不算冻结, 但我们调回
        # -0.2 以腾出预算给我们隔离时的阻尼.
        "joint_torque_rate_l2": -0.2,
    }
    for name, weight in expected.items():
        assert name in cfg.rewards, f"缺失起身 reward: {name}"
        assert cfg.rewards[name].weight == weight, f"{name} 上的权重不符"


def test_recovery_rewards_use_roller_heights_not_walker_heights():
    from mjlab_microduck.tasks.microduck_roller_standup_env_cfg import (
        ROLLER_PRONE_Z,
        ROLLER_STAND_Z,
    )

    cfg = make_microduck_roller_standup_env_cfg()
    assert ROLLER_STAND_Z == 0.138  # 不是无轮模型的 0.115
    for name in ("height_stand", "height_stand_sharp", "height_stand_l1"):
        assert cfg.rewards[name].params["target_height"] == ROLLER_STAND_Z
    assert cfg.rewards["standing_composite"].params["target_height"] == ROLLER_STAND_Z
    # com_upward_velocity 在恰好高于目标时切断 (10 mm 余量),
    # 否则 policy 会停在切断高度而不完成上行.
    assert cfg.rewards["com_upward_velocity"].params["max_height"] == ROLLER_STAND_Z + 0.010
    # upright_sharp 在趴地和站立之间被 gate.
    assert cfg.rewards["upright_sharp"].params["height_low"] == ROLLER_PRONE_Z
    assert cfg.rewards["upright_sharp"].params["height_high"] == ROLLER_STAND_Z


def test_pose_rewards_target_legs_only_at_roller_indices():
    from mjlab_microduck.tasks.microduck_roller_standup_env_cfg import _LEG_JOINTS

    cfg = make_microduck_roller_standup_env_cfg()
    for name in ("pose_stand_legs", "pose_stand_l1", "standing_composite"):
        assert cfg.rewards[name].params["joint_indices"] == _LEG_JOINTS
        # target_overrides=None -> 目标是 HOME (default_joint_pos).
        assert cfg.rewards[name].params["target_overrides"] is None


def test_trunk_asset_cfgs_are_distinct_objects():
    """mjlab 会就地解析并 MUTE SceneEntityCfg: 多个 term 共享一个对象
    会触发 stale indices.每个 term 必须有自己的一份.
    """
    cfg = make_microduck_roller_standup_env_cfg()
    names = (
        "height_stand",
        "height_stand_sharp",
        "height_stand_l1",
        "com_upward_velocity",
        "gentle_rise",
        "upright_linear",
        "upright_sharp",
        "standing_composite",
    )
    seen = [id(cfg.rewards[n].params["asset_cfg"]) for n in names]
    assert len(set(seen)) == len(seen), "asset_cfg 被多个 term 共享"


def test_starts_from_ground_states():
    # 趴 + 仰 + 站.没有 "坐" 桶: 它在 standup 里只用于与 sit policy 的
    # hand-off, roller 没有对应物 —— 而且它的 sitting_joint_overrides 是
    # 无轮模型的索引.
    cfg = make_microduck_roller_standup_env_cfg()
    assert "set_ground_state" in cfg.events
    params = cfg.events["set_ground_state"].params
    assert params["sitting_prob"] == 0.0
    assert params["sitting_joint_overrides"] is None
    assert params["face_down_prob"] > 0.0
    assert params["standing_prob"] > 0.0
    # face_up (仰面) 起步为 0: 由 curriculum 晚期引入.
    assert params["face_up_prob"] == 0.0


def test_ground_state_heights_are_roller_specific():
    cfg = make_microduck_roller_standup_env_cfg()
    params = cfg.events["set_ground_state"].params
    # 趴与仰共用一个 z 范围, 但接触不同:
    # 趴姿态从 0.0752 起才离地, 仰在 0.0475 贴地.
    # prone_z_min = 0.076 以彻底消除趴姿侧的互穿.
    assert (params["prone_z_min"], params["prone_z_max"]) == (0.076, 0.09)
    # 低于 0.0752 (实测接触, HOME pose) 时, 趴姿起跑是 DANS 地面 ——
    # 一个 policy 会通过 gentle_rise / joint_torque_rate_l2 付出的接触
    # 推出.prone_z_min 必须保持在其上方.
    assert params["prone_z_min"] >= 0.0752
    # 站立: roller 高度 (相比无轮模型 +23 mm, 后者是 0.11-0.12).
    assert params["standing_z_min"] == 0.134
    assert params["standing_z_max"] == 0.144
    assert params["standing_z_min"] < 0.138 < params["standing_z_max"]


def test_ground_state_event_runs_after_base_reset():
    # set_ground_state 会覆盖 reset_base / reset_robot_joints 放下的 pose:
    # 事件顺序遵循插入顺序, 它必须排在之后.
    cfg = make_microduck_roller_standup_env_cfg()
    order = list(cfg.events.keys())
    assert order.index("set_ground_state") > order.index("reset_base")
    assert order.index("set_ground_state") > order.index("reset_robot_joints")


def test_no_fall_termination():
    # 机器人从摔倒开始: 一个倾角终止会在第一步就杀掉 episode.
    # nan_state (继承) 留着.
    cfg = make_microduck_roller_standup_env_cfg()
    assert "fell_over" not in cfg.terminations
    assert "nan_state" in cfg.terminations


def test_ground_state_curriculum_ramps_easy_to_hard():
    cfg = make_microduck_roller_standup_env_cfg()
    assert "ground_state_mix" in cfg.curriculum
    stages = cfg.curriculum["ground_state_mix"].params["param_stages"]
    assert cfg.curriculum["ground_state_mix"].params["event_name"] == "set_ground_state"
    # 步数单调递增, 从 0 起步.
    steps = [s["step"] for s in stages]
    assert steps[0] == 0 and steps == sorted(steps) and len(set(steps)) == len(steps)
    # 仰面 (face_up) 晚期引入, 之后单调增长.
    face_up = [s["params"]["face_up_prob"] for s in stages]
    assert face_up[0] == 0.0
    assert face_up == sorted(face_up)
    assert face_up[-1] >= 0.35
    # 每段都是一个合法分布, 且 "已经站立" 永不消失
    # (否则 policy 起身后又会因没学过站立而摔).
    for stage in stages:
        p = stage["params"]
        total = p["standing_prob"] + p["sitting_prob"] + p["face_down_prob"] + p["face_up_prob"]
        assert abs(total - 1.0) < 1e-9
        assert p["sitting_prob"] == 0.0
        assert p["standing_prob"] > 0.0


def test_wheel_friction_curriculum_is_decreasing():
    """新增项: 轮子从 制动 -> 自由.

    轮子是滚动的, 没有任何纵向附着力来推地面.我们用接近锁死的
    轴承 bootstrap (像用脚那样起身), 再爬向真实值.roller env 反而
    让这个摩擦上升 (0 -> 0.0015): 这里方向相反.
    """
    cfg = make_microduck_roller_standup_env_cfg()
    stages = cfg.curriculum["wheel_friction"].params["ranges_stages"]
    assert cfg.curriculum["wheel_friction"].params["event_name"] == "randomize_wheel_friction"

    steps = [s["step"] for s in stages]
    assert steps[0] == 0 and steps == sorted(steps) and len(set(steps)) == len(steps)

    lows = [s["ranges"][0] for s in stages]
    assert lows == sorted(lows, reverse=True), "摩擦必须 递减"
    assert lows[0] >= 0.02, "起步要确实制动以 bootstrap 动作"
    # 终点落在真实轴承值 (roller env 的那个值).
    assert stages[-1]["ranges"] == (0.0015, 0.0015)
    for stage in stages:
        assert stage["ranges"][0] == stage["ranges"][1]


def test_wheel_friction_event_default_matches_stage_zero():
    # curriculum manager 在每次 reset 之前运行 (包括第一次), 且
    # wheel_friction_curriculum 自身默认到 stage 0: 所以这个 event 的默认
    # 值实际从不会被读.这里只校验它与 curriculum 的 stage 0 保持一致
    # —— 如果某天 curriculum 被删掉但 event 留下, 这层冗余防御有用.
    cfg = make_microduck_roller_standup_env_cfg()
    stage0 = cfg.curriculum["wheel_friction"].params["ranges_stages"][0]["ranges"]
    assert cfg.events["randomize_wheel_friction"].params["ranges"] == stage0


def test_action_rate_ramp_is_the_standup_one_not_the_roller_one():
    # roller env 升到 -2.0 (平稳 gait): 这是 motion blocker, 它会拖慢
    # 仰面起身所需的快速动作.我们沿用 standup 的 ramp, 它封顶 -1.0.
    cfg = make_microduck_roller_standup_env_cfg()
    weights = [s["weight"] for s in cfg.curriculum["action_rate_weight"].params["weight_stages"]]
    assert weights == [-0.4, -0.8, -1.0]
    assert cfg.rewards["action_rate_l2"].weight == -0.6


def test_push_curriculum_ramps_from_zero():
    # 继承的推力 (±0.2 m/s), 但是 ramp: 第 0 步就推一下会干扰起身
    # 的 bootstrap.
    cfg = make_microduck_roller_standup_env_cfg()
    assert "push_robot" in cfg.events
    stages = cfg.curriculum["push_magnitude"].params["push_stages"]
    assert cfg.curriculum["push_magnitude"].params["event_name"] == "push_robot"
    assert stages[0]["velocity_range"]["x"] == (0.0, 0.0)
    assert stages[-1]["velocity_range"]["x"] == (-0.2, 0.2)
    highs = [s["velocity_range"]["x"][1] for s in stages]
    assert highs == sorted(highs), "推力必须 递增"


def test_inherited_dr_curricula_survive():
    # 继承自 roller env 的 DR 不能在路上丢失.
    cfg = make_microduck_roller_standup_env_cfg()
    for name in ("com_range", "head_com_range"):
        assert name in cfg.curriculum, f"丢失的 DR curriculum: {name}"
    for name in (
        "randomize_com",
        "randomize_head_com",
        "randomize_armature",
        "randomize_joint_friction",
        "randomize_mass_inertia",
        "randomize_wheel_friction",
        "encoder_bias",
    ):
        assert name in cfg.events, f"丢失的 DR event: {name}"


# ── play override: 强制仰面起跑 ──────────────────────────────────────────
# 没有 override 的话, play 永远不会显示仰面起跑: play env 是全新构建的,
# 所以 common_step_counter 回 0, curriculum 套上它的 stage 0, 那里
# face_up_prob = 0.可那恰恰是最难、最该肉眼检查的情况.STANDUP_PLAY_FACE_UP
# 强制混合, 仿照 roller_slope 里的 SLOPE_PLAY_DIFFICULTY.


def test_play_face_up_override_forces_back_starts(monkeypatch):
    monkeypatch.setenv("STANDUP_PLAY_FACE_UP", "1.0")
    cfg = make_microduck_roller_standup_env_cfg(play=True)
    params = cfg.events["set_ground_state"].params
    assert params["face_up_prob"] == 1.0
    assert params["face_down_prob"] == 0.0
    assert params["standing_prob"] == 0.0
    # 没有它, curriculum 会在第一次 reset 就改写概率
    # (event_param_curriculum 跑在 reset 事件之前).
    assert "ground_state_mix" not in cfg.curriculum


def test_play_face_up_override_splits_remainder_like_final_stage(monkeypatch):
    # 0.4 应重现 curriculum 的最后一段 (0.40 趴 / 0.20 站 / 0.40 仰):
    # 余额按那一段的 2:1 比例分配.
    monkeypatch.setenv("STANDUP_PLAY_FACE_UP", "0.4")
    params = make_microduck_roller_standup_env_cfg(play=True).events["set_ground_state"].params
    assert params["face_up_prob"] == pytest.approx(0.40)
    assert params["face_down_prob"] == pytest.approx(0.40)
    assert params["standing_prob"] == pytest.approx(0.20)
    total = params["face_up_prob"] + params["face_down_prob"] + params["standing_prob"]
    assert total == pytest.approx(1.0)


def test_play_face_up_override_is_clamped(monkeypatch):
    monkeypatch.setenv("STANDUP_PLAY_FACE_UP", "3.0")
    params = make_microduck_roller_standup_env_cfg(play=True).events["set_ground_state"].params
    assert params["face_up_prob"] == 1.0


def test_play_face_up_override_ignored_during_training(monkeypatch):
    # 护栏: 这个变量绝不能影响训练, 否则会在不知不觉中破坏 easy->hard
    # curriculum.
    monkeypatch.setenv("STANDUP_PLAY_FACE_UP", "1.0")
    cfg = make_microduck_roller_standup_env_cfg(play=False)
    assert cfg.events["set_ground_state"].params["face_up_prob"] == 0.00
    assert "ground_state_mix" in cfg.curriculum


def test_play_without_override_keeps_curriculum_mix(monkeypatch):
    # 默认行为不变: stage 0, 没有仰面起跑.
    monkeypatch.delenv("STANDUP_PLAY_FACE_UP", raising=False)
    cfg = make_microduck_roller_standup_env_cfg(play=True)
    assert cfg.events["set_ground_state"].params["face_up_prob"] == 0.00
    assert "ground_state_mix" in cfg.curriculum


def test_play_face_up_override_invalid_value_falls_back(monkeypatch):
    monkeypatch.setenv("STANDUP_PLAY_FACE_UP", "pouet")
    cfg = make_microduck_roller_standup_env_cfg(play=True)
    assert cfg.events["set_ground_state"].params["face_up_prob"] == 0.00
    assert "ground_state_mix" in cfg.curriculum


def test_play_face_up_override_none_keyword_disables(monkeypatch):
    monkeypatch.setenv("STANDUP_PLAY_FACE_UP", "none")
    cfg = make_microduck_roller_standup_env_cfg(play=True)
    assert cfg.events["set_ground_state"].params["face_up_prob"] == 0.00
    assert "ground_state_mix" in cfg.curriculum


# ── Anti-violence: 机器人上测试后的修正 ───────────────────────────────────
# 观察到的症状 (checkpoint 4000+, 在 sim 里也有, 所以不是 sim2real):
# 动作非常突兀, 头撞地, 真机仰面起身失败.诊断在 wandb 里测过
# (run vweolw91, iter 7500).


def test_already_negative_penalties_use_positive_weights():
    """锁住让 policy 变暴力那类 bug.

    mdp.py 混用两种符号约定: 某些 penalty 函数返回正的量级 (乘负权重),
    另一些已经返回负值 (乘正权重).trunk_vertical_accel_penalty 返回
    -|a_z|: 配上从 standup 继承的 -0.02 权重, 双重负号就在奖励垂直
    加速度 —— 实测 Episode_Reward/gentle_rise = +0.0118, 是唯一一个
    记为正的 penalty 项.
    """
    cfg = make_microduck_roller_standup_env_cfg()
    # 这三项调用的函数已经返回负值
    # (height_l1_penalty, pose_l1_penalty, trunk_vertical_accel_penalty).
    for name in ("height_stand_l1", "pose_stand_l1", "gentle_rise"):
        assert cfg.rewards[name].weight > 0, f"{name} 调用的函数已经返回负值: 负权重会变成奖励"
    # 而这些项返回正量级 -> 负权重.
    for name in ("joint_torques_l2", "joint_torque_rate_l2", "action_rate_l2"):
        assert cfg.rewards[name].weight < 0, f"{name} 期望负权重"


def test_no_ungated_head_impact_penalty():
    """没有未 gate 的头部 impact 惩罚 —— 它会冻结 policy.

    在 -1.0 (velstand 的值) 试过: policy 收敛到趴着一动不动.在 run
    d8rnko6p 上实测: head_impact_penalty -1.01/步, 是最大的负项, 同时
    standing_composite 从 +14.3 崩到 +3.3.

    推理错误在于相信一个 "针对性" 惩罚不会制动动作.这里错了: 这个
    机器人从仰面起身是 PIVOT 在它的头和肩上.头是翻身时的支点, 不是
    附带伤害 —— 惩罚它就是惩罚唯一可用的机制.

    如果修正 gentle_rise 符号后 slam 回潮, 重做应该是一个按高度 GATE
    的惩罚 (像 upright_sharp 那样), 那能在地面翻身阶段松开.不是这种.
    """
    cfg = make_microduck_roller_standup_env_cfg()
    assert "head_impact_penalty" not in cfg.rewards
    assert "head_impact_contact" not in [s.name for s in cfg.scene.sensors]


def test_inherited_sensors_intact():
    # 继承自 roller env 的 sensor 被保留下的 reward 用 (self_collisions)
    # 以及 observation 用.
    cfg = make_microduck_roller_standup_env_cfg()
    names = [s.name for s in cfg.scene.sensors]
    assert "feet_ground_contact" in names
    assert "self_collision" in names


def test_lazy_prone_optimum_is_documented_risk():
    """冻结来自一个懒惰 optimum: 趴着, 腿在 HOME, 它就付费.

    pose_stand_legs 在机器人躺下时仍停在 +7.72/8 —— 腿在趴姿态下
    处于 HOME, 所以 pose reward 几乎免费就拿到.一旦给动作加成本,
    这就是让 "什么也不做" 可行的反向配重.height_stand_l1 (权重 +30)
    是那个本该把 "留在地面" 拉成净负的项: 它必须保持强.
    """
    cfg = make_microduck_roller_standup_env_cfg()
    assert cfg.rewards["height_stand_l1"].weight >= 30.0
    assert cfg.rewards["com_upward_velocity"].weight > 0.0


def test_damping_terms_are_not_numerically_negligible():
    """专用阻尼器字面上毫无贡献.

    实测收敛时: joint_torque_rate_l2 -0.0002/步, joint_torques_l2
    -0.0001/步, 对应 ~+41.6 的任务 reward (所有阻尼器加起来约 35:1).
    joint_torque_rate_l2 是确定该上调的杠杆: 它惩罚的是力矩的 变化,
    而非动作, 所以它不会像 motion blocker 那样作用 —— standup 文档说
    body_ang_vel 和 action_rate 才会冻结仰面起身.
    """
    cfg = make_microduck_roller_standup_env_cfg()
    assert abs(cfg.rewards["joint_torque_rate_l2"].weight) >= 0.1
    # motion blocker 留在 "能从任何姿态起身" 的值上.
    assert cfg.rewards["body_ang_vel"].weight == -0.05
    weights = [s["weight"] for s in cfg.curriculum["action_rate_weight"].params["weight_stages"]]
    assert min(weights) >= -1.0, "action_rate 超过 -1.0 会冻结起身 (standup)"
