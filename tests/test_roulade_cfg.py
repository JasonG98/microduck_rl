"""Roulade (前滚翻) env cfg 的回归测试.

锁定: 新 home-pose 税的符号/闸门/关节覆盖, 61D obs 契约, 以及注册/时长等
不变式. 纯 CPU, 无 sim.
"""

import mujoco
import pytest

from mjlab_microduck.robot.microduck_constants import get_standup_spec
from mjlab_microduck.tasks.microduck_roulade_env_cfg import (
    _BODY_JOINTS,
    _LEG_JOINTS,
    _NECK_JOINTS,
    LANDING_GATE_HI,
    LANDING_GATE_LO,
    make_microduck_roulade_env_cfg,
)


def test_env_builds_train_and_play():
    assert make_microduck_roulade_env_cfg() is not None
    assert make_microduck_roulade_env_cfg(play=True) is not None


def test_task_is_registered():
    from mjlab.tasks.registry import list_tasks

    import mjlab_microduck.tasks  # noqa: F401  (import 触发注册)

    assert "Mjlab-Roulade-Flat-MicroDuck" in list_tasks()


def test_episode_length_is_5s():
    # Run-3 变更: 4 s 在节奏翻滚后无空间起立 → 5 s.
    cfg = make_microduck_roulade_env_cfg()
    assert cfg.episode_length_s == 5.0


def test_obs_layout_keeps_61d_command_block_last():
    # 61D obs 契约: actor 必须是 [48 本体感知] + [13D 指令块 (twist/head/body)],
    # 且指令块在最后且顺序不变 (runtime 栈硬依赖).
    cfg = make_microduck_roulade_env_cfg()
    keys = list(cfg.observations["actor"].terms.keys())
    assert keys[-3:] == ["command", "head_command", "body_command"], f"指令块不在尾部: {keys}"
    assert "height_scan" not in keys
    assert "base_lin_vel" not in keys  # roulade 无速度指令, actor 不观测线速度


def test_no_fall_termination():
    # 摔倒就是任务: 只有 NaN 守卫 + 超时.
    cfg = make_microduck_roulade_env_cfg()
    assert "fell_over" not in cfg.terminations
    assert "nan_state" in cfg.terminations


def test_body_joints_cover_all_14_servo():
    assert sorted(_BODY_JOINTS) == list(range(14))
    assert len(set(_BODY_JOINTS)) == 14
    assert set(_BODY_JOINTS) == set(_LEG_JOINTS) | set(_NECK_JOINTS)


def test_body_joints_match_actual_plain_model_order():
    """锁: plain (ground-contact) 模型的关节顺序是 左腿 0-4, 头颈 5-8,
    右腿 9-13. 若模型插入关节, 这份索引必须跟着修 (同 roller 测试).
    """
    model = get_standup_spec().compile()
    articulated = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in range(model.njnt)
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
    ]
    assert [articulated[i] for i in _BODY_JOINTS] == [
        "left_hip_yaw",
        "left_hip_roll",
        "left_hip_pitch",
        "left_knee",
        "left_ankle",
        "neck_pitch",
        "head_pitch",
        "head_yaw",
        "head_roll",
        "right_hip_yaw",
        "right_hip_roll",
        "right_hip_pitch",
        "right_knee",
        "right_ankle",
    ]


def test_home_pose_l1_term_wired_with_positive_weight():
    """roulade_home_pose_l1 自否定 (返回 <= 0) → 权重必须为正.
    负权重会把 "站定后偏离 HOME" 变成奖励 (penalty sign 坑, 仓库踩过多次).
    """
    cfg = make_microduck_roulade_env_cfg()
    term = cfg.rewards["roulade_home_pose_l1"]
    assert term.func.__name__ == "roulade_home_pose_l1"
    assert term.weight > 0
    assert term.weight == 2.5  # 1.0 实测残差 ~10° 后停止, 提高继续压 (run yt8i7lif)


def test_home_pose_l1_is_gated_to_landing_window():
    cfg = make_microduck_roulade_env_cfg()
    p = cfg.rewards["roulade_home_pose_l1"].params
    # 翻滚完成闸与 landing 系一致 (260-330 smoothstep + require_head).
    assert p["gate_lo"] == LANDING_GATE_LO
    assert p["gate_hi"] == LANDING_GATE_HI
    # 站立窗: 高度窗必须在 STAND_Z 之下开启、且倾斜窗先于 >45 度关闭,
    # 否则起身过渡期 (腿蜷头摆) 会被征税.
    assert p["z_low"] < p["z_high"] <= 0.115  # STAND_Z
    assert p["tilt_zero_deg"] > p["tilt_full_deg"] >= 10.0
    # 关节覆盖 = 全身 14 servo, 目标为 HOME (无 target_overrides).
    assert p["joint_indices"] == _BODY_JOINTS
    assert "target_overrides" not in p or p["target_overrides"] is None


def test_core_reward_weights_unchanged():
    # 核心任务堆不允许在新项加入时漂移.
    cfg = make_microduck_roulade_env_cfg()
    expected = {
        "roulade_progress": 8.0,
        "roulade_landing_composite": 4.0,
        "roulade_landing_sharp": 2.0,
        "roulade_upright_after_roll": 1.5,
        "roulade_height_after_roll": 1.0,
        "roulade_stand_tax": 5.0,
    }
    for name, weight in expected.items():
        assert name in cfg.rewards, f"缺失 reward: {name}"
        assert cfg.rewards[name].weight == weight, f"{name} 权重漂移"


def test_tuck_overrides_cover_neck_and_head():
    # latch 的物理前提是 "头顶朝下" 的抱头: 中途生成的 tuck 锚点必须仍在.
    cfg = make_microduck_roulade_env_cfg()
    tuck = cfg.events["set_roulade_state"].params["tuck_overrides"]
    assert 5 in tuck and 6 in tuck  # neck_pitch, head_pitch
