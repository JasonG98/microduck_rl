from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_spin_env_cfg import MicroduckSpinRlCfg, make_microduck_spin_env_cfg


def test_cfg_uses_phase_command_with_runtime_default_period():
    cfg = make_microduck_spin_env_cfg()
    cmd = cfg.commands["twist"]
    assert isinstance(cmd, microduck_mdp.GroundPickPhaseCommandCfg)
    # 4.0 s = --ground-pick-period 的默认: runtime 不用传任何东西
    assert cmd.period == 4.0
    # 每个 episode 从 phase 0 起跑 (站立), 像部署时的按钮
    assert cmd.randomize_phase is False


def test_cfg_has_the_spin_rewards():
    cfg = make_microduck_spin_env_cfg()
    for name in (
        "spin_rate_track",
        "spin_rate_l1",
        "spin_stay_in_place",
        "spin_wheel_differential",
        "spin_grounded",
        "leg_antisymmetry",
    ):
        assert name in cfg.rewards, name
    # 主目标带一个主导权重
    assert cfg.rewards["spin_rate_track"].weight == 6.0
    # 原地是 COST
    assert cfg.rewards["spin_stay_in_place"].weight < 0.0


def test_stay_in_place_is_attenuated_during_the_launch_ramp():
    # 加强到 -3.0 后, 如果启动斜坡里全价, 这项会反对角动量注入:
    # 在那里必须衰减.
    cfg = make_microduck_spin_env_cfg()
    params = cfg.rewards["spin_stay_in_place"].params
    assert 0.0 < params["launch_scale"] < 1.0
    assert params["accel_end"] == microduck_mdp.SPIN_ACCEL_END
    # 正目标 = 逆时针 (方向由包络携带)
    assert microduck_mdp.SPIN_RATE_MAX > 0.0


def test_angular_momentum_reward_is_removed():
    # 回归: angular_momentum_penalty 惩罚角动量的 3D 范数, 它会直接对抗
    # 旋转.它必须不存在.
    cfg = make_microduck_spin_env_cfg()
    assert "angular_momentum" not in cfg.rewards
    # body_ang_vel 只罚 x/y -> 它留下, 压住甩摆
    assert "body_ang_vel" in cfg.rewards


def test_head_yaw_is_free_to_act_as_a_flywheel():
    cfg = make_microduck_spin_env_cfg()
    pattern = cfg.rewards["neck_joint_pos_l2"].params["pattern"]
    assert "head_yaw" not in pattern


def test_entry_velocity_allows_standstill_and_slow_roll():
    cfg = make_microduck_spin_env_cfg()
    # 永远不能走 mode="reset" 的 push (crouch 的 NaN 回归)
    assert "entry_velocity" not in cfg.events
    lo, hi = cfg.events["reset_base"].params["velocity_range"]["x"]
    assert lo == 0.0 and hi > 0.0


def test_symmetry_augmentation_is_disabled():
    # G/D 对称会把一个左旋变成右旋
    assert MicroduckSpinRlCfg.algorithm.symmetry_cfg is None


def test_leg_antisymmetry_shaping_decays():
    cfg = make_microduck_spin_env_cfg()
    stages = cfg.curriculum["leg_antisym_weight"].params["weight_stages"]
    weights = [s["weight"] for s in stages]
    assert weights[0] == cfg.rewards["leg_antisymmetry"].weight
    assert weights == sorted(weights, reverse=True)
    assert weights[-1] < weights[0]


def test_actor_observation_keeps_the_61d_slot_layout():
    # ONNX 能加载进 runtime slot 的条件.与 crouch 维度精确相等由下面
    # 的 test_obs_parity_with_roller_crouch 检查; 这里查结构.
    cfg = make_microduck_spin_env_cfg()
    terms = cfg.observations["actor"].terms
    assert "base_lin_vel" not in terms
    assert "height_scan" not in terms
    for padded in ("head_command", "body_command"):
        assert padded in terms
    assert terms["head_command"].params["dim"] == 4
    assert terms["body_command"].params["dim"] == 6


def test_obs_parity_with_roller_crouch():
    # layout 对齐是硬性的: 否则导出的 ONNX 加载不进 runtime slot.
    # 与上面的结构测试相对, 这个按组比较 term 的 精确 顺序.
    from mjlab_microduck.tasks.microduck_roller_crouch_env_cfg import (
        make_microduck_roller_crouch_env_cfg,
    )

    spin = make_microduck_spin_env_cfg()
    crouch = make_microduck_roller_crouch_env_cfg()
    for grp in ("actor", "critic"):
        assert list(spin.observations[grp].terms.keys()) == list(crouch.observations[grp].terms.keys()), (
            f"observation layout 在 {grp} 组上分叉了"
        )
