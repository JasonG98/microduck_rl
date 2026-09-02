from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_roller_crouch_env_cfg import make_microduck_roller_crouch_env_cfg


def test_cfg_uses_phase_command():
    cfg = make_microduck_roller_crouch_env_cfg()
    cmd = cfg.commands["twist"]
    assert isinstance(cmd, microduck_mdp.GroundPickPhaseCommandCfg)
    # period 必须匹配部署时的 --ground-pick-period
    # (0.5s 下蹲 + 2s 低姿 + 0.5s 上行 + 2s 站立 = 5s)
    assert cmd.period == 5.0
    # 每个 episode 从站立开始 (phase 0), 与 runtime 触发一致
    assert cmd.randomize_phase is False


def test_cfg_has_crouch_and_forward_rewards():
    cfg = make_microduck_roller_crouch_env_cfg()
    # pose 目标 (站立<->蹲) + L1 bootstrap
    assert "crouch_glide_pose" in cfg.rewards
    assert "crouch_glide_pose_l1" in cfg.rewards
    assert "forward_speed" in cfg.rewards
    # 蹲下过程中轻微前倾 (target 正值 = 向前)
    assert "crouch_forward_lean" in cfg.rewards
    assert cfg.rewards["crouch_forward_lean"].params["target_pitch"] > 0.0
    # 蹲姿按名携带并含腿弯折
    cp = cfg.rewards["crouch_glide_pose"].params["crouch_pose"]
    assert "left_knee" in cp and "right_knee" in cp
    # 主动滑行的 reward 已移除 (trick 期间不迈步)
    for gone in (
        "braking",
        "skating_air_time",
        "single_support",
        "glide",
        "wheel_speed",
    ):
        assert gone not in cfg.rewards


def test_entry_velocity_applied_safely_via_reset_base():
    # 回归: 入场动量必须通过 reset_base 的 velocity_range 注入
    # (reset_root_state_uniform 从干净默认状态设置它), 而不是用一个
    # mode="reset" 的 push_by_setting_velocity 事件, 后者会加到当前
    # (可能已发散的) root velocity 上, 把 base free-joint 顶到 NaN.
    # 见 env cfg 上 ENTRY_VELOCITY_X 的注释.
    cfg = make_microduck_roller_crouch_env_cfg()
    # 出 bug 的 reset-push 事件必须不存在
    assert "entry_velocity" not in cfg.events
    # 向前入场速度必须由 reset_base 携带, 且范围是正值
    vr = cfg.events["reset_base"].params.get("velocity_range")
    assert vr and "x" in vr
    lo, hi = vr["x"]
    assert lo > 0.0 and hi >= lo
