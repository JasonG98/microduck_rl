from mjlab_microduck.tasks.microduck_roller_slope_env_cfg import make_microduck_roller_slope_env_cfg
from mjlab_microduck.tasks.slope_terrain import FlatRampTerrainCfg


def test_terrain_is_flat_ramp_generator():
    cfg = make_microduck_roller_slope_env_cfg()
    assert cfg.scene.terrain.terrain_type == "generator"
    gen = cfg.scene.terrain.terrain_generator
    assert gen is not None and gen.curriculum is True
    assert any(isinstance(st, FlatRampTerrainCfg) for st in gen.sub_terrains.values())


def test_command_is_neutralised():
    cfg = make_microduck_roller_slope_env_cfg()
    cmd = cfg.commands["twist"]
    assert cmd.rel_standing_envs == 1.0
    assert cmd.rel_heading_envs == 0.0
    assert cmd.ranges.lin_vel_x == (0.0, 0.0)
    assert cmd.ranges.lin_vel_y == (0.0, 0.0)
    if getattr(cmd.ranges, "ang_vel_z", None) is not None:
        assert cmd.ranges.ang_vel_z == (0.0, 0.0)


def test_rolling_entry_no_base_push():
    # 助推以滚动方式给 (reset_rolling_entry), 不是 base 推力
    # (只有 base + 静止轮 = 滑动急停).所以 reset_base 不设
    # 任何 base 速度.
    cfg = make_microduck_roller_slope_env_cfg()
    assert cfg.events["reset_base"].params["velocity_range"] == {}
    assert "reset_rolling_entry" in cfg.events
    lo, hi = cfg.events["reset_rolling_entry"].params["speed_range"]
    assert 0.0 < lo <= hi <= 0.6


def test_has_heading_hold_reward():
    # 走直线: 维持 spawn 时的 yaw
    cfg = make_microduck_roller_slope_env_cfg()
    assert "heading_hold" in cfg.rewards
    assert cfg.rewards["heading_hold"].weight > 0.0


def test_balance_rewards_no_fixed_pose():
    # 自由平衡: upright/alive/滑行 在, 但不施加固定 pose
    # (它必须能挪重心才能稳在坡上).
    cfg = make_microduck_roller_slope_env_cfg()
    for name in ("upright", "alive", "feet_flat", "wheel_glide", "neck_joint_pos_l2"):
        assert name in cfg.rewards
    assert "standing_pose" not in cfg.rewards
    assert "standing_pose_l1" not in cfg.rewards


def test_has_wheel_glide_reward_not_base_speed():
    # "顺坡滑" = 滚动 (轮子), 不是奖励 base 速度
    # (它靠跑来达到那个速度).wheel_glide 在, descent_speed 不在.
    cfg = make_microduck_roller_slope_env_cfg()
    assert "wheel_glide" in cfg.rewards
    assert cfg.rewards["wheel_glide"].weight > 0.0
    assert "descent_speed" not in cfg.rewards


def test_no_roller_skating_rewards_survive():
    # roller 的滑行 reward 不能残留 (heading_hold 是为了直走特意重新加的,
    # 所以不在这份清单里).
    cfg = make_microduck_roller_slope_env_cfg()
    for name in ("wheel_speed", "braking", "skating_air_time", "glide", "forward_lean"):
        assert name not in cfg.rewards


def test_spawn_yaw_faces_downhill():
    # yaw 固定为 0: 始终面朝坡底 (+x), 不是继承来的 -pi/+pi
    cfg = make_microduck_roller_slope_env_cfg()
    assert cfg.events["reset_base"].params["pose_range"]["yaw"] == (0.0, 0.0)


def test_void_termination_present_no_edge_termination():
    cfg = make_microduck_roller_slope_env_cfg()
    assert "fell_into_void" in cfg.terminations
    assert "fell_over" in cfg.terminations
    # 不再有 "地形边缘" 终止 (被出口平地取代)
    assert "reached_bottom" not in cfg.terminations
    assert "out_of_terrain_bounds" not in cfg.terminations


def test_obs_nan_policy_sanitize():
    # obs 已 sanitize: 罕见 contact 的 NaN 不应杀掉训练
    cfg = make_microduck_roller_slope_env_cfg()
    assert cfg.observations["actor"].nan_policy == "sanitize"
    assert cfg.observations["critic"].nan_policy == "sanitize"


def test_curriculum_present_and_starts_gentle():
    # curriculum 从平缓到陡: 起步在最缓的坡, 主动晋升
    cfg = make_microduck_roller_slope_env_cfg()  # play=False (训练)
    assert "terrain_levels" in cfg.curriculum
    assert cfg.scene.terrain.max_init_terrain_level == 0


def test_terrain_tile_fits_geometry():
    # 地砖必须容下 平地 + ramp_max + 出口
    cfg = make_microduck_roller_slope_env_cfg()
    gen = cfg.scene.terrain.terrain_generator
    st = next(iter(gen.sub_terrains.values()))
    assert st.flat_length + st.ramp_length_range[1] + st.runout_length <= gen.size[0]
