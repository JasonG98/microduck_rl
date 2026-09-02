from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks.microduck_velocity_rollers_env_cfg import make_microduck_velocity_rollers_env_cfg
from mjlab_microduck.tasks.microduck_velocity_swizzle_env_cfg import make_microduck_velocity_swizzle_env_cfg


def test_swizzle_head_control_wired():
    cfg = make_microduck_velocity_swizzle_env_cfg()
    roller_cfg = make_microduck_velocity_rollers_env_cfg()

    # head-pose 命令项存在.
    assert "head_pose" in cfg.commands

    # head_command obs 是真实命令 (非零填充), 在两个组上都是.
    for group in ("actor", "critic"):
        term = cfg.observations[group].terms["head_command"]
        assert term.func is mdp.generated_commands
        assert term.params["command_name"] == "head_pose"

    # head_pose_tracking reward 存在.
    assert "head_pose_tracking" in cfg.rewards

    # 两个会与 head 命令对抗的 HOME-puller 已被处理:
    #  - neck_joint_pos_l2 已移除
    assert "neck_joint_pos_l2" not in cfg.rewards
    #  - pose reward 通过一个负向 lookahead regex 限定到腿关节, 排除
    #    neck/head (以及被动轮)
    pose_joints = cfg.rewards["pose"].params["asset_cfg"].joint_names
    assert any("(?!" in j and "neck" in j and "head" in j for j in pose_joints), (
        f"pose reward 没有从 neck/head 上 scope 走: {pose_joints}"
    )

    # pose reward 函数未变 (没被换成另一个函数)
    assert cfg.rewards["pose"].func is roller_cfg.rewards["pose"].func, (
        "pose reward 函数被换了 (应该只 scope asset_cfg)"
    )

    # 晚期 head curriculum 存在.
    assert "head_pose_tracking_weight" in cfg.curriculum
    assert "head_pose_range" in cfg.curriculum
