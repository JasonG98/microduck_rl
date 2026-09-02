"""head_pose_bias_penalty: 惩罚持续的站立下垂, 永不惩罚恢复.

velocity-env 的教训 (run 5yay13u4): 瞬时姿态精度是对运动的不可避免税.
standup 的教训 (退役的 head_impact_penalty): 任何在地面阶段激活的头部
代价都会阻止 head-pivot 翻转.所以这项必须 (a) 只计 DC 偏置, (b) 倒地
时绝不累积, (c) 站立到达后从 ~0 起表.
"""

import math

import torch

from mjlab_microduck.tasks import mdp as microduck_mdp


class _Data:
    def __init__(self, n):
        self.joint_pos = torch.zeros(n, 6)
        self.default_joint_pos = torch.zeros(n, 6)
        self.root_link_pos_w = torch.zeros(n, 3)
        self.root_link_quat_w = torch.zeros(n, 4)
        self.root_link_quat_w[:, 0] = 1.0  # upright


class _Asset:
    def __init__(self, data):
        self.data = data


class _Terrain:
    def __init__(self, n):
        self.env_origins = torch.zeros(n, 3)


class _Scene:
    def __init__(self, asset, n):
        self._asset = asset
        self.terrain = _Terrain(n)

    def __getitem__(self, _):
        return self._asset


class _Cmd:
    def __init__(self, n, dim=4):
        self.cmd = torch.zeros(n, dim)

    def get_command(self, _):
        return self.cmd


class _Env:
    def __init__(self, n):
        self.num_envs = n
        self.device = "cpu"
        self.step_dt = 0.02
        self.episode_length_buf = torch.full((n,), 10, dtype=torch.long)
        self.scene = _Scene(_Asset(_Data(n)), n)
        self.command_manager = _Cmd(n)
        # 预填 neck-id 缓存 (正常情况下从真实 asset 构建).
        self._head_pose_neck_ids = torch.tensor([0, 1, 2, 3])
        self._head_pose_bl_ids = torch.tensor([0, 0, 0, 0])
        self._head_pose_bl_mask = torch.zeros(4)


GATE = {
    "gate_height_low": 0.09,
    "gate_height_high": 0.11,
    "gate_tilt_full_deg": 20.0,
    "gate_tilt_zero_deg": 45.0,
}


def _set_pose(env, z, pitch_deg):
    env.scene._asset.data.root_link_pos_w[:, 2] = z
    half = math.radians(pitch_deg) / 2
    q = torch.tensor([math.cos(half), 0.0, math.sin(half), 0.0])
    env.scene._asset.data.root_link_quat_w[:] = q


def _run(env, steps, **kw):
    out = None
    for _ in range(steps):
        out = microduck_mdp.head_pose_bias_penalty(env, tau_s=1.0, **kw)
    return out


def test_prone_accumulates_nothing():
    env = _Env(2)
    _set_pose(env, z=0.05, pitch_deg=90.0)  # 趴在地上
    env.scene._asset.data.joint_pos[:, :4] = 0.5  # 巨大头部 "误差" (28°)
    out = _run(env, 200, **GATE)  # 4 s 的趴地挣扎
    assert torch.allclose(out, torch.zeros(2), atol=1e-9), out


def test_arrival_starts_from_zero_then_charges_true_bias():
    env = _Env(2)
    _set_pose(env, z=0.05, pitch_deg=90.0)
    env.scene._asset.data.joint_pos[:, :4] = 0.5
    _run(env, 200, **GATE)  # 地面阶段: EMA 保持 0
    _set_pose(env, z=0.117, pitch_deg=0.0)  # 恢复完成
    env.scene._asset.data.joint_pos[:, :4] = math.radians(15)  # 15° 下垂
    first = _run(env, 1, **GATE)
    assert first.abs().max() < 0.01, f"终点墙: {first}"  # 没有到达尖峰
    settled = _run(env, 300, **GATE)  # 6 s 站立
    assert abs(-settled[0].item() - math.radians(15)) < 0.01  # 计下垂


def test_fall_stops_the_charge_immediately():
    env = _Env(2)
    _set_pose(env, z=0.117, pitch_deg=0.0)
    env.scene._asset.data.joint_pos[:, :4] = math.radians(15)
    _run(env, 300, **GATE)
    _set_pose(env, z=0.05, pitch_deg=90.0)  # 摔倒
    out = _run(env, 1, **GATE)
    assert torch.allclose(out, torch.zeros(2), atol=1e-9), out


def test_ungated_matches_velocity_env_behavior():
    # 无 gate 参数 -> 原始误差的普通 EMA (velocity-env 的项).
    env = _Env(2)
    _set_pose(env, z=0.05, pitch_deg=90.0)  # 姿态必须无关
    env.scene._asset.data.joint_pos[:, :4] = math.radians(15)
    out = _run(env, 300)
    assert abs(-out[0].item() - math.radians(15)) < 0.01


def test_reset_clears_the_ema():
    env = _Env(2)
    _set_pose(env, z=0.117, pitch_deg=0.0)
    env.scene._asset.data.joint_pos[:, :4] = math.radians(15)
    _run(env, 300, **GATE)
    env.episode_length_buf[0] = 1  # env 0 刚重置
    out = _run(env, 1, **GATE)
    # 重置后一步 EMA 恰好持有 alpha*err (~0.005), 而未重置 env 仍带完整
    # 稳定偏置 (~0.26).
    assert out[0].abs() < 0.01 and out[1].abs() > 0.2


def test_standup_cfg_wiring():
    from mjlab_microduck.tasks.microduck_standup_env_cfg import (
        make_microduck_standup_env_cfg,
    )

    cfg = make_microduck_standup_env_cfg()
    term = cfg.rewards["head_pose_bias"]
    assert term.weight == 0.0  # 探索阶段未动
    assert term.params["gate_height_low"] is not None
    # gate 值与 arrival_damping 相同, 所以 "站立" 是同一回事.
    ad = cfg.rewards["arrival_damping"].params
    assert term.params["gate_height_low"] == ad["height_low"]
    assert term.params["gate_tilt_full_deg"] == ad["tilt_full_deg"]
    stages = cfg.curriculum["head_pose_bias_weight"].params["weight_stages"]
    assert stages[0]["weight"] == 0.0
    assert min(s["step"] for s in stages if s["weight"] > 0) >= 3000 * 24


def test_velocity_cfg_unchanged_no_gate():
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
        make_microduck_velocity_env_cfg,
    )

    cfg = make_microduck_velocity_env_cfg()
    assert "gate_height_low" not in cfg.rewards["head_pose_bias"].params


def test_velstand_inherited_term_is_gated():
    # Velstand episode 能挺过摔倒 —— 继承自 velocity-env 的项不能计地面阶段.
    from mjlab_microduck.tasks.microduck_velstand_env_cfg import (
        make_microduck_velstand_env_cfg,
    )

    cfg = make_microduck_velstand_env_cfg()
    params = cfg.rewards["head_pose_bias"].params
    assert params.get("gate_height_low") is not None
    assert params["gate_tilt_zero_deg"] == 40.0  # REWARD_GATE_TILT_DEG
