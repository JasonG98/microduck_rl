"""robot_state_is_nan 必须捕捉任何地方 (关节 或 base 或 轮子) 的非有限
状态, 不能只看 joint_pos —— 否则一个 free-joint 在 NaN 上发散会逃过
reset 并污染 critic obs (base_lin_vel/wheel_vel), 然后通过 rsl_rl 的
check_nan 杀掉训练.
"""

import torch

from mjlab_microduck.tasks.mdp import robot_state_is_nan


class _Data:
    def __init__(self, n):
        self.joint_pos = torch.zeros(n, 4)
        self.joint_vel = torch.zeros(n, 4)
        self.root_link_pos_w = torch.zeros(n, 3)
        self.root_link_quat_w = torch.zeros(n, 4)
        self.root_link_lin_vel_w = torch.zeros(n, 3)
        self.root_link_ang_vel_w = torch.zeros(n, 3)


class _Asset:
    def __init__(self, data):
        self.data = data


class _Scene:
    def __init__(self, asset):
        self._a = asset

    def __getitem__(self, _key):
        return self._a


class _Env:
    def __init__(self, data):
        self.scene = _Scene(_Asset(data))


def test_catches_base_linear_velocity_nan():
    # env 1: base 速度 NaN (free-joint 发散) —— joint_pos 仍有限.
    d = _Data(3)
    d.root_link_lin_vel_w[1, 0] = float("nan")
    out = robot_state_is_nan(_Env(d))
    assert out.tolist() == [False, True, False]


def test_catches_base_velocity_inf():
    # base 角速度里的 inf (在它变 NaN 之前).
    d = _Data(2)
    d.root_link_ang_vel_w[0, 2] = float("inf")
    out = robot_state_is_nan(_Env(d))
    assert out.tolist() == [True, False]


def test_still_catches_joint_pos_nan():
    # 历史行为保留.
    d = _Data(2)
    d.joint_pos[0, 1] = float("nan")
    out = robot_state_is_nan(_Env(d))
    assert out.tolist() == [True, False]


def test_clean_state_is_not_flagged():
    out = robot_state_is_nan(_Env(_Data(4)))
    assert out.tolist() == [False, False, False, False]
