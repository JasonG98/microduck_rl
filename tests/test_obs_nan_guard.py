"""critic obs 必须能扛住非有限的 sensor 读数.

针对 2026-08-21 崩溃的回归: rsl_rl 的 check_nan 用 "observation group
'critic' contains NaN" 杀掉了一次 Velocity2-Rough-Backlash 训练.
`nan_state` (robot_state_is_nan) 只覆盖 joint + root state, 但 critic
还带三个 SENSOR 衍生项 (raycast heights, contact air-time, contact
forces).MuJoCo 可以在积分后的机器人状态仍然干净时返回一个非有限
的 contact force, 所以 env 永不 reset, NaN 流到了 runner.
"""

import torch

from mjlab_microduck.tasks import mdp as microduck_mdp


class _SensorData:
    def __init__(self, force=None, heights=None):
        self.force = force
        self.heights = heights


class _Sensor:
    def __init__(self, data):
        self.data = data


class _Scene:
    def __init__(self, sensors, asset):
        self.sensors = sensors
        self._asset = asset

    def __getitem__(self, key):
        return self.sensors.get(key, self._asset)


class _AssetData:
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


class _Env:
    def __init__(self, n, force):
        self.num_envs = n
        self.device = "cpu"
        asset = _Asset(_AssetData(n))
        self.scene = _Scene({"feet": _Sensor(_SensorData(force=force))}, asset)


def _force(n, bad_env=None, value=float("nan")):
    f = torch.ones(n, 2, 3)
    if bad_env is not None:
        f[bad_env, 0, 0] = value
    return f


def test_state_only_check_misses_bad_contact_force():
    # 这就是杀掉训练的缺口: 机器人状态干净, force 不干净.
    env = _Env(3, _force(3, bad_env=1))
    assert not microduck_mdp.robot_state_is_nan(env).any()


def test_termination_catches_nan_contact_force():
    env = _Env(3, _force(3, bad_env=1))
    out = microduck_mdp.robot_state_is_nan(env, sensor_names=("feet",))
    assert out.tolist() == [False, True, False]


def test_termination_catches_inf_contact_force():
    env = _Env(3, _force(3, bad_env=2, value=float("inf")))
    out = microduck_mdp.robot_state_is_nan(env, sensor_names=("feet",))
    assert out.tolist() == [False, False, True]


def test_termination_ignores_missing_sensor():
    env = _Env(2, _force(2))
    assert not microduck_mdp.robot_state_is_nan(env, sensor_names=("nope",)).any()


def test_finite_helper_sanitizes_nan_and_inf():
    x = torch.tensor([[1.0, float("nan"), float("inf"), float("-inf")]])
    out = microduck_mdp._finite(x)
    assert torch.isfinite(out).all()
    assert out[0, 0] == 1.0


def test_safe_obs_wrappers_are_wired_into_the_critic():
    # guard 必须真的装在 env cfg 上, 不能只是存在.
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
        make_microduck_velocity_env_cfg,
    )

    cfg = make_microduck_velocity_env_cfg(rough=True)
    terms = cfg.observations["critic"].terms
    for name in ("foot_contact_forces", "foot_height", "foot_air_time"):
        assert terms[name].func.__name__.endswith("_safe"), f"critic/{name} 丢了 NaN guard"


def test_nan_state_termination_watches_the_contact_sensor():
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
        make_microduck_velocity_env_cfg,
    )

    cfg = make_microduck_velocity_env_cfg(rough=True)
    params = cfg.terminations["nan_state"].params
    assert params.get("sensor_names"), "nan_state 不再监视 contact forces"


def test_standup_env_is_also_guarded():
    # 部署的站立 policy 在 StandUp 上训练, 后者基于 mjlab 的 base env (不
    # 是 microduck velocity env) 构建, 因此不继承那里接的 guard.
    from mjlab_microduck.tasks.microduck_standup_env_cfg import (
        make_microduck_standup_env_cfg,
    )

    cfg = make_microduck_standup_env_cfg()
    terms = cfg.observations["critic"].terms
    for name in ("foot_contact_forces", "foot_air_time"):
        assert terms[name].func.__name__.endswith("_safe"), f"standup critic/{name} 丢了 NaN guard"
    assert cfg.terminations["nan_state"].params.get("sensor_names"), (
        "standup nan_state 不再监视 contact forces"
    )
