import math

import torch

from mjlab_microduck.tasks import mdp

# 包络: 4s 上 加速 0.5s / 稳态 1.6s / 制动 0.5s / 静止 1.4s.
_ENV = {"rate_max": 6.0, "accel_end": 0.125, "hold_end": 0.525, "brake_end": 0.650}


def test_spin_rate_segment_boundaries():
    # 4 个段的边界: 起步 0, [accel_end, hold_end] 上满转速,
    # 制动一开始仍是满转速, 静止段一开始就是 0.
    phase = torch.tensor([0.0, 0.125, 0.30, 0.525, 0.650, 0.80, 0.999])
    w = mdp.spin_rate_by_phase(phase, **_ENV)
    expected = torch.tensor([0.0, 6.0, 6.0, 6.0, 0.0, 0.0, 0.0])
    assert torch.allclose(w, expected, atol=1e-6)


def test_spin_rate_accel_ramp_is_increasing():
    phase = torch.linspace(0.0, 0.125, 20)
    w = mdp.spin_rate_by_phase(phase, **_ENV)
    assert torch.all(w[1:] >= w[:-1])
    # 启动斜坡中点 -> 目标的一半
    mid = mdp.spin_rate_by_phase(torch.tensor([0.0625]), **_ENV)
    assert torch.allclose(mid, torch.tensor([3.0]), atol=1e-6)


def test_spin_rate_brake_ramp_is_decreasing():
    phase = torch.linspace(0.525, 0.6499, 20)
    w = mdp.spin_rate_by_phase(phase, **_ENV)
    assert torch.all(w[1:] <= w[:-1])
    # 制动中点 -> 目标的一半
    mid = mdp.spin_rate_by_phase(torch.tensor([0.5875]), **_ENV)
    assert torch.allclose(mid, torch.tensor([3.0]), atol=1e-6)


def test_spin_rate_integral_matches_trapezoid_shape_at_rate_max_6():
    # 这个测试保护梯形的 形状 (每周期 2.1 * rate_max rad), 而不是真正
    # 派发的目标: 在 rate_max=6.0 (假设值, 见上面 _ENV) 下它约等于
    # 4*pi rad = 2 圈.精确包络 = 12.6 rad, 4*pi = 12.566 -> 1% 容差.
    # 真正生效的目标由下一个测试覆盖.
    n = 100_000
    phase = (torch.arange(n, dtype=torch.float64) + 0.5) / n
    w = mdp.spin_rate_by_phase(phase, **_ENV)
    integral = float(w.mean()) * 4.0
    assert abs(integral - 4 * math.pi) / (4 * math.pi) < 0.01


def test_spin_rate_max_integrates_to_2_1_times_itself_per_cycle():
    # 这个测试保护真正 派发 的目标 (mdp.SPIN_RATE_MAX), 与上面那个
    # 仅测 rate_max=6.0 形状的测试相对.一周期内包络下的面积等于
    # 2.1 * rate_max rad, 与 rate_max 无关 (0.25 + 1.6 + 0.25 = 2.1,
    # 见 mdp.py 常量上方的注释).当前设置 (SPIN_RATE_MAX = 3.0) 下
    # 它是 6.3 rad, 约 1 圈 —— 不是 2 圈.这个测试在有人不经思考就
    # 改目标 (那意味着多少圈) 时会响亮失败.
    n = 100_000
    phase = (torch.arange(n, dtype=torch.float64) + 0.5) / n
    w = mdp.spin_rate_by_phase(
        phase,
        rate_max=mdp.SPIN_RATE_MAX,
        accel_end=mdp.SPIN_ACCEL_END,
        hold_end=mdp.SPIN_HOLD_END,
        brake_end=mdp.SPIN_BRAKE_END,
    )
    integral = float(w.mean()) * mdp.SPIN_PERIOD
    expected = 2.1 * mdp.SPIN_RATE_MAX
    assert abs(integral - expected) / expected < 0.01


def test_spin_gate_is_normalized_rate():
    phase = torch.tensor([0.0, 0.0625, 0.30, 0.5875, 0.80])
    gate = mdp.spin_gate_by_phase(phase, **_ENV)
    rate = mdp.spin_rate_by_phase(phase, **_ENV)
    assert torch.allclose(gate, rate / 6.0, atol=1e-6)
    assert torch.all(gate >= 0.0) and torch.all(gate <= 1.0)


def test_spin_gate_is_zero_over_the_whole_rest_segment():
    # 静止段里任何 amorce 都不该推剪叉 -> gate 为零,
    # 这才能干净地从 trick 出口过渡到 roller policy.
    phase = torch.linspace(0.650, 0.999, 50)
    gate = mdp.spin_gate_by_phase(phase, **_ENV)
    assert torch.allclose(gate, torch.zeros_like(gate), atol=1e-6)


# ── 最小假 env: 让我们能在没有 MuJoCo 的情况下测 reward wrapper ────────────
class _FakeData:
    def __init__(self, ang_vel_b=None, lin_vel_b=None, joint_pos=None, joint_vel=None):
        self.root_link_ang_vel_b = ang_vel_b
        self.root_link_lin_vel_b = lin_vel_b
        self.joint_pos = joint_pos
        self.joint_vel = joint_vel


class _FakeEntity:
    """最小 Entity: find_joints() 按 {name: index} 字典按名解析."""

    def __init__(self, data, joint_ids=None):
        self.data = data
        self._joint_ids = joint_ids or {}

    def find_joints(self, pattern):
        import re

        names = list(self._joint_ids.keys())
        if isinstance(pattern, (list, tuple)):
            matched = [n for n in names if n in pattern]
        else:
            matched = [n for n in names if re.fullmatch(pattern, n)]
        assert matched, f"在 {names} 中没有 joint 匹配 {pattern!r}"
        return [self._joint_ids[n] for n in matched], matched


class _FakeCommandManager:
    def __init__(self, cmd):
        self._cmd = cmd

    def get_command(self, name):
        return self._cmd


class _FakeSensorData:
    def __init__(self, current_contact_time):
        self.current_contact_time = current_contact_time


class _FakeSensor:
    def __init__(self, current_contact_time):
        self.data = _FakeSensorData(current_contact_time)


class _FakeEnv:
    def __init__(self, entity, cmd=None, sensors=None):
        self.scene = {"robot": entity, **(sensors or {})}
        self.command_manager = _FakeCommandManager(cmd)
        self.device = "cpu"


def _phase_cmd(phases):
    """policy 看到的 slot 命令: [cos(2*pi*phi), sin(...), 0]."""
    p = torch.as_tensor(phases, dtype=torch.float32)
    return torch.stack(
        [torch.cos(2 * math.pi * p), torch.sin(2 * math.pi * p), torch.zeros_like(p)],
        dim=-1,
    )


# ── phase recover ────────────────────────────────────────────────────────────
def test_spin_phase_from_command_roundtrip():
    phases = torch.tensor([0.0, 0.125, 0.4, 0.65, 0.9])
    got = mdp.spin_phase_from_command(_phase_cmd(phases))
    assert torch.allclose(got, phases, atol=1e-5)


# ── spin_rate_track ──────────────────────────────────────────────────────────
def test_spin_rate_reward_peaks_on_exact_match():
    w = torch.tensor([6.0, 6.0])
    target = torch.tensor([6.0, 4.5])
    r = mdp.spin_rate_reward_from_values(w, target, std=1.5)
    # 误差 0 -> 1.0; 误差 = 1 std -> exp(-1)
    assert torch.allclose(r, torch.tensor([1.0, math.exp(-1.0)]), atol=1e-6)


def test_spin_rate_track_uses_yaw_and_phase():
    # phase 0.30 = 满转速 -> 目标 SPIN_RATE_MAX (3.0 rad/s, 这里隐式
    # 调用默认值).一个按目标转速的机器人应该得 1.0; 一个静止的机器
    # 人应该明显更低 (exp(-(3/1.5)^2) = 0.018, 在当前设置下: std=1.5
    # 在该目标上仍然校准良好, 见 mdp.py).
    ang = torch.tensor([[0.0, 0.0, mdp.SPIN_RATE_MAX], [0.0, 0.0, 0.0]])
    env = _FakeEnv(_FakeEntity(_FakeData(ang_vel_b=ang)), cmd=_phase_cmd([0.30, 0.30]))
    r = mdp.spin_rate_track(env, std=1.5)
    assert r[0] > 0.99
    assert r[1] < 0.05


def test_spin_rate_track_wants_stillness_during_rest():
    # phase 0.80 = 静止 -> 目标 0: 还在转就被罚, 静止才付费.
    ang = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 6.0]])
    env = _FakeEnv(_FakeEntity(_FakeData(ang_vel_b=ang)), cmd=_phase_cmd([0.80, 0.80]))
    r = mdp.spin_rate_track(env, std=1.5)
    assert r[0] > 0.99
    assert r[1] < 0.01


def test_spin_rate_track_penalizes_wrong_direction():
    # 要求 +SPIN_RATE_MAX 时以 -SPIN_RATE_MAX (顺时针) 转应该比静止更糟.
    ang = torch.tensor([[0.0, 0.0, -mdp.SPIN_RATE_MAX], [0.0, 0.0, 0.0]])
    env = _FakeEnv(_FakeEntity(_FakeData(ang_vel_b=ang)), cmd=_phase_cmd([0.30, 0.30]))
    r = mdp.spin_rate_track(env, std=1.5)
    assert r[0] < r[1]


# ── spin_rate_l1 ─────────────────────────────────────────────────────────────
def test_spin_rate_l1_is_negative_absolute_error():
    # phase 0.30 = 满转速 -> 目标 SPIN_RATE_MAX (3.0 rad/s, 默认).
    ang = torch.tensor([[0.0, 0.0, mdp.SPIN_RATE_MAX], [0.0, 0.0, 1.0]])
    env = _FakeEnv(_FakeEntity(_FakeData(ang_vel_b=ang)), cmd=_phase_cmd([0.30, 0.30]))
    r = mdp.spin_rate_l1(env)
    expected = torch.tensor([0.0, -(mdp.SPIN_RATE_MAX - 1.0)])
    assert torch.allclose(r, expected, atol=1e-5)


# ── spin_stay_in_place ───────────────────────────────────────────────────────
def test_spin_stay_in_place_is_squared_planar_speed():
    # phase 0.30 = 满转速 -> 全价成本
    lin = torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.4, 9.0]])
    env = _FakeEnv(_FakeEntity(_FakeData(lin_vel_b=lin)), cmd=_phase_cmd([0.30, 0.30]))
    c = mdp.spin_stay_in_place(env)
    # 0.3^2 + 0.4^2 = 0.25; z 分量被忽略
    assert torch.allclose(c, torch.tensor([0.0, 0.25]), atol=1e-6)


def test_spin_stay_in_place_is_attenuated_during_the_launch_ramp():
    # 同一速度, 两个 phase: 启动斜坡 (0.05 < accel_end) 里成本乘以
    # launch_scale, 稳态 (0.30) 全价.这才能让该项不阻止角动量注入.
    lin = torch.tensor([[0.3, 0.4, 0.0], [0.3, 0.4, 0.0]])
    env = _FakeEnv(_FakeEntity(_FakeData(lin_vel_b=lin)), cmd=_phase_cmd([0.05, 0.30]))
    c = mdp.spin_stay_in_place(env, launch_scale=0.2, accel_end=0.125)
    # 0.25 * 0.2 = 0.05
    assert torch.allclose(c, torch.tensor([0.05, 0.25]), atol=1e-6)
    assert c[0] < c[1]


def test_spin_stay_in_place_is_full_price_during_rest():
    # 静止段我们要机器人 IMMOBILE: 这项绝不能被关, 与 amorce
    # (spin_wheel_differential, spin_grounded, 剪叉) 相反.
    lin = torch.tensor([[0.3, 0.4, 0.0]])
    env = _FakeEnv(_FakeEntity(_FakeData(lin_vel_b=lin)), cmd=_phase_cmd([0.80]))
    c = mdp.spin_stay_in_place(env)
    assert torch.allclose(c, torch.tensor([0.25]), atol=1e-6)


# ── spin_wheel_differential ──────────────────────────────────────────────────
_WHEEL_IDS = {
    "passive_LF_wheel": 0,
    "passive_LR_wheel": 1,
    "passive_RF_wheel": 2,
    "passive_RR_wheel": 3,
}


def _wheel_env(vel_rows, phases):
    vel = torch.tensor(vel_rows, dtype=torch.float32)
    entity = _FakeEntity(_FakeData(joint_vel=vel), joint_ids=_WHEEL_IDS)
    return _FakeEnv(entity, cmd=_phase_cmd(phases))


def test_wheel_differential_rewards_counter_rolling_wheels():
    # 逆时针: 左轮为负 (滑动向后), 右轮为正 -> omega_D - omega_G > 0 -> 被奖励.
    env = _wheel_env(
        [
            [-10.0, -10.0, 10.0, 10.0],  # 好的差速
            [10.0, 10.0, 10.0, 10.0],  # 直行: 差速为零
            [10.0, 10.0, -10.0, -10.0],  # 反向差速 (顺时针)
        ],
        [0.30, 0.30, 0.30],
    )
    r = mdp.spin_wheel_differential(env, omega_scale=20.0)
    assert r[0] > 0.5
    assert torch.allclose(r[1], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(r[2], torch.tensor(0.0), atol=1e-6)


def test_wheel_differential_is_gated_off_during_rest():
    # 同样好的差速, 但在静止段 -> gate 为零 -> 不付费.
    env = _wheel_env([[-10.0, -10.0, 10.0, 10.0]], [0.80])
    r = mdp.spin_wheel_differential(env, omega_scale=20.0)
    assert torch.allclose(r, torch.zeros(1), atol=1e-6)


def test_wheel_differential_saturates():
    # tanh: 超过 omega_scale 后 reward 饱和, 不会出现速度军备竞赛.
    env = _wheel_env([[-10.0, -10.0, 10.0, 10.0], [-100.0, -100.0, 100.0, 100.0]], [0.30, 0.30])
    r = mdp.spin_wheel_differential(env, omega_scale=20.0)
    assert r[1] > r[0]
    assert r[1] <= 1.0


def test_wheel_differential_from_values_is_pure():
    diff = torch.tensor([20.0, 0.0, -20.0])
    gate = torch.ones(3)
    r = mdp.spin_wheel_differential_from_values(diff, gate, omega_scale=20.0)
    expected = torch.tensor([math.tanh(1.0), 0.0, 0.0])
    assert torch.allclose(r, expected, atol=1e-6)


# ── spin_grounded ────────────────────────────────────────────────────────────
def test_spin_grounded_rewards_both_blades_down_and_is_gated():
    contact = torch.tensor([[0.2, 0.3], [0.2, 0.0], [0.0, 0.0], [0.2, 0.3]])
    entity = _FakeEntity(_FakeData())
    env = _FakeEnv(
        entity,
        cmd=_phase_cmd([0.30, 0.30, 0.30, 0.80]),
        sensors={"feet_ground_contact": _FakeSensor(contact)},
    )
    r = mdp.spin_grounded(env, sensor_name="feet_ground_contact")
    # 稳态下两片刀片着地 -> gate 1.0; 只有一片或零片 -> 0;
    # 两片着地但在静止段 -> gate 0.
    assert torch.allclose(r, torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-6)


# ── leg_antisymmetry ─────────────────────────────────────────────────────────
_LEG_IDS = {
    "left_hip_pitch": 0,
    "left_knee": 1,
    "right_hip_pitch": 2,
    "right_knee": 3,
}


def _leg_env(pos_rows, phases):
    pos = torch.tensor(pos_rows, dtype=torch.float32)
    entity = _FakeEntity(_FakeData(joint_pos=pos), joint_ids=_LEG_IDS)
    return _FakeEnv(entity, cmd=_phase_cmd(phases))


def test_leg_antisymmetry_prefers_scissor_over_mirror():
    # 镜像约定: q_G = -q_D 是 SYMMETRIC pose (这里不好),
    # q_G = q_D 是 SCISSOR (这里好).值 = -mean|q_G - q_D|, 所以 <= 0.
    env = _leg_env(
        [
            [0.4, 0.3, 0.4, 0.3],  # 完美剪叉: q_G == q_D -> 0.0
            [0.4, 0.3, -0.4, -0.3],  # 镜像: 偏差 0.8 和 0.6 -> -0.7
        ],
        [0.30, 0.30],
    )
    r = mdp.leg_antisymmetry(env)
    assert torch.allclose(r, torch.tensor([0.0, -0.7]), atol=1e-6)
    assert r[0] > r[1]


def test_leg_antisymmetry_is_gated_off_during_rest():
    # 静止段 gate 为零: 没什么推剪叉, 中性自由站位.
    env = _leg_env([[0.4, 0.3, -0.4, -0.3]], [0.80])
    r = mdp.leg_antisymmetry(env)
    assert torch.allclose(r, torch.zeros(1), atol=1e-6)


# ── neck_joint_pos_l2: pattern 参数 ──────────────────────────────────────────
_NECK_IDS = {
    "neck_pitch": 0,
    "head_pitch": 1,
    "head_roll": 2,
    "head_yaw": 3,
}


def test_neck_joint_pos_l2_pattern_can_exclude_head_yaw():
    class _NeckData(_FakeData):
        def __init__(self, joint_pos, default_joint_pos):
            super().__init__(joint_pos=joint_pos)
            self.default_joint_pos = default_joint_pos

    pos = torch.tensor([[0.0, 0.0, 0.0, 1.0]])  # 只有 head_yaw 偏 1 rad
    default = torch.zeros(1, 4)
    entity = _FakeEntity(_NeckData(pos, default), joint_ids=_NECK_IDS)
    env = _FakeEnv(entity)

    # 默认 pattern: head_yaw 算 -> 成本 1.0
    assert torch.allclose(mdp.neck_joint_pos_l2(env), torch.tensor([1.0]), atol=1e-6)
    # spin 的 pattern: head_yaw 被排除 -> 成本 0.0 (头可自由偏航)
    assert torch.allclose(
        mdp.neck_joint_pos_l2(env, pattern=r"^(neck_pitch|head_pitch|head_roll)$"),
        torch.tensor([0.0]),
        atol=1e-6,
    )
