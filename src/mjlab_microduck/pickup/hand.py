"""批量 virtual hand: 抓住 trunk 把机器人提起, 搬运/倾斜/摇晃, 然后放下
(下放到脚触地, 短暂延迟后松手) 或直接丢下.

纯 torch 的 (N,) 状态, 因此同一份代码既驱动 MuJoCo-Warp 数据生成器, 也驱动单 env 的
CPU demo. 手是一个 mocap body, 通过 SOFT equality constraint (``add_hand_to_spec``) 焊接到
trunk, 由 MuJoCo 在每个物理子步隐式求解 — 像人一样的柔顺抓握. (以 50 Hz 更新的显式 PD wrench
不稳定: 经过软舵机后 trunk 在高频下的等效惯量极小, 任何有用的转动阻尼都会过冲.)
``VirtualHand.step`` 只产生手的目标姿态 + active 掩码.

源自 ``handled_robot`` 分支的 ``virtual_hand_pickup`` step event, 并按人实际摆弄鸭子的方式
扩展: 任意倾斜 (直到倒置), 摇晃, 抖动, 快速短促的提起, 以及轻缓的放下.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

IDLE, CARRY, LOWER = 0, 1, 2


@dataclass
class HandCfg:
    pickup_rate_hz: float = 0.08        # idle env 以此速率发起一次 pick-up
    min_idle_s: float = 1.0             # 松手后 / episode 开始后
    lift_z: tuple = (0.03, 0.35)        # 相对抓取时 trunk 高度的抬升量
    lift_s: tuple = (0.25, 3.0)          # v2: 也包含慢速提起 (抓头就是一次轻缓的拉拽)
    hold_s: tuple = (0.5, 8.0)
    # 倾斜幅度分桶 (deg, 概率): 多数随手, 偶尔剧烈
    tilt_buckets: tuple = ((15.0, 0.35), (45.0, 0.30), (90.0, 0.25), (160.0, 0.10))
    tilt_freq_hz: tuple = (0.05, 0.6)
    yaw_rate: tuple = (-1.0, 1.0)
    drift_amp: tuple = (0.0, 0.25)      # m, 平滑的搬运位移
    drift_freq_hz: tuple = (0.05, 0.5)
    shake_prob: float = 0.35
    shake_amp: tuple = (0.005, 0.03)    # m
    shake_freq_hz: tuple = (1.5, 5.0)
    tremor_amp: tuple = (0.0003, 0.002)  # m, 生理性抖动 8-12 Hz
    tremor_rot_deg: tuple = (0.1, 0.8)
    drop_prob: float = 0.10             # 在高处直接松手, 而不是放下
    lower_speed: tuple = (0.04, 0.4)    # m/s, 触地前的下降速度
    putdown_tilt_deg: float = 8.0       # 放下时的残余倾斜
    touch_release_s: tuple = (0.05, 0.6)  # 脚触地后手再过这么久才松开
    touch_force_frac: float = 0.25      # 脚承担该比例体重即判定 "touched"
    max_lower_s: float = 6.0
    pickup_fallen: bool = True          # 人也会把倒地的机器人捡起来
    touch_press: tuple = (0.0, 0.01)    # m, 触地后手继续下压的量
    # v2 (真机反馈 2026-10-03: 被 HEAD 提起时从不暂停, 转 180° 反而恢复了)
    head_grip_prob: float = 0.35        # 抓头 (jaw_soft) 而不是 trunk
    orient_prob: float = 0.35           # 在球面上任意位置保持一个持续朝向搬运
    orient_pitch: tuple = (-math.pi, math.pi)
    orient_roll: tuple = (-math.pi / 2, math.pi / 2)
    orient_ramp_s: tuple = (0.4, 2.0)   # 手把鸭子转到该朝向的速度
    yaw_turn_prob: float = 0.35         # 绕竖直轴的快速旋转, 最大 ±180°
    yaw_turn: tuple = (-math.pi, math.pi)
    yaw_turn_s: tuple = (0.4, 1.5)


def _u(n, lo_hi, device):
    lo, hi = lo_hi
    return torch.empty(n, device=device).uniform_(lo, hi)


def _quat_from_euler_xyz(roll, pitch, yaw):
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    return torch.stack([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dim=-1)


def _quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def _quat_conj(q):
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _axis_angle(q):
    """单位四元数的旋转向量 (最短路径)."""
    q = torch.where(q[..., :1] < 0, -q, q)
    v = q[..., 1:]
    s = v.norm(dim=-1, keepdim=True)
    ang = 2.0 * torch.atan2(s, q[..., :1])
    return torch.where(s > 1e-8, v / s.clamp_min(1e-8) * ang, 2.0 * v)


def yaw_of(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class VirtualHand:
    """状态机 IDLE → CARRY → (LOWER → touch → release | drop) → IDLE."""

    # 平滑噪声通道: roll, pitch, x, y, z ; 每个由两条正弦组成
    _NCH = 5

    def __init__(self, num_envs: int, device, cfg: HandCfg | None = None):
        self.cfg = cfg or HandCfg()
        self.N, self.dev = num_envs, device
        z = lambda *s: torch.zeros(num_envs, *s, device=device)
        self.phase = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.t = z(); self.idle_t = z(); self.t_hold = z(); self.t_lift = z()
        self.anchor = z(3)            # 搬运锚点 (抬升目标, 无噪声)
        self.z0 = z(); self.yaw = z(); self.yaw_rate = z()
        self.tilt_amp = z(); self.drift_amp = z()
        self.sin_a = z(self._NCH, 2); self.sin_f = z(self._NCH, 2); self.sin_p = z(self._NCH, 2)
        self.shake_amp = z(); self.shake_f = z(); self.shake_dir = z(3)
        self.trem_amp = z(); self.trem_rot = z(); self.trem_f = z(); self.trem_dir = z(6); self.trem_p = z()
        self.drop = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.lower_v = z(); self.touch_t = z(); self.touch_release = z(); self.touched = torch.zeros_like(self.drop)
        self.end_tilt = z(2); self.lower_from_tilt = z(2); self.lower_t0 = z()
        self.press = z(); self.z_touch = z()
        self.cur_tilt = z(2)
        self.forced = torch.zeros_like(self.drop)
        self.grip = torch.zeros(num_envs, dtype=torch.long, device=device)  # 0 trunk, 1 head
        self.base_q = torch.zeros(num_envs, 4, device=device); self.base_q[:, 0] = 1.0  # 抓取时的无 yaw 坐标系
        self.orient = z(2); self.orient_s = z()
        self.yaw0 = z(); self.yaw_turn = z(); self.yaw_turn_t0 = z(); self.yaw_turn_s = z()

    # ------------------------------------------------------------------ api
    @property
    def active(self) -> torch.Tensor:
        return self.phase > IDLE

    def reset(self, env_ids):
        self.phase[env_ids] = IDLE
        self.t[env_ids] = 0.0
        self.idle_t[env_ids] = 0.0

    def select(self, pos, quat, pos_head=None, quat_head=None):
        """每个 env 被抓握物体的姿态 (trunk, 或 ``grip`` 为 1 时的 head)."""
        if pos_head is None:
            return pos, quat
        head = (self.grip == 1).unsqueeze(-1)
        return torch.where(head, pos_head, pos), torch.where(head, quat_head, quat)

    def start(self, mask: torch.Tensor, pos: torch.Tensor, quat: torch.Tensor, pos_head=None, quat_head=None):
        """在 ``mask`` 指定的 env 上开始一次 pick-up (demo 中也用于强制触发).
        ``pos/quat`` 为 trunk 的; ``pos_head/quat_head`` 为 head 的, 抓头时需要."""
        mask = mask & (self.phase == IDLE)
        n = int(mask.sum())
        if n == 0:
            return
        c, d = self.cfg, self.dev
        can_head = pos_head is not None
        self.grip[mask] = ((torch.rand(n, device=d) < c.head_grip_prob) & can_head).long()
        pos, quat = self.select(pos, quat, pos_head, quat_head)
        # 抓取时被抓物体的无 yaw 朝向: 手的倾斜相对它定义, 因此抓头时从头自身的姿态出发,
        # 而不是把它瞬间掰平. 抓 trunk 时保持绝对 (单位) — 把倒地的机器人捡起来会把它
        # 重新放回站立姿态.
        yq = _quat_from_euler_xyz(torch.zeros(n, device=d), torch.zeros(n, device=d), -yaw_of(quat[mask]))
        bq = _quat_mul(yq, quat[mask])
        ident = torch.zeros_like(bq); ident[:, 0] = 1.0
        self.base_q[mask] = torch.where((self.grip[mask] == 1).unsqueeze(-1), bq, ident)
        orient = torch.rand(n, device=d) < c.orient_prob
        o = torch.stack([_u(n, c.orient_roll, d), _u(n, c.orient_pitch, d)], -1)
        self.orient[mask] = o * orient.unsqueeze(-1).float()
        self.orient_s[mask] = _u(n, c.orient_ramp_s, d)
        turn = torch.rand(n, device=d) < c.yaw_turn_prob
        self.yaw_turn[mask] = _u(n, c.yaw_turn, d) * turn.float()
        self.yaw_turn_s[mask] = _u(n, c.yaw_turn_s, d)
        self.yaw_turn_t0[mask] = _u(n, (0.5, 4.0), d)
        self.phase[mask] = CARRY
        self.t[mask] = 0.0
        self.t_hold[mask] = _u(n, c.hold_s, d)
        self.t_lift[mask] = _u(n, c.lift_s, d)
        self.z0[mask] = pos[mask, 2]
        anc = pos[mask].clone()
        anc[:, 2] = pos[mask, 2] + _u(n, c.lift_z, d)
        self.anchor[mask] = anc
        self.yaw[mask] = yaw_of(quat[mask]); self.yaw0[mask] = self.yaw[mask]
        self.yaw_rate[mask] = _u(n, c.yaw_rate, d) * (torch.rand(n, device=d) < 0.6).float()
        # 倾斜分桶
        amps = torch.tensor([b[0] for b in c.tilt_buckets], device=d)
        probs = torch.tensor([b[1] for b in c.tilt_buckets], device=d)
        k = torch.multinomial(probs, n, replacement=True)
        self.tilt_amp[mask] = torch.deg2rad(amps[k]) * torch.rand(n, device=d).sqrt()
        self.drift_amp[mask] = _u(n, c.drift_amp, d)
        a = torch.rand(n, self._NCH, 2, device=d)
        self.sin_a[mask] = a / a.sum(-1, keepdim=True)
        f = torch.empty(n, self._NCH, 2, device=d)
        f[:, :2].uniform_(*c.tilt_freq_hz); f[:, 2:].uniform_(*c.drift_freq_hz)
        self.sin_f[mask] = f
        self.sin_p[mask] = torch.rand(n, self._NCH, 2, device=d) * 2 * math.pi
        shake = torch.rand(n, device=d) < c.shake_prob
        self.shake_amp[mask] = _u(n, c.shake_amp, d) * shake.float()
        self.shake_f[mask] = _u(n, c.shake_freq_hz, d)
        sd = torch.randn(n, 3, device=d); self.shake_dir[mask] = sd / sd.norm(dim=1, keepdim=True)
        self.trem_amp[mask] = _u(n, c.tremor_amp, d)
        self.trem_rot[mask] = torch.deg2rad(_u(n, c.tremor_rot_deg, d))
        self.trem_f[mask] = _u(n, (8.0, 12.0), d)
        td = torch.randn(n, 6, device=d); self.trem_dir[mask] = td / td.norm(dim=1, keepdim=True)
        self.trem_p[mask] = torch.rand(n, device=d) * 2 * math.pi
        self.drop[mask] = torch.rand(n, device=d) < c.drop_prob
        self.lower_v[mask] = _u(n, c.lower_speed, d)
        self.touch_release[mask] = _u(n, c.touch_release_s, d)
        self.touched[mask] = False
        self.end_tilt[mask] = (torch.rand(n, 2, device=d) * 2 - 1) * math.radians(c.putdown_tilt_deg)
        self.press[mask] = _u(n, c.touch_press, d)

    def release(self, mask: torch.Tensor):
        """强制结束搬运: 放下 (LOWER) — demo 控制用."""
        m = mask & (self.phase == CARRY)
        self.t_hold[m] = self.t[m]

    def step(self, dt, pos, quat, feet_force, mass, upright, pos_head=None, quat_head=None):
        """推进手一个控制步. 返回焊接 mocap 手的 (target_pos (N,3),
        target_quat (N,4) wxyz, active (N,) bool) — 即被抓物体
        (``grip``: 0 trunk, 1 head) 的目标姿态.
        ``feet_force`` = 脚上的总法向力 (N), ``mass`` = 每个 env 的机器人
        质量, ``upright`` = 用于门控倒地机器人 pick-up 的掩码."""
        c, d, N = self.cfg, self.dev, self.N

        # ── 启动 pick-up ──
        idle = self.phase == IDLE
        self.idle_t = torch.where(idle, self.idle_t + dt, torch.zeros_like(self.idle_t))
        elig = idle & (self.idle_t >= c.min_idle_s) & (upright | c.pickup_fallen)
        self.start(elig & (torch.rand(N, device=d) < c.pickup_rate_hz * dt), pos, quat, pos_head, quat_head)
        pos, quat = self.select(pos, quat, pos_head, quat_head)

        self.t = self.t + dt
        carry = self.phase == CARRY
        turn_a = ((self.t - self.yaw_turn_t0) / self.yaw_turn_s).clamp(0, 1)
        turn_a = 0.5 - 0.5 * torch.cos(math.pi * turn_a)
        self.yaw = torch.where(carry, self.yaw0 + self.yaw_rate * self.t + self.yaw_turn * turn_a, self.yaw)

        # ── carry → lower / drop ──
        done = carry & (self.t >= self.t_hold)
        if done.any():
            drop = done & self.drop
            self.phase[drop] = IDLE
            put = done & ~self.drop
            self.phase[put] = LOWER
            self.lower_t0[put] = self.t[put]
            self.lower_from_tilt[put] = self.cur_tilt[put]
            self.touched[put] = False
            self.touch_t[put] = 0.0
        lower = self.phase == LOWER
        # 触地检测 (手感觉到地面) → 延迟后松手
        touch_now = lower & ~self.touched & (feet_force > c.touch_force_frac * mass * 9.81)
        self.z_touch = torch.where(touch_now, pos[:, 2], self.z_touch)
        self.touched = self.touched | touch_now
        self.touch_t = torch.where(lower & self.touched, self.touch_t + dt, self.touch_t)
        lt = self.t - self.lower_t0
        rel = lower & ((self.touched & (self.touch_t >= self.touch_release)) | (lt > c.max_lower_s))
        self.phase[rel] = IDLE
        lower = self.phase == LOWER
        carry = self.phase == CARRY
        active = carry | lower

        # ── 目标姿态 ──
        t = self.t.unsqueeze(-1).unsqueeze(-1)
        sn = (self.sin_a * torch.sin(2 * math.pi * self.sin_f * t + self.sin_p)).sum(-1)  # (N, 5)
        lift_a = (self.t / self.t_lift).clamp(0, 1)
        lift_a = 0.5 - 0.5 * torch.cos(math.pi * lift_a)  # 平滑
        ramp = lift_a.unsqueeze(-1)
        orient_a = ((self.t - self.t_lift) / self.orient_s).clamp(0, 1)
        orient_a = (0.5 - 0.5 * torch.cos(math.pi * orient_a)).unsqueeze(-1)
        tilt = self.tilt_amp.unsqueeze(-1) * sn[:, :2] * ramp + self.orient * orient_a
        tgt = self.anchor.clone()
        tgt[:, :2] = tgt[:, :2] + self.drift_amp.unsqueeze(-1) * sn[:, 2:4] * ramp
        tgt[:, 2] = self.z0 + lift_a * (self.anchor[:, 2] - self.z0) + 0.3 * self.drift_amp * sn[:, 4] * lift_a
        # 下放: 以 lower_v 从当前搬运姿态下降, 倾斜 → end_tilt
        lo_a = (lt / 1.0).clamp(0, 1)
        tilt_lo = self.lower_from_tilt + (self.end_tilt - self.lower_from_tilt) * lo_a.unsqueeze(-1)
        z_lo = self.anchor[:, 2] - self.lower_v * lt
        z_lo = torch.where(self.touched, torch.maximum(z_lo, self.z_touch - self.press), z_lo)
        tilt = torch.where(lower.unsqueeze(-1), tilt_lo, tilt)
        tgt[:, 2] = torch.where(lower, z_lo, tgt[:, 2])
        self.cur_tilt = torch.where(carry.unsqueeze(-1), tilt, self.cur_tilt)
        # 摇晃 + 抖动
        ph = 2 * math.pi * self.shake_f * self.t
        tgt = tgt + (self.shake_amp * torch.sin(ph)).unsqueeze(-1) * self.shake_dir * carry.float().unsqueeze(-1)
        tph = torch.sin(2 * math.pi * self.trem_f * self.t + self.trem_p)
        tgt = tgt + (self.trem_amp * tph).unsqueeze(-1) * self.trem_dir[:, :3]
        rot_trem = (self.trem_rot * tph).unsqueeze(-1) * self.trem_dir[:, 3:]
        q_tgt = _quat_from_euler_xyz(tilt[:, 0] + rot_trem[:, 0], tilt[:, 1] + rot_trem[:, 1], self.yaw + rot_trem[:, 2])
        q_tgt = _quat_mul(q_tgt, self.base_q)
        if lower.any():  # 放下过程中保持机器人自身的 yaw
            self.yaw = torch.where(lower, yaw_of(quat), self.yaw)

        self.tgt_pos, self.tgt_quat = tgt, q_tgt
        return tgt, q_tgt, active


def add_hand_to_spec(spec, trunk_body: str = "robot/trunk_base", timeconst: float = 0.03, dampratio: float = 1.0,
                     head_body: str | None = None):
    """添加 mocap 手 + (未激活的) hand↔trunk 软焊接 ("pickup_hand_weld"), 以及当给出
    ``head_body`` 时 (默认: trunk 的前缀 + "jaw_soft") 的 hand↔head
    ("pickup_hand_weld_head"). 相对位姿为单位: 手的姿态就是被抓物体的目标."""
    import mujoco
    hand = spec.worldbody.add_body(name="pickup_hand", mocap=True)
    hand.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.012, 0, 0], contype=0, conaffinity=0,
                  rgba=[1.0, 0.6, 0.2, 0.0], group=5)
    if head_body is None:
        head_body = trunk_body.rsplit("trunk_base", 1)[0] + "jaw_soft"
    for name, body in (("pickup_hand_weld", trunk_body), ("pickup_hand_weld_head", head_body)):
        eq = spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                               name1="pickup_hand", name2=body, active=False)
        eq.name = name
        eq.solref = [timeconst, dampratio]
        eq.data[:] = 0.0
        eq.data[6] = 1.0   # relpose 四元数 (w) = 单位
        eq.data[10] = 1.0  # torquescale
    return spec
