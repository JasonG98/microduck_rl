"""带逐环境摩擦力幅度域随机化的 BAM 执行器.

标准的 ``bam.mjlab.BamActuator`` 暴露了逐环境的增益缩放 (kp/kd) 但没有摩擦力钩子,
而在 BAM 下 MuJoCo 的 ``dof_frictionloss`` 在 ``edit_spec`` 中被置零 (BAM 在
``compute()`` 中自行计算摩擦). 因此原生的 ``dr.dof_frictionloss`` 在此处是空操作.

这个薄子类增加了一个逐环境的 ``friction_scale``, 在 ``_compute_friction_budget``
内部乘到 BAM 的与速度无关的摩擦预算上 (Coulomb + Stribeck + 负载相关) —
该项承载了 sim2real 中主要的摩擦不确定性 (静摩擦 / 齿轮箱). 粘性 (与速度成正比)
项保持标称值; 如有需要可通过覆写 ``compute`` 来一并缩放.

非累积: ``friction_scale`` 在每个 episode 由 ``randomize_bam_friction``
事件先重置为 1.0 再设为新采样值 (见 tasks/mdp.py).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch
from bam.mjlab import BamActuator, BamActuatorCfg
from mjlab.actuator.actuator import ActuatorCmd


class FrictionDRBamActuator(BamActuator):
    """BamActuator + 对 BAM 摩擦预算逐环境的 friction_scale."""

    def initialize(self, mj_model, model, data, device) -> None:
        """分配逐环境的 ``friction_scale`` 缓冲区 (默认 1.0)."""
        super().initialize(mj_model, model, data, device)
        # kp_scale 是 (num_envs, 1); 镜像它以得到逐环境的摩擦乘子.
        self.friction_scale = torch.ones_like(self.kp_scale)
        self.default_friction_scale = self.friction_scale.clone()

    def _compute_friction_budget(
        self,
        motor_torque: torch.Tensor,
        external_torque: torch.Tensor,
        stribeck_coeff: torch.Tensor,
    ) -> torch.Tensor:
        base = super()._compute_friction_budget(motor_torque, external_torque, stribeck_coeff)
        fs = getattr(self, "friction_scale", None)
        return base if fs is None else base * fs  # (N, J) * (N, 1)

    def set_friction_scale(self, env_ids, friction_scale: torch.Tensor) -> None:
        """为给定环境设置逐环境的摩擦缩放."""
        self.friction_scale[env_ids] = friction_scale

    def reset_friction_scale(self, env_ids) -> None:
        """将逐环境的摩擦缩放重置为编译时默认值."""
        self.friction_scale[env_ids] = self.default_friction_scale[env_ids]


@dataclass(kw_only=True)
class FrictionDRBamActuatorCfg(BamActuatorCfg):
    """BamActuatorCfg 的直接替换, 构建一个支持摩擦域随机化的执行器."""

    def build(self, entity, target_ids, target_names) -> FrictionDRBamActuator:
        """构建支持摩擦域随机化的 BAM 执行器实例."""
        return FrictionDRBamActuator(self, entity, target_ids, target_names)


class BacklashEncoderBamActuator(FrictionDRBamActuator):
    """固件 PD 通过 backlash 读取编码器的 FrictionDRBamActuator.

    Backlash 模型 (robot_groundcontact_backlash.xml) 在每个 servo 关节串联
    了一个非驱动的 ``passive_<joint>_backlash`` 铰链: servo 关节是电机输出,
    backlash 关节是它与连杆之间的间隙, 连杆角度为两者之和.

    真实 servo 上磁性编码器位于该间隙的输出侧, 所以固件位置环在
    main+backlash 上闭环 — 当 servo 穿过死区时测量位置 (因而 PD 误差)
    不变. 本子类重现这一行为: 送入 BAM 电压控制律的 ``cmd.pos`` 变为
    qpos[main] + qpos[backlash].

    ``cmd.vel`` 刻意留在电机侧: 在 BAM 中它驱动反电动势和摩擦,
    属于转子物理量, 不是编码器派生的固件信号.

    在没有 backlash 关节的模型上退化为普通的 FrictionDRBamActuator
    (逐关节掩码), 因此可安全用于任何 microduck 模型.
    """

    def initialize(self, mj_model, model, data, device) -> None:
        """解析逐关节的 backlash 铰链 id 和编码器反馈掩码."""
        super().initialize(mj_model, model, data, device)
        name_to_local = {n: i for i, n in enumerate(self.entity.joint_names)}
        ids, mask = [], []
        for name in self._target_names:
            bl_id = name_to_local.get(f"passive_{name}_backlash")
            ids.append(0 if bl_id is None else bl_id)
            mask.append(0.0 if bl_id is None else 1.0)
        self._backlash_joint_ids = torch.tensor(ids, dtype=torch.long, device=device)
        self._backlash_mask = torch.tensor(mask, dtype=torch.float32, device=device)
        n_backlash = int(self._backlash_mask.sum().item())
        print(f"[BacklashEncoderBamActuator] {n_backlash}/{len(mask)} 个关节启用穿过 backlash 的编码器反馈")

    def get_command(self, data) -> ActuatorCmd:
        """返回位置反馈经 backlash 铰链偏移后的命令."""
        cmd = super().get_command(data)
        pos = cmd.pos + data.joint_pos[:, self._backlash_joint_ids] * self._backlash_mask
        return dataclasses.replace(cmd, pos=pos)


@dataclass(kw_only=True)
class BacklashEncoderBamActuatorCfg(FrictionDRBamActuatorCfg):
    """PD 反馈穿过 backlash 关节读取的 FrictionDRBamActuatorCfg."""

    def build(self, entity, target_ids, target_names) -> BacklashEncoderBamActuator:
        """构建穿过 backlash 读取编码器的 BAM 执行器实例."""
        return BacklashEncoderBamActuator(self, entity, target_ids, target_names)
