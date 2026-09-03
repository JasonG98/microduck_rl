"""infer_policy.py 在 CPU MuJoCo 排演中使用与策略在 warp 中训练时完全相同的
BAM M6 执行器 (bam.mujoco.MujocoController, 作用在电机转换后的模型上).
这些测试把两端固定在一起:

* 脚本中硬编码的 BAM 常量镜像 microduck_constants 中的 ``_BAM_ACTUATOR_KWARGS``
  (这里不导入, 是为了让脚本保持不依赖 torch/warp);
* 电机转换与 ``bam.mjlab.BamActuator.edit_spec`` 所做的工作一致
  (torque 电机, 带电压边界的 forcerange, armature, 清零的 XML 摩擦,
  刚性摩擦约束), 并且一个带真实摩擦预算的步进循环可以跑起来.
"""

import importlib.util
from pathlib import Path

import mujoco
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ip():
    spec = importlib.util.spec_from_file_location(
        "infer_policy", REPO / "scripts" / "infer_policy.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cpu_bam_constants_mirror_training_cfg(ip):
    from bam.mjlab import BamActuator
    from mjlab_microduck.robot.microduck_constants import _BAM_ACTUATOR_KWARGS as k

    assert ip.BAM_MOTOR_NAME == k["motor_name"]
    assert ip.BAM_MODEL == k["model"]
    assert ip.BAM_KP_FW == k["kp_fw"]
    assert ip.BAM_VIN_RANGE == k["vin_range"]
    assert ip.BAM_VIN_DROP_GAIN_RANGE == k["vin_drop_gain_range"]
    assert ip.BAM_VIN_MIN == k["vin_min"]
    assert ip.BAM_MAX_CURRENT == k.get("max_current")
    assert ip.BAM_STIFF_SOLREF_FRICTION == BamActuator._STIFF_SOLREF_FRICTION
    assert ip.BAM_STIFF_SOLIMP_FRICTION == BamActuator._STIFF_SOLIMP_FRICTION


@pytest.fixture(scope="module")
def bam_sim(ip):
    bam_model = ip.load_bam_model(ip.BAM_KP_FW, 7.4, ip.BAM_MAX_CURRENT)
    model, data, ctrl, names = ip.load_mujoco_with_bam(
        str(REPO / ip.MICRODUCK_XML), bam_model, 0.005, 0.1, ip.BAM_VIN_MIN
    )
    return ip, bam_model, model, data, ctrl, names


def test_actuators_converted_like_warp(bam_sim):
    ip, bam_model, model, data, ctrl, names = bam_sim
    kt, R = bam_model.kt.value, bam_model.R.value
    assert len(names) == 14 and model.nu == 14
    assert not any(n.startswith("passive_") for n in names)
    # Torque 电机: ctrl 就是 BAM 转矩, 没有多余的 MuJoCo PD.
    assert (model.actuator_gaintype == mujoco.mjtGain.mjGAIN_FIXED).all()
    assert (model.actuator_biastype == mujoco.mjtBias.mjBIAS_NONE).all()
    assert np.allclose(model.actuator_gainprm[:, 0], 1.0)
    # (set_to_motor 会遗留旧的 PD biasprm 字节; 在 BIAS_NONE 下处于非激活状态,
    # 与 warp 的 edit_spec 中的情况完全一致.)
    assert (model.actuator_forcelimited == 1).all()
    assert np.allclose(model.actuator_forcerange[:, 1], 7.4 * kt / R)
    dofs = model.jnt_dofadr[model.actuator_trnid[:, 0]]
    assert np.allclose(model.dof_armature[dofs], bam_model.actuator.get_extra_inertia())
    assert np.allclose(model.dof_solref[dofs], ip.BAM_STIFF_SOLREF_FRICTION)
    assert np.allclose(model.dof_solimp[dofs], ip.BAM_STIFF_SOLIMP_FRICTION)
    assert bam_model.actuator.kp == ip.BAM_KP_FW
    assert bam_model.actuator.max_current is None  # training has no current limiter


def test_bam_step_loop_runs_with_live_friction(bam_sim):
    ip, bam_model, model, data, ctrl, names = bam_sim
    fj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qa = model.jnt_qposadr[fj]
    mujoco.mj_resetData(model, data)
    data.qpos[qa + 2] = 0.125
    data.qpos[qa + 3 : qa + 7] = [1, 0, 0, 0]
    jq = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    data.qpos[jq] = ip.DEFAULT_POSE
    ctrl.reset(data.qpos)
    ctrl.q_target[:] = ip.DEFAULT_POSE
    mujoco.mj_forward(model, data)
    dofs = model.jnt_dofadr[model.actuator_trnid[:, 0]]
    for _ in range(100):
        ctrl.update()
        mujoco.mj_step(model, data)
    assert not np.isnan(data.qpos).any()
    limit = model.actuator_forcerange[0, 1]
    assert (np.abs(data.ctrl) <= limit + 1e-9).all()  # ctrl 就是电机转矩
    assert (model.dof_frictionloss[dofs] > 0).all()  # BAM 摩擦预算每一步都会被写入
    assert np.allclose(model.dof_damping[dofs], bam_model.friction_viscous.value)
