"""交互式蹲姿编辑器 (roller robot).

打开 MuJoCo viewer, 加载站立的 rollers 机器人. 在 viewer 的 "Control"
面板中拖动滑块 (膝盖/髋关节/踝关节...) 来组合想要的蹲姿.
重力被关闭, 基座保持竖直 + 下沉, 使最低点始终贴地 (因此当你弯曲
膝盖时会看到躯干下降). 关闭窗口时, 姿态会以 CROUCH_POSE dict
 {关节名: 弧度} 的形式打印出来, 可直接粘贴使用.

Usage:
    uv run python scripts/crouch_pose_editor.py
"""

import re
import time

import mujoco
import mujoco.viewer

from mjlab_microduck.robot.microduck_constants import (
    HOME_FRAME,
    get_walk_rollers_spec,
)


def home_value(joint_name: str):
    """返回指定关节的 home 位置."""
    for pattern, val in HOME_FRAME.joint_pos.items():
        if re.search(pattern, joint_name):
            return float(val)
    return 0.0


# 直接从 robot spec 构建模型 (XML 中有 14 个 <position> 执行器).
model = get_walk_rollers_spec().compile()
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)
model.opt.gravity[:] = [0, 0, 0]  # 没有重力作用: 只有滑块会动

has_free = model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE

# 执行关节 (排除被动轮子), 记录 qpos 地址.
joints = []
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    if not name or "freejoint" in name or "passive_" in name:
        continue
    joints.append((name, model.jnt_qposadr[i]))

# 初始 ctrl = HOME 姿态 (位置执行器保持这个目标).
for a in range(model.nu):
    aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
    data.ctrl[a] = home_value(aname or "")

if has_free:
    data.qpos[0:3] = [0.0, 0.0, 0.14]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    base_xy = data.qpos[0:2].copy()
    base_quat = data.qpos[3:7].copy()

robot_geoms = [g for g in range(model.ngeom) if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE]

mujoco.mj_forward(model, data)

print("=== 蹲姿编辑器 (rollers) ===")
print(f"执行器: {model.nu} | 浮动基座: {has_free}")
print("打开 viewer 的 'Control' 面板, 拖动滑块来组合")
print("蹲姿. 完成后关闭窗口.\n")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        if has_free:
            data.qpos[0:2] = base_xy
            data.qpos[3:7] = base_quat
            data.qvel[0:6] = 0.0
        mujoco.mj_step(model, data)  # 位置执行器 -> 关节跟随 ctrl
        if has_free:
            data.qpos[0:2] = base_xy
            data.qpos[3:7] = base_quat
            data.qvel[0:6] = 0.0
            mujoco.mj_forward(model, data)
            try:
                zmin = min(float(data.geom_xpos[g, 2] - model.geom_rbound[g]) for g in robot_geoms)
                data.qpos[2] -= zmin
                mujoco.mj_forward(model, data)
            except Exception:
                pass
        viewer.sync()
        time.sleep(1.0 / 60.0)

print("\n=== 已捕获蹲姿 ===\n")
print("CROUCH_POSE = {")
for name, adr in joints:
    print(f'    "{name}": {float(data.qpos[adr]):.4f},')
print("}")
if has_free:
    print(f"\n# 最终基座高度 (信息): z = {float(data.qpos[2]):.4f}")
print("# 将 CROUCH_POSE 粘贴到这里, 交给 Claude 来配置 reward.")
