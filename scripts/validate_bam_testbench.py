"""用真实测试台数据验证 BAM M6 执行器 kernel.

加载真实测试台录制数据, 在 MuJoCo 中用 BAM M6 执行器回放, 并对比仿真与真实的位置轨迹.
同时运行 BAM 自带的 Python 仿真器作为参考.

用法:
    uv run python3 scripts/validate_bam_testbench.py [--plot] [--max-files N]
"""

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

# ── 路径 ──
BAM_DIR = Path("~/Rhoban/bam").expanduser()
DATA_DIR = BAM_DIR / "bam" / "data" / "processed"
PARAMS_FILE = BAM_DIR / "params" / "xl330" / "m6_new.json"
TESTBENCH_XML = (
    Path(__file__).resolve().parent.parent / "src" / "mjlab_microduck" / "robot" / "xl330_test_bench" / "scene.xml"
)

# ── 加载 M6 参数 ──
with PARAMS_FILE.open() as f:
    M6 = json.load(f)

# XL330 固件常数
ERROR_GAIN = (4096 / (2 * np.pi)) / (256 * 885)
VIN = 7.4
MAX_PWM = 1.0


def bam_python_rollout(log: dict) -> list[float]:
    """参考: BAM 自带的 Python 仿真器."""
    sys.path.insert(0, str(BAM_DIR))
    from bam.model import load_model
    from bam.simulate import Simulator

    # BAM 期望 log 里有 arm_mass (臂本身的质量, 不是负载)
    if "arm_mass" not in log:
        log = dict(log)
        log["arm_mass"] = 0.0

    model = load_model(str(PARAMS_FILE))
    sim = Simulator(model)
    result = sim.rollout_log(log, simulate_control=True)
    return result[0]  # 位置


def compute_m6_friction(motor_torque, external_torque, dq):
    """与我们的 kernel (以及 BAM 的 model.py) 一致的 M6 摩擦计算."""
    p = M6
    stribeck_coeff = np.exp(-(np.abs(dq / p["dtheta_stribeck"]) ** p["alpha"]))

    gearbox_torque = np.abs(external_torque * p["load_friction_external"] - motor_torque * p["load_friction_motor"])
    gearbox_torque_stribeck = np.abs(
        external_torque * p["load_friction_external_stribeck"] - motor_torque * p["load_friction_motor_stribeck"]
    )

    frictionloss = p["friction_base"]
    frictionloss += gearbox_torque
    frictionloss += stribeck_coeff * p["friction_stribeck"]
    frictionloss += gearbox_torque_stribeck * stribeck_coeff
    # 二次项 (很小, 为清晰起见跳过)

    damping = p["friction_viscous"]
    friction_budget = frictionloss + damping * np.abs(dq)
    return friction_budget


def mujoco_rollout(log: dict) -> list[float]:
    """在 MuJoCo 中用我们的 BAM M6 执行器逻辑运行测试台."""
    mass = log["mass"]
    kp_fw = log["kp"]
    dt = log["dt"]
    entries = log["entries"]

    # 加载并修改测试台模型
    spec = mujoco.MjSpec.from_file(str(TESTBENCH_XML))

    # 将执行器转为 motor (同我们 kernel 的 edit_spec)
    for act in spec.actuators:
        act.set_to_motor()
        act.forcelimited = True
        force_limit = VIN * M6["kt"] / M6["R"]
        act.forcerange = (-force_limit, force_limit)
        act.gear = [1.0, 0, 0, 0, 0, 0]

    # 清零 MuJoCo 关节摩擦 (我们自行处理)
    for joint in spec.joints:
        if joint.type == mujoco.mjtJoint.mjJNT_HINGE:
            joint.damping = 0.0
            joint.frictionloss = 0.0
            joint.armature = M6["armature"]

    # 设置臂的质量以匹配 BAM 录制
    for body in spec.bodies:
        if body.name == "arm":
            # 按比例缩放质量和惯量
            original_mass = body.mass
            scale = mass / original_mass if original_mass > 0 else 1.0
            body.mass = mass
            # 惯量按质量等比缩放
            body.fullinertia = [x * scale for x in body.fullinertia]
            break

    model = spec.compile()
    data = mujoco.MjData(model)
    model.opt.timestep = dt

    # 查找关节和执行器 ID
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "1")
    dof_id = model.jnt_dofadr[joint_id]

    # 初始化状态
    data.qpos[dof_id] = entries[0]["position"]
    data.qvel[dof_id] = entries[0].get("speed", 0.0)
    mujoco.mj_forward(model, data)

    positions = []
    for entry in entries:
        positions.append(float(data.qpos[dof_id]))

        if not entry["torque_enable"]:
            data.ctrl[0] = 0.0
            mujoco.mj_step(model, data)
            continue

        goal = entry["goal_position"]
        q = data.qpos[dof_id]
        dq = data.qvel[dof_id]

        # ── BAM M6 执行器逻辑 (同我们 kernel) ──

        # 1. 固件控制律
        duty = (goal - q) * kp_fw * ERROR_GAIN
        duty = np.clip(duty, -MAX_PWM, MAX_PWM)
        voltage = VIN * duty

        # 2. DC 电机力矩
        motor_torque = M6["kt"] * voltage / M6["R"] - M6["kt"] ** 2 * dq / M6["R"]

        # 3. 外部力矩 (来自 MuJoCo 偏置力)
        # BAM 约定: bias_torque = m*g*l*sin(q), 其中 g=-9.81 (重力为负)
        # MuJoCo 约定: qfrc_bias 符号相反
        external_torque = -data.qfrc_bias[dof_id]

        # 4. M6 摩擦
        friction_budget = compute_m6_friction(motor_torque, external_torque, dq)

        # 5. 静摩擦截断
        eff_inertia = 1.0 / model.dof_invweight0[dof_id] if model.dof_invweight0[dof_id] > 0 else 1e6
        net_no_friction = motor_torque + external_torque
        tau_stop = (eff_inertia / dt) * dq + net_no_friction
        friction_mag = min(abs(tau_stop), friction_budget)
        friction_torque = -np.sign(tau_stop) * friction_mag

        # 6. 设置 ctrl = motor + friction (MuJoCo 会加上 qfrc_bias)
        data.ctrl[0] = motor_torque + friction_torque

        mujoco.mj_step(model, data)

    return positions


def main():
    """用录制的测试台轨迹验证 BAM M6 kernel."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="显示图表")
    parser.add_argument("--max-files", type=int, default=5)
    args = parser.parse_args()

    data_files = sorted(DATA_DIR.glob("*.json"))
    if args.max_files:
        data_files = data_files[: args.max_files]

    print(f"用 {len(data_files)} 条测试台录制验证 BAM M6 kernel")
    print(f"M6 参数: kt={M6['kt']:.4f} R={M6['R']:.4f}")
    print(f"测试台 XML: {TESTBENCH_XML}")
    print()

    results = []
    for fpath in data_files:
        with fpath.open() as f:
            log = json.load(f)
        name = f"{log['trajectory']}_m{log['mass']}_kp{log['kp']}"
        print(f"  {name}...", end=" ", flush=True)

        real_pos = [e["position"] for e in log["entries"]]

        # BAM Python 参考
        bam_pos = bam_python_rollout(log)

        # 我们的 MuJoCo M6 kernel
        mj_pos = mujoco_rollout(log)

        # 计算 MAE
        real_np = np.array(real_pos)
        bam_np = np.array(bam_pos)
        mj_np = np.array(mj_pos[: len(real_np)])

        mae_bam = np.mean(np.abs(bam_np - real_np))
        mae_mj = np.mean(np.abs(mj_np - real_np))
        mae_bam_vs_mj = np.mean(np.abs(bam_np - mj_np))

        print(f"MAE  bam_vs_real={mae_bam:.5f}  mj_vs_real={mae_mj:.5f}  bam_vs_mj={mae_bam_vs_mj:.5f}")

        results.append(
            {
                "name": name,
                "real": real_np,
                "bam": bam_np,
                "mj": mj_np,
                "mae_bam": mae_bam,
                "mae_mj": mae_mj,
                "mae_bam_vs_mj": mae_bam_vs_mj,
            }
        )

    print()
    avg_bam = np.mean([r["mae_bam"] for r in results])
    avg_mj = np.mean([r["mae_mj"] for r in results])
    avg_diff = np.mean([r["mae_bam_vs_mj"] for r in results])
    print(f"平均 MAE  bam_vs_real={avg_bam:.5f}  mj_vs_real={avg_mj:.5f}  bam_vs_mj={avg_diff:.5f}")

    if avg_diff > 0.01:
        print("\n⚠ BAM 与 MuJoCo 差异显著 — 可能是 kernel bug!")
    elif avg_mj > avg_bam * 1.5:
        print("\n⚠ MuJoCo 比 BAM 差 — MuJoCo 动力学与 BAM 的简单积分器不同")
    else:
        print("\n✓ BAM 与 MuJoCo 一致 — kernel 正确")

    if args.plot:
        try:
            import matplotlib.pyplot as plt

            n = len(results)
            fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=False)
            if n == 1:
                axes = [axes]

            for ax, r in zip(axes, results, strict=False):
                t = np.arange(len(r["real"])) * 0.005
                ax.plot(t, r["real"], "k-", lw=1.5, label="Real")
                ax.plot(t, r["bam"], "b--", lw=1.2, label=f"BAM (MAE={r['mae_bam']:.4f})")
                ax.plot(t, r["mj"], "r:", lw=1.2, label=f"MuJoCo M6 (MAE={r['mae_mj']:.4f})")
                ax.set_title(r["name"])
                ax.set_ylabel("Position (rad)")
                ax.legend(fontsize=8)
                ax.grid(alpha=0.3)

            axes[-1].set_xlabel("Time (s)")
            plt.tight_layout()
            plt.savefig("bam_validation.png", dpi=150)
            print("已保存 bam_validation.png")
            plt.show()
        except ImportError:
            print("matplotlib 不可用, 跳过绘图")


if __name__ == "__main__":
    main()
