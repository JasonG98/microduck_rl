"""用 CPU MuJoCo/BAM 扫描对称蹬地动作, 测量起跳的可行证据, 不估计全局上限.

从静止蹲姿稳定保持 3 秒后起跳. 只通过关节目标驱动, 不加基座外力/速度.
使用 walking 或 groundcontact 模型, 镜像训练碰撞参数, 控制频率 50 Hz.
"""

import argparse
import contextlib
import csv
import io
import json
from collections import deque
from pathlib import Path

import infer_policy as ip
import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]


class JumpProbe:
    """保持独立 BAM 状态的单机器人起跳探针."""

    def __init__(self, scene, voltage=7.4, timestep=0.005):
        """创建无随机化仿真, 使用固定 20 ms 指令延迟."""
        with contextlib.redirect_stdout(io.StringIO()):
            self.model, self.data, self.ctrl, self.names = ip.load_mujoco_with_bam(
                str(ROOT / scene),
                ip.load_bam_model(ip.BAM_KP_FW, voltage, ip.BAM_MAX_CURRENT),
                timestep,
                0.1,
                ip.BAM_VIN_MIN,
            )
        m = self.model
        for g in range(m.ngeom):
            name = m.geom(g).name
            if name.endswith("_collision"):
                m.geom_contype[g] = m.geom_conaffinity[g] = 1
                m.geom_condim[g] = 3 if name in ("left_foot_collision", "right_foot_collision") else 1
                m.geom_priority[g] = 1 if "foot_collision" in name else 0
                if "foot_collision" in name:
                    m.geom_friction[g, 0] = 1.0
            elif m.geom_bodyid[g] != 0:
                m.geom_contype[g] = m.geom_conaffinity[g] = 0
        self.index = {n: i for i, n in enumerate(self.names)}
        self.qadr = m.jnt_qposadr[m.actuator_trnid[:, 0]]
        self.free = int(m.jnt_qposadr[m.joint("trunk_base_freejoint").id])
        self.body = m.body("trunk_base").id
        self.floor = m.geom("floor").id
        self.feet = {m.geom(n).id for n in ("left_foot_collision", "right_foot_collision")}
        self.sid = m.site("left_foot").id
        canonical_names = (
            "left_hip_yaw",
            "left_hip_roll",
            "left_hip_pitch",
            "left_knee",
            "left_ankle",
            "neck_pitch",
            "head_pitch",
            "head_yaw",
            "head_roll",
            "right_hip_yaw",
            "right_hip_roll",
            "right_hip_pitch",
            "right_knee",
            "right_ankle",
        )
        home_by_name = dict(zip(canonical_names, ip.DEFAULT_POSE, strict=False))
        self.home = np.array([home_by_name[n] for n in self.names], dtype=float)
        self.decimation = round(0.02 / timestep)
        self.delay = round(0.02 / timestep)
        self.queue = deque([self.home.copy()] * (self.delay + 1), maxlen=self.delay + 1)
        self.reset(self.home, 0.125)
        self.foot_ref = self.data.site_xpos[self.sid].copy()
        self.rot_ref = self.data.site_xmat[self.sid].copy()

    def reset(self, pose, height):
        """重新初始化物理和执行器, 不携带上一试验的速度或电压状态."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.free : self.free + 7] = [0, 0, height, 1, 0, 0, 0]
        self.data.qpos[self.qadr] = pose
        self.ctrl.reset(self.data.qpos)
        self.queue = deque([pose.copy()] * (self.delay + 1), maxlen=self.delay + 1)
        mujoco.mj_forward(self.model, self.data)

    def solve_pose(self, knee):
        """固定左右对称膝角, 求保持脚位置/朝向的髋踝和基座高度."""
        self.reset(self.home, 0.125)

        def make(x):
            q = self.home.copy()
            for name, value in zip(("hip_pitch", "knee", "ankle"), (x[0], knee, x[1]), strict=False):
                q[self.index["left_" + name]] = value
                q[self.index["right_" + name]] = -value
            return q

        def residual(x):
            self.data.qpos[self.qadr] = make(x)
            self.data.qpos[self.free + 2] = x[2]
            mujoco.mj_forward(self.model, self.data)
            return np.r_[
                10 * (self.data.site_xpos[self.sid][[0, 2]] - self.foot_ref[[0, 2]]),
                self.data.site_xmat[self.sid] - self.rot_ref,
            ]

        result = least_squares(residual, [-0.45, 0.45, 0.12], bounds=([-1.55, -1.55, 0.02], [1.55, 1.55, 0.2]))
        if np.linalg.norm(result.fun) > 1e-4:
            raise ValueError("逆运动学姿态未达到误差要求")
        return make(result.x), float(result.x[2])

    def snapshot(self):
        """读全机器人质心及所有地面接触, 避免把收腿或跌倒当起跳."""
        m, d = self.model, self.data
        mujoco.mj_forward(m, d)
        mujoco.mj_subtreeVel(m, d)
        contacts = []
        force = np.zeros(6)
        normal = 0.0
        for contact_id in range(d.ncon):
            contact = d.contact[contact_id]
            if self.floor in contact.geom and contact.dist <= 0:
                other = int(contact.geom[1] if contact.geom[0] == self.floor else contact.geom[0])
                contacts.append(other)
                mujoco.mj_contactForce(m, d, contact_id, force)
                normal += max(float(force[0]), 0.0)
        com = d.subtree_com[self.body]
        vel = d.subtree_linvel[self.body]
        tilt = np.degrees(np.arccos(np.clip(d.xmat[self.body].reshape(3, 3)[2, 2], -1, 1)))
        return {
            "time_s": float(d.time),
            "com_z_m": float(com[2]),
            "com_vz_m_s": float(vel[2]),
            "pitch_rate_rad_s": float(d.cvel[self.body, 1]),
            "angular_momentum_y": float(d.subtree_angmom[self.body, 1]),
            "tilt_deg": float(tilt),
            "ground_contact": bool(contacts),
            "both_feet_contact": self.feet.issubset(contacts),
            "nonfoot_contact": any(g not in self.feet for g in contacts),
            "ground_force_n": normal,
        }

    def step(self, target):
        """以物理步延迟目标, 运行 BAM 和物理积分."""
        self.queue.append(target.copy())
        self.ctrl.q_target[:] = self.queue[0]
        self.ctrl.update()
        mujoco.mj_step(self.model, self.data)

    def prepare(self, knee):
        """让蹲姿在无外力下保持 3 秒, 验证最后半秒的稳定性."""
        pose, z = self.solve_pose(knee)
        self.reset(pose, z)
        tail = []
        for step in range(round(3 / self.model.opt.timestep)):
            self.step(pose)
            if step >= round(2.5 / self.model.opt.timestep):
                tail.append(self.snapshot())
        stable = all(
            s["both_feet_contact"] and not s["nonfoot_contact"] and s["tilt_deg"] < 10 and abs(s["com_vz_m_s"]) < 0.02
            for s in tail
        )
        return pose, stable, tail[-1]

    def trial(self, knee, gain, hold_s, ramp_s, hip_scale=1.0, ankle_scale=1.0, record=False):
        """扫描伸腿幅度和时序; 腾空只统计第一次由脚蹬地进入的无接触段."""
        extended, _ = self.solve_pose(-1.0)
        squat, stable, initial = self.prepare(knee)
        params = {
            "knee_rad": knee,
            "gain": gain,
            "hold_s": hold_s,
            "ramp_s": ramp_s,
            "hip_scale": hip_scale,
            "ankle_scale": ankle_scale,
        }
        if not stable:
            return {**params, "stable_start": False, "initial": initial}, [], []
        dt = self.model.opt.timestep
        start = self.data.time
        target = squat.copy()
        delta = extended - squat
        for side in ("left_", "right_"):
            delta[self.index[side + "hip_pitch"]] *= hip_scale
            delta[self.index[side + "ankle"]] *= ankle_scale
        trace, qposes = [], []
        for step in range(round(1.2 / dt)):
            t = step * dt
            if step % self.decimation == 0:
                if t < hold_s:
                    blend = 1.0 if ramp_s == 0 else min(t / ramp_s, 1.0)
                    target = squat + blend * gain * delta
                else:
                    target = self.home.copy()
            snap = self.snapshot()
            snap["time_s"] -= start
            trace.append(snap)
            if record:
                qposes.append(self.data.qpos.copy())
            self.step(target)
        result = {
            **params,
            "stable_start": True,
            "initial": initial,
            "flight_s": 0.0,
            "flight_rise_m": 0.0,
            "takeoff_vz_m_s": 0.0,
            "max_com_rise_from_start_m": max(s["com_z_m"] for s in trace) - initial["com_z_m"],
            "clean_takeoff": False,
            "landed_on_feet": False,
        }
        bad_before = False
        for i in range(1, len(trace)):
            bad_before |= trace[i - 1]["nonfoot_contact"]
            if trace[i - 1]["ground_contact"] and not trace[i]["ground_contact"]:
                j = i
                while j < len(trace) and not trace[j]["ground_contact"]:
                    j += 1
                flight = trace[i:j]
                if len(flight) * dt < 0.02:
                    continue
                valid = not bad_before and trace[i]["com_vz_m_s"] > 0.05
                result.update(
                    flight_s=len(flight) * dt,
                    flight_rise_m=max(s["com_z_m"] for s in flight) - flight[0]["com_z_m"],
                    takeoff_vz_m_s=flight[0]["com_vz_m_s"],
                    clean_takeoff=valid,
                    takeoff_s=flight[0]["time_s"],
                    takeoff_angular_momentum_y=flight[0]["angular_momentum_y"],
                    landed_on_feet=j < len(trace) and not trace[j]["nonfoot_contact"],
                    landing_tilt_deg=trace[j]["tilt_deg"] if j < len(trace) else None,
                    max_flight_tilt_deg=max(s["tilt_deg"] for s in flight),
                )
                break
        return result, trace, qposes


def main():
    """执行有限参数扫描, 保存可复现的参数和完整最佳轨迹."""
    parser = argparse.ArgumentParser(description="扫描 CPU/BAM 对称起跳动作, 不运行 RL 训练")
    parser.add_argument("--output", type=Path, required=True, help="结果目录")
    parser.add_argument("--scene", default="src/mjlab_microduck/robot/microduck/scene_walk.xml", help="机器人场景路径")
    parser.add_argument("--voltage", type=float, default=7.4, help="电池电压, 单位 V")
    parser.add_argument("--timestep", type=float, default=0.005, help="物理步长, 单位秒")
    parser.add_argument("--replay", type=Path, help="仅重放指定 JSON 中的最佳参数")
    parser.add_argument("--validate-candidates", type=Path, help="复核指定 JSON 中所有曾有效起跳的候选")
    parser.add_argument("--random-trials", type=int, default=0, help="增加独立髋踝发力比例的随机搜索次数")
    args = parser.parse_args()
    if args.timestep <= 0 or not np.isclose(round(0.02 / args.timestep) * args.timestep, 0.02):
        parser.error("物理步长必须正值且整除 0.02 秒")
    args.output.mkdir(parents=True, exist_ok=True)
    probe = JumpProbe(args.scene, args.voltage, args.timestep)
    results = []
    if args.replay and args.validate_candidates:
        parser.error("重放最佳参数和候选复核不能同时选择")
    if args.validate_candidates:
        source = json.loads(args.validate_candidates.read_text())
        for row in source["trials"]:
            if not row.get("clean_takeoff"):
                continue
            result, _, _ = probe.trial(
                *(row[k] for k in ("knee_rad", "gain", "hold_s", "ramp_s")),
                hip_scale=row.get("hip_scale", 1),
                ankle_scale=row.get("ankle_scale", 1),
            )
            results.append(result)
        if not results:
            parser.error("输入结果没有可复核的起跳候选")
        best = max(results, key=lambda r: r.get("flight_rise_m", -1) if r.get("clean_takeoff") else -1)
    elif args.replay:
        best = json.loads(args.replay.read_text())["best"]
    else:
        for knee in (0.0, 0.4, 0.8, 1.2, 1.5):
            for gain in (1.0, 2.0, 4.0):
                for hold in (0.08, 0.16, 0.28):
                    for ramp in (0.0, 0.08):
                        result, _, _ = probe.trial(knee, gain, hold, ramp)
                        results.append(result)
            clean = [r for r in results if r.get("clean_takeoff")]
            print(
                f"已扫描 {len(results)} 组, 有效起跳 {len(clean)} 组, 最佳离地质心升高 "
                f"{max((r['flight_rise_m'] for r in clean), default=0) * 100:.2f} cm",
                flush=True,
            )
        rng = np.random.default_rng(42)
        for i in range(args.random_trials):
            result, _, _ = probe.trial(
                float(rng.choice([0.4, 0.8, 1.2, 1.5])),
                float(rng.uniform(1, 4)),
                float(rng.uniform(0.08, 0.32)),
                float(rng.choice([0, 0.04, 0.08])),
                float(rng.uniform(0.2, 1.8)),
                float(rng.uniform(-0.5, 2)),
            )
            results.append(result)
            if (i + 1) % 40 == 0:
                clean = [r for r in results if r.get("clean_takeoff")]
                print(
                    f"已完成额外搜索 {i + 1} 组, 有效起跳 {len(clean)} 组, 最佳升高 "
                    f"{max((r['flight_rise_m'] for r in clean), default=0) * 100:.2f} cm",
                    flush=True,
                )
        clean = [r for r in results if r.get("clean_takeoff")]
        if not clean:
            best = max(results, key=lambda r: r.get("flight_rise_m", -1))
        else:
            best = max(clean, key=lambda r: r["flight_rise_m"])
    best, trace, poses = probe.trial(
        *(best[k] for k in ("knee_rad", "gain", "hold_s", "ramp_s")),
        hip_scale=best.get("hip_scale", 1),
        ankle_scale=best.get("ankle_scale", 1),
        record=True,
    )
    payload = {
        "scene": args.scene,
        "voltage_v": args.voltage,
        "timestep_s": args.timestep,
        "control_hz": 50,
        "fixed_command_delay_s": 0.02,
        "vin_drop_gain": 0.1,
        "mass_kg": float(probe.model.body_mass.sum()),
        "current_limit_a": ip.BAM_MAX_CURRENT,
        "solver_iterations": int(probe.model.opt.iterations),
        "random_seed": 42,
        "candidate_source": str(args.validate_candidates or args.replay)
        if args.validate_candidates or args.replay
        else None,
        "method": "对称开环蹬地有限网格搜索, 稳定蹲姿起始, 无外力/基座初速度, 无 DR, 非 RL, 非全局上限",
        "best": best,
        "trials": results,
    }
    (args.output / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    if trace:
        with (args.output / "best_trace.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(trace[0]))
            writer.writeheader()
            writer.writerows(trace)
        np.save(args.output / "best_qpos.npy", np.array(poses))
    print("最佳结果:", json.dumps(best, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
