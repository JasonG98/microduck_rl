#!/usr/bin/env python3
"""记录一次固定关节目标在四个 BAM/MuJoCo 物理子步中的变化."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.utils.buffers import DelayBuffer
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "docs" / "learning" / "assets" / "u2-3"
JOINT_NAME = "left_knee"
ACTION_OFFSET_RAD = 0.05
PHYSICS_DT_S = 0.005
NUM_SUBSTEPS = 4
DELAY_BOUNDARY_LAGS = (3, 6)
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def load_infer_module():
    """加载现有 CPU BAM 排演器, 保持执行器转换与项目一致."""
    path = ROOT / "scripts" / "infer_policy.py"
    spec = importlib.util.spec_from_file_location("infer_policy", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def probe_delay_boundaries() -> list[dict[str, float | int]]:
    """用固定的 3/6 步延迟展示新目标最早和最晚何时被读到."""
    records: list[dict[str, float | int]] = []
    old_target = torch.tensor([[0.0]])
    new_target = torch.tensor([[ACTION_OFFSET_RAD]])
    for lag in DELAY_BOUNDARY_LAGS:
        buffer = DelayBuffer(min_lag=lag, max_lag=lag, batch_size=1, device="cpu")
        for _ in range(lag + 2):
            buffer.append(old_target)
            buffer.compute()
        for substep in range(1, lag + 2):
            buffer.append(new_target)
            bam_seen_target = float(buffer.compute()[0, 0])
            records.append(
                {
                    "fixed_lag_substeps": lag,
                    "substep_after_new_target": substep,
                    "elapsed_ms_before_substep": (substep - 1) * PHYSICS_DT_S * 1000,
                    "bam_seen_target_rad": bam_seen_target,
                }
            )
    return records


def draw_summary_image() -> Path:
    """生成可直接预览的一页 PNG 图, 复述本节已核对的执行链."""
    image = Image.new("RGB", (1200, 800), "#f8fbff")
    draw = ImageDraw.Draw(image)
    title = ImageFont.truetype(FONT_PATH, 34)
    subtitle = ImageFont.truetype(FONT_PATH, 18)
    box_title = ImageFont.truetype(FONT_PATH, 22)
    box_text = ImageFont.truetype(FONT_PATH, 17)
    small = ImageFont.truetype(FONT_PATH, 15)
    mono = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)

    def text(x: int, y: int, value: str, font, color="#243b53") -> None:
        draw.text((x, y), value, font=font, fill=color)

    def box(x: int, y: int, width: int, height: int, color: str, border: str, heading: str, lines: list[str]) -> None:
        draw.rounded_rectangle((x, y, x + width, y + height), radius=16, fill=color, outline=border, width=2)
        text(x + 24, y + 28, heading, box_title, "#102a43")
        for index, line in enumerate(lines):
            text(x + 24, y + 67 + 29 * index, line, box_text if index < 2 else small)

    def arrow(start: tuple[int, int], end: tuple[int, int]) -> None:
        draw.line((start, end), fill="#334155", width=3)
        ex, ey = end
        if abs(end[0] - start[0]) >= abs(end[1] - start[1]):
            direction = 1 if end[0] > start[0] else -1
            points = [(ex, ey), (ex - direction * 12, ey - 7), (ex - direction * 12, ey + 7)]
        else:
            direction = 1 if end[1] > start[1] else -1
            points = [(ex, ey), (ex - 7, ey - direction * 12), (ex + 7, ey - direction * 12)]
        draw.polygon(points, fill="#334155")

    text(56, 25, "U2.3: 一次策略动作如何变成真实关节运动", title, "#102a43")
    text(56, 72, "policy 50 Hz: 每 20 ms 一次. Physics/BAM 200 Hz: 每 5 ms 一次.", subtitle, "#486581")
    box(52, 125, 212, 178, "#dbeafe", "#60a5fa", "1. Policy", ["输出 14D raw action a_t", "每 20 ms 一次", "不是 torque, 不是 qpos"])
    box(326, 125, 232, 178, "#dcfce7", "#4ade80", "2. ActionManager", ["latest target =", "default + a_t × 1.0", "history 立即记为 a_t"])
    box(620, 125, 240, 178, "#f3e8ff", "#c084fc", "3. DelayBuffer", ["BAM 实际读到的 target", "来自 3--6 步历史", "可能仍是旧 target"])
    box(922, 125, 226, 178, "#fef3c7", "#fbbf24", "4. BAM", ["target + qpos/qvel", "→ 电压, 摩擦, torque", "每 5 ms 重算"])
    arrow((264, 214), (318, 214))
    arrow((558, 214), (612, 214))
    arrow((860, 214), (914, 214))
    box(830, 365, 318, 140, "#fee2e2", "#fb7185", "5. MuJoCo physics", ["torque + 重力 + 接触 + 摩擦", "→ 新 qpos / qvel"])
    arrow((1035, 303), (1035, 357))
    box(318, 365, 250, 140, "#e0f2fe", "#38bdf8", "6. 新 observation", ["qpos/qvel: 真实新状态", "actions: raw a_t"])
    arrow((830, 435), (576, 435))
    draw.arc((82, 291, 326, 540), 95, 235, fill="#334155", width=3)
    arrow((108, 365), (108, 305))
    text(62, 322, "下一次 policy", small)

    draw.rounded_rectangle((52, 548, 1148, 746), radius=16, fill="#ffffff", outline="#cbd5e1", width=2)
    text(78, 570, "实测: 左膝 +0.050 rad 目标偏移, 固定 target 的前 20 ms", box_title, "#102a43")
    headers = [(80, "子步"), (193, "时刻"), (301, "target"), (475, "实际 qpos"), (676, "BAM torque"), (893, "读法")]
    for x, value in headers:
        text(x, 615, value, small, "#102a43")
    draw.line((78, 641, 1120, 641), fill="#cbd5e1", width=2)
    values1 = [(80, "1"), (193, "5 ms"), (301, "0.04510"), (475, "-0.00467"), (676, "0.02772 Nm")]
    values4 = [(80, "4"), (193, "20 ms"), (301, "0.04510"), (475, "-0.00353"), (676, "0.02355 Nm")]
    for x, value in values1:
        text(x, 660, value, mono)
    for x, value in values4:
        text(x, 696, value, mono)
    text(893, 660, "刚开始向目标移动", small)
    text(893, 696, "误差变小, torque 下降", small)
    text(78, 766, "要点: latest target 固定, 但 BAM 读到的 target、torque 和 qpos 都可能每个子步不同.", small)

    path = OUTPUT_DIR / "u2-3-one-page-summary.png"
    image.save(path)
    return path


def main() -> None:
    """从站立初态施加小目标偏移, 输出四个物理子步的可复核记录."""
    infer = load_infer_module()
    bam_model = infer.load_bam_model(infer.BAM_KP_FW, 7.4, infer.BAM_MAX_CURRENT)
    model, data, bam_ctrl, names = infer.load_mujoco_with_bam(
        str(ROOT / "src" / "mjlab_microduck" / "robot" / "microduck" / "scene_walk.xml"),
        bam_model,
        PHYSICS_DT_S,
        vin_drop_gain=0.0,
        vin_min=infer.BAM_VIN_MIN,
    )

    actuator_id = names.index(JOINT_NAME)
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, JOINT_NAME)
    qpos_id = int(model.jnt_qposadr[joint_id])
    qvel_id = int(model.jnt_dofadr[joint_id])
    freejoint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    free_qpos_id = int(model.jnt_qposadr[freejoint_id])

    mujoco.mj_resetData(model, data)
    data.qpos[free_qpos_id + 2] = 0.125
    data.qpos[free_qpos_id + 3 : free_qpos_id + 7] = [1.0, 0.0, 0.0, 0.0]
    qpos_ids = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    data.qpos[qpos_ids] = infer.DEFAULT_POSE
    bam_ctrl.reset(data.qpos)
    bam_ctrl.q_target[:] = infer.DEFAULT_POSE
    bam_ctrl.q_target[actuator_id] += ACTION_OFFSET_RAD
    mujoco.mj_forward(model, data)

    records: list[dict[str, float | int]] = []
    for substep in range(1, NUM_SUBSTEPS + 1):
        bam_ctrl.update()
        torque_nm = float(data.ctrl[actuator_id])
        qpos_before = float(data.qpos[qpos_id])
        mujoco.mj_step(model, data)
        records.append(
            {
                "substep": substep,
                "time_ms": substep * PHYSICS_DT_S * 1000,
                "target_rad": float(bam_ctrl.q_target[actuator_id]),
                "qpos_before_rad": qpos_before,
                "qpos_after_rad": float(data.qpos[qpos_id]),
                "qvel_after_rad_s": float(data.qvel[qvel_id]),
                "bam_torque_nm": torque_nm,
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "u2-3-left-knee-substeps.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    delay_records = probe_delay_boundaries()
    delay_csv_path = OUTPUT_DIR / "u2-3-delay-boundaries.csv"
    with delay_csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(delay_records[0]))
        writer.writeheader()
        writer.writerows(delay_records)
    summary_path = draw_summary_image()

    print(f"关节: {JOINT_NAME}")
    print(f"目标偏移: {ACTION_OFFSET_RAD:.3f} rad, 物理步长: {PHYSICS_DT_S * 1000:.1f} ms")
    for row in records:
        print(
            f"子步 {row['substep']}: target={row['target_rad']:.5f} rad, "
            f"qpos={row['qpos_after_rad']:.5f} rad, "
            f"qvel={row['qvel_after_rad_s']:.4f} rad/s, "
            f"torque={row['bam_torque_nm']:.5f} Nm"
        )
    for lag in DELAY_BOUNDARY_LAGS:
        first_new = next(
            row for row in delay_records if row["fixed_lag_substeps"] == lag and row["bam_seen_target_rad"] > 0.0
        )
        print(
            f"固定延迟 {lag} 步: BAM 在第 {first_new['substep_after_new_target']} 个子步 "
            f"首次读到新目标, 此前经过 {first_new['elapsed_ms_before_substep']:.1f} ms"
        )
    print(f"已保存记录: {csv_path.relative_to(ROOT)}")
    print(f"已保存延迟边界记录: {delay_csv_path.relative_to(ROOT)}")
    print(f"已保存一页总结图: {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
