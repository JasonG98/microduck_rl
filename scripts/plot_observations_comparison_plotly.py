#!/usr/bin/env python3
"""使用 Plotly 绘制真实与仿真观测值的对比图."""

import argparse
import pickle
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_observations(pkl_path: str):
    """从 pickle 文件加载观测值."""
    with Path(pkl_path).open("rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        if "observations" in data and "timestamps" in data:
            observations = data["observations"]
            timestamps = data["timestamps"]
        else:
            raise ValueError("字典必须包含 'observations' 和 'timestamps' 键")
    elif isinstance(data, list):
        if len(data) == 0:
            raise ValueError("数据列表为空")

        if isinstance(data[0], dict) and "timestamp" in data[0] and "observation" in data[0]:
            timestamps = [item["timestamp"] for item in data]
            observations = [item["observation"] for item in data]
        elif isinstance(data[0], tuple):
            timestamps = [item[0] for item in data]
            observations = [item[1] for item in data]
        else:
            observations = data
            timestamps = [i * 0.02 for i in range(len(observations))]
    else:
        raise ValueError(f"不支持的数据格式: {type(data)}")

    return np.array(observations), np.array(timestamps)


def plot_comparison(real_obs, real_ts, sim_obs=None, sim_ts=None):
    """使用 Plotly 绘制真实与仿真观测值的对比图.

    如果 sim_obs 为 None, 则只绘制真实数据.
    """
    # 关节名称
    joint_names = [
        "L_hip_yaw",
        "L_hip_roll",
        "L_hip_pitch",
        "L_knee",
        "L_ankle",
        "neck_pitch",
        "head_pitch",
        "head_yaw",
        "head_roll",
        "R_hip_yaw",
        "R_hip_roll",
        "R_hip_pitch",
        "R_knee",
        "R_ankle",
    ]

    obs_dim = real_obs.shape[1] if sim_obs is None else min(real_obs.shape[1], sim_obs.shape[1])

    # Velocity (51D): ang_vel (3) + proj_grav (3) + joint_pos (14) + joint_vel (14) + actions (14) + command (3)
    base_ang_vel_start = 0
    gravity_start = 3
    joint_pos_start = 6
    joint_vel_start = 20
    action_start = 34

    # 创建带分区的子图标题
    subplot_titles = []

    # 基座角速度 (3)
    subplot_titles.extend(["<b>BASE ANG VEL</b><br>ω_x", "ω_y", "ω_z", ""])

    # 原始加速度计 (3)
    subplot_titles.extend(["<b>Raw Accelero</b><br>g_x", "g_y", "g_z", ""])

    # 关节位置 (14 + 2 个空位)
    subplot_titles.append(f"<b>JOINT POSITIONS</b><br>{joint_names[0]}")
    subplot_titles.extend(joint_names[1:14])
    subplot_titles.extend(["", ""])

    # 关节速度 (14 + 2 个空位)
    subplot_titles.append(f"<b>JOINT VELOCITIES</b><br>{joint_names[0]}")
    subplot_titles.extend(joint_names[1:14])
    subplot_titles.extend(["", ""])

    # 动作 (14 + 2 个空位)
    subplot_titles.append(f"<b>ACTIONS</b><br>{joint_names[0]}")
    subplot_titles.extend(joint_names[1:14])
    subplot_titles.extend(["", ""])

    num_rows = 14
    fig = make_subplots(
        rows=num_rows,
        cols=4,
        subplot_titles=subplot_titles,
        vertical_spacing=0.02,
        horizontal_spacing=0.05,
        row_heights=[1] * num_rows,
    )

    plot_idx = 0

    # 跟踪数据以便统一缩放

    def add_traces(row, col, real_data, sim_data=None, y_range=None):
        """向子图添加真实和仿真 trace 的辅助函数."""
        fig.add_trace(
            go.Scatter(
                x=real_ts,
                y=real_data,
                name="Real",
                line={"color": "blue", "width": 1.5},
                showlegend=(plot_idx == 0),
            ),
            row=row,
            col=col,
        )
        if sim_data is not None:
            fig.add_trace(
                go.Scatter(
                    x=sim_ts,
                    y=sim_data,
                    name="Sim",
                    line={"color": "red", "width": 1.5, "dash": "dash"},
                    showlegend=(plot_idx == 0),
                ),
                row=row,
                col=col,
            )
        if y_range:
            fig.update_yaxes(range=y_range, row=row, col=col)

    base_ang_vel_data = []
    gravity_data = []
    joint_pos_data = []
    joint_vel_data = []
    action_data = []

    # 1. 基座角速度 (3 个子图)
    for i in range(3):
        row, col = divmod(plot_idx, 4)
        row += 1
        col += 1
        base_ang_vel_data.append(real_obs[:, base_ang_vel_start + i])
        if sim_obs is not None:
            base_ang_vel_data.append(sim_obs[:, base_ang_vel_start + i])
        add_traces(
            row,
            col,
            real_obs[:, base_ang_vel_start + i],
            None if sim_obs is None else sim_obs[:, base_ang_vel_start + i],
        )
        fig.update_yaxes(title_text="rad/s", row=row, col=col)
        plot_idx += 1

    # 空位
    plot_idx += 1

    # 2. 原始加速度计 (3 个子图)
    for i in range(3):
        row, col = divmod(plot_idx, 4)
        row += 1
        col += 1
        gravity_data.append(real_obs[:, gravity_start + i])
        if sim_obs is not None:
            gravity_data.append(sim_obs[:, gravity_start + i])
        add_traces(
            row,
            col,
            real_obs[:, gravity_start + i],
            None if sim_obs is None else sim_obs[:, gravity_start + i],
        )
        fig.update_yaxes(title_text="g", row=row, col=col)
        plot_idx += 1

    # 空位
    plot_idx += 1

    # 3. 关节位置 (14 个子图)
    for i in range(14):
        row, col = divmod(plot_idx, 4)
        row += 1
        col += 1
        if joint_pos_start + i < obs_dim:
            joint_pos_data.append(real_obs[:, joint_pos_start + i])
            if sim_obs is not None:
                joint_pos_data.append(sim_obs[:, joint_pos_start + i])
            add_traces(
                row,
                col,
                real_obs[:, joint_pos_start + i],
                None if sim_obs is None else sim_obs[:, joint_pos_start + i],
            )
        fig.update_yaxes(title_text="rad", row=row, col=col)
        plot_idx += 1

    # 跳过 2 个空位
    plot_idx += 2

    # 4. 关节速度 (14 个子图)
    for i in range(14):
        row, col = divmod(plot_idx, 4)
        row += 1
        col += 1
        if joint_vel_start + i < obs_dim:
            joint_vel_data.append(real_obs[:, joint_vel_start + i])
            if sim_obs is not None:
                joint_vel_data.append(sim_obs[:, joint_vel_start + i])
            add_traces(
                row,
                col,
                real_obs[:, joint_vel_start + i],
                None if sim_obs is None else sim_obs[:, joint_vel_start + i],
            )
        fig.update_yaxes(title_text="rad/s", row=row, col=col)
        plot_idx += 1

    # 跳过 2 个空位
    plot_idx += 2

    # 5. 动作 (14 个子图)
    for i in range(14):
        row, col = divmod(plot_idx, 4)
        row += 1
        col += 1
        if action_start + i < obs_dim:
            action_data.append(real_obs[:, action_start + i])
            if sim_obs is not None:
                action_data.append(sim_obs[:, action_start + i])
            add_traces(
                row,
                col,
                real_obs[:, action_start + i],
                None if sim_obs is None else sim_obs[:, action_start + i],
            )
        fig.update_yaxes(title_text="action", row=row, col=col)
        fig.update_xaxes(title_text="Time (s)", row=row, col=col)
        plot_idx += 1

    # 为每组设置统一的 y 轴范围
    def compute_range(data_list):
        if not data_list:
            return None
        all_data = np.concatenate([d.flatten() for d in data_list])
        y_min, y_max = np.min(all_data), np.max(all_data)
        margin = (y_max - y_min) * 0.1
        return [y_min - margin, y_max + margin]

    base_ang_vel_range = compute_range(base_ang_vel_data)
    gravity_range = compute_range(gravity_data)
    joint_pos_range = compute_range(joint_pos_data)
    joint_vel_range = compute_range(joint_vel_data)
    action_range = compute_range(action_data)

    # 应用统一范围
    plot_idx = 0

    for _ in range(3):  # 基座角速度
        row, col = divmod(plot_idx, 4)
        fig.update_yaxes(range=base_ang_vel_range, row=row + 1, col=col + 1)
        plot_idx += 1
    plot_idx += 1

    for _ in range(3):  # 重力
        row, col = divmod(plot_idx, 4)
        fig.update_yaxes(range=gravity_range, row=row + 1, col=col + 1)
        plot_idx += 1
    plot_idx += 1

    for _ in range(14):  # 关节位置
        row, col = divmod(plot_idx, 4)
        fig.update_yaxes(range=joint_pos_range, row=row + 1, col=col + 1)
        plot_idx += 1
    plot_idx += 2

    for _ in range(14):  # 关节速度
        row, col = divmod(plot_idx, 4)
        fig.update_yaxes(range=joint_vel_range, row=row + 1, col=col + 1)
        plot_idx += 1
    plot_idx += 2

    for _ in range(14):  # 动作
        row, col = divmod(plot_idx, 4)
        fig.update_yaxes(range=action_range, row=row + 1, col=col + 1)
        plot_idx += 1

    # 更新布局
    title = "Real vs Simulated Observations Comparison" if sim_obs is not None else "Real Robot Observations"

    fig.update_layout(
        title_text=title,
        title_font_size=24,
        height=4600,
        width=1600,
        showlegend=True,
        legend={"x": 0.85, "y": 0.99, "bgcolor": "rgba(255,255,255,0.8)"},
        hovermode="x unified",
    )

    fig.show()


def main():
    """用交互式 Plotly 图表对比真实与仿真观测值."""
    parser = argparse.ArgumentParser(description="对比真实与仿真观测值 (Plotly 版本)")
    parser.add_argument("real_pkl", type=str, help="真实机器人观测值的 .pkl 文件路径")
    parser.add_argument(
        "sim_pkl",
        type=str,
        nargs="?",
        default=None,
        help="仿真观测值的 .pkl 文件路径 (可选)",
    )
    args = parser.parse_args()

    # 检查文件是否存在
    if not Path(args.real_pkl).exists():
        print(f"错误: 未找到 {args.real_pkl}")
        return 1

    # 加载观测值
    print(f"正在从 {args.real_pkl} 加载真实观测值...")
    real_obs, real_ts = load_observations(args.real_pkl)
    print(f"已加载 {len(real_obs)} 条真实观测值 (shape: {real_obs.shape})")

    if args.sim_pkl:
        if not Path(args.sim_pkl).exists():
            print(f"错误: 未找到 {args.sim_pkl}")
            return 1
        print(f"正在从 {args.sim_pkl} 加载仿真观测值...")
        sim_obs, sim_ts = load_observations(args.sim_pkl)
        print(f"已加载 {len(sim_obs)} 条仿真观测值 (shape: {sim_obs.shape})")
    else:
        print("未提供仿真数据, 只绘制真实数据")
        sim_obs, sim_ts = None, None

    # 绘制对比图
    print("\n正在生成交互式对比图...")
    plot_comparison(real_obs, real_ts, sim_obs, sim_ts)

    return 0


if __name__ == "__main__":
    exit(main())
