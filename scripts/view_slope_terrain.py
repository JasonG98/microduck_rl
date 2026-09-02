"""在 MuJoCo viewer 中观察坡度模式 (roller_slope) 的斜坡地形.

只构建 "平地 + 斜坡" 地形 (FlatRampTerrainCfg), 分多行展示难度递增 (坡度 2° -> 20°),
并打开 MuJoCo 原生 viewer. 不需要训练好的策略 — 用于肉眼验证几何 (平地/斜坡接缝, 下坡方向).

用法:
    uv run python scripts/view_slope_terrain.py
    uv run python scripts/view_slope_terrain.py --rows 6 --ramp-max 8 --runout 4
    uv run python scripts/view_slope_terrain.py --build-only   # 无 GUI 测试

在 viewer 中: 滚轮缩放, 左键拖动旋转视角, 右键拖动平移. 每一行是一条越来越陡的斜坡
(难度 0 -> 1), 长度在 [ramp-min, ramp-max] 中随机抽取, 末尾接一段平地. "前方" (+x) 应向下.
"""

import argparse

import mujoco
import mujoco.viewer
from mjlab.terrains.terrain_generator import TerrainGenerator, TerrainGeneratorCfg

from mjlab_microduck.tasks.slope_terrain import (
    RAMP_DEG_MAX,
    RAMP_DEG_MIN,
    FlatRampTerrainCfg,
)


def build_model(rows, size, flat_length, ramp_range, runout, deg_min, deg_max):
    """构建仅含地形的 MuJoCo 模型 (rows 条坡度递增的斜坡)."""
    cfg = TerrainGeneratorCfg(
        seed=0,
        size=size,
        num_rows=rows,
        num_cols=1,
        curriculum=True,  # 沿行方向难度递增
        difficulty_range=(0.0, 1.0),
        add_lights=True,
        sub_terrains={
            "flat_ramp": FlatRampTerrainCfg(
                flat_length=flat_length,
                ramp_length_range=ramp_range,
                runout_length=runout,
                deg_min=deg_min,
                deg_max=deg_max,
            )
        },
    )
    generator = TerrainGenerator(cfg)
    spec = mujoco.MjSpec()
    generator.compile(spec)
    model = spec.compile()
    return model


def main():
    """打开 MuJoCo viewer 展示生成的斜坡地形瓦片."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--rows",
        type=int,
        default=5,
        help="行数 = 展示的坡度数量 (默认 5)",
    )
    p.add_argument(
        "--size",
        type=float,
        nargs=2,
        default=(15.0, 4.0),
        help="单个瓦片的尺寸 (x y), 单位 m",
    )
    p.add_argument("--flat-length", type=float, default=2.0, help="起始平地长度 (m)")
    p.add_argument(
        "--ramp-min",
        type=float,
        default=3.0,
        help="斜坡最小水平长度 (m)",
    )
    p.add_argument(
        "--ramp-max",
        type=float,
        default=8.0,
        help="斜坡最大水平长度 (m)",
    )
    p.add_argument("--runout", type=float, default=4.0, help="末端平地长度 (m)")
    p.add_argument(
        "--deg-min",
        type=float,
        default=RAMP_DEG_MIN,
        help=f"最小坡度, 单位度 (默认 {RAMP_DEG_MIN})",
    )
    p.add_argument(
        "--deg-max",
        type=float,
        default=RAMP_DEG_MAX,
        help=f"最大坡度, 单位度 (默认 {RAMP_DEG_MAX})",
    )
    p.add_argument(
        "--build-only",
        action="store_true",
        help="只构建模型后退出 (无 GUI 测试)",
    )
    args = p.parse_args()

    model = build_model(
        rows=args.rows,
        size=tuple(args.size),
        flat_length=args.flat_length,
        ramp_range=(args.ramp_min, args.ramp_max),
        runout=args.runout,
        deg_min=args.deg_min,
        deg_max=args.deg_max,
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print(
        f"地形已构建: {args.rows} 条斜坡, 坡度 {args.deg_min}°->{args.deg_max}°, "
        f"斜坡长度 {args.ramp_min}-{args.ramp_max}m + 末端 {args.runout}m, "
        f"共 {model.ngeom} 个几何体."
    )
    if args.build_only:
        print("--build-only: OK, 无 GUI.")
        return

    print("正在打开 MuJoCo viewer (Ctrl+C 退出)...")
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        while viewer.is_running():
            mujoco.mj_forward(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
