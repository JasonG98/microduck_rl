"""roller_slope 任务的自定义地形 « 平地 + 下坡 ».

机器人在平地区出生, 沿 +x 方向受一冲量, 滚到坡道后顺势下滑.坡道
角度由 difficulty (curriculum) 在 [RAMP_DEG_MIN, RAMP_DEG_MAX] 度之间插值.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np
from mjlab.terrains.terrain_generator import (
    SubTerrainCfg,
    TerrainGeometry,
    TerrainOutput,
)

RAMP_DEG_MIN = 2.0
RAMP_DEG_MAX = 20.0


def ramp_angle_by_difficulty(difficulty: float, deg_min: float = RAMP_DEG_MIN, deg_max: float = RAMP_DEG_MAX) -> float:
    """由 difficulty [0,1] 线性插值得到的坡道角度 (弧度)."""
    d = float(np.clip(difficulty, 0.0, 1.0))
    return math.radians(deg_min + d * (deg_max - deg_min))


@dataclass(kw_only=True)
class FlatRampTerrainCfg(SubTerrainCfg):
    """起始平地 → 下坡 → 出口平地.

    沿 +x 方向对齐的三个 box:
      1. 起始平地 (z=0 表面), 机器人在此出生;
      2. 下坡, 角度由 difficulty 插值, 水平长度在
         ``ramp_length_range`` 中随机取值 (每块地形一个值, 生成时固定);
      3. 出口平地位于坡底, 确保机器人落在实地而非虚空.
    """

    flat_length: float = 2.0  # 起始平地 (m)
    ramp_length_range: tuple = (
        3.0,
        8.0,
    )  # 坡道水平长度 (m), 随机取值
    runout_length: float = 4.0  # 坡底出口平地 (m)
    spawn_on_ramp: float = 0.3  # 在坡道上沿出生的距离 (m) (重力 => 滚动)
    deg_min: float = RAMP_DEG_MIN
    deg_max: float = RAMP_DEG_MAX
    thickness: float = 0.5  # box 厚度 (m)

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng) -> TerrainOutput:
        """生成平地、坡道与出口平地的 geom 以及出生原点."""
        total_max = self.flat_length + self.ramp_length_range[1] + self.runout_length
        assert total_max <= self.size[0], f"flat+ramp_max+runout ({total_max}) 必须能放进 size[0] ({self.size[0]})"
        body = spec.body("terrain")
        angle = ramp_angle_by_difficulty(difficulty, self.deg_min, self.deg_max)
        width = self.size[1]
        t = self.thickness
        # 随机抽取的坡道长度 (对给定 rng 是确定的).
        ramp_length = float(rng.uniform(self.ramp_length_range[0], self.ramp_length_range[1]))
        drop = ramp_length * math.tan(angle)  # 落差 (m), 正值

        # 1) 起始平地: 表面 z=0, x 范围 [0, flat_length].
        flat = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(self.flat_length / 2.0, width / 2.0, t / 2.0),
            pos=(self.flat_length / 2.0, 0.0, -t / 2.0),
        )

        # 2) 坡道: 绕 +y 轴旋转 +angle 的 box (+x 一侧下降).
        # 沿 x 方向偏移 -(t/2)·sin(angle): 若无此偏移, 倾斜表面的上沿
        # 会落在 x=flat_length+(t/2)sin(a) -> 起始平地 (终点 flat_length) 与
        # 坡道之间出现小缝.加上偏移后, 坡道顶恰好贴平起始平地的边缘
        # (干净衔接), 底部恰好贴住出口平地.
        surf_len = ramp_length / math.cos(angle)
        ramp_cx = self.flat_length + ramp_length / 2.0 - (t / 2.0) * math.sin(angle)
        ramp_cz = -(drop / 2.0) - (t / 2.0) * math.cos(angle)
        half = angle / 2.0
        ramp = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(surf_len / 2.0, width / 2.0, t / 2.0),
            pos=(ramp_cx, 0.0, ramp_cz),
            quat=(math.cos(half), 0.0, math.sin(half), 0.0),
        )

        # 3) 出口平地: 表面位于坡底高度 (z = -drop).
        runout_cx = self.flat_length + ramp_length + self.runout_length / 2.0
        runout = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(self.runout_length / 2.0, width / 2.0, t / 2.0),
            pos=(runout_cx, 0.0, -drop - t / 2.0),
        )

        # 在坡道上往里一点出生: 重力让轮子立刻开始滚动 (动量在轮子上,
        # 而非打滑的底盘推力), 且机器人已在斜坡上.z 取该距离处的倾斜
        # 表面高度.
        spawn_x = self.flat_length + self.spawn_on_ramp
        spawn_z = -self.spawn_on_ramp * math.tan(angle)
        origin = np.array([spawn_x, 0.0, spawn_z])
        return TerrainOutput(
            origin=origin,
            geometries=[
                TerrainGeometry(geom=flat, color=(0.5, 0.5, 0.5, 1.0)),
                TerrainGeometry(geom=ramp, color=(0.45, 0.55, 0.75, 1.0)),
                TerrainGeometry(geom=runout, color=(0.5, 0.5, 0.5, 1.0)),
            ],
        )
