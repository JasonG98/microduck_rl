"""一个模拟的 VL53L5CX, 让鸭子能看到一堵墙.

在 45 度的方形视场上分布 8x8 的区块, 最远 4 m, 15 Hz -- 与真实传感器的形态一致, 因为
`tofd` 发布的正是这种帧, 且 `maploc` 也假定如此进行重投影.

以 `~/MISC/microduck_maploc` 的 `sim/tof_sensor.py` 为模型, 其数据来自数据手册和桌面上的真机
测量: 噪声随距离增大 (近距离为毫米级, 接近极限时为厘米级), 并且每个区块带有一个状态而非仅有
距离.

**状态字节与距离同样重要.** 真实传感器区分了"前方什么都没有"和"无法测量", 而 `maploc` 对两者
的处理不同 -- 没有目标的区块在地图上是待清空的空白区域, 而测量失败的区块则完全不含信息.
一个只上报距离的模拟器会让真机才能发现的那种 bug 溜过去.
"""

from __future__ import annotations

import mujoco
import numpy as np

# 传感器, 与 `tof/src/lib.rs` 发布的一致.
ROWS = 8
COLS = 8
ZONES = ROWS * COLS
STATUS_VALID = 5
STATUS_NO_TARGET = 255

# VL53L5CX: 每轴 45 度 (对角线 63 度), 量程 4 m.
FOV_DEG = 45.0
MAX_RANGE = 4.0


class Tof:
    """一只鸭子 `tof` 位置上的 8x8 传感器.

    光线在 site 坐标系中发射 -- +x 向前, +y 向左, +z 向上 -- 所以转动的头部会带着传感器一起转,
    这正是 `robot.look` 扫描房间的意义所在.
    """

    def __init__(self, model: mujoco.MjModel, site: int, seed: int = 0):
        self.model = model
        self.site = site
        self.random = np.random.default_rng(seed)

        # 区块中心, 只计算一次. 方位角绕 +z 轴 (正方向朝向 +y, 即鸭子的左侧), 高度向上,
        # 两者都覆盖整个视场 -- 第 0 行是画面顶部, 第 0 列是传感器的左侧, 与
        # `kinematics::tof` 读取真实传感器缓冲区的方式一致. 如果第 0 列在右侧, 每一帧到达
        # 映射器时都会被镜像: 斜墙会被画在头部轴线的镜像位置, 且孪生体上永远无法闭合闭环.
        half = np.radians(FOV_DEG) / 2.0
        edges = np.linspace(-half, half, COLS + 1)
        centres = (edges[:-1] + edges[1:]) / 2.0
        self.directions = np.zeros((ZONES, 3))
        for row in range(ROWS):
            elevation = -centres[row]
            for col in range(COLS):
                azimuth = -centres[col]
                self.directions[row * COLS + col] = [
                    np.cos(elevation) * np.cos(azimuth),
                    np.cos(elevation) * np.sin(azimuth),
                    np.sin(elevation),
                ]

    def frame(self, data: mujoco.MjData) -> tuple[list[int], list[int]]:
        """一次采样: 每个区块的距离 (毫米) 和状态.

        自身碰撞会被上报而不是过滤掉. 当鸭子的喙位于传感器前方时, 真实传感器能看到自己的喙,
        而一个悄悄跳过自身几何的模拟器恰恰会掩盖这里本要捕获的那种安装问题.
        """
        origin = data.site_xpos[self.site].copy()
        rotation = data.site_xmat[self.site].reshape(3, 3)
        world = rotation @ self.directions.T  # 形状 (3, ZONES)

        distance_mm = [0] * ZONES
        status = [STATUS_NO_TARGET] * ZONES
        geom = np.zeros(1, dtype=np.int32)
        for zone in range(ZONES):
            hit = mujoco.mj_ray(
                self.model,
                data,
                origin,
                np.ascontiguousarray(world[:, zone]),
                None,
                1,
                -1,
                geom,
            )
            if hit < 0 or hit > MAX_RANGE:
                continue
            # 随距离增大的噪声, 与数据手册一致: 近距离为几毫米, 远距离为几厘米. 没有它,
            # 模拟地图会过于干净, 下游的每个滤波器也都不会得到检验.
            sigma = 0.003 + 0.02 * (hit / MAX_RANGE)
            measured = max(0.0, hit + self.random.normal(0.0, sigma))
            distance_mm[zone] = int(measured * 1000.0)
            status[zone] = STATUS_VALID
        return distance_mm, status
