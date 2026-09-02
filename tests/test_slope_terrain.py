import math

import mujoco
import numpy as np

from mjlab_microduck.tasks.slope_terrain import RAMP_DEG_MAX, RAMP_DEG_MIN, FlatRampTerrainCfg, ramp_angle_by_difficulty


def test_ramp_angle_endpoints():
    assert math.isclose(ramp_angle_by_difficulty(0.0), math.radians(RAMP_DEG_MIN), abs_tol=1e-9)
    assert math.isclose(ramp_angle_by_difficulty(1.0), math.radians(RAMP_DEG_MAX), abs_tol=1e-9)


def test_ramp_angle_midpoint():
    mid_deg = (RAMP_DEG_MIN + RAMP_DEG_MAX) / 2.0
    assert math.isclose(ramp_angle_by_difficulty(0.5), math.radians(mid_deg), abs_tol=1e-9)


def test_ramp_angle_clamps_out_of_range():
    assert math.isclose(ramp_angle_by_difficulty(-1.0), math.radians(RAMP_DEG_MIN), abs_tol=1e-9)
    assert math.isclose(ramp_angle_by_difficulty(2.0), math.radians(RAMP_DEG_MAX), abs_tol=1e-9)


def _empty_terrain_spec():
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    return spec


def test_flat_ramp_builds_geoms_and_origin_on_flat():
    cfg = FlatRampTerrainCfg(flat_length=2.0)
    cfg.size = (15.0, 4.0)  # 由生成器正常设置
    spec = _empty_terrain_spec()
    out = cfg.function(difficulty=0.5, spec=spec, rng=np.random.default_rng(0))
    # 三块几何: 起始平地 + 坡道 + 出口平地
    assert len(out.geometries) == 3
    # origin 在坡道上 (越过起始平地), 所以 x > flat_length 且 z < 0
    assert out.origin[0] == cfg.flat_length + cfg.spawn_on_ramp
    assert out.origin[2] < 0.0
    # z = 坡道边 spawn_on_ramp 处的倾斜表面 (落差 = d * tan(angle))
    angle = ramp_angle_by_difficulty(0.5, cfg.deg_min, cfg.deg_max)
    assert abs(out.origin[2] - (-cfg.spawn_on_ramp * math.tan(angle))) < 1e-9


def test_flat_ramp_steeper_at_higher_difficulty():
    # 难度更高时, 坡道末端更低
    cfg = FlatRampTerrainCfg()
    cfg.size = (15.0, 4.0)
    easy = cfg.function(0.0, _empty_terrain_spec(), np.random.default_rng(0))
    hard = cfg.function(1.0, _empty_terrain_spec(), np.random.default_rng(0))
    # 坡道 (第 2 块几何) 在困难时更低 (z 中心更负)
    # (同 rng -> 抽到同样的长度 -> 只有坡度变了)
    assert hard.geometries[1].geom.pos[2] < easy.geometries[1].geom.pos[2]


def test_ramp_joins_flat_platform_no_gap():
    # 坡顶必须碰到平地块的边缘 (x=flat_length):
    # 坡道中心在 x 上偏移 -(t/2)*sin(angle).
    cfg = FlatRampTerrainCfg()
    cfg.size = (15.0, 4.0)
    out = cfg.function(0.5, _empty_terrain_spec(), np.random.default_rng(0))
    ramp = out.geometries[1].geom
    angle = ramp_angle_by_difficulty(0.5, cfg.deg_min, cfg.deg_max)
    surf_half = ramp.size[0]
    ramp_len = surf_half * 2.0 * math.cos(angle)
    expected_cx = cfg.flat_length + ramp_len / 2.0 - (cfg.thickness / 2.0) * math.sin(angle)
    assert abs(ramp.pos[0] - expected_cx) < 1e-6


def test_flat_ramp_runout_at_ramp_bottom():
    # 出口平地 (第 3 块几何) 在坡底水平 (z<0),
    # 它的表面是平的 (没旋转的 box: identity 四元数).
    cfg = FlatRampTerrainCfg()
    cfg.size = (15.0, 4.0)
    out = cfg.function(1.0, _empty_terrain_spec(), np.random.default_rng(0))
    runout = out.geometries[2].geom
    assert runout.pos[2] < 0.0  # 下沉到起始平地之下
    # identity 四元数 (平, 不倾斜)
    assert math.isclose(runout.quat[0], 1.0, abs_tol=1e-9)


def test_ramp_length_within_range():
    cfg = FlatRampTerrainCfg(ramp_length_range=(3.0, 8.0))
    cfg.size = (15.0, 4.0)
    # 坡道表面 = ramp_length / cos(angle); difficulty 0 时 angle=2°,
    # 所以 surf_len ~= ramp_length.在多次抽样上校验.
    for seed in range(20):
        out = cfg.function(0.0, _empty_terrain_spec(), np.random.default_rng(seed))
        surf_half = out.geometries[1].geom.size[0]
        ramp_len = surf_half * 2.0 * math.cos(math.radians(2.0))
        assert 3.0 - 1e-6 <= ramp_len <= 8.0 + 1e-6
