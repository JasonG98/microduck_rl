"""标注左膝 actuator 到 joint 的实际映射, 不运行动力学."""

import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('MESA_SHADER_CACHE_DIR', '/tmp/microduck-mesa-cache')
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parents[3]
out = root / 'docs/learning/assets/u2-1'
m = mujoco.MjModel.from_xml_path(str(root / 'src/mjlab_microduck/robot/microduck/scene_walk.xml'))
m.vis.global_.offwidth = 1000
m.vis.global_.offheight = 800
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, m.key('STAND').id)
mujoco.mj_forward(m, d)
joint = m.joint('left_knee').id
actuator = m.actuator('left_knee').id
assert m.actuator_trnid[actuator, 0] == joint
assert m.jnt_bodyid[joint] == m.body('leg').id
assert m.nu == 14
origin = d.xanchor[joint].copy()
left_bodies = [m.body(name).id for name in ('upper_leg_left', 'leg', 'ankle_left')]
keep = (np.isin(m.geom_bodyid, left_bodies) & (m.geom_group == 2)) | (np.arange(m.ngeom) == m.geom('floor').id)
m.geom_rgba[~keep, 3] = 0
for name, color in [('leg', [1, .65, .25, 1]), ('ankle_left', [.15, .85, .7, 1])]:
    ids = np.flatnonzero((m.geom_bodyid == m.body(name).id) & (m.geom_group == 2))
    m.geom_rgba[ids] = color
    m.geom_matid[ids] = -1
r = mujoco.Renderer(m, height=700, width=950)
c = mujoco.MjvCamera()
c.lookat[:] = origin + [0, 0, -.02]
c.azimuth, c.elevation, c.distance = 140, -18, .34
opt = mujoco.MjvOption()
opt.geomgroup[:] = 0
opt.geomgroup[[0, 2]] = 1
opt.sitegroup[:] = 0
r.update_scene(d, camera=c, scene_option=opt)
canvas = Image.new('RGB', (1600, 1000), '#f3f6fb')
canvas.paste(Image.fromarray(r.render()), (30, 160))
draw = ImageDraw.Draw(canvas)


def text(xy, value, size=25, bold=False, color='#15263e'):
    weight = 'Bold' if bold else 'Regular'
    font = ImageFont.truetype(f'/usr/share/fonts/opentype/noto/NotoSansCJK-{weight}.ttc', size)
    draw.text(xy, value, font=font, fill=color)


def project(point):
    cams = r.scene.camera
    pos = (np.array(cams[0].pos) + np.array(cams[1].pos)) / 2
    forward, up = np.array(cams[0].forward), np.array(cams[0].up)
    right = np.cross(forward, up)
    v = np.asarray(point) - pos
    focal = r.height / (2 * np.tan(np.deg2rad(m.vis.global_.fovy) / 2))
    return np.array([30 + r.width / 2 + focal * np.dot(v, right) / np.dot(v, forward),
                     160 + r.height / 2 - focal * np.dot(v, up) / np.dot(v, forward)])


text((40, 25), 'actuator: 驱动关节, 让机器人产生运动', 40, True)
text((42, 87), '真实左腿模型 · 橙色: 左小腿 · 绿色: 左脚 · 圆点按 left_knee 关节锚点定位', 23)
p = project(origin)
draw.ellipse((p[0]-17, p[1]-17, p[0]+17, p[1]+17), fill='#ff6383', outline='white', width=3)
draw.line([tuple(p), (900, 535), (1010, 535)], fill='#ff6383', width=4)
text((1015, 172), '模型 XML: 包含与引用', 30, True)
text((1025, 225), '├─ worldbody', 25)
text((1055, 267), '│  └─ … → body: leg', 25)
text((1055, 309), '│          └─ joint: left_knee', 25)
text((1025, 359), '└─ actuator', 25)
text((1055, 401), '    └─ position: left_knee', 25, True)
for y, title, desc in [(465, '引用: joint = left_knee', '名字相同, 但 joint 与 actuator 是不同对象'), (605, 'joint: 允许怎样运动', '左膝允许小腿相对大腿绕轴转动'), (745, 'actuator: 提供驱动力矩', '本项目训练时由 BAM 模拟 XL330 执行器')]:
    draw.rounded_rectangle((1010,y,1570,y+115), radius=16, fill='white')
    text((1030,y+10), title, 28, True)
    text((1030,y+61), desc, 22)
text((42, 900), '策略动作 → 关节目标位置 → BAM 电压/电机模型 → 力矩与摩擦 → 物理运动', 27, True)
text((42, 949), 'XML 中的 position 执行器会在训练构建时转换为 motor 模式; 目标位置不会直接覆盖实际关节位置。', 22)
canvas.save(out / '06-actuator-left-knee.png')
r.close()
print('已核对执行器关节映射并保存图:', out / '06-actuator-left-knee.png')
