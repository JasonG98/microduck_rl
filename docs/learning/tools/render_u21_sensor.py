"""在真实躯干模型上标注 imu site 及 gyro 的引用关系, 并核对瞬时读数."""

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
site = m.site('imu').id
body = m.body('trunk_base').id
assert m.site_bodyid[site] == body
sensor = m.sensor('imu_ang_vel').id
assert m.sensor_objid[sensor] == site
assert m.sensor_dim[sensor] == 3
address = m.sensor_adr[sensor]
assert np.allclose(d.sensordata[address:address+3], 0)
free = m.joint('trunk_base_freejoint').id
d.qvel[m.jnt_dofadr[free]+5] = 1.0
mujoco.mj_forward(m, d)
assert np.allclose(d.sensordata[address:address+3], [0, 0, 1])
d.qvel[:] = 0
mujoco.mj_forward(m, d)
origin = d.site_xpos[site].copy()
rotation = d.site_xmat[site].reshape(3, 3).copy()
assert np.allclose(rotation.T @ rotation, np.eye(3))
keep = ((m.geom_bodyid == body) & (m.geom_group == 2)) | (np.arange(m.ngeom) == m.geom('floor').id)
m.geom_rgba[~keep, 3] = 0
r = mujoco.Renderer(m, height=700, width=950)
c = mujoco.MjvCamera()
c.lookat[:] = origin + [0, 0, .012]
c.azimuth, c.elevation, c.distance = 135, -18, .29
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


text((40, 25), 'sensor: 在 imu 坐标系中读取角速度', 40, True)
text((42, 87), '真实 trunk_base 躯干模型 · 白点与坐标轴为 imu site · 陀螺仪通过名称引用它', 23)
p = project(origin)
for axis, name, color in [(0, 'X', '#ff6b7f'), (1, 'Y', '#45e3b5'), (2, 'Z', '#72baff')]:
    tip = project(origin + rotation[:, axis] * .05)
    unit = (tip-p) / np.linalg.norm(tip-p)
    side = np.array([-unit[1], unit[0]])
    draw.line([tuple(p), tuple(tip)], fill=color, width=7)
    draw.polygon([tuple(tip), tuple(tip-unit*19+side*8), tuple(tip-unit*19-side*8)], fill=color)
    text(tuple(tip + unit*12 - [8, 16]), name, 31, True, color)
draw.ellipse((p[0]-10, p[1]-10, p[0]+10, p[1]+10), fill='white', outline='#172a44', width=3)
draw.line([tuple(p), (900, 517), (1010, 517)], fill='white', width=3)
text((1015, 172), '模型 XML: 包含关系', 30, True)
text((1025, 225), '├─ worldbody', 25)
text((1055, 267), '└─ body: trunk_base', 25)
text((1090, 309), '└─ site: imu', 25, True)
text((1025, 359), '└─ sensor', 25)
text((1055, 401), '└─ gyro: imu_ang_vel', 25, True)
for y, title, desc in [(465, 'site = imu', '引用左图的参考位置和方向'), (605, '测量结果: 三个数', '绕 imu 的 X / Y / Z 轴转得多快'), (745, '单位: rad/s (弧度每秒)', '输出角速度, 不输出转过的角度')]:
    draw.rounded_rectangle((1010,y,1570,y+115), radius=16, fill='white')
    text((1030,y+10), title, 28, True)
    text((1030,y+61), desc, 24)
text((42, 900), '导师瞬时读数核对: 静止 → [0, 0, 0]; 人工设置绕 Z 轴 1 rad/s → [0, 0, 1]。', 26, True)
text((42, 949), '这是模型状态/传感器映射核对, 非策略或动力学实验; 标记透过外壳显示, 箭头示意长度 5 cm。', 22)
canvas.save(out / '05-sensor-imu.png')
r.close()
print('已核对陀螺仪读数并保存图:', out / '05-sensor-imu.png')
