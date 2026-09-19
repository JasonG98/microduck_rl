"""在真实左脚模型上标注 site 原点与局部坐标轴, 不运行动力学."""

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
site = m.site('left_foot').id
body = m.body('ankle_left').id
assert m.site_bodyid[site] == body
origin = d.site_xpos[site].copy()
rotation = d.site_xmat[site].reshape(3, 3).copy()
assert np.allclose(rotation.T @ rotation, np.eye(3))
keep = ((m.geom_bodyid == body) & (m.geom_group == 2)) | (np.arange(m.ngeom) == m.geom('floor').id)
m.geom_rgba[~keep, 3] = 0
r = mujoco.Renderer(m, height=700, width=950)
c = mujoco.MjvCamera()
c.lookat[:] = origin + [0, 0, .012]
c.azimuth, c.elevation, c.distance = 130, -28, .20
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


text((40, 25), 'site: 给左脚贴一个带方向的参考标记', 40, True)
text((42, 87), '真实 left_foot 坐标 · 不是关节, 也不是碰撞形状 · 其余机器人部件仅隐藏显示', 23)
p = project(origin)
for axis, name, color in [(0, 'X', '#ff6b7f'), (1, 'Y', '#45e3b5'), (2, 'Z', '#72baff')]:
    tip = project(origin + rotation[:, axis] * .04)
    unit = (tip-p) / np.linalg.norm(tip-p)
    side = np.array([-unit[1], unit[0]])
    draw.line([tuple(p), tuple(tip)], fill=color, width=7)
    draw.polygon([tuple(tip), tuple(tip-unit*19+side*8), tuple(tip-unit*19-side*8)], fill=color)
    text(tuple(tip + unit*12 - [8, 16]), name, 31, True, color)
draw.ellipse((p[0]-10, p[1]-10, p[0]+10, p[1]+10), fill='white', outline='#172a44', width=3)
draw.line([tuple(p), (900, 490), (1010, 490)], fill='white', width=3)
text((1015, 175), 'body: ankle_left', 31, True)
text((1035, 230), '├─ joint: left_ankle', 27)
text((1035, 281), '├─ geom: 外观 / 碰撞形状', 27)
text((1035, 332), '└─ site: left_foot', 28, True)
for y, title, desc in [(430, '位置', '白点: 这个参考坐标系的原点'), (585, '方向', '三根箭头: 它自己的 X / Y / Z 轴'), (740, '归属', '固定在左脚 body 上, 随脚一起运动')]:
    draw.rounded_rectangle((1010,y,1570,y+120), radius=16, fill='white')
    text((1030,y+12), title, 29, True)
    text((1030,y+62), desc, 24)
text((42, 902), '用途: 为“脚在哪里、朝向哪里”提供统一参考; 本项目用它定位脚部地形高度传感器。', 26, True)
text((42, 949), '箭头按 site_xmat 真实方向绘制, 示意长度 4 cm; 为便于观察透过外观显示, 不产生任何碰撞。', 22)
canvas.save(out / '04-site-left-foot.png')
r.close()
print('已保存左脚 site 标注图:', out / '04-site-left-foot.png')
