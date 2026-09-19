"""渲染机器人实际状态的 qpos/qvel 布局, 按名字核对地址和维度."""

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
free = m.joint('trunk_base_freejoint').id
knee = m.joint('left_knee').id
hinges = np.flatnonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
assert len(hinges) == 14 and m.nq == 7 + len(hinges) and m.nv == 6 + len(hinges)
assert np.isclose(np.linalg.norm(d.qpos[m.jnt_qposadr[free]+3:m.jnt_qposadr[free]+7]), 1)
origin = d.xanchor[free].copy()
for name, color in [('trunk_base', [.3,.6,1,1]), ('leg', [1,.65,.25,1])]:
    ids = np.flatnonzero((m.geom_bodyid == m.body(name).id) & (m.geom_group == 2))
    m.geom_rgba[ids] = color
    m.geom_matid[ids] = -1
r = mujoco.Renderer(m, height=760, width=680)
c = mujoco.MjvCamera()
c.lookat[:] = [0, 0, .16]
c.azimuth, c.elevation, c.distance = 135, -15, .60
opt = mujoco.MjvOption()
opt.geomgroup[:] = 0
opt.geomgroup[[0, 2]] = 1
opt.sitegroup[:] = 0
r.update_scene(d, camera=c, scene_option=opt)
canvas = Image.new('RGB', (1600, 1060), '#f3f6fb')
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


text((40, 25), 'qpos / qvel: 机器人此刻的实际状态', 40, True)
text((42, 88), '当前 walk 模型: 一个自由基座 + 14 个铰链关节 · 目标位置另存, 不在这两组状态里', 24)
for point, color, y, caption in [(d.xanchor[free], '#559eff', 500, '自由基座'), (d.xanchor[knee], '#ffac50', 700, '左膝关节')]:
    p = project(point)
    draw.ellipse((p[0]-12,p[1]-12,p[0]+12,p[1]+12), fill=color, outline='white', width=3)
    draw.line([tuple(p),(680,y)], fill=color, width=3)
    text((475,y+10),caption,24,True,color)
text((760,175), 'qpos: 位置 / 姿态 / 关节角', 32, True)
text((760,225), '数组长度 nq = 21', 25)
text((760,425), 'qvel: 线速度 / 角速度', 32, True)
text((760,475), '数组长度 nv = 20', 25)
for y, labels, counts in [(280, ['基座位置','基座姿态','14 个关节角'], [3,4,14]),(530,['基座线速度','基座角速度','14 个关节角速度'],[3,3,14])]:
    x = 760
    for width, color, title, count in zip([180,180,370],['#dceaff','#e8e0fb','#ffe6c5'],labels,counts):
        draw.rounded_rectangle((x,y,x+width,y+105),radius=13,fill=color)
        text((x+15,y+13),title,23,True)
        text((x+15,y+52),f'{count} 个数',26)
        x += width + 15
text((760,680),'为什么多了一个数?',29,True)
text((760,729),'姿态用 quaternion(四元数) 的 4 个数表示;',24)
text((760,771),'转动速度用 XYZ 三轴角速度的 3 个数表示。',24)
text((760,820),'四元数有单位长度约束, 不是 4 个独立自由度。',23)
text((42,960),f'当前 STAND 参考状态: 左膝实际角度 {d.qpos[m.jnt_qposadr[knee]]:.5f} rad, 角速度 {d.qvel[m.jnt_dofadr[knee]]:.0f} rad/s。',27,True)
text((42,1008),'名称到数组地址由模型解析; 此处只演示 walk 模型布局, 不对其他模型硬编码下标。',23)
canvas.save(out / '07-qpos-qvel.png')
r.close()
print('已核对状态维度并保存图:', out / '07-qpos-qvel.png')
