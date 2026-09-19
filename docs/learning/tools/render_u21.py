"""用真实 MJCF 生成 U2.1 标注图及运动学动画, 不运行策略或动力学实验."""

import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('MESA_SHADER_CACHE_DIR', '/tmp/microduck-mesa-cache')
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'docs/learning/assets/u2-1'
OUT.mkdir(parents=True, exist_ok=True)
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
BOLD = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
BG = '#f3f6fb'
INK = '#15263e'
BLUE, ORANGE, TEAL, RED = '#559eff', '#ffac50', '#31d9bd', '#ff6383'
m = mujoco.MjModel.from_xml_path(str(ROOT / 'src/mjlab_microduck/robot/microduck/scene_walk.xml'))
m.vis.global_.offwidth = 1000
m.vis.global_.offheight = 850
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, m.key('STAND').id)
mujoco.mj_forward(m, d)
base = d.qpos.copy()
rgba = m.geom_rgba.copy()
matid = m.geom_matid.copy()
foot = m.geom('left_foot_collision').id
ankle = m.body('ankle_left').id
knee = m.joint('left_knee').id
free = m.joint('trunk_base_freejoint').id
floor = m.geom('floor').id


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def label(canvas, xy, text, size=24, color=INK, bold=False):
    ImageDraw.Draw(canvas).text(xy, text, font=font(size, bold), fill=color)


def rgb(color):
    return np.array([int(color[i:i+2], 16) / 255 for i in (1, 3, 5)] + [1.0])


def restore():
    m.geom_rgba[:] = rgba
    m.geom_matid[:] = matid
    d.qpos[:] = base
    mujoco.mj_forward(m, d)


def tint_body(name, color):
    ids = np.flatnonzero((m.geom_bodyid == m.body(name).id) & (m.geom_group == 2))
    m.geom_rgba[ids] = rgb(color)
    m.geom_matid[ids] = -1


def camera(lookat, distance, azimuth=135, elevation=-15):
    c = mujoco.MjvCamera()
    c.lookat[:] = lookat
    c.distance, c.azimuth, c.elevation = distance, azimuth, elevation
    return c


def render(renderer, cam, groups=(0, 2)):
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[list(groups)] = 1
    opt.sitegroup[:] = 0
    renderer.update_scene(d, camera=cam, scene_option=opt)
    return Image.fromarray(renderer.render())


def project(renderer, xyz, offset=(0, 0)):
    cs = renderer.scene.camera
    pos = (np.array(cs[0].pos) + np.array(cs[1].pos)) / 2
    forward = np.array(cs[0].forward)
    up = np.array(cs[0].up)
    right = np.cross(forward, up)
    v = np.array(xyz) - pos
    depth = np.dot(v, forward)
    focal = renderer.height / (2 * np.tan(np.deg2rad(m.vis.global_.fovy) / 2))
    return (offset[0] + renderer.width / 2 + focal * np.dot(v, right) / depth,
            offset[1] + renderer.height / 2 - focal * np.dot(v, up) / depth)


def marker(canvas, point, color, number):
    draw = ImageDraw.Draw(canvas)
    x, y = point
    draw.ellipse((x-17, y-17, x+17, y+17), fill=color, outline='white', width=3)
    draw.text((x, y-1), str(number), font=font(20, True), fill=INK, anchor='mm')


# 图一: 仅为教学着色, 坐标锚点从模型运行数据读取.
restore()
for name, color in [('trunk_base', BLUE), ('leg', ORANGE), ('ankle_left', TEAL)]:
    tint_body(name, color)
r = mujoco.Renderer(m, height=800, width=950)
cam = camera([0, 0, .155], .64)
canvas = Image.new('RGB', (1600, 1060), BG)
label(canvas, (40, 24), 'U2.1 | 在真实模型上认识 body 与 joint', 40, bold=True)
label(canvas, (42, 84), 'scene_walk.xml / STAND 参考姿态 · 蓝、橙、绿是教学着色', 23)
canvas.paste(render(r, cam), (30, 140))
items = [
    ('body: trunk_base', '蓝色部分属于躯干刚体', BLUE, d.xipos[m.body('trunk_base').id].copy()),
    ('freejoint: 躯干的自由关节', '锚点在躯干原点; 不是额外电机', BLUE, d.xanchor[free].copy()),
    ('joint: left_knee', '红点是左膝关节位置', RED, d.xanchor[knee].copy()),
    ('body: leg', '橙色部分属于左小腿刚体', ORANGE, d.xipos[m.body('leg').id].copy()),
    ('body: ankle_left', '绿色部分属于左脚刚体', TEAL, d.geom_xpos[foot].copy()),
]
for i, (title, desc, color, xyz) in enumerate(items):
    y = 180 + i * 143
    point = project(r, xyz, (30, 140))
    draw = ImageDraw.Draw(canvas)
    draw.line([point, (990, y+35), (1020, y+35)], fill=color, width=3)
    marker(canvas, point, color, i+1)
    draw.rounded_rectangle((1020, y, 1570, y+113), radius=16, fill='white')
    label(canvas, (1040, y+13), f'{i+1}  {title}', 26, bold=True)
    label(canvas, (1040, y+64), desc, 23)
label(canvas, (42, 969), 'body 包含形状描述; joint 描述该 body 相对父 body 的运动方式。', 27, bold=True)
label(canvas, (42, 1011), '圆点表示模型锚点/示意定位, 不代表新增实体。此图没有运行策略或物理实验。', 22)
canvas.save(OUT / '01-body-joint.png')
r.close()

# 图二: 同一个脚部刚体, 同一机位, 对照外观与脚底碰撞形状.
restore()
look = d.geom_xpos[foot].copy() + [0, 0, .012]
cam = camera(look, .245, azimuth=135, elevation=-24)
r = mujoco.Renderer(m, height=650, width=740)
canvas = Image.new('RGB', (1600, 980), BG)
label(canvas, (40, 24), '同一个 body, 两种 geom 用途', 40, bold=True)
label(canvas, (42, 82), '左脚 ankle_left 放大 · 相同机位 · 其余机器人形状仅在画面中隐藏', 23)
for panel in range(2):
    restore()
    m.geom_rgba[:, 3] = 0
    m.geom_rgba[floor] = rgba[floor]
    if panel == 0:
        ids = np.flatnonzero((m.geom_bodyid == ankle) & (m.geom_group == 2))
        m.geom_rgba[ids] = rgba[ids]
        title, subtitle = '外观 geom', '显示脚底、外壳与零件'
    else:
        m.geom_rgba[foot] = rgb(TEAL)
        m.geom_matid[foot] = -1
        title, subtitle = '碰撞 geom: left_foot_collision', '保留脚底接触用的形状'
    x = 40 + panel * 780
    label(canvas, (x+5, 134), title, 29, bold=True)
    canvas.paste(render(r, cam, (0, 2, 3)), (x, 185))
    label(canvas, (x+8, 855), subtitle, 26, bold=True)
label(canvas, (42, 922), '这两个脚底 geom 引用同一份 sole_left 网格; 本例的碰撞形状没有换成盒子。', 24)
canvas.save(OUT / '02-geom-comparison.png')
r.close()

# 视频只改变关节位置并更新几何关系, 不调用 mj_step, 不检验 BAM 或稳定性.
restore()
r = mujoco.Renderer(m, height=540, width=960)
cam = camera([0, 0, .155], .65)
writer = imageio.get_writer(str(OUT / '03-model-tour.mp4'), fps=24, codec='libx264', quality=8, macro_block_size=16)
preview = []
for frame in range(288):
    restore()
    chapter = frame // 96
    t = (frame % 96) / 95
    groups = (0, 2)
    if chapter == 0:
        tint_body('leg', ORANGE)
        tint_body('ankle_left', TEAL)
        d.qpos[m.jnt_qposadr[knee]] += .55 * np.sin(2*np.pi*t)
        title = '01 / joint: 左膝转动, 小腿与脚跟着改变位置'
        detail = '橙色 = 左小腿 body    绿色 = 左脚 body    红点 = left_knee'
        cam = camera([0, 0, .155], .65)
    elif chapter == 1:
        cam = camera(look, .245, azimuth=135, elevation=-24)
        m.geom_rgba[:, 3] = 0
        m.geom_rgba[floor] = rgba[floor]
        if t < .5:
            ids = np.flatnonzero((m.geom_bodyid == ankle) & (m.geom_group == 2))
            m.geom_rgba[ids] = rgba[ids]
            title = '02 / geom: 左脚的外观形状'
        else:
            m.geom_rgba[foot] = rgb(TEAL)
            m.geom_matid[foot] = -1
            title = '02 / geom: 同一只脚的脚底碰撞形状'
        groups = (0, 2, 3)
        detail = '仅切换显示; 没有删除 body、关节或碰撞设置'
    else:
        tint_body('trunk_base', BLUE)
        d.qpos[m.jnt_qposadr[free]] += .045 * np.sin(2*np.pi*t)
        title = '03 / freejoint: 躯干位置变化, 整棵子树一起移动'
        detail = '关节角保持不变; 手动改变自由基座位置, 不是机器人自己走路'
        cam = camera([0, 0, .155], .65)
    mujoco.mj_forward(m, d)
    canvas = Image.new('RGB', (960, 720), BG)
    label(canvas, (22, 15), title, 28, bold=True)
    label(canvas, (22, 61), '真实 MJCF · 运动学示意 · 未运行策略或动力学', 21)
    canvas.paste(render(r, cam, groups), (0, 100))
    if chapter == 0:
        marker(canvas, project(r, d.xanchor[knee], (0, 100)), RED, '膝')
    label(canvas, (22, 665), detail, 21)
    writer.append_data(np.array(canvas))
    if frame % 3 == 0:
        preview.append(canvas.resize((640, 480)))
writer.close()
r.close()
preview[0].save(OUT / '03-model-tour.gif', save_all=True, append_images=preview[1:], duration=125, loop=0)
print('已生成两张标注图、12 秒视频与动图预览:', OUT)
