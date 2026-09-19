"""渲染真实 walk 模型的站立姿态, 标注躯干与整机质心."""

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MESA_SHADER_CACHE_DIR", "/tmp/microduck-mesa-cache")

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/learning/assets/u3-4"
OUT.mkdir(parents=True, exist_ok=True)
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
BG = "#f3f6fb"
INK = "#15263e"
RED = "#e55353"
GREEN = "#25a47a"
BLUE = "#4d8cf5"

model = mujoco.MjModel.from_xml_path(
    str(ROOT / "src/mjlab_microduck/robot/microduck/scene_walk.xml")
)
model.vis.global_.offwidth = 1050
model.vis.global_.offheight = 860
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, model.key("STAND").id)
mujoco.mj_forward(model, data)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(BOLD if bold else FONT, size)


def label(canvas: Image.Image, xy: tuple[int, int], text: str, size: int = 24, color: str = INK, bold: bool = False) -> None:
    ImageDraw.Draw(canvas).text(xy, text, font=font(size, bold), fill=color)


def camera() -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, 0.135]
    cam.distance = 0.62
    cam.azimuth = 135
    cam.elevation = -17
    return cam


def project(renderer: mujoco.Renderer, xyz: np.ndarray, offset: tuple[int, int]) -> tuple[float, float]:
    cameras = renderer.scene.camera
    position = (np.array(cameras[0].pos) + np.array(cameras[1].pos)) / 2
    forward = np.array(cameras[0].forward)
    up = np.array(cameras[0].up)
    right = np.cross(forward, up)
    vector = np.asarray(xyz) - position
    depth = np.dot(vector, forward)
    focal = renderer.height / (2 * np.tan(np.deg2rad(model.vis.global_.fovy) / 2))
    return (
        offset[0] + renderer.width / 2 + focal * np.dot(vector, right) / depth,
        offset[1] + renderer.height / 2 - focal * np.dot(vector, up) / depth,
    )


renderer = mujoco.Renderer(model, height=820, width=1050)
option = mujoco.MjvOption()
option.geomgroup[:] = 0
option.geomgroup[[0, 2]] = 1
option.sitegroup[:] = 0
renderer.update_scene(data, camera=camera(), scene_option=option)

canvas = Image.new("RGB", (1600, 1040), BG)
canvas.paste(Image.fromarray(renderer.render()), (30, 145))
label(canvas, (40, 25), "U3.4 | CoM: 质量加权后的整机平衡点", 40, bold=True)
label(canvas, (42, 84), "真实 scene_walk.xml 的 STAND 姿态. 圆点由 MuJoCo 运行数据计算, 不是新增零件.", 22)

trunk_id = model.body("trunk_base").id
trunk_com = data.xipos[trunk_id].copy()
robot_com = data.subtree_com[trunk_id].copy()
ground_projection = robot_com.copy()
ground_projection[2] = 0.002
trunk_xy = project(renderer, trunk_com, (30, 145))
robot_xy = project(renderer, robot_com, (30, 145))
ground_xy = project(renderer, ground_projection, (30, 145))
draw = ImageDraw.Draw(canvas)

# 红线是总 CoM 沿世界竖直方向投到地面的示意, 不表示实体杆件.
draw.line((robot_xy, ground_xy), fill=RED, width=4)
for point, color, radius in ((trunk_xy, GREEN, 13), (robot_xy, RED, 15), (ground_xy, RED, 8)):
    x, y = point
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="white", width=3)

draw.line((robot_xy[0] + 12, robot_xy[1] - 8, 1135, 278), fill=RED, width=3)
draw.line((trunk_xy[0] + 12, trunk_xy[1] + 8, 1135, 482), fill=GREEN, width=3)
draw.line((ground_xy[0] + 8, ground_xy[1], 1135, 650), fill=RED, width=3)

label(canvas, (1155, 220), "整机 CoM", 30, RED, bold=True)
label(canvas, (1155, 264), "所有 body 的质量加权平均位置.", 22)
label(canvas, (1155, 300), "关节运动时, 它也会移动.", 22)

label(canvas, (1155, 424), "trunk_base 的 CoM", 30, GREEN, bold=True)
label(canvas, (1155, 468), "只是躯干这个刚体自己的平衡点.", 22)
label(canvas, (1155, 504), "它不等于整机 CoM.", 22)

label(canvas, (1155, 602), "红线落点", 30, RED, bold=True)
label(canvas, (1155, 646), "整机 CoM 的竖直投影示意.", 22)
label(canvas, (1155, 682), "静态支撑时, 它与脚底支撑区域", 22)
label(canvas, (1155, 716), "的关系决定是否容易倾倒.", 22)

label(canvas, (42, 988), "教学标注只显示 STAND 瞬时几何关系. 未运行 policy、BAM 或动力学 rollout.", 20)
canvas.save(OUT / "01-center-of-mass.png")
