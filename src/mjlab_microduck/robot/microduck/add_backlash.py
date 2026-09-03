#!/usr/bin/env python3
"""向 onshape-to-robot 导出的 MJCF 注入齿轮箱 backlash 关节.

对每个驱动的 servo 关节 (``class="chosen_actuator"``) 在同一 body / 同一轴
上紧随其后插入一个非驱动的铰链:

    <joint axis="0 0 1" name="left_hip_yaw" ... class="chosen_actuator"/>
    <joint axis="0 0 1" name="passive_left_hip_yaw_backlash" class="backlash"/>

复合连杆旋转为 main + backlash: main 关节是 servo
输出 (由 BAM 驱动), backlash 关节是 servo 与连杆之间的间隙,
在 ±(backlash/2) 范围内自由游动.

命名: ``passive_`` 前缀使新关节自动被
任务配置中所有现有 regex 排除 (执行器 ``^(?!passive_).*``,
关节观测, pose 奖励). 穿过 backlash 的编码器处理在
mjlab 侧完成 (BacklashEncoderBamActuatorCfg + joint_pos/vel_rel_backlash 观测).

用作 onshape-to-robot 配置的最后一个 post_import_command
(见 config_mjcf_groundcontact_backlash.json), 但也可独立用于任何
已导出的机器人 xml:

    python3 add_backlash.py robot_groundcontact_backlash.xml --backlash-deg 2.0

``--backlash-deg`` 是总的峰峰值间隙 (按住 servo 摇动 horn 时
测得的值); 关节范围对称 ±deg/2.
"""

import argparse
import math
import re
import sys
from pathlib import Path

JOINT_RE = re.compile(r"^(\s*)<joint\b[^>]*/>\s*$")
ATTR_RE = re.compile(r'(\w+)="([^"]*)"')


def build_backlash_default(
    half_range_rad: float,
    damping: float,
    armature: float,
    frictionloss: float,
    total_deg: float,
) -> str:
    """构建 MJCF ``<default class="backlash">`` 块字符串."""
    return (
        f"  <!-- Backlash injected by add_backlash.py: {total_deg:g} deg total play"
        f" (symmetric +/-{total_deg / 2:g} deg) -->\n"
        f"  <default>\n"
        f'    <default class="backlash">\n'
        f"      <!-- 刚性极限约束: 在这么小的 range 下默认\n"
        f"           solref (0.02,1) 会让关节在负载下越限 ~2x.\n"
        f"           0.01 = 2*sim_dt (mjlab velocity 任务运行 dt=0.005),\n"
        f"           最刚且稳定的设置; solimp 提高阻抗使\n"
        f"           齿面接触近于刚性. -->\n"
        f'      <joint damping="{damping:g}" frictionloss="{frictionloss:g}"'
        f' armature="{armature:g}" limited="true"'
        f' range="{-half_range_rad:.17g} {half_range_rad:.17g}"'
        f' solreflimit="0.01 1" solimplimit="0.95 0.999 0.0001 0.5 2"/>\n'
        f"    </default>\n"
        f"  </default>\n"
    )


def main() -> int:
    """将 backlash 关节就地注入 MJCF 文件."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml", help="要就地修改的 MJCF 文件")
    parser.add_argument(
        "--backlash-deg",
        type=float,
        default=2.0,
        help="TOTAL backlash 间隙, 单位度 (峰峰值); 关节范围对称 +/-deg/2 (默认: 2.0)",
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=0.01,
        help="backlash 关节阻尼 (默认: 0.01)",
    )
    parser.add_argument(
        "--armature",
        type=float,
        default=0.001,
        help="backlash 关节 armature, 保持小但非零以利于求解器条件数 (默认: 0.001)",
    )
    parser.add_argument(
        "--frictionloss",
        type=float,
        default=0.0,
        help="backlash 关节 frictionloss (默认: 0)",
    )
    parser.add_argument(
        "--joint-class",
        default="chosen_actuator",
        help="被注入 backlash 的关节的默认 class (默认: chosen_actuator)",
    )
    parser.add_argument(
        "--exclude",
        default=None,
        help="可选的 regex, 匹配的关节名将被跳过 (如 '.*(neck|head).*')",
    )
    args = parser.parse_args()

    half_range = math.radians(args.backlash_deg) / 2.0
    exclude = re.compile(args.exclude) if args.exclude else None

    with Path(args.xml).open() as f:
        lines = f.readlines()

    if any('class="backlash"' in line for line in lines):
        print(f"[add_backlash] {args.xml} 已包含 backlash 关节 — 中止.")
        return 1

    out = []
    added = []
    default_inserted = False
    for line in lines:
        # 在 <worldbody> 之前插入 defaults 块.
        if not default_inserted and "<worldbody>" in line:
            out.append(
                build_backlash_default(
                    half_range,
                    args.damping,
                    args.armature,
                    args.frictionloss,
                    args.backlash_deg,
                )
            )
            default_inserted = True

        out.append(line)

        m = JOINT_RE.match(line)
        if m is None:
            continue
        attrs = dict(ATTR_RE.findall(line))
        if attrs.get("class") != args.joint_class:
            continue
        name = attrs.get("name")
        if not name or (exclude and exclude.match(name)):
            continue
        indent = m.group(1)
        axis = attrs.get("axis", "0 0 1")
        pos = f' pos="{attrs["pos"]}"' if "pos" in attrs else ""
        out.append(
            f'{indent}<joint axis="{axis}"{pos} name="passive_{name}_backlash" type="hinge" class="backlash"/>\n'
        )
        added.append(name)

    if not default_inserted:
        print("[add_backlash] 错误: 未找到 <worldbody> — 这是一个 MJCF 文件吗?")
        return 1
    if not added:
        print(f'[add_backlash] 错误: 未找到 class="{args.joint_class}" 的关节.')
        return 1

    with Path(args.xml).open("w") as f:
        f.writelines(out)

    print(
        f"[add_backlash] 向 {args.xml} 添加了 {len(added)} 个 backlash 关节 "
        f"(+/-{args.backlash_deg / 2:g} deg = +/-{half_range:.5f} rad): "
        f"{', '.join(added)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
