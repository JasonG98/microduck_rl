"""把 MuJoCo 中的一个 microduck 机体经 TCP 提供给真实的 `robotd`.

    uv run duck-body                      # 单只鸭子, 一个查看器
    uv run duck-body --ducks 4            # 四只, 共享同一个世界和同一个窗口
    uv run duck-body --port 7801
    uv run duck-body --headless           # 无窗口, 用于测试和大量鸭子的场景

然后, 在 daemon 一侧:

    robotd --sim 127.0.0.1:7801

每只鸭子一个 TCP 端口, 从 `--port` 起递增, 并且所有鸭子共用一次 `mj_step` -- 因此鸭子们共享
同一块地板, 可以互相碰撞, 这就是"一个房间里有四台机器人"与"四台机器人出现在同一个屏幕上"的
区别.

`duck_control::io::RobotIo` 之上的所有代码都是跑在真实机器人上的 -- 50 Hz 的循环, ONNX 策略,
安全, 摔倒检测, 里程计, 运动学, 每一次 IPC 调用. 本进程是唯一知道这里并没有机器人的部分.

**为什么是这个仓库.** 它已经拥有关键场景, 拟合到真实 XL330 上的 BAM 执行器模型以及 mjlab; 向
daemon 提供机体与它今天所做的 sim2real 是镜像关系. daemon 一侧的实现留在 `microduck` 中, 因为
它实现的是针对仓库内协议的一个仓库内 trait.

## 协议

基于 TCP 的换行分隔 JSON, 每行一个请求和应答, 握手时校验 `protocol` -- 两部分分散在两个仓库中,
因此"你的模拟器是旧的"和"你的 daemon 是旧的"绝不能表现为同一种症状. 另一端是
`duck_control::sim`, 它承载着为什么用 TCP 而非 unix socket、用 JSON 而非紧凑结构体的理由.

## 这一侧主动拥有的两个映射

**这里十五个关节, 模型里十四个.** daemon 按 `JOINT_NAMES` 索引关节, 其中下标 9 是 `mouth`;
没有任何 alpha 策略驱动它, 行走模型里也没有它. daemon 不必需要知道这一点, 因此这里插入并丢弃它.
关于模型自身形状的知识放在哪里, 正是协议携带机器人单位而非 MuJoCo 单位的全部原因.

**是重力, 不只是朝向.** 策略在躯干坐标系中观测投影重力. MuJoCo 给出的是朝向四元数, 因此这里做
旋转 -- 与 IMU 的 SFLP 滤波器在机器人上做的运算相同, 只是在同一根线的另一端.
"""

from __future__ import annotations

import argparse
import json
import socket
import socketserver
import threading
import time
from pathlib import Path

import mujoco
import numpy as np

from mjlab_microduck.sim.tof import COLS, ROWS, Tof

PROTOCOL = 1

# 策略训练所用的步长, 也是 `scripts/infer_policy.py` 设置的. 场景自带的为 0.002; 加上该脚本
# 4 倍的下采样, 恰好是 daemon 控制循环运行的 50 Hz. 这不是一个性能旋钮: BAM 执行器拟合、
# 接触 solref 和关节转动惯量都按这个步长调校, 所以 0.002 会让鸭子腿能转到正确的角度却依然
# 撑不起自己.
TIMESTEP = 0.005

# `scripts/infer_policy.py` 在开始前安置鸭子的位置: 躯干这么高, 直立, 每个关节都在 home 位姿.
# 不是关键帧 -- 关键帧是一堆位姿, 而这是一个*安放*位置.
HOME_TRUNK_Z = 0.125

# `duck_ipc_proto::JOINT_NAMES`, 它是协议的一部分: 线上每个位置数组都按它来索引. 在这里复制
# 而不是共享, 因为两个仓库无法共享常量 -- 启动时再与模型比对, 是仅次于共享的次优方案.
JOINT_NAMES = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "mouth",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)
MOUTH_INDEX = JOINT_NAMES.index("mouth")

# `duck_control::DEFAULT_POSITION`, 以及 `infer_policy.py` 中把 mouth 放回去后的 `DEFAULT_POSE`.
# 右腿是镜像而非对称 -- 值得阅读而不是想当然.
HOME_POSE = (
    0.0,
    -0.0873,
    -0.4579,
    -0.0049,
    0.4530,
    0.3491,
    0.3491,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0873,
    0.4579,
    0.0049,
    -0.4530,
)

SCENES = Path(__file__).resolve().parents[1] / "robot" / "microduck"
# 用 `scene.xml` 而不是 `scene_walk.xml`: 行走场景包含 RL 工作所训练的模型, 其执行器默认类携带
# `contype="0" conaffinity="0"`, 因此机器人不会与任何东西碰撞, 会穿过场景中确实存在的地板下沉.
DEFAULT_SCENE = SCENES / "scene.xml"
# 没有地板和场景布置的机器人 -- 额外的鸭子从它上面附加出来.
ROBOT_ONLY = SCENES / "robot_allcollisions.xml"

# 距离足够远, 静止时互不相碰, 又足够近, 能放在一个屏幕内.
SPACING = 0.5

# 机器人会上报、而这里并不模拟的东西. 用常量而非省略, 这样 `robotctl health` 显示的是一个
# 合理的机器人而不是一个令人担忧的机器人.
NOMINAL_VOLTS = 7.4
NOMINAL_TEMP_C = 32.0


def duck_prefix(index: int) -> str:
    """第一只鸭子 (场景自带的) 前缀为 `""`, 附加的鸭子为 `d1_`, `d2_` ... ."""
    return "" if index == 0 else f"d{index}_"


def build_world(scene: Path, count: int) -> mujoco.MjModel:
    """一个模型里放 `count` 只鸭子, 这样它们共享一块地板并能互相碰撞.

    场景里已经包含一只鸭子; 其余的在名字前缀下附加到它上面, 这正是 `MjSpec` 的用途. 共享一个
    世界而不是运行 N 个模拟器正是关键: 两只鸭子如果在各自独立的物理中, 可以屏幕上紧挨着却永远
    不接触, 而"彼此挨着"正是所有社交行为的核心.
    """
    if count == 1:
        return mujoco.MjModel.from_xml_path(str(scene))

    spec = mujoco.MjSpec.from_file(str(scene))
    for index in range(1, count):
        # 每次都是全新的子节点: `attach` 会重命名传入的 spec, 因此复用同一个会导致第三只鸭子
        # 得到 `d2_d1_left_hip_yaw` 这样的名字, 并因为不兼容的 id 而编译报错.
        robot = mujoco.MjSpec.from_file(str(ROBOT_ONLY))
        frame = spec.worldbody.add_frame(pos=[0.0, index * SPACING, 0.0])
        spec.attach(robot, prefix=duck_prefix(index), frame=frame)
    return spec.compile()


def pose_table(scene: Path, keyframe: str) -> tuple[dict[str, float] | None, float]:
    """场景关键帧中的一个具名位姿, 表示为 关节名 -> 角度.

    从单只鸭子的模型读取并按名字应用, 因为附加机器人并不会把场景的关键帧带过来 -- 位姿是关于
    机器人的事实, 与房间里有几只无关.
    """
    if keyframe.upper() == "HOME":
        return None, HOME_TRUNK_Z
    model = mujoco.MjModel.from_xml_path(str(scene))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i) for i in range(model.nkey)]
    if keyframe not in names:
        raise SystemExit(f"{scene.name} 中没有关键帧 {keyframe!r}. 它有: {', '.join(n for n in names if n)}")
    qpos = model.key_qpos[names.index(keyframe)]
    table = {}
    for joint in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if name in JOINT_NAMES:
            table[name] = float(qpos[model.jnt_qposadr[joint]])
    # 关键帧自带的躯干高度, 而非 home 的高度: 一只坐着的鸭子若被安放在站立高度, 就会悬在空中,
    # 一旦有人启用它就会坠落.
    return table, float(qpos[2])


def gravity_in_trunk(quat: np.ndarray) -> np.ndarray:
    """躯干坐标系中的世界重力. 直立时为 `[0, 0, -1]`.

    这是策略实际观测的量, 放在这里而不放在 daemon 里, 因为 daemon 的 IMU 从传感器自身的滤波器
    交付的正是这个已经旋转过的量.
    """
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, quat)
    return -rotation.reshape(3, 3).T[:, 2]


class World:
    """被其中每只鸭子共享的物理.

    一次 `mj_step` 推进所有的鸭子, 这正是让一次推挤真实而非装饰的原因. 锁只用于把数字读进或
    写出, 从不跨越一次 step: 一个请求传感器的 daemon 绝不能等待求解器, 正如真实的总线读不会
    等待某个伺服.
    """

    def __init__(self, scene: Path, count: int = 1):
        self.model = build_world(scene, count)
        self.model.opt.timestep = TIMESTEP
        self.data = mujoco.MjData(self.model)
        self.lock = threading.Lock()
        self.bodies: list[Body] = []

    def step(self, times: int = 1) -> None:
        """推进世界, 对整批只取一次锁.

        **因为瓶颈是锁而不是求解器, 所以要分批.** 四个 daemon 在 50 Hz 下每秒发出 400 个请求,
        每个都想要这把锁, 而 Python 在它们之间来回移交 GIL -- 因此一个每次都必须取放锁 200 次的
        物理循环会失败. 5 ms 的四步就是 20 ms 的世界时间, 恰好是一个控制 tick, 所以没有任何
        传感器会比它所属的 tick 更陈旧.
        """
        with self.lock:
            for _ in range(times):
                mujoco.mj_step(self.model, self.data)
                # 还没有被任何人启用的鸭子, 会被放回原处. 物理是共享的, 所以不能简单地把
                # 它排除在 step 之外 -- 一只手稳住一只机器人, 而另一只在其间走动, 是房间里
                # 的寻常情形.
                for body in self.bodies:
                    if not body.released:
                        body.restore()


class Body:
    """一只鸭子对世界的视角: 它的关节, 它的执行器, 它的躯干."""

    def __init__(self, world: World, index: int, limp: bool = False, kp: float = 200.0):
        self.world = world
        self.index = index
        self.prefix = duck_prefix(index)
        model = world.model

        def ident(kind, name):
            found = mujoco.mj_name2id(model, kind, self.prefix + name)
            if found < 0:
                raise SystemExit(f"模型中没有鸭子 {index} 的关节 {name!r}")
            return found

        # 按名字而非下标: 一次 MJCF 编辑会静默地重新排列.
        self.actuators = []
        self.to_wire = []
        for wire_index, name in enumerate(JOINT_NAMES):
            found = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.prefix + name)
            if found >= 0:
                self.actuators.append(found)
                self.to_wire.append(wire_index)
        if not self.actuators:
            raise SystemExit(f"鸭子 {index} (前缀 {self.prefix!r}) 没有任何可驱动的关节")

        self.qpos_adr = np.array([model.jnt_qposadr[model.actuator_trnid[a, 0]] for a in self.actuators])
        self.qvel_adr = np.array([model.jnt_dofadr[model.actuator_trnid[a, 0]] for a in self.actuators])
        # 深度传感器, 位于模型自己的 `tof` site 上 -- 所以转动的头部会带着它一起转, 这正是
        # `robot.look` 扫描房间的方式.
        self.tof = Tof(model, ident(mujoco.mjtObj.mjOBJ_SITE, "tof"), seed=index)
        self.trunk = int(model.jnt_qposadr[ident(mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")])
        self.trunk_dof = int(model.jnt_dofadr[ident(mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")])

        self.actuator_slice = np.array(self.actuators)
        self._gain = model.actuator_gainprm[self.actuator_slice, 0].copy()

        # **在 daemon 接走之前一直保持不放.** 处于静态位姿的双足并不稳定: 仅用位置控制保持 home
        # 位姿, 会让这只鸭子在不到一秒内倒在地上, 无论什么步长、从什么安放位置开始 --
        # `infer_policy.py` 从不这么做, 因为它从第零步起就有策略在平衡. `robotd` 故意在启动时
        # 不使能力矩, 所以那些秒必须被花费在不摔倒上.
        self.released = limp
        self.torque_on = not limp
        self.kp = kp
        self.held = None

    # ── 安放 ───────────────────────────────────────────────────────

    def place(self, pose: dict[str, float] | None, trunk_z: float, offset_y: float) -> None:
        data = self.world.data
        data.qpos[self.trunk + 0] = 0.0
        data.qpos[self.trunk + 1] = offset_y
        data.qpos[self.trunk + 2] = trunk_z
        data.qpos[self.trunk + 3 : self.trunk + 7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[self.trunk_dof : self.trunk_dof + 6] = 0.0
        for slot, wire_index in enumerate(self.to_wire):
            name = JOINT_NAMES[wire_index]
            value = HOME_POSE[wire_index] if pose is None else pose.get(name, HOME_POSE[wire_index])
            data.qpos[self.qpos_adr[slot]] = value
            data.qvel[self.qvel_adr[slot]] = 0.0
        data.ctrl[self.actuator_slice] = data.qpos[self.qpos_adr]
        self._apply_torque()
        self.remember()

    def remember(self) -> None:
        data = self.world.data
        self.held = (
            data.qpos[self.trunk : self.trunk + 7].copy(),
            data.qpos[self.qpos_adr].copy(),
        )

    def restore(self) -> None:
        """把这只鸭子放回原处, 用于那只尚未被启用的鸭子."""
        if self.held is None:
            return
        trunk, joints = self.held
        data = self.world.data
        data.qpos[self.trunk : self.trunk + 7] = trunk
        data.qpos[self.qpos_adr] = joints
        data.qvel[self.trunk_dof : self.trunk_dof + 6] = 0.0
        data.qvel[self.qvel_adr] = 0.0

    # ── daemon 看到的内容 ───────────────────────────────────────

    def sensors(self) -> dict:
        data = self.world.data
        with self.world.lock:
            sim_time = float(data.time)
            positions = data.qpos[self.qpos_adr].copy()
            velocities = data.qvel[self.qvel_adr].copy()
            force = data.actuator_force[self.actuator_slice].copy()
            quat = data.qpos[self.trunk + 3 : self.trunk + 7].copy()
            gyro = data.qvel[self.trunk_dof + 3 : self.trunk_dof + 6].copy()
            trunk = [float(v) for v in data.qpos[self.trunk : self.trunk + 3]]
            trunk_z = trunk[2]

        wire_pos = [0.0] * len(JOINT_NAMES)
        wire_vel = [0.0] * len(JOINT_NAMES)
        wire_cur = [0.0] * len(JOINT_NAMES)
        for slot, wire_index in enumerate(self.to_wire):
            wire_pos[wire_index] = float(positions[slot])
            wire_vel[wire_index] = float(velocities[slot])
            # 未经真实伺服校准: 一个形状正确的替身, 让盯着负载的消费者能看到负载. 由模拟力矩换算出的
            # 安培是一个带单位的虚构值.
            wire_cur[wire_index] = abs(float(force[slot])) * 100.0

        return {
            "positions": wire_pos,
            "velocities": wire_vel,
            "currents_ma": wire_cur,
            # **不是协议的一部分, 而是故意多给的.** 没有机器人能量出自己躯干有多高, serde 在 daemon 侧
            # 会忽略这些字段. 它们存在, 是因为一个询问"它站起来了吗?"的工具别无他法获知 -- 一只
            # 用屁股坐在地上、躯干竖直的鸭子也同样是重力 [0, 0, -1] -- 而且因为一个秒并不是秒的
            # 模拟器会静默地毁掉某个策略.
            "trunk_z": trunk_z,
            # 这只鸭子在房间里的位置. 机器人也不知道这个 -- 它在这里, 是为了让模拟的
            # 无线电能决定谁离得近到足以听见谁, 而这是一次真实 BLE 广播免费就能得到、伪造的
            # 却必须被告知的东西.
            "trunk": trunk,
            "sim_time": sim_time,
            "imu": {
                "gyro": [float(v) for v in gyro],
                "gravity": [float(v) for v in gravity_in_trunk(quat)],
                "quat": [float(v) for v in quat],
            },
        }

    def slow_sensors(self) -> dict:
        return {"volts": NOMINAL_VOLTS, "temps_c": [NOMINAL_TEMP_C] * len(JOINT_NAMES)}

    def depth(self) -> dict:
        """一帧 8x8 的深度帧, 单位与 `tofd` 发布的一致.

        六十四次光线投射, 所以这是最昂贵的一项 -- 按传感器自身的 15 Hz 而非控制循环的 50 Hz 请求,
        正如硬件那样.
        """
        with self.world.lock:
            distance_mm, status = self.tof.frame(self.world.data)
        return {"rows": ROWS, "cols": COLS, "distance_mm": distance_mm, "status": status}

    # ── daemon 下发的命令 ──────────────────────────────────────

    def set_targets(self, wire_targets: list[float]) -> None:
        if len(wire_targets) != len(JOINT_NAMES):
            raise ValueError(f"预期 {len(JOINT_NAMES)} 个目标, 实际收到 {len(wire_targets)}")
        with self.world.lock:
            for slot, wire_index in enumerate(self.to_wire):
                self.world.data.ctrl[self.actuators[slot]] = wire_targets[wire_index]

    def set_gain(self, kp: int) -> None:
        with self.world.lock:
            self.kp = float(kp)
            self._apply_torque()

    def set_torque(self, on: bool) -> None:
        with self.world.lock:
            self.torque_on = bool(on)
            if on:
                self.released = True
                # 施加力矩时绝不能把机器人甩向一个过期的目标.
                self.world.data.ctrl[self.actuator_slice] = self.world.data.qpos[self.qpos_adr]
            self._apply_torque()

    def _apply_torque(self) -> None:
        """力矩关闭意味着无力 (limp), 而非冻结.

        拒绝指挥一只摔倒的机器人只会把它冻在摔倒时的位姿 -- 这正是 `RobotIo::set_gain`
        存在的原因. 零增益是切断电源的模拟等价. daemon 的 kp 是一个 Dynamixel 寄存器值, 其
        200 正是 BAM 用来拟合模型自身增益的数值, 因此它是一个相对 200 的比值, 而不是同一单位
        下的一个数.
        """
        scale = (self.kp / 200.0) if self.torque_on else 0.0
        model = self.world.model
        model.actuator_gainprm[self.actuator_slice, 0] = self._gain * scale
        model.actuator_biasprm[self.actuator_slice, 1] = -self._gain * scale


class Handler(socketserver.StreamRequestHandler):
    """一只鸭子的 daemon. 一次一个连接, 这也是真实的关系."""

    def handle(self) -> None:
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        body: Body = self.server.body
        print(f"== 鸭子 {body.index}: daemon 已从 {self.client_address} 连接", flush=True)
        for raw in self.rfile:
            try:
                answer = self.dispatch(body, json.loads(raw))
            except Exception as error:  # 一帧坏数据绝不能把模拟器一起拖垮
                answer = {"error": str(error)}
            self.wfile.write((json.dumps(answer) + "\n").encode())
            self.wfile.flush()
        print(f"== 鸭子 {body.index}: daemon 已断开连接", flush=True)

    def dispatch(self, body: Body, request: dict) -> dict:
        op = request.get("op")
        if op == "hello":
            asked = request.get("protocol")
            if asked != PROTOCOL:
                raise ValueError(f"daemon 讲协议 {asked}, 而这个模拟器讲 {PROTOCOL}")
            return {"protocol": PROTOCOL}
        if op == "read":
            return body.sensors()
        if op == "write":
            body.set_targets(request["targets"])
            return {}
        if op == "gain":
            body.set_gain(int(request["kp"]))
            return {}
        if op == "torque":
            body.set_torque(bool(request["on"]))
            return {}
        if op == "slow":
            return body.slow_sensors()
        if op == "tof":
            return body.depth()
        raise ValueError(f"未知的 op {op!r}")


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def run(world: World, headless: bool) -> None:
    """按真实时间推进.

    **真实时间, 而不是尽可能快.** daemon 的循环是墙钟时间, 其健康门槛低于 45 Hz 的 50 Hz 就判为
    失败, 所以一个按自己节奏运行的模拟器不仅仅是看起来不对 -- 它会让每只鸭子上报不健康, 并让
    更新器开始回滚版本.
    """
    viewer = None
    if not headless:
        try:
            import mujoco.viewer

            # 没有侧边面板: 这个窗口用于观察鸭子, 而面板本来要驱动的所有东西都属于 daemon.
            viewer = mujoco.viewer.launch_passive(world.model, world.data, show_left_ui=False, show_right_ui=False)
        except Exception as error:
            print(f"== 无查看器 ({error}); 以降无头模式运行", flush=True)

    dt = world.model.opt.timestep
    # 每次传值一个控制 tick 的世界: 20 ms, 与 `infer_policy.py` 使用的下采样一致.
    batch = max(1, round(0.020 / dt))
    period = batch * dt
    # 每隔 N 次传值一帧, 用计数 -- 而不是 `data.time % 0.033`, 那是对累积值做浮点运算,
    # 会在它想触发的时候触发. 用 30 而非 60: `viewer.sync()` 在这个线程上拷贝场景, 并且当
    # 场景里有好几只鸭子时, 这正是能否保持真实时间的区别.
    passes_per_frame = max(1, round((1.0 / 30.0) / period))
    step = 0
    next_step = time.perf_counter()
    behind = 0
    try:
        while True:
            world.step(batch)
            if viewer is not None and not viewer.is_running():
                break
            next_step += period
            slack = next_step - time.perf_counter()
            # 只在值得睡的时候才睡: 对几毫秒调用 `time.sleep` 会睡得比它等待的还多.
            if slack > 0.002:
                time.sleep(slack)
            elif slack < -0.25:
                behind += 1
                print(
                    f"== 落后于真实时间 {-slack:.2f}s (x{behind}) -- 减少鸭子, 或使用 --headless",
                    flush=True,
                )
                next_step = time.perf_counter()
            step += 1
            if viewer is not None and step % passes_per_frame == 0:
                viewer.sync()
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            viewer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--ducks", type=int, default=1, help="多少只, 共享同一个世界")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7801, help="第一只鸭子的端口; 每只 +1")
    parser.add_argument("--headless", action="store_true", help="不显示查看器窗口")
    parser.add_argument(
        "--limp",
        action="store_true",
        help="以无力矩启动, 这样鸭子会就地瘫倒 -- 一台被发现倒在地上的机器人, 正是 `robotd` 的坐姿启动路径所处理的",
    )
    parser.add_argument(
        "--keyframe",
        default="SIT",
        help="从哪里开始. SIT 是一只折叠在地板上的鸭子, 这在它等待时是稳定的, 也是站立策略"
        "自行起立的起点. HOME 是 infer_policy.py 的安放方式 -- home 位姿, 躯干 0.125 m, "
        "直立 -- 而 STAND 和 FOLD 是场景中的其它位姿",
    )
    args = parser.parse_args()

    if not args.scene.exists():
        raise SystemExit(
            f"{args.scene} 处没有场景. 可用的:\n  " + "\n  ".join(sorted(p.name for p in SCENES.glob("scene*.xml")))
        )
    if args.ducks < 1:
        raise SystemExit("--ducks 至少需要一只鸭子")

    world = World(args.scene, args.ducks)
    pose, trunk_z = pose_table(args.scene, args.keyframe)
    servers = []
    for index in range(args.ducks):
        body = Body(world, index, limp=args.limp)
        body.place(pose, trunk_z, offset_y=index * SPACING)
        world.bodies.append(body)
        server = Server((args.host, args.port + index), Handler)
        server.body = body
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)

    mujoco.mj_forward(world.model, world.data)
    print(f"== {args.scene.name}: {args.ducks} 只鸭子, 从 {args.keyframe} 开始", flush=True)
    for index in range(args.ducks):
        print(f"==   鸭子 {index}: robotd --sim {args.host}:{args.port + index}", flush=True)

    run(world, headless=args.headless)


if __name__ == "__main__":
    main()
