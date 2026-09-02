# Microduck RL

<img width="2215" height="884" alt="image" src="https://github.com/user-attachments/assets/5db7cc83-b3ce-4f7c-83f0-0572a63baed7" />


为 [Microduck](https://github.com/pollen-robotics/microduck) —
一台约 800 g, 约 25 cm 高的双足机器人 — 构建的 RL 训练环境, 基于
[mjlab](https://github.com/mujocolab/mjlab) (MuJoCo Warp) 与 PPO.
策略在此以 50 Hz 训练, 导出为 ONNX, 并由
[pollen-robotics/microduck](https://github.com/pollen-robotics/microduck) 中的运行时部署到真实机器人上.

<!-- HERO VIDEO — real robot montage: walking, standup, roulade, roller skating.
     Keep it short (~30 s) and real-robot-first: this is the "why should I care" shot. -->

https://github.com/user-attachments/assets/50c3d537-8db2-4005-9d9c-3472faeec4d0

本仓库编码了完整的 sim2real 方案: [BAM](https://github.com/Rhoban/bam)
执行器物理, 域随机化, 齿隙仿真, 以及让它真正可用的奖励设计经验
(精炼版见 [AGENTS.md](AGENTS.md)).

## 快速开始

需要 CUDA GPU (训练通过 MuJoCo Warp 运行) 和 [uv](https://docs.astral.sh/uv/).

> **在 ARM 设备上 (DGX Spark / GB10, Jetson):** `uv sync` 首次会拉取约 2 GB 的
> CUDA wheel, uv 默认 30 s 的 HTTP 超时可能在中途中断下载.
> 首次同步请设置 `UV_HTTP_TIMEOUT=600`.

```bash
git clone https://github.com/pollen-robotics/microduck_rl
cd microduck_rl

# 训练行走策略 (使用 GPU; 4096 envs 下约 1-2 小时得到可用的步态)
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096

# 在 viewer 中观看训练好的策略
uv run play Mjlab-Velocity-Flat-MicroDuck --wandb-run-path <entity/project/run_id>

# 导出为 ONNX 用于部署
uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck --wandb-run-path <...>

# 在 CPU MuJoCo 中用键盘驱动导出的策略
uv run scripts/infer_policy.py --walking output.onnx
```

从 checkpoint 恢复训练:

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 \
    --agent.run-name resume --agent.load-checkpoint model_29999.pt --agent.resume True
```

没有 GPU? 给任何 train 命令加上 `--hf-jobs` 即可在 Hugging Face Jobs 上运行,
而不必在本地跑 (见 [scripts/hf/README.md](scripts/hf/README.md)).

## 任务

`uv run list-envs` 打印实时注册表. 标注处存在 Flat/Rough 变体.

<!-- SHOWCASE GRID — one short GIF per task family (sim or real), 3 per row.
     Priority order if you only record a few: Velocity, VelStand (fall+recover),
     Roulade, SitStand, Rollers/Swizzle, BallKick. -->

| Task id | 地形 | 描述 |
|---|---|---|
| `Mjlab-Velocity-{Flat,Rough}-MicroDuck` | 平地/崎岖 | **主任务**: 速度命令 + 头部姿态命令行走 |
| `Mjlab-VelStand-{Flat,Rough}-MicroDuck` | 平地/崎岖 | 行走 + 摔倒恢复集成在一条策略中 |
| `Mjlab-StandUp-{Flat,Rough}-MicroDuck` | 平地/崎岖 | 从俯卧/仰卧/坐姿站起, 然后保持站立 + 身体姿态控制 |
| `Mjlab-SitStand-{Flat,Rough}-MicroDuck` | 平地/崎岖 | 单策略命令式坐 ↔ 站, 动作轻柔, 头部可命令 |
| `Mjlab-GroundPick-{Flat,Rough}-MicroDuck` | 平地/崎岖 | 蹲下用嘴尖触地, 再回到站立 |
| `Mjlab-BallKick-Flat-MicroDuck` | 平地 | 向前踢一个 70 mm / 15 g 的球 (actor 看不到球) |
| `Mjlab-Roulade-Flat-MicroDuck` | 平地 | 头顶前滚翻, 落回双脚 |
| `Mjlab-Velocity-Flat-MicroDuck-Rollers` | 平地 | 轮滑速度跟踪 (脚底带被动轮) |
| `Mjlab-Velocity-Swizzle-MicroDuck` | 平地 | 经典对称 swizzle 滑行 |
| `Mjlab-RollerCrouch-Flat-MicroDuck` | 平地 | 轮滑过程中蹲下 |
| `Mjlab-RollerSlope-Flat-MicroDuck` | 斜坡 | 轮滑下坡 |
| `Mjlab-RollerStandUp-Flat-MicroDuck` | 平地 | 从地面站起并站到轮子上 |
| `Mjlab-Spin-Flat-MicroDuck` | 平地 | 在轮子上快速原地旋转 |

部署时运行时在一个共享的 61 维观测契约下热插拔这些策略 (行走 / 恢复 / 技巧),
因此其中任何一条随时可以接管机器人. `scripts/infer_policy.py` 正是对这一流程的排演:

```bash
uv run scripts/infer_policy.py --walking walk.onnx --standing stand.onnx \
    --sitstand sitstand.onnx --roulade roulade.onnx --new-cmd-obs
```

键盘驱动 (速度命令, `G` 拾取, `Y` 坐/站, `R` 前滚翻,
`K`/`L` 踢球); `--debug`, `--save-csv`, `--record` 支持 sim2real 对比.

### 齿隙变体

每个主任务都有一个 **Backlash** 孪生体, 在 14 个舵机关节上各串联 ±1° (共 2°)
的齿轮间隙进行训练: 在 task id 中 `MicroDuck` 之前插入 `-Backlash`,
例如 `Mjlab-Velocity-Flat-Backlash-MicroDuck`.

齿隙按 sim2real 要求正确建模: 每个舵机配一个未驱动的
`passive_<joint>_backlash` 铰链, 由于真实编码器位于间隙输出侧,
固件 PD 仿真 (`BacklashEncoderBamActuator`) 和 `joint_pos`/`joint_vel` 观测都
*穿过* 齿隙读取 (`qpos[servo] + qpos[backlash]`). 观测和动作维度不变,
因此 ONNX 导出和运行时无需改动. 见 `src/mjlab_microduck/tasks/backlash.py`.

## 执行器模型

所有任务使用 [BAM](https://github.com/Rhoban/bam) M6 执行器模型建模
Dynamixel XL330 (电压控制律, 反电动势, 库仑/Stribeck/负载相关摩擦),
并对电池电压, 负载下电压跌落, 命令延迟和摩擦幅值做 per-env 域随机化
(`FrictionDRBamActuator`, 见 `src/mjlab_microduck/actuator/`).

在这个尺度上 — 微型舵机驱动约 800 g 双足 — 执行器保真度是 sim2real 差距的主要来源,
这就是为什么执行器被建模到电压控制律层面, 而非理想 PD.

## 机器人模型

MJCF 模型位于 `src/mjlab_microduck/robot/microduck/`, 由
[onshape-to-robot](https://github.com/Rhoban/onshape-to-robot) 从 Onshape 导出,
每个模型对应一个 `config_mjcf_*.json`:

| XML | 使用者 |
|---|---|
| `robot_walk.xml` | Velocity (去掉躯干/头部碰撞 — 摔倒代价低) |
| `robot_allcollisions.xml` | VelStand, StandUp, SitStand, GroundPick, BallKick, Roulade (身体可以物理上躺在地面) |
| `robot_allcollisions_rollers.xml` | Roller 任务 (被动轮) |
| `robot_*_backlash.xml` | Backlash 任务变体 (由 `add_backlash.py` 生成) |

`scene*.xml` 文件用地板 + 关键帧 (STAND/SIT/FOLD) 包装机器人,
便于快速查看和用于 `infer_policy.py`.

<!-- IMAGE — side-by-side render: walk model vs rollers model (or a collision-geom
     visualization). One image here makes the model-variant story instant. -->

## 项目结构

```
src/mjlab_microduck/
├── robot/
│   ├── microduck/                    # MJCF 导出, 导出配置, 场景, add_backlash.py
│   └── microduck_constants.py        # 机器人配置, HOME 帧, BAM 执行器配置
├── actuator/friction_dr_bam.py       # BAM + 摩擦 DR + 齿隙编码器反馈
├── tasks/
│   ├── __init__.py                   # 任务注册 (基础 + 齿隙变体)
│   ├── mdp.py                        # 奖励, 事件, 观测, 自定义类
│   ├── backlash.py                   # make_backlash_variant() env-cfg 包装器
│   └── microduck_*_env_cfg.py        # 每个任务族一个 cfg 模块
├── train_cli.py                      # `train` 脚本 (与 mjlab 的相同)
├── train_hook.py                     # 拦截 `train ... --hf-jobs`
└── hf_jobs.py                        # Hugging Face Jobs 提交
```

值得了解的约定:

- 观测布局在所有策略间共享 (61 维 actor obs:
  48 本体感知 + 命令 `[twist(3), head_pose(4), body_pose(6)]`), 这正是
  运行时策略热插拔得以实现的基础. 不使用某个命令槽的 env 对其零填充, 而不是删掉.
- 未驱动关节全部命名为 `passive_*` (轮子, 齿隙铰链); 执行器, 关节观测和姿态奖励
  用 `^(?!passive_).*` 选取舵机关节.
- 域随机化开关是每个 env cfg 文件顶部的 `ENABLE_*` 布尔值.
- 关节布局 (14 个舵机): 0–4 左腿 (hip_yaw, hip_roll, hip_pitch, knee,
  ankle), 5–8 颈/头 (neck_pitch, head_pitch, head_yaw, head_roll),
  9–13 右腿.
- 导出器把观测归一化器烤进 ONNX 图 — 务必部署由
  `scripts/export.py` 产出的 ONNX, 绝不要手工转换的
  checkpoint, 否则策略在运行时看到的是未归一化的观测.

[AGENTS.md](AGENTS.md) 记录了 env 构建工作流和整个项目过程中积累的奖励设计
规则 (同样面向在此仓库工作的 AI 编程 agent).

## 测试

```bash
uv run --with pytest pytest tests/
```

仅 CPU 的配置不变量和奖励函数回归测试 — 它们锁定关节索引映射, 奖励符号约定和 NaN 防护.

## 相关项目

- [microduck](https://github.com/pollen-robotics/microduck) — Microduck 项目主页, 包括运行导出策略的板载运行时
- [mjlab](https://github.com/mujocolab/mjlab) — 训练框架 (MuJoCo Warp + rsl_rl)
- [BAM](https://github.com/Rhoban/bam) — 更好的执行器模型, 由 Rhoban 开发

## 许可证

本项目基于 Apache 2.0 许可证. 详情见 [LICENSE](LICENSE) 文件.
3D 模型文件采用 Creative Commons BY-SA-NC 许可证.
