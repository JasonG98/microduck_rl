# Session 2: 训练命令如何找到 Microduck 任务?

- 日期: 2026-09-05.
- 单元/小节: U1.1/U1.2.
- 状态: 已暂停 (2026-09-08 收课); U2.1 词汇范围已覆盖, 下一次从文末收课总结接续.
- 入口: 学习体系重整后的接续点; 用户请求继续学习.

## 本节实际讲了什么

本仓库定义机器人和任务 (观测, 动作, 奖励, 指令等), mjlab 负责组织环境/仿真和训练接线, rsl_rl 提供 PPO 采样更新流程. MuJoCo/MuJoCo Warp 计算物理, BAM 建模执行器响应. 本仓库的 runner 继承依赖实现, 并非重写 PPO.

以 velocity 训练命令为线索: 控制台脚本进入 train_cli.main, 导入 mjlab 时加载 mjlab.tasks 插件入口, 从而导入本项目 tasks 模块并注册任务. mjlab 训练 main 选择 task ID, 从注册表取环境和 RL 配置副本, 应用命令行覆盖, 后续创建环境和 runner 并调用 learn.

注册表把一个字符串关联到训练环境配置, play 环境配置, RL 配置对象及 runner 类. 配置工厂在注册时已经被调用, 但这一步不等于创建仿真环境. 类比 CV 的配置/注册表, 同时强调 RL 训练数据来自当前策略与环境的交互, 不直接等同于固定图片数据集.

## 源码与证据

导师静态核对当前工作区及本机实际安装的 mjlab 1.3.0 源码:

- `pyproject.toml`: `train = "mjlab_microduck.train_cli:main"`; `mjlab.tasks` 插件指向 `mjlab_microduck.tasks`. 本机生成的 train 脚本也指向此包装器.
- `src/mjlab_microduck/train_cli.py`: `main` 委托 `mjlab.scripts.train.main`.
- mjlab `__init__._import_registered_packages`: 在包导入时加载插件; 文档字符串提到 gymnasium, 但本项目实际调用的是 mjlab 自己的注册表, 不按该注释推断实现.
- `src/mjlab_microduck/tasks/__init__.py`: Flat velocity 注册关联 `make_microduck_velocity_env_cfg()`, 同工厂 `play=True`, `MicroduckRlCfg`, `MicroduckOnPolicyRunner`.
- mjlab `tasks.registry`: `_REGISTRY` 保存配置, `load_env_cfg` / `load_rl_cfg` 返回深拷贝.
- mjlab `scripts.train`: `main` 先选任务, `TrainConfig.from_task` 取配置, 再解析覆盖参数; `run_train` 创建 `ManagerBasedRlEnv`, wrapper 和 runner, 调用 `runner.learn`.
- runner 继承链: `MicroduckOnPolicyRunner → VelocityOnPolicyRunner → MjlabOnPolicyRunner → rsl_rl.OnPolicyRunner`.
- `src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py`: `MicroduckRlCfg` 是配置实例, actor/critic 各有 `hidden_dims=(512, 256, 128)`.

本节只有导师执行源码检索, 没有启动训练或仿真, 不作为用户实践证据. 工作区已有工程改动, 本课未修改这些代码.

## 练习与理解检查

待用户回答: 如果只把 actor 隐藏层从 `(512, 256, 128)` 改为 `(256, 128)`, 应沿注册块的 `env_cfg` 还是 `rl_cfg` 查找? 从对应符号找出 `actor.hidden_dims` 即可, 不必实际改文件.

目标是验证环境配置与学习器配置的职责区分. 状态: 已讲解, 导师已示范静态追踪; 用户待回答, 未判定独立掌握.

## 停在哪里, 下次做什么

先接用户对上述定位题的回答/追问. 随后进入 U1.3, 从 `make_microduck_velocity_env_cfg` 看基础 cfg 如何覆盖为 Microduck cfg, 用定位一个已有命令范围验证. 原因: 已有入口地图后再展开配置内容. 头部姿态问题继续搁置.

## 2026-09-06 接续: U1.3 配置如何成为对象

本次继续 Session 2. 主问题: 配置工厂返回什么, 修改配置副本如何影响后续环境创建?

已讲解: `make_microduck_velocity_env_cfg` 先调用 mjlab 的 `make_velocity_env_cfg()`, 再替换机器人、传感器及动作/奖励等配置, 最终返回 `ManagerBasedRlEnvCfg`. 配置本身也是 Python 对象, 但还不是运行中的环境. 注册时保存工厂返回的配置; `load_env_cfg` 深拷贝它, 包括嵌套的 scene. 训练入口 `TrainConfig.from_task` 读取配置, CLI 应用覆盖后, `run_train` 才调用 `ManagerBasedRlEnv(cfg=cfg.env, ...)` 创建运行环境. play 环境配置与 RL 配置是注册表中的独立入口.

源码核对: 本项目 `tasks/microduck_velocity_env_cfg.py` 的工厂、`tasks/__init__.py` 的 Flat 注册块; 本机 mjlab 的 `tasks.registry.load_env_cfg`、`scripts.train.TrainConfig.from_task` 和 `run_train`. 路径均以 `src/mjlab_microduck/` 或依赖包为起点, 不依赖历史行号.

导师已执行配置级示范 (使用本机 .venv/bin/python, 未构造运行环境、未启动训练):

```python
from mjlab.tasks.registry import load_env_cfg
name = "Mjlab-Velocity-Flat-MicroDuck"
a = load_env_cfg(name)
b = load_env_cfg(name)
a.scene.num_envs = 64
```

实测: 配置类型为 `ManagerBasedRlEnvCfg`; 修改前 a/b 的 num_envs 均为 1; 修改后分别为 64/1; 再次 load 得到 1; `a is b` 与 `a.scene is b.scene` 均为 False. 这是当前注册默认值, 不代表历史训练只使用了 1 个环境. 验证范围仅为配置副本隔离, 未验证 CLI 解析或仿真运行.

用户小验证待回答: 如果随后用 b 创建环境, 会使用多少个并行环境? 要让该环境使用 32 个, 应在创建前修改哪一行? 本次导师示范不计为用户实践或独立掌握. 原 actor.hidden_dims 定位题继续保留, 不阻塞推进.

下一步: 接用户对副本练习的回答, 必要时追踪 `load_env_cfg` 的 deepcopy; 理解后转入 U2.1, 从机器人配置选中的 MJCF 区分 body/joint/actuator. 头部姿态调试继续搁置.

## 2026-09-07 复习回合: U1 换一种方式复述

用户请求把之前讲的内容换一种方式重讲, 未要求开新课, 也未回答旧题.

- 方式: 从"运行中的 4096 个并行环境"倒推到一行命令, 反向重建调用链; 每个检查站同时问"这段代码谁写的 / 手里是什么对象".
- 复述范围: U1.1 分工 (本仓库定义任务+机器人, mjlab 组织环境与仿真, rsl_rl 提供 PPO, MuJoCo 计算物理, BAM 建模执行器; runner 继承链使本仓库不重写 PPO) 与 CV 的对照差异 (RL 数据由当前策略与环境交互现场产生, 非固定数据集); U1.2 注册表与插件入口 (`pyproject.toml` 的 train 脚本与 `mjlab.tasks` 插件, `tasks/__init__.py` 的 `register_mjlab_task` Flat 注册块, 注册时工厂参数即被求值); U1.3 配置对象到运行对象 (`TrainConfig.from_task` 深拷贝 + CLI 覆盖, `ManagerBasedRlEnv(cfg=cfg.env)` 创建活环境), 顺带重述副本隔离示范的含义与 rl_cfg/env_cfg 职责区分.
- 本轮证据: 导师静态核对 `pyproject.toml` 入口与 `tasks/__init__.py` 注册块, 与历史笔记一致. 未运行训练或仿真.
- 用户掌握证据: 无新增. 两道理解题 (副本 num_envs 小验证, actor.hidden_dims 定位) 仍待用户回答; 复习回合不验收为掌握.
- 下一步不变: 先答 U1.3 副本小验证, 随后进入 U2.1. 头部姿态调试继续搁置.

## 2026-09-07 晚 副本练习作答与澄清

用户回答 U1.3 副本练习: ① 用 b 创建环境会用 1 个并行环境 (正确); ② 让环境用 32 个: 用户写出 `--env.scene.num-envs 32`.

判定与澄清: ① 该点验收通过 (副本互不影响 → b 仍是默认值). ② CLI 覆盖在真实训练路径中确实有效: 训练入口从注册表取自己的副本, 覆盖落在该副本上, 与脚本里 load 出的 b 是不同对象; 但题干是对象级问题 — 若代码为 `ManagerBasedRlEnv(cfg=b)`, 环境只认构造那一刻 `b.scene.num_envs` 的值 (1), 命令行参数碰不到 b, 要 32 需在构造前执行 `b.scene.num_envs = 32`.

导师配置级复核 (本机 .venv, 未建环境未训练): `load_env_cfg` 默认 1; `b.scene.num_envs = 32` 后读回 32; 再次 load 的新对象仍 1. 用户尚未用自己的话复述"CLI 覆盖落在训练入口的副本"与"对象赋值落在 b 自身"的归属区别, 该项待一句话确认, 不阻塞.

下一步: 澄清无异议后进入 U2.1 模型词汇. 头部姿态调试继续搁置.

## 2026-09-07 深夜 面试式选择题测验 (首轮)

用户主动要求"像面试官一样拷问", 免文字作答, 用 GUI 选择题作答 (10 题单选). 结果 6 对 3 错 1 未答: 答对副本数值 (copy-1), PPO 分工 (div-1), hidden_dims 定位 (dims-1), 无深拷贝后果 (copy-2), on-policy (onpol-1), CV 数据本质 (cvdata-1); 答错注册求值时机 (reg-1), CLI 覆盖落点 (cli-1), 运行中改权重途径 (live-1); 未答迭代采样算术 (unit-1). 错题共同主题: 对象生命周期的时间点 + 单位区分, 经讲解后进入复测. 逐题题面/答案/考点见 [U1 题库](../quiz/questions/u1-bank.md), 作答与判定见 [答题记录](../quiz/records/2026-09-07-u1-quiz-rounds.md).

## 2026-09-07 深夜 Q8 追问: 单位区分

用户提问: 为何 velocity 例是 4096x24x4, roulade 例只写 8192x24 没有再乘 4. 讲解: 两个表达式数的是不同单位 — transition 按决策粒度 (每环境步 1 条, PPO buffer 记账), 物理子步是引擎内部积分 (每环境步 decimation 次). 上轮 roulade 行只报了 transition 数未报物理步, 故无 x4, 并非规则不同. 实测核对 (load 配置): velocity 与 roulade 的 decimation 均 4, num_steps_per_env 均 24; roulade 物理步/迭代 = 8192x24x4 = 786432. 附带锚点: 课程配置以 NUM_STEPS_PER_ENV 缩放, 证明课程/日志按环境步 (决策粒度) 计时. 用户尚未独立重算或复述, 待复测验证.

## 2026-09-07 深夜 术语追问: velocity 与 roulade 是什么

用户问 velocity 与 roulade 是什么. 讲解: 二者是注册表里的任务族 (task family), task_id 结构为 Mjlab-<行为>-<地形>-MicroDuck. velocity = 连续速度指令跟踪行走 (基于 mjlab velocity 模板, 由本项目覆盖机器人/奖励; 用户 Session 1 训练并 play 的 model_4998.pt 即此任务); roulade = 本项目独立设计的一次性前滚翻特技, 注册注释为"越过平头向前滚翻, 落回双脚" (用户 2026-09-07 重训的 4pkjvk56 即此任务). Flat/Rough 为地形难度 (velocity 两者皆注册, roulade 目前仅 Flat), MicroDuck 为机器人名. 上一轮算术例并举两者, 因为实测共享 decimation=4 与 num_steps_per_env=24. 用户对任务命名体系尚不熟, 记为待复习词汇, 未验收.

## 2026-09-07 深夜 复测 (第二轮): 5/5 全对

上轮错题/未答题换场景复测, 5 题单选全对: 导入期工厂执行 (reg-2), CLI 覆盖跨进程隔离 (cli-2), 分段权重表 (live-2), 采样单位算术 (unit-2, 512x24=12288 transition/迭代, x4=49152 物理子步), 任务族词汇 (vocab-1). 判定: 识别级验收通过, 未做文字复述, 不虚报独立复述; U1 主线具备进入 U2.1 的条件. 题面/作答/判定见 [答题记录](../quiz/records/2026-09-07-u1-quiz-rounds.md).

## 2026-09-07 深夜 收工总结与夜间训练建议

本节收工 (用户要去休息). 本会话 (导师主线) 当日完成: U1 倒推法复习; 副本小验证作答 (第 1 问正确, 第 2 问 CLI 方案, 已澄清对象级赋值); 首轮面试式测验 10 题 (6 对 3 错 1 未答) + 逐题讲评 (含 Q8 单位区分追问与 velocity/roulade 术语追问); 复测 5/5 全对 (识别级). 另完成面试拷问模式内化: SKILL.md 新增专节与意图行, recording.md/plan.md/学习 README 同步, quiz/ 目录落地 (题库 [u1-bank](../quiz/questions/u1-bank.md) 15 题 + 答题记录 [rounds](../quiz/records/2026-09-07-u1-quiz-rounds.md)), 会话记录中的逐题细节已压缩为摘要. 所有学习文档改动未提交 git.

夜间训练建议 (导师建议, 用户自行在服务器执行, 结果明日核对):

- 任务: roulade 头颈修复配置 (w=2.5) 的**第二种子完整重训**, 与 4pkjvk56 形成双种子对照.
- 理由: 4pkjvk56 是单种子; neck_pitch −10° 残差在两条路径 (resume/重训) 收敛到同一值, 疑似 reward 平衡点, 但完成率 95.7% 与 −10° 是否种子无关仍无证据. 第二种子若复现 (完成率接近 + 残差仍约 −10°) → 确认 reward 塑形所致, 再决定是否做 EMA-DC 直流税; 若完成率大幅波动 → 种子敏感, 修复结论需打折. 该实验无代码改动, 属 U4.5/U5.5 多种子实践, 不计课程号.
- 命令: `uv run train Mjlab-Roulade-Flat-MicroDuck --env.scene.num-envs 8192 --agent.seed 2025` (先跑 64 env/5 iter smoke 再放长跑; 建议 tmux/nohup 后台). seed 字段默认 42, 已核对 mjlab `rl/config.py` (`seed: int = 42`, train 入口同时设置 agent.seed 与 env.seed).
- 对照基线: 4pkjvk56 (seed 42 默认), 站立桶完成 95.7%, 头颈偏差 −10/+0.6/−0.5/−0.7°.
- 明日核对: 新 run 站立桶完成率与头颈偏差是否复现; 同时用户做 4pkjvk56 视频观感确认.

明日起点: U2.1 模型词汇 (robot/microduck_constants.py + velocity 任务实际使用的 MJCF), 主线暂停结束由用户说"继续"触发.

## 2026-09-08 接续: U2.1 模型词汇 (walk 模型上的 MuJoCo 词汇)

用户说"继续学习", 主线从 U1 收工点进入 U2.1 (progress 与 plan 记录的明日起点).

- 主问题: 策略实际控制的是怎样一台机器人? 本节先建立模型词汇: body, joint, freejoint, actuator, geom, site, sensor 及 qpos/qvel, 全部锚定在 velocity (Flat) 任务实际使用的 `robot_walk.xml` 上.
- 已讲解: 每个词汇的职责、为何 RL 侧按名字引用它们、以及该模型上的实测清单 (body 15 个机器人刚体 + 世界, 14 个 hinge 舵机关节 + 1 个浮动基座, 14 执行器 ↔ 14 关节一一对应, 75 geom 中仅 2 个带名碰撞脚底 + 3 个自碰撞, 7 site, 6 sensor; nq 21 = 7 + 14, nv 20 = 6 + 14). 指明 qpos/qvel 维度不同的根源是浮动基座姿态用四元数 (7/6), hinge 各 1/1.
- 源码与证据 (导师实测, 2026-09-08 本机 .venv): 用 `MjModel.from_xml_path` 实测 `robot_walk.xml`: nq 21, nv 20, nbody 16, njnt 15, nu 14, ngeom 75, nsite 7, nsensor 6; joint/actuator 名字列表一致且 `actuator_trnid` 显示每个执行器绑定到同名关节; site 与 sensor 名如上. XML 锚点: `<freejoint>` robot_walk.xml L85, hinge 关节自 L113, sensor 块 L60-67, `<actuator>` 块 L411-426 (14 个 position 执行器), 带名脚底碰撞 geom L164/L319, self_collision_only L99/L154/L309. 澄清"XML 有 14 个 position 执行器、项目又配了 BAM 执行器是否重复": 依据 bam 包 `edit_spec` (bam/mjlab.py L234-293) 与 mjlab Entity `_add_actuators` (entity.py L176 起), BAM cfg 按目标关节名把 XML 已有执行器就地 `set_to_motor` 转成电机模式并清零关节 frictionloss/damping, 未新增 → 训练模型仍 14 执行器; BAM 电压/摩擦细节留 U2.3. velocity 场景确实使用 `MICRODUCK_WALK_ROBOT_CFG` (velocity cfg L261) → `MICRODUCK_WALK_XML`.
- 练习 (待用户作答): 3 个文件定位题 (左腿 5 关节名 / 负责"躯干不焊死"的元素 / 角速度 sensor 名) + 从 XML 预测 nq/nv/nu 再与导师实测核对; 可选终端小命令列出 14 关节 range.
- 掌握证据: 无新增. 本节为讲解 + 导师静态/实测核对, 用户练习与独立复述待作答, 未验收.
- 停点与下一步: 接用户练习作答; 随后 U2.2 项目映射 (HOME_FRAME 与关节命名, passive_* 前缀, 为何不能对模型硬编码关节索引, `_servo_joint_ids` 辅助). 头部姿态调试与 roulade 收尾等搁置项不变.

## 2026-09-08 U2.1 从头重讲: body 与 joint

用户明确要求“继续 U2.1, 从头来”. 保留之前讲解与导师实测的历史证据, 当前接续改为小步重讲, 不要求先完成原整组练习, 不直接进入 U2.2.

本轮主问题: 仿真如何表达机器人由哪些刚性部分组成, 以及这些部分之间怎样运动? 先讲 body(刚体) 与 joint(关节): body 是运动时作为整体的一块, 可包含多个固定装配的零件; joint 声明该 body 相对父 body 允许的运动. 用大腿/小腿和膝关节作直观例子, 并说明关节定义运动自由度, 执行器负责驱动, 两者职责不同. 本轮不展开执行器实现.

导师静态核对 `src/mjlab_microduck/robot/microduck/robot_walk.xml`: `trunk_base` 下嵌套 `yaw2roll`, 后者内的 `left_hip_yaw` 为 hinge, axis="0 0 1"; 此轴在关节所在局部坐标系中表达, 不能直接当世界竖直方向. 讲解使用省略其他属性与节点的 XML 摘录. 机器人 XML 的入口也由 `microduck_constants.py` 的 `get_walk_spec` 核对. 未运行新仿真或训练, 没有新增用户实践/掌握证据.

理解检查待回答: 若腿上的两个 body 之间没有 joint, 也没有其他特殊连接设置, 它们是否能相对弯曲? 允许一句直觉回答, 不列为正式题库测验. 下一步先接该回答或困惑, 再用 trunk_base 下的 freejoint 解释整台机器人为何能移动/跌倒; 随后逐步补 geom/site/sensor/actuator 与 qpos/qvel. 原 U2.1 整组练习保留为历史待完成项, 不作为当前门槛.


### 同日追问: 间接连接中有 joint 是否能相对运动

用户回答“可以吧, 如果是间接的有 joint 呢?”. 原题没有明确限定直接父子 body, 存在歧义, 不判用户错误. 用户主动考虑中间连接含 joint 的情况, 是局部概念推理证据, 不代表整个 U2.1 已掌握.

澄清: 子 body 自身不声明 joint/freejoint 时, 它相对直接父 body 固定; 这不等于相对所有祖先或世界固定. 示例 A --hinge-- B --固定-- C: hinge 转动时 B/C 一起相对 A 转动, C 相对 B 不变, C 相对 A 可以变化. 整体在世界里移动与两个 body 之间的相对运动要区分. 静态复核 robot_walk.xml 的 trunk_base → yaw2roll → hip_l 嵌套链, 两段分别由 left_hip_yaw/left_hip_roll 定义; 未执行新仿真或用户操作.

当前问题已澄清, 无新增必答题. 下一步从 trunk_base/freejoint 讲根 body 如何相对世界运动, 延续“相对谁”的判断.


### 同日接续: freejoint 与自由基座

用户说“继续”, 按当前接续点讲 freejoint(自由关节)/floating base(浮动基座). 主问题: 躯干相对世界能怎样运动? `worldbody` 是世界参考, `trunk_base` 是其直接子 body; 该 body 内声明 `trunk_base_freejoint`, 允许三维平移与三维转动, 共 6 个自由度. 与 hinge 只允许绕一根轴转动对照. 此处自由度是独立运动方式的数量, 不等于舵机数量; 自由基座不是额外的 6 个电机, 运动由重力、接触和机器人内部运动等共同影响. freejoint 本身不提供推进力, 脚底接触仍能约束实际运动.

导师静态核对 `src/mjlab_microduck/robot/microduck/robot_walk.xml` 的 worldbody/trunk_base/freejoint 声明. 摘录 pos="0 0 0.12" 表达 XML 定义的参考位置, 不是运行中锁定高度, 也不保证训练 reset 使用该值; 本轮不展开 reset 或 qpos/qvel. 未执行新仿真或训练.

理解检查待回答: 只删除 trunk_base 内的 freejoint, 不添加替代 joint、不改其他连接, 腿部关节仍保留; 机器人还能通过走路让躯干相对世界向前移动吗? 用户尚未作答, 不判定 freejoint 已掌握. 下一步先接该题或追问, 然后讲 geom(几何体) 如何表达外观与碰撞, 以脚底接触承接 freejoint 仍受物理作用的概念.


### 同日追问: XML 父子节点与关节声明规则

补记上一回合: 用户对删除 freejoint 的预测题回答“不知道诶”, 记未答. 导师已解释删除后躯干相对 worldbody 固定, 腿部仍可运动但不能改变躯干相对世界的固定关系. 未执行物理实验, 不计为用户掌握.

本轮用户说“上一层就是他的父关节”, 并问为何 freejoint 定义在 body 内. 澄清该词应是父 body(刚体), 并区分 XML 节点包含关系与机械运动关系. MJCF 通过 body 嵌套建立刚体树; 子 body 内直接声明的 joint 定义该 body 相对直接父 body 的运动自由度. 因而 joint 写在子 body 中, 但描述的是父子之间的运动关系, 不是再挂一个子刚体. 无 joint 时两 body 相对固定, 连接关系本身已由 body 嵌套建立. freejoint 沿用这一规则, 定义 trunk_base 相对 worldbody 的自由运动.

示例为当前 robot_walk.xml 的简化摘录: worldbody → trunk_base (内含 trunk_base_freejoint) → yaw2roll (内含 left_hip_yaw). 对应运动关系 worldbody --freejoint-- trunk_base --left_hip_yaw-- yaw2roll. 已静态核对仓库声明, 并查 MuJoCo 官方 XML Reference 的 body/joint 定义: https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-joint . XML 只规定文档结构, 此处物理语义来自 MuJoCo 的 MJCF 规则. 本轮未运行仿真.

状态: 父 body 与 joint 的归属规则已澄清, 用户尚未独立复述, 无新增必答题. 下一步如仍困惑先读同一简化 XML 的两条边; 否则讲 geom 外观与碰撞.


### 同日用户纠正: 先按对象属性理解 XML

用户明确纠正导师的表达: 自己关注 XML 的包含结构, 将子项类比为 Python 对象的属性, 嵌套 body 是其中的嵌套对象; trunk_base_freejoint 是 trunk_base 所拥有的描述项, 表示该 body 有一个 freejoint. 此为用户自主表达的结构理解证据, 不应继续归因为不懂父 body/关节关系. 上一轮导师过早转向机械连接图, 没有充分回应用户的结构视角; 本条修正上一条“已澄清”的单方面判断.

回应采用概念性 Python 配置示意: Body(name=..., joints=[FreeJoint(...)], children=[Body(...)]) 放入 WorldBody 的 children. 明确这不是项目真实 API; 更精确类比是对象组合, 不是 Python 的嵌套类定义. 用户将子项统称为属性作为阅读类比可行, 但 XML 正式语法区分 name 等 attribute 与 freejoint/body 等 child element. 仅保留必要物理语义: 该 body 拥有的 freejoint 描述它相对父 body 的自由运动. 本轮未运行仿真或新增题目.

后续教学先按对象/字段归属解释 XML, 再按需连接物理含义. 下一步可沿这个视角讲 body 的 geom 描述项. 用户证据限于对象包含关系与 freejoint 归属, 不扩张成全节掌握.


### 同日表达偏好更新: 用树结构

用户要求“不要用代码表达, 又复杂又不直观, 用树结构就好”. 后续用树展示包含关系, 节点标明 body/joint 类型与简短作用, 不再用 Python 配置示例重写 XML. 本轮仅调整表达并重画已有 worldbody/trunk_base/freejoint/yaw2roll/left_hip_yaw 结构, 无新知识验收或实践, 不占新课号. 下一步仍为用树结构介绍 geom.


### 同日接续: geom 的外观与碰撞作用

用户要求继续下一步. 按偏好用包含关系树讲 geom(几何体), 不用 XML/Python 代码示例. 主问题: body 如何描述自己的形状? geom 是附着于 body 的形状描述, 同一 body 可有多个 geom, 它们随 body 一起运动; 是否参与碰撞由配置决定, 不是每个看得见的形状都负责碰撞.

导师静态核对 `src/mjlab_microduck/robot/microduck/robot_walk.xml`: body ankle_left 直接包含 left_ankle joint, 使用 sole_left 的 visual geom, 具名 left_foot_collision 的 collision geom, 使用 foot_left 的 visual geom 等. 树中“脚底外观/脚部外观”为教学标签, 非 XML 的 geom 名称, 且省略惯性/site/其他 geom. 两个脚底 geom 都引用 sole_left 网格; 本例没有把碰撞形状简化成盒子. visual 默认 contype/conaffinity 均为 0, 当前模型关闭其自动碰撞参与; left_foot_collision 作为脚底碰撞形状. 碰撞 geom 也可以显示, visual/collision 是本项目配置分工, 不是互斥的几何类型. 未运行仿真或用户实践.

小判断待回答: 只在画面中隐藏脚底外观 geom, 保留 left_foot_collision 与其他物理设置, 脚是否还能受地面支撑? 用于区分显示与物理碰撞, 未提前判用户掌握. 下一步接该题或追问, 随后用同一 body 内的 left_foot site 讲参考点, 再衔接 sensor.


### 同日图文增强: 真实模型标注图与 12 秒视频

用户指出纯文字/树缺少有效图例, 明确要求在真实模型上标注、图文并茂并附在学习笔记, 能做视频更好. 导师据此制作本节教学素材, 不另外开课号. 后续以实际模型标注图为主, 树表示包含关系; 有运动关系时优先补短动画, 不把用户“不要代码”误读为只需要文字树.

![body 与 joint 真实模型标注](../assets/u2-1/01-body-joint.png)

蓝色 trunk_base 是躯干, 橙色 leg 是左小腿, 绿色 ankle_left 是左脚. 红点对应实际 left_knee 关节锚点; freejoint 标记在躯干原点, 不是额外电机. 着色只服务教学, 没有改机器人源 XML.

![左脚外观与碰撞形状对照](../assets/u2-1/02-geom-comparison.png)

同一脚部刚体、同一机位对比: 左侧显示外观零件, 右侧只显示 left_foot_collision. 两个脚底 geom 共享 sole_left 网格, 区别在配置用途. 其余形状仅隐藏显示, 不代表删除物理模型.

[观看 12 秒视频](../assets/u2-1/03-model-tour.mp4) · [动图预览](../assets/u2-1/03-model-tour.gif) · [完整图解笔记与复现说明](../assets/u2-1/README.md).

视频分三段: 0–4 秒左膝转动、4–8 秒外观/碰撞显示切换、8–12 秒自由基座带动整棵子树移动. 是手动设 qpos 后 mj_forward 的运动学示意, 没有 mj_step、策略推理或 BAM 动力学, 不证明支撑、稳定性或策略效果.

导师执行证据: MuJoCo 3.10.0 EGL 离屏渲染 scene_walk.xml 的 STAND 参考姿态; 按名字解析关节/body/geom, 由 xanchor/xipos/geom_xpos 投影标注. 检查两张静态图与视频三个阶段的抽帧, 视频解码确认 H.264、960×720、24 fps、288 帧、12 秒. 生成器与图片/视频/动图均在 docs/learning 内, 无需临时路径接续; 未提交/推送 Git. 素材由导师制作, 不算用户独立实践或掌握证据.

当前接续: 先看图比较左脚外观与碰撞覆盖范围; 随后在实际模型上标 site, 用树辅助说明其归属. 原 geom 支撑判断没有收到用户答案, 保留未验收, 不强制重问.


### 同日看图追问: 碰撞 geom 是否就是外观的底部

用户从图提出“当前项目的碰撞 geom 实际就外观 geom 的底部对吗?”. 说明用户已关注外观与碰撞的空间覆盖差异, 但属求证, 不计全概念验收. 导师核对 robot_walk.xml: 脚底 visual geom 与 left_foot_collision 独立声明, 同用 sole_left 网格与同一 pos/quat. 对该脚图可理解为选择脚底零件形状, 不是自动裁切整脚外观; 也不是只定义一个二维底面. 范围不扩大为整个项目: walk 模型另有右脚底和躯干/左右小腿 self_collision_only 几何配置, 其他任务模型另论. 已补入图解页, 本轮仅静态核对, 无新物理实测. 下一步仍为在模型上标 site.


### 同日追问: 为什么不让整个脚部都参与碰撞

用户问为何只用脚底而不是整个脚部. 导师核对 config_mjcf_walk.json 的碰撞零件筛选与 config_mjcf_groundcontact.json 的扩展集合, 并核对 microduck_constants.py 的任务用途注释. 讲解正常行走主要需要脚底支撑/摩擦, 未接触环境的上部外壳碰撞对该阶段帮助有限, 增加碰撞形状可能增加计算工作; 明确这属于工程解释, 非作者完整历史动机或提速实测. 整脚碰撞合法, 涉及脚侧/脚背接触时应补足; 缺省碰撞的外壳可能穿过环境. 显式惯性已定义, 省碰撞不是省掉质量. 已同步至图解笔记并复用左脚对照图. 无新仿真或用户验收, site 仍是后续主线.


### 同日接续: site 的位置与方向

用户要求继续下一步. 按图文偏好制作真实 left_foot site 标注图, 用白点表示原点、RGB 箭头表示局部 XYZ, 旁配 body/joint/geom/site 包含树. 已讲解 site 是附着于 body 的参考坐标系, 同时有位置与方向, 随脚移动/转动; 不自动等于脚尖/质心/实际接触点. 不参与碰撞或惯性, 自身不是 sensor, 后者可以引用它. 用“脚的位置具体指哪里”解释定义命名参考点的职责.

![left_foot 的参考位置和方向](../assets/u2-1/04-site-left-foot.png)

导师核对 robot_walk.xml 中 left_foot 直属 ankle_left, 并核对 velocity cfg 的 foot_height_scan.frame 引用左右脚 site. 通过 CPU MuJoCo 加载 STAND 姿态, 读取实际 site_xpos/site_xmat, 检查所属 body 与旋转正交性, 渲染并目视检查标注图; 修正原点引线对齐“位置”说明卡. 无策略/动力学实验. 详见 [完整图解与复现](../assets/u2-1/README.md), 新生成器为 docs/learning/tools/render_u21_site.py.

状态: site 已讲解与导师图示, 用户尚无新复述/操作证据, 本轮不强制新增题. 下一步先接图上疑问, 随后讲 sensor 与 site 分工, 从 trunk_base 的 imu site 和引用它的 gyro 入手.


### 同日用户理解证据: site 与 ROS TF 类比

补记用户自主复述: site 就是坐标系, 类似 ROS 世界/机器人/相机坐标系, 通过 TF 做转换. 导师确认核心类比成立, 补充 body 本身也有 frame, site 是附加参考 frame, 变换由 MuJoCo 计算, 不依赖 ROS TF. 用户随后说“明白了, 继续下一步”. 记 site 坐标系概念的局部复述证据, 非独立 TF/仿真操作或全 U2.1 验收.

### 同日接续: sensor 与 site 的分工

本轮用真实躯干模型标出 imu site 的原点/三轴, 并画 sensor 配置与 site 的名字引用关系. 主问题: 有了参考坐标系, 如何定义测量? site 决定参考位置/方向, gyro sensor imu_ang_vel 测局部三轴角速度, rad/s; 与 ROS 消息 frame_id/测量字段类比. sensor 块与 worldbody 是模型下的不同区域, gyro 通过 site="imu" 引用, 不把 sensor 画成 site 内子节点. 简短说明 site 本身不自动产生 IMU 测量, 不把所有 sensor 都限定为 site 引用.

![imu site 与 gyro 传感器](../assets/u2-1/05-sensor-imu.png)

导师实测: scene_walk.xml / STAND, MuJoCo 3.10.0, 名字解析 imu_ang_vel 为 3D gyro 并核对其引用 imu. 静止读数 [0,0,0], 人工设置自由基座 Z 角速度 1 rad/s 后 mj_forward 得 [0,0,1]. 图由实际 site_xpos/site_xmat 投影, 已目视检查. 未运行策略、BAM、动力学积分或训练噪声; 验证仅为瞬时状态到传感器的映射. 代码与详细解释见 [图解页](../assets/u2-1/README.md) 和 docs/learning/tools/render_u21_sensor.py.

状态: sensor 已讲解和导师示范, 用户理解待验证, 本轮不强制新增题. 下一步先接 gyro/site 疑问, 随后用图认识 actuator, 再继续 qpos/qvel; 不自动进入未完成的 U2.2.


### 同日接续: actuator 与被驱动的关节

用户对 sensor 说“明白了, 继续下一步”, 记用户自述理解, 不等同于独立复述或实践. 本轮讲 actuator 执行器, 在真实左腿模型上以 left_knee 锚点标明驱动关系, 对照 joint 的运动自由度与 actuator 的驱动力矩. XML actuator/position:left_knee 显式引用 joint:left_knee, 同名不是同一对象. 图与说明已同步到图解页.

![左膝 actuator 与 joint](../assets/u2-1/06-actuator-left-knee.png)

导师核对原始 XML 模型 nu=14, left_knee 执行器的 actuator_trnid 及目标 joint 的 body 归属. 当前源码使用 JointPositionActionCfg, 本机 mjlab JointPositionAction.apply_actions 写位置目标; FrictionDRBamActuatorCfg 继承 BAM, edit_spec 就地将 XML position 转为 motor, compute 用目标/状态计算电压与力矩并更新摩擦. 用目标 0.2 rad/当前 0 rad 作概念例子区分目标与响应, 未声称该动作已动态实测. 图已生成并目视检查, 仅静态核对, 没有训练/BAM 动力学实验.

状态: actuator 已讲解与导师图示, 用户理解待验证. 后续先接目标/实际状态的疑问, 再讲 qpos/qvel, U2.3 再细看 BAM 和控制频率.


### 同日接续: qpos / qvel 的状态布局

用户对 actuator 表示“明白了, 继续下一步”, 记自述理解, 无独立作答/操作. 本轮配真实模型与状态分块图, 讲实际位置状态 qpos 和实际速度 qvel, 与位置目标分开. 当前 walk 自由基座贡献 7/6, 14 个 hinge 贡献 14/14, 总计 nq=21/nv=20. 差一维来自姿态四元数 4 个数与转动速度 3 个数的表示区别; 四元数有单位长度约束, 不等于额外转动自由度. 不展开四元数推导.

![qpos/qvel 布局](../assets/u2-1/07-qpos-qvel.png)

导师 CPU MuJoCo 加载 STAND 关键帧并 mj_forward: nq=21/nv=20, 基座位置/姿态 [0,0,0.12,1,0,0,0], qvel 全零, left_knee 实际角度 -0.00494 rad、角速度 0 rad/s, 按名字解析的 qpos/qvel 地址为 10/9. 仅为当前模型核对, 不跨模型硬编码. 已核对官方自由关节 7/6 约定, 生成标注图并目视检查. 图/生成器/详细说明见 [图解页](../assets/u2-1/README.md). 未运行动力学或训练.

理解检查待回答: 左膝目标 0.2 rad、实际 0.1 rad, qpos 对应项存哪个? 状态: qpos/qvel 已讲解与导师实测, 用户待回答; U2.1 重讲范围已覆盖, 不宣称全节掌握. 下一步先接问题/疑问, 然后 U2.2 HOME_FRAME 与关节名映射; 已搁置项目不恢复.


## 2026-09-08 收课总结: U2.1 模型词汇从头重讲

用户说“今天先到这里, 总结整理一下”. 本轮只收课与整理, 不引入新知识、不催答题、不启动 U2.2. 继续保留 Session 2 编号, 本节状态改为已暂停.

### 今日讲解主线

body/joint → freejoint → geom 外观与碰撞 → site → sensor → actuator → qpos/qvel. 全部通过当前 walk 模型核对, 后半程形成“真实模型标注图 + 包含关系树 + 简短解释”的教学方式. 已澄清直接父子与间接运动链的范围、脚底碰撞复用与按任务取舍、site 的 ROS/TF 类比、sensor 的测量引用、actuator 目标不等于实际运动、自由基座 7/6 导致 qpos/qvel 21/20.

### 用户证据与待确认项

- 有局部推理/复述证据: 主动提出间接 joint 反例; 以对象包含关系解释 XML 和 freejoint 归属; 自主用 ROS 坐标系/TF 类比 site.
- 用户自述理解: sensor 与 actuator 讲解后表示“明白了”. 不扩大为独立操作验收.
- 未答但已讲解: 删除 freejoint 后躯干是否还能前进, 用户曾答“不知道”, 导师解释完成.
- 仍未答: 只隐藏脚底外观是否仍有支撑; 目标 0.2 rad/实际 0.1 rad 时 qpos 存哪个. 保留为待确认, 不判错误, 不强制全部补答.
- 原 U2.1 大组文件定位/nq-nv-nu 练习保留为历史待完成, 不作为强制门槛. 四元数仅解释表示维度, 未推公式.

### 图文、视频与导师核对

学习者复习入口为 [U2.1 图解笔记](../assets/u2-1/README.md), 已增加收课速览与逐图索引. 共 6 张标注图 (body/joint、geom、site、sensor、actuator、qpos/qvel), 12 秒 MP4 与 GIF 预览, 5 个可复现生成器. 已检查图片、视频抽帧与媒体参数. 模型维度/按名映射、gyro 两组瞬时读数均为导师执行; 没有用户独立实践、BAM 动力学实验或新训练. 不把视频手动摆动当策略行为证据.

### 下次直接从哪里开始

先接 qpos 目标/实际区别的一句确认或用户疑问, 随后进入 U2.2, 从 `src/mjlab_microduck/robot/microduck_constants.py` 的 HOME_FRAME 与实际 joint 名字映射开始; 验证方式是按名字找到一个初始关节角并对应模型位置. 先建立配置与模型的映射, 再学 U2.3 BAM/控制频率. 用户可跳过未答题, 不重新从 U2.1 开头讲起. 头部观感与 roulade 收尾等搁置项保持搁置.

已保存到工作区学习文档, 未提交/推送 Git. 本收课整理不改变其他任务或既有训练记录.


### 收课后管理: 将图文教学内化到 skill

用户明确要求把图文并茂的讲解写入导师 skill. 已在 `.agents/skills/microduck-mentor/SKILL.md` 新增“图文并茂的讲解”, 覆盖真实模型标注、正文展示、树辅助、必要动画、模型核对与视觉检查、复用生成器、媒体持久化及环境受限替代; 同步调整原教学顺序/追问讲法, 在 references/recording.md 补媒体记录约定. 使用 skill-creator 做范围审查与格式验证, quick_validate 通过, git diff --check 通过. 本次为管理更新, 没有授课、训练或新掌握证据, 课程继续暂停, 下次起点保持 U2.2 前的 qpos 简短确认.


### 收课后恢复: U2.1 选择式复测, 第一批待作答

用户明确要求针对今天知识复测, 恢复测验活动, 不进入新课. 首批 5 道换场景题覆盖连接关系、碰撞分工、gyro 测量、执行器目标与实际状态、状态维度. 题面/答案已入 [U2.1 题库](../quiz/questions/u2-1-bank.md), 作答/判定单列 [记录](../quiz/records/2026-09-08-u2-1-quiz.md). 尚无用户选择或成绩, 不提升能力等级.

本轮出题前核对 robot_walk.xml visual/碰撞/site/gyro 与惯性声明, 以及当前 mjlab JointPositionAction.apply_actions; 使用 MjSpec 内存副本移除 freejoint 并编译, 验证维度从 21/20 变为 14/14, 未改源模型、未训练. 下一步接 GUI 作答后逐题讲评, 错题和明确不知道的题换场景复测; 无提交保持待回答.


### U2.1 首批复测讲评: 4/5, 状态维度待复测

GUI 卡片曾消失, 重发后用户最终用正文提交首批答案. 结果4对1错0未答: 连接关系、碰撞分工、gyro角速度、目标/实际状态正确 (识别级证据); 删除 freejoint 后维度题错误. 重申 freejoint 贡献7/6, 删除后从21/20变为14/14, 区别于保留但静止. 使用已保存状态布局图辅助讲评, 无新课程内容. 两题换场景复测已入库并待回答: 保留结构但速度全零, 六个hinge新增自由基座. 本轮MuJoCo内存模型核对结果21/20与6/6→13/12, 未改项目模型、未训练. 题面与成绩分离存储, 见 [作答记录](../quiz/records/2026-09-08-u2-1-quiz.md). 下一步接两题复测, 全对再销项, U2.2仍未开始.


### U2.1 维度复测通过: 2/2

用户提交“B、D”, 两道换场景题全对, 逐题讲评为“零速度不改变维度”与“六铰链6/6加自由基座7/6得到13/12”. 首批4/5, 唯一错题的针对性复测2/2, 维度主题销项. 本次抽查范围通过, 仅为识别级证据, 不宣称独立建模操作或全部课程能力已验收. 逐题结果见 [作答记录](../quiz/records/2026-09-08-u2-1-quiz.md). 本轮不新增知识或实践. 今日测验结束, 新课保持暂缓; 下次从U2.2 HOME_FRAME与joint名字映射开始, 不再把qpos小判断作为未解决阻塞.


### 最终收课: 今晚行走训练建议与明日 W&B 读图课

用户再次收课去休息, 要求直接给训练命令, 明天教看 W&B 数据. 推荐 Flat velocity, 4096 环境、5000 迭代、seed42, 从头训练; 原因是行走主线便于解释速度跟踪、分项奖励与 episode 长度, 当前 action_rate/head_pose_bias 等权重在约500–1500迭代分阶段变化, 可结合权重表解释曲线变化. 不承诺具体耗时或最终步态质量. 当前默认 max_iterations 为50000, 命令显式限制为5000.

先64环境/5迭代 smoke, 退出成功后再开始长训练. smoke 用 tensorboard 避免混淆 W&B 主 run; 主 run 显式 wandb logger, project=mjlab_microduck, run_name=velocity_wandb_lesson. 用户需实际执行; 导师没有启动训练, 尚无 run ID/链接或结果, 不记为已实践. 参考命令:

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 64 --agent.max-iterations 5 --agent.logger tensorboard && \
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max-iterations 5000 --agent.seed 42 --agent.logger wandb --agent.run-name velocity_wandb_lesson
```

已静态核对任务注册、MicroduckRlCfg、mjlab runner logger/run_name/seed 字段与 tyro 入口, 没有执行该命令或新的 smoke. 项目既有 W&B 登录状态未验证; 不输出或读取凭据.

明日起点按用户新目标改为 W&B 读图实践, U2.2 暂后移而非取消: 先定位本次 run (名称后缀 velocity_wandb_lesson, 若有多个以实际时间/配置区分), 确认环境数/迭代数/种子/是否完整结束; 再依次看总奖励与 episode 长度、主要速度跟踪与惩罚分项、课程权重阶段、loss; 最后结合实际 rollout/视频评估行为. 先从少量曲线讲起, 不用高总奖励或小 loss 单独断定学会. 使用实际 run 数据与标注图, 若用户未运行或失败则先核对状态, 不虚构曲线. 不创建定时提醒或后台监控. 用户下次回来提供链接或从本地日志定位后再开课.


### 夜间任务更换: SitStand 坐站切换

用户指出 velocity 已跑过, 要换其他模型. 本条替代上面的 velocity 夜间建议. 改推荐现有 `Mjlab-SitStand-Flat-MicroDuck`, 按指令坐下/站起的双向任务, 4096环境、6000迭代、seed42, run_name=sitstand_wandb_lesson. 并非确定独立结课项目, 不恢复 roulade 调试.

静态核对 tasks 注册、microduck_sitstand_env_cfg.py 任务语义和 runner 配置 (原默认15000迭代, 项目mjlab_microduck). 上升速度惩罚在1500/2500迭代引入/增强, 动作平滑项在500–1500分阶段变化, 推力在2500达到后期范围; 6000是本次建议预算, 不保证行为成功或特定耗时. 先64环境/5迭代 smoke, 成功才放长跑, 导师未实际执行:

```bash
uv run train Mjlab-SitStand-Flat-MicroDuck --env.scene.num-envs 64 --agent.max-iterations 5 --agent.logger tensorboard && \
uv run train Mjlab-SitStand-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max-iterations 6000 --agent.seed 42 --agent.logger wandb --agent.run-name sitstand_wandb_lesson
```

明天优先定位此 SitStand run, 看任务奖励、动作平滑和上升/下降惩罚随课程变化的关系, 再结合视频确认坐下和站起是否都完成. 速度跟踪不再作为本次主任务曲线. 尚无实际启动、run链接或结果, 记录为建议待用户执行, U2.2仍延后.


### 2026-09-14 复习回合: U1 + U2.1 主线回顾

- 用户请求“复习一下先”, 本回合不进入 W&B 新课, 先回顾已讲过的项目启动链与机器人模型词汇.
- 导师重新核对当前源码: `tasks/__init__.py` 的注册参数会在模块导入时调用任务工厂并保存配置对象; `MicroduckOnPolicyRunner` 继承上游 runner, 只做项目级序列化接线; `HOME_FRAME` 在 `microduck_constants.py` 中按 joint 名称正则描述初始状态.
- 复习主线已展示为: 训练命令 → `train_cli.py`/插件加载 → 任务注册 → 配置副本与 CLI 覆盖 → 构造环境与 Manager → PPO 收集交互数据并更新. 机器人侧为 body/joint → geom/site/sensor/actuator → qpos/qvel 与控制目标.
- 本回合未执行训练或测试; 用户的复述检查待回答, 不提升为可独立完成.
- 下次根据用户回答逐项讲评; 若复习通过, 再回到 W&B 读图或 U2.2, 由用户选择.

#### 复述检查结果

- Q1: 用户回答“运行中的环境”, 判定错误. CLI 覆盖发生在构造环境之前, 写入训练进程取出的配置副本; 运行中的 Manager 已根据配置构造, 不是每步重新读取 CLI.
- Q2: 用户回答“原来的角度”, 判定正确. 用户能区分 actuator 控制目标与物理状态中的实际 qpos.
- Q3: 用户回答“不知道”, 记为未答. 已解释 61D 是策略输入与部署端共同遵守的接口合同: actor 网络按固定位置读取各维, ONNX 输入形状也固定; 随意增加第62维会造成训练模型输入形状、归一化、运行时观测写入位置不一致, 不能热插拔.
- 证据等级: 1 对 / 1 错 / 1 未答, 识别与解释已完成; 尚无用户独立复述“配置副本”和“共享观测合同”的证据.

#### 复述补答

- 用户补答“环境构造之前”, 正确概括 CLI 覆盖时机.
- 用户补答“61D 是训练时的环境维度, 模型训练好后已经固定”, 核心正确. 导师补精确为 actor observation 输入合同, 不等于整个环境或 critic 的全部维度; 61 个位置及其语义需与部署保持一致.
- 本轮两个复习点均完成用户复述级确认; 运行时配置副本边界与 actor observation 合同的本轮卡点销项.

### 2026-09-14 U2.2: HOME_FRAME 与 joint 名字映射（讲解中）

- 用户选择进入 U2.2. 本节主问题是: `HOME_FRAME` 如何通过 joint 名称模式把初始姿态应用到实际 MJCF 模型.
- 当前源码调用链已核对: `make_microduck_velocity_env_cfg()` 设置 `cfg.scene.entities["robot"] = MICRODUCK_WALK_ROBOT_CFG`; 后者设置 `init_state=HOME_FRAME`.
- `HOME_FRAME` 是 `EntityCfg.InitialStateCfg`, 用 `joint_pos` 保存初始关节角, 用 `joint_vel={".*": 0.0}` 将初始关节速度设为零. `joint_pos` 的 key 是正则表达式, 例如 `r".*left_hip_pitch.*"` 匹配 walk 模型的 `left_hip_pitch`.
- 当前 walk XML 已核对包含 14 个主动关节: 左右两侧各 hip_yaw/roll/pitch、knee、ankle, 以及 neck_pitch、head_pitch、head_yaw、head_roll. 本节强调按名字匹配而不是假设固定索引, 因为 roller/backlash 模型会插入或包含 `passive_*` 关节.
- 已解释 `BACKLASH_HOME_FRAME` 在普通模式之前加入 `r".*_backlash$": 0.0`, 使 backlash 被动关节保持零位, 避免普通的 `.*left_hip_roll.*` 等模式误匹配到被动 backlash 关节.
- 本节尚未完成用户小练习; 尚无用户复述或独立映射证据.

#### 用户追问: 为什么使用正则表达式

- 已澄清: 正则表达式不是 RL 必需概念, 也不是所有机器人项目的通用做法; 它属于 mjlab 的配置选择器风格, Microduck 按该接口使用.
- 当前安装的 mjlab `EntityCfg.InitialStateCfg` 将 `joint_pos` 定义为 `dict[str, float]`, 并在 `Entity._add_initial_state_keyframe()` 中调用 `resolve_expr(...)` 按 joint 名称解析; mjlab 的 actuator、geom 等配置也有 `*_names_expr` 选择器. mjlab asset zoo 的 Unitree Go1 配置同样使用 joint 名称正则, 说明不是 Microduck 独有.
- 正则的工程价值: 同一个模式可选择一组带共同命名结构的对象, 对导入模型的前缀/后缀更有弹性, 并减少为每个关节写重复配置. 但宽模式也可能误匹配, 所以 backlash 变体用 `r".*_backlash$": 0.0` 先处理被动关节.
- 需要区分: RL 算法只关心最终得到的状态/动作张量; 它不要求正则. 其他框架也可能使用固定索引、精确名称、URDF joint handle 或显式对象引用.
- 本节用户尚未完成“正则匹配范围”小练习.

#### 用户追问: walk 模型的所有实际 joint 名字

- 已按 `robot_walk.xml` 当前声明顺序展开完整清单: `trunk_base_freejoint`; 左腿 `left_hip_yaw`, `left_hip_roll`, `left_hip_pitch`, `left_knee`, `left_ankle`; 颈头 `neck_pitch`, `head_pitch`, `head_yaw`, `head_roll`; 右腿 `right_hip_yaw`, `right_hip_roll`, `right_hip_pitch`, `right_knee`, `right_ankle`.
- 已说明父 body 链: 左腿为 `trunk_base → yaw2roll → hip_l → upper_leg_left → leg → ankle_left`; 颈头为 `trunk_base → neck → neck_pitch → yaw_roll_motion → jaw_soft`; 右腿为 `trunk_base → bearing_roll → hip_l_2 → upper_leg_right → leg_2 → ankle_right`.
- 已区分 1 个 freejoint 与 14 个 `class="chosen_actuator"` 的 hinge: 前者负责浮动基座状态, 不属于 14D 舵机动作; 后者是当前 walk 模型的主动关节. XML 中每个 hinge 的局部 `axis="0 0 1"`, 实际空间方向还受其所在 body 的姿态影响.
- 本节讲解已完成, 用户仍可通过按名字写出父 body 或正则匹配范围来完成实践级确认.

### 2026-09-14 U2.2: 一次 HOME_FRAME 映射的实际追踪

- 用户要求继续学习. 本段沿当前安装的 mjlab 实现追踪 `Entity._add_initial_state_keyframe()` 到 `resolve_expr()`.
- `resolve_expr()` 的规则已核对: 按模型 `joint_names` 逐个处理; 按配置字典声明顺序寻找第一个 `pat.match(name)` 成功的模式; 没有匹配时使用传入默认值 `0.0`.
- 使用当前 walk joint 名字做静态验证: `left_hip_pitch -> -0.4579`, `right_hip_pitch -> 0.4579`, `passive_left_hip_pitch_backlash -> 0.0`, `head_yaw` 未被专门设置而使用默认 `0.0`.
- 已解释得到的 joint 角度会组成模型 init keyframe 的关节部分; 对有 actuator 的实体, mjlab 还按 actuator 的 target joint name 生成对应初始 ctrl, 但这仍是初始控制目标, 不等于物理已经运动到该角度.
- 验证命令为导师执行的 Python 正则静态核对, 未启动仿真; 用户尚未独立完成同口径映射练习.

### 2026-09-14 U2.2: 从名字映射到 servo-only 索引视图

- 用户说“继续学习”. 已核对 `mdp.py` 的 `_servo_joint_ids`, `_servo_joint_pos`, `_servo_joint_vel`, `_servo_default_joint_pos`.
- `_servo_joint_ids` 使用 `asset.find_joints(r"^(?!passive_).*")` 找出所有不以 `passive_` 开头的 joint, 并缓存 entity 局部索引. 普通 walk 模型上它等价于主动关节顺序; backlash/rollers 模型上它把交错的被动关节过滤掉, 恢复 14-servo 视图.
- 已解释索引分两层: 模型原始 joint 数组的索引可能因模型变体变化; 任务逻辑使用 helper 返回的 servo-only 索引, 后续 `[:, joint_indices]` 才能继续使用项目约定的 0-13 servo 布局.
- 本段已完成源码阅读, 未运行仿真; 用户尚未完成按模型变体预测 helper 输出的练习.

### 2026-09-14 U2.2 追问: backlash 模型

- 用户正确回答 `_servo_joint_ids` 示例为 `0, 2, 4`, 并追问 backlash 模型.
- 已解释 backlash 是执行器齿轮/传动间隙的仿真, 不是新的动作或新的 RL 算法. 当前 Microduck backlash XML 给每个 servo 关节串联一个未驱动的 `passive_<joint>_backlash` hinge, 其范围为 `±1°` (总间隙约2°); 原 servo joint 表示电机输出侧, passive joint 表示电机与连杆之间的回差.
- 已说明 `BacklashEncoderBamActuator` 让反馈位置读取 `qpos[servo] + qpos[backlash]`, 模拟真实编码器位于齿轮间隙输出侧; `joint_pos_rel_backlash` 与 `joint_vel_rel_backlash` 也使用相同输出侧视角. 策略和动作维度仍为14, 但观测看到的是含回差的测量.
- 已区分基础任务与 backlash A/B 变体: `robot_walk_backlash.xml` 对应 Velocity walk, 其他任务按规则使用 groundcontact backlash 模型; 目的是检验同一策略/任务在执行器非理想条件下的鲁棒性, 不是把 passive 关节当作额外舵机.
- 本段为源码讲解, 未运行动力学实验; 回差下的实际运动曲线仍待实践.

### 2026-09-14 U2.2 总结与图解

- 用户要求对 U2.2 做总结. 已制作并保存图解 [U2.2 HOME_FRAME 与 backlash](../assets/u2-2/README.md), 以当前 walk/backlash XML 和源码为依据.
- 图解覆盖: `trunk_base_freejoint`、左腿/颈头/右腿 joint-body 链, 14D servo 顺序, `HOME_FRAME → resolve_expr → init keyframe` 数据流, 普通与 backlash 模型的观测/控制差异.
- U2.2 本轮用户复述证据: 能给出 `_servo_joint_ids` 的 `0,2,4`; 能说明 backlash 观测为 servo 与 passive backlash 的输出侧合成值, action 仍只控制主动 servo. “第一个匹配规则优先”已解释, 但此前换场景练习中曾把具体规则优先于声明顺序, 该细节保留为待巩固.
- 图解是结构/数据流示意, 导师已核对文件与源码; 未进行动力学或策略 rollout 实验.

#### 用户追问: 是否有两个观察模型

- 已澄清: 不是同一次运行中同时存在两套 policy observation model. 区分“物理模型内部的状态数量”和“观测函数输出给策略的张量”.
- 普通模型中每个 servo 只有一个 joint 状态; backlash 模型内部每个 servo 多一个 `passive_*_backlash` 状态, 但该 passive 状态没有独立动作.
- 在单个 backlash 任务中, actor/critic 的 `joint_pos` 与 `joint_vel` 观测项被替换为 `joint_pos_rel_backlash` / `joint_vel_rel_backlash`, 每个 servo 输出一个经过回差的值, 即位置约为 `qpos[servo]+qpos[backlash]`, 仍是14个关节观测而非两套观测.
- 因此是两种可分别运行的任务变体: 普通任务读取普通 joint view, `-Backlash-` 任务读取 output-side encoder view; 不是一条策略同时接收两份输入.

#### 用户复述确认

- 用户复述: 观测端看到 `left_hip_pitch + passive_left_hip_pitch_backlash` 的合成值, 控制端仍只控制 `left_hip_pitch`.
- 判定: 核心理解正确. 精确实现是 backlash joint position 之和再减去主动 servo 的 default joint position, 之后可能叠加 encoder bias/归一化; action 仍是14D并只绑定非`passive_`的 servo targets. passive backlash joint 不接收独立动作, 但受物理约束随输出链运动.
- 该点达到用户复述级证据.

#### U2.2 小练习结果

- 题目给定模式顺序为: `.*hip_pitch.*`, `.*_backlash$`, `.*left_hip_pitch.*`; 用户回答 `1, 3, 1`.
- A `left_hip_pitch` 判定正确, C `right_hip_pitch` 判定正确.
- B `passive_left_hip_pitch_backlash` 判定错误: 按当前 `resolve_expr()` 的先到先得规则, 第一个模式 `.*hip_pitch.*` 已经匹配, 因此结果是规则1, 不是规则3. 只有把 `.*_backlash$` 放到普通 `hip_pitch` 模式之前, 它才会优先得到0.0.
- 已指出本题的关键不是“谁更具体”, 而是“谁先匹配”; 这是 backlash 初始状态配置的实际安全条件.

#### 用户追问: `.*hip_pitch.*` 中的点为何不要求句号

- 已解释正则语法: `.` 是“任意一个字符”的通配符, `*` 表示前一个表达式重复零次或多次, 因此 `.*` 表示任意长度的字符序列, 也可以为空; 它不是要求实际名字中出现句号.
- 因而 `.*hip_pitch.*` 可匹配 `left_hip_pitch` (前面是 `left_`, 后面为空) 和 `passive_left_hip_pitch_backlash` (前面有 `passive_left_`, 后面有 `_backlash`).
- 若要匹配真正的句号, 需要写 `\\.`; 若要只匹配完整的 `left_hip_pitch`, 可写 `^left_hip_pitch$`.

### 2026-09-14 U2.3: 14D 策略动作、BAM 与 50 Hz/200 Hz 时间关系

- 用户指定继续 U2.3. 本节主问题: 一次 14D policy action 如何成为关节目标, 经过 BAM 得到实际力矩, 并在一段 policy interval 内推进机器人状态.
- 当前源码/实际安装依赖已核对: velocity 基础配置为 MuJoCo `timestep=0.005 s` 与 `decimation=4`, 所以 physics step 为 5 ms (200 Hz), environment step 为 20 ms (50 Hz). `ManagerBasedRlEnv.step()` 每个 environment step 只执行一次 `ActionManager.process_action()`, 随后四次循环 `apply_action() -> scene.write_data_to_sim() -> sim.step() -> scene.update()`.
- 当前 Microduck velocity factory 令 `JointPositionActionCfg.scale=1.0`, 并继承 `use_default_offset=True`. 因此 raw policy action 经过 `target = action * 1.0 + default_joint_pos`; `JointPositionAction.apply_actions()` 再减 encoder bias 后写入关节位置 target. 它不是直接给 MuJoCo 力矩, 也不是实际 qpos.
- 四个 physics substep 中, position target 保持为同一次已处理 action 的结果, 但 BAM 每个子步读取新的实际位置/速度, 重新计算固件位置误差对应的电压控制、直流电机力矩和摩擦预算; MuJoCo 将力矩与重力、接触、约束共同积分成下一时刻状态. 因而“同一个目标写四遍”不等于力矩不变.
- `MICRODUCK_WALK_ROBOT_CFG` 的 XL330 m6 BAM 当前配置 `delay_min_lag=3`, `delay_max_lag=6`, 单位为 physics timestep. 这表示 position target 进入 BAM 控制律前有 15--30 ms 的随机命令延迟, 约为 0.75--1.5 个 20 ms 的 policy interval. 这是训练中显式覆盖的执行链非理想性, 不可将 policy 输出误读成即时电机响应.
- 已保存 [U2.3 时间关系图解](../assets/u2-3/README.md). 该图为源码对应的流程/时间示意, 不是动力学实验或策略 rollout.

#### 小理解检查, 待用户回答

- 若 policy 刚在 t=0 给出一次新 action, 在下一个 policy 输出到来前, `apply_action()` 和 BAM `compute()` 各会运行几次? 该 action 会直接成为 torque 吗? 请用一句话回答.
- 本节未执行单关节动态实验、CPU MuJoCo 或训练; 用户理解尚待回答, 不记录为实践或独立掌握.

#### 用户追问: BAM 与 MuJoCo 的分工

- 用户提出的主线理解正确: policy 给关节目标, BAM 根据目标和当前关节状态形成执行器力矩, MuJoCo 结合环境物理得到新的实际状态. 已补正两个边界.
- 当前 `JointPositionAction` 的 raw action 是相对默认关节姿态的偏移, 因为 `scale=1.0` 且 `use_default_offset=True`: `position target = default_joint_pos + raw_action * 1.0`, 再按 encoder bias 修正后写入目标. 它不是世界坐标中的绝对位置, 也不是直接的 torque.
- BAM 是 XL330 m6 的电机/固件模型: 它从 position target 与当前 `qpos`/`qvel` 计算电压控制、反电动势、电机 torque 及关节内部摩擦预算. BAM 本身不积分关节运动, 也不单独把关节“执行到目标”.
- MuJoCo 是物理引擎: 每个 5 ms 子步将 BAM 给出的 torque 同重力、质量/惯性、接触、关节约束和环境摩擦一起求解并积分, 产生新的 `qpos`/`qvel`. 更精确地说, 它不是先判断某个“可到达终点”, 而是在每个很短的子步推进一点状态. BAM 的齿轮/电机摩擦预算也写给 MuJoCo 的关节摩擦约束; 脚与地面的摩擦则由 MuJoCo 场景 geom 的接触参数处理.
- 用户当前为概念澄清与提问, 尚未完成原定时间链小检查; 不升级为用户复述或实践证据.

#### U2.3 时间链复述确认

- 用户回答: “4次，不会”. 判定正确: 一个 20 ms policy interval 由四个 5 ms physics substep 组成, 因而 BAM `compute()` 在该 interval 内运行四次; raw action 先转换为 position target, 不会直接成为 torque.
- 该点达到用户复述级证据. 尚未完成单关节动态实验, 因此不记录为用户实践或独立操作能力.
- 下一个最小验证: 从稳定姿态给一个关节小幅 position target, 记录 target、`qpos`、BAM torque 与倾角/接触. 目的不是训练, 而是亲眼确认“目标恒定的四个子步中, 实际状态与力矩仍会变化”.

#### U2.3 最小动态验证, 导师执行

- 使用新增 `docs/learning/tools/probe_u23_bam_substeps.py` 复用 `scripts/infer_policy.py` 的 CPU MuJoCo/BAM 路径, 从 stand 初态给 `left_knee` 增加 `+0.050 rad` position target, 按四个 5 ms substep 记录结果. BAM 为训练同款 XL330 m6, `kp_fw=200`; 本次固定 `vin=7.4 V`、`vin_drop_gain=0.0`, 不引入随机命令延迟或 domain randomization, 以隔离执行器与物理积分链.
- 导师实测: target 在 5/10/15/20 ms 均为 `0.04510 rad`, 而 `qpos` 从 `-0.00467` 逐步到 `-0.00353 rad`, `qvel` 从 `0.0451` 升到 `0.0893 rad/s`, BAM torque 从 `0.02772` 降到 `0.02355 Nm`. 完整数据与范围说明见 [U2.3 图解](../assets/u2-3/README.md) 和 CSV.
- 结论限于这段 20 ms 的单关节响应: 固定 target 并不会冻结实际状态或 torque; 当前 qpos 越接近 target, 本次固件位置误差驱动的 torque 越小. 未据此断言完整机器人稳定、策略质量或延迟 DR 行为.
- 脚本成功及数据为导师实践, 不计为用户实践. 用户理解检查待回答: 为什么 target 不变时, 这四行 BAM torque 仍然下降?

#### U2.3 动态验证复述确认

- 用户复述两点: “action 50hz, 仿真 200hz，4个物理步内 target 都不会变”；“target 逐渐接近时，BAM 所需力矩就逐渐减小了”. 第一条正确, 对应一个 20 ms environment step 内的四个 5 ms physics substep.
- 第二条的因果理解正确, 措辞校正为: 固定的 target 被实际 `qpos` 逐步接近, 位置误差缩小, 因而 BAM 的固件位置控制所需 torque 下降. 不是 target 自己向 `qpos` 变化.
- 用户已达到本节“固定 position target 不等于固定 torque”的复述级证据. 当前 CPU 实验没有注入训练配置的 3--6 substep 命令延迟; 下一小节可专门追踪 delay buffer 如何使 BAM 在某些子步仍读取前一次 target, 再讨论为什么要做该 domain randomization.

#### U2.3: command delay buffer 的边界追踪

- 当前安装的 `DelayBuffer` 与 `Actuator.apply_delay()` 已核对. 每个 physics step, BAM position command 先 append 到缓冲区, 再从 `T-lag` 取出送进控制律. Microduck 配置 `delay_min_lag=3`, `delay_max_lag=6`, `delay_update_period=0`, `delay_hold_prob=0.0`; 因而它不是每 episode 固定延迟, 而是每物理步对每并行环境重新采样 3--6 步历史 target.
- 导师扩展 `probe_u23_bam_substeps.py`, 用固定 3 与固定 6 步作为可解释的边界对照: 写入新 target 后, lag=3 时 BAM 前三个子步读旧 target、第 4 子步在经过 15 ms 后首次读新 target; lag=6 时前六个子步读旧 target、第 7 子步在经过 30 ms 后才首次读新 target. 数据保存为 `assets/u2-3/u2-3-delay-boundaries.csv`.
- 这澄清两层 target: ActionManager 在一个 environment step 的四个子步都写同一个最新 target, 但 BAM 控制律可能仍读取历史 target. 随机 delay 的工程目的, 是避免 policy 依赖仿真中理想的即时动作响应, 从而对真实通信/总线/执行器时序不确定性更鲁棒.
- 用户理解检查待回答: 固定 lag=3 时, 新 target 在一个 20 ms environment step 的第几个子步首次进入 BAM 控制律? 固定 lag=6 时, 前四个子步会读到新 target 吗?

#### U2.3 delay buffer 复述确认

- 用户回答 “4、不会”. 判定正确: 固定 lag=3 时新 target 于第 4 个子步首次进入 BAM 控制律; 固定 lag=6 时前四个子步仍读取旧 target.
- 用户已达到“ActionManager 写入最新 target 与 BAM 实际读取延迟 target 可能不同”的复述级证据. 未完成用户亲手运行或更长 rollout, 因此不记录为实践.
- 下一小节: 归拢 U2.3 的完整单步链, 重点区分 action history、position target、delay 后 target、`qpos`、torque 与下一帧 observation; 再决定是否做用户操作的单关节实验.

#### U2.3: 从 action history 回到下一帧 observation

- 已核对当前安装的 velocity actor 配置与 `mdp.last_action()`: `actions` observation 项返回 `env.action_manager.action`, 即 ActionManager 保存的 raw policy action. `process_action(a_t)` 在缩放/默认姿态偏移前立即写入它; 四个物理子步结束、MuJoCo 状态更新后才计算 observation.
- 对普通未 termination/reset 的 environment step, `o_{t+1}` 中的 `actions` 项是刚刚输出的 `a_t`; `joint_pos`/`joint_vel` 是四个 5 ms 子步后的实际状态. `actions` 项不是 latest position target, 不是 DelayBuffer 实际取到的历史 target, 也不是 BAM torque.
- 已说明延迟下的观测解释边界: `qpos` 变化小不能单独说明 policy 未发动作, 因为 command delay、惯性、接触与摩擦都会影响短时响应. 若该步 terminated 且 auto-reset, 返回的是 reset 后 observation, action history 清零, 不按普通 `o_t -> a_t -> o_{t+1}` 链解释.
- 用户理解检查待回答: 在普通未 reset 的 `o_{t+1}` 中, `actions` 项记录的是 (A) policy 刚输出的 raw `a_t`, (B) BAM 延迟后实际使用的 target, 还是 (C) BAM torque? 若这帧 `qpos` 几乎未变, 能否仅据此断定 policy 没有发动作?

#### U2.3 action history 复述, 部分完成

- 用户选择 A, 正确: 普通未 reset 的 `o_{t+1}` 的 `actions` 项是 policy 刚输出的 raw `a_t`.
- “`qpos` 几乎未变能否直接断定 policy 没发动作”尚未回答, 保留为当前最小检查; 不把该点记为完整复述证据.

#### U2.3 action history 复述确认

- 用户回答 “不能，可能是 actions 延时了”, 正确. command delay 会使 policy 已输出并已记入 action history 的 raw `a_t`, 尚未成为 BAM 实际使用的 target; 即使 BAM 已读到新目标, 惯性、接触和摩擦也可能使 20 ms 内的 `qpos` 变化很小.
- 用户已达到本段复述级证据: 能区分 `actions` history、BAM 延迟 target 与短时物理状态, 不以单帧 `qpos` 变化判断 policy 是否发动作.
- U2.3 的阅读/导师排演主线已完整覆盖. 若用户继续, 下一步优先由用户亲手运行 `docs/learning/tools/probe_u23_bam_substeps.py`, 先预测固定 target、`qpos`、torque 与 delay 边界的输出, 再对照实际 CSV; 这把当前复述升级为操作证据, 不启动训练.

#### U2.3 图文总结

- 用户要求图文并茂地总结 U2.3. 已制作 [U2.3 一页总结 PNG](../assets/u2-3/u2-3-one-page-summary.png) 与可编辑 [SVG 源](../assets/u2-3/u2-3-one-page-summary.svg), 并嵌入专题图解首页. 图以当前源码与 CPU 四子步实测为依据, 清晰区分 raw action、latest position target、DelayBuffer 实际输出、BAM torque、MuJoCo `qpos`/`qvel` 和下一帧 action history; PNG 已目视检查字体、箭头、表格与数值可读.
- 图中下方数值仅来自此前导师执行的固定电压、无随机延迟 CPU 单关节实验; 上方 delay 模块描述训练的真实 3--6 步随机缓冲路径. 图不是策略 rollout、完整机器人稳定性实验或真机数据.

### 2026-09-14 U2 综合复习测验: 首轮 5 对 / 1 未答, 复测 2/2

- 用户请求回顾 U2 并进行高难测验. 导师依当前源码和既有 U2.1--U2.3 讲解范围建立 6 道换场景单选题, 不引入未讲过的 reward、PPO 或训练运维内容.
- 题目覆盖: 下一帧 observation 中 raw action 与实际状态的来源, `resolve_expr()` 的 first-match 规则, backlash output-side 观测与 14D action 合同, 固定 6 个 physics step delay 的 20 ms 边界, 固定 target 下 torque 的重算, 以及静止与模型自由度的区别.
- 用户回答 `B、C、B、B、B、E(不知道)`. 前五题均正确, 说明在新场景中能区分 raw action/延迟 target/实际状态/torque, 应用 first-match 规则, 并保持 backlash 的 14D action 合同. 选择题均为识别级证据, 不替代已有复述证据或独立操作.
- 第6题 `structure-1` 判为未答: 用户明确表示忘记 `nq/nv`. 已复习 `nq` 是广义位置向量长度, `nv` 是广义速度向量长度; 它们由 joint 结构而非当前静止状态决定. freejoint 贡献 `7 / 6`, hinge 各贡献 `1 / 1`.
- 用户提交复测 `C、C`, 对应“1 freejoint + 2 hinge”为 `9/8` 和“固定基座 + 4 hinge”为 `4/4`, 两题全对. `nq/nv` 主题本轮销项 (识别级): 维度来自 joint 结构, 不随静止、目标等于当前值或固定基座而消失.
- 题面与标准答案仅在 [U2 题库](../quiz/questions/u2-bank.md); 用户作答、判定与复测在 [本轮答题记录](../quiz/records/2026-09-14-u2-review-quiz.md). 本轮 U2 综合测验最终为首轮5对1未答、针对性复测2/2; 不据此升级为独立操作能力.
- 下一步: 若继续 U2, 由用户运行 `docs/learning/tools/probe_u23_bam_substeps.py`, 先预测四子步与3/6步延迟边界再核对输出, 将已通过的概念识别升级为操作证据.
