# Session 2: 训练命令如何找到 Microduck 任务?

- 日期: 2026-09-05.
- 单元/小节: U1.1/U1.2.
- 状态: 进行中, 等待用户完成小验证.
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
