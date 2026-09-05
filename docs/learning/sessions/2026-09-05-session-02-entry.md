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
