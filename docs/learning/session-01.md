# 学习笔记

## Session 1: 项目概览与首次训练

> 日期: 2026-09-02
> 时长: ~1 小时
> 内容: 项目结构概览 + 首次 smoke test

---

### 1. 项目是什么

为 Microduck 双足机器人 (~800g, ~25cm) 构建的 RL 训练环境。

- **训练框架**: mjlab (MuJoCo Warp) + PPO (rsl_rl)
- **策略维度**: 61 维 actor obs → 14 维 action
- **部署方式**: 训练 → ONNX 导出 → 真实机器人运行
- **核心目标**: sim2real transfer — 仿真中训练, 真实中部署

### 2. 目录结构

```
src/mjlab_microduck/
├── robot/                        # 机器人 MJCF 模型 + 配置
│   ├── microduck/                # MJCF 文件 (robot_walk.xml 等)
│   └── microduck_constants.py    # 机器人配置, HOME 姿态, BAM 执行器
├── actuator/                     # BAM 执行器 + 摩擦 DR
├── tasks/
│   ├── __init__.py               # 任务注册 (13+ 个任务)
│   ├── mdp.py                    # 所有自定义 MDP 函数
│   ├── backlash.py               # 齿隙变体包装器
│   └── microduck_*_env_cfg.py    # 每个任务族一个配置文件
├── train_cli.py                  # train 入口 (委托 mjlab)
└── train_hook.py                 # --hf-jobs 拦截
```

### 3. 训练流程

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5
```

执行链:
1. `train_cli.py:main()` → mjlab 的 `train` 入口
2. 加载 `mjlab_microduck.tasks` → 注册所有任务
3. 创建 `ManagerBasedRlEnv` → 初始化所有 Manager
4. PPO 训练循环: Collection (24步) → Learning (5 epochs) → Logging

### 4. 输出关键字段

#### 物理参数
```
Number of environments: 64
Physics step-size: 0.005      # MuJoCo 仿真 200Hz
Environment step-size: 0.02   # 策略执行 50Hz
```
每控制步 = 4 物理步 (0.02/0.005=4)

#### BAM 执行器
```
[BamActuator] model='m6' joints=14 kt=0.3660 R=2.8114 vin=range=(6.5, 8.2)
```
真实 XL330 舵机的电压控制律, 不是理想 PD。

#### 观测空间
```
Actor obs (61 维):
  base_ang_vel (3) + projected_gravity (3) + joint_pos (14) + joint_vel (14)
  + actions (14) + command (3) + head_command (4) + body_command (6)

Critic obs (76 维): 多 15 维特权信息
  base_lin_vel (3) + foot_height (2) + foot_air_time (2)
  + foot_contact (2) + foot_contact_forces (6)
```

#### 奖励函数 (16 个)
```
track_linear_velocity  +2.0    速度跟踪
track_angular_velocity +2.0    角速度跟踪
upright                +2.0    直立
pose                   +1.0    接近 HOME 姿态
body_ang_vel           -0.05   抑制角速度
action_rate_l2         -0.1    动作平滑
air_time               +3.0    脚悬空时间
head_pose_tracking     +2.0    头部姿态跟踪
...
```

#### 训练指标
```
Mean reward: 0.15              总奖励
Mean episode length: 18.67     平均 episode 长度
Mean action std: 1.00          动作标准差 (初始=1.0)
Mean value loss: 0.0366        Critic 损失
Mean surrogate loss: -0.0330   PPO 策略损失
Mean entropy loss: 19.8564     策略随机性
```

#### Termination
```
time_out: 0.5833               58% 超时结束
fell_over: 0.2083              21% 摔倒
```

#### Curriculum (课程进度)
```
action_rate_weight: -0.1       动作平滑权重 (初始)
standing_envs: 0.02            站立环境比例 2%
com_range: 0.003               质心随机化 ±3mm
```

### 5. 每次迭代的数据流

```
Collection (采样):
  for step in range(24):
      obs(61) → Actor网络 → action(14)
      action → BAM → MuJoCo × 4 物理步
      新 obs, reward, done → buffer
  总数据: 64 × 24 = 1536 条

Learning (学习):
  1536 条 → 4 mini-batches (384/batch)
  每 batch: 5 PPO epochs
  计算: surrogate_loss, value_loss, entropy_loss
```

### 6. 关键设计决策

1. **61 维观测共享**: 所有任务 (行走/站立/恢复) 共享同一观测格式 → 策略热插拔
2. **被动关节命名 `passive_*`**: 所有选择器用 `^(?!passive_).*` 排除
3. **BAM 执行器**: 建模真实电压控制, 不是理想 PD
4. **ONNX 导出烘焙归一化器**: 必须用 `scripts/export.py`, 不能手转
5. **负权重 = 惩罚**: `action_rate_l2` 权重 -0.1 意味着动作变化越小越好

### 7. 已跑通的命令

```bash
# Smoke test (5 步, 64 环境)
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5
# 耗时: ~3 秒
# wandb run: https://wandb.ai/jayguo-/mjlab_microduck/runs/eqbupglt
```

### 8. 下次学习目标

- [ ] 用 `play` 命令查看 5 步训练的结果 (基本是随机动作)
- [ ] 跑 1000 步训练, 观察步态从随机到行走的变化
- [ ] 开始阅读 `microduck_velocity_env_cfg.py` 的配置细节

---

## 参考资源

- mjlab 文档: https://github.com/mujocolab/mjlab
- AGENTS.md: 本项目的 AI agent 工程规范
- README.md: 项目概览和快速开始
- wandb 项目: https://wandb.ai/jayguo-/mjlab_microduck
