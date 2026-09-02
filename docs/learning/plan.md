# Microduck RL 学习计划

> 背景: CV 工程师, 懂 RL 原理, 无 RL 工程经验
> 目标: 掌握本项目的代码结构和开发流程

---

## 第一阶段: 环境跑通 (第 1 周)

### 1.1 跑通训练流程 ✅
- [x] 运行 smoke test: `uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5`
- [ ] 理解终端输出的每个字段含义
- [ ] 用 `play` 命令查看训练结果

### 1.2 跑通完整训练
- [ ] 运行 1000 步训练 (~1 分钟): `uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 256 --agent.max_iterations 1000`
- [ ] 在 wandb 上查看训练曲线
- [ ] 用 `play` 观察训练好的步态

### 1.3 跑通测试
- [ ] 运行 `uv run --with pytest pytest tests/ -v`
- [ ] 理解测试在验证什么

---

## 第二阶段: 代码结构理解 (第 2 周)

### 2.1 核心文件阅读
- [ ] `src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py` — 行走任务配置
  - [ ] ENABLE_* 开关区域 (第 1-94 行)
  - [ ] make_microduck_velocity_env_cfg() 工厂函数 (第 193-925 行)
  - [ ] 奖励配置区域 (第 274-348 行)
  - [ ] 观测配置区域 (第 521-608 行)
  - [ ] 课程学习区域 (第 753-922 行)

- [ ] `src/mjlab_microduck/tasks/mdp.py` — 自定义 MDP 函数
  - [ ] 辅助函数: _servo_joint_ids, _servo_joint_pos (第 148-175 行)
  - [ ] Reset 函数: reset_with_forward_velocity (第 177-247 行)
  - [ ] 奖励函数: upright_progress, height_progress (第 554-602 行)
  - [ ] 终止函数: fallen_too_long, robot_state_is_nan (第 911-982 行)

- [ ] `src/mjlab_microduck/robot/microduck_constants.py` — 机器人配置
  - [ ] HOME_FRAME 姿态 (第 83-106 行)
  - [ ] BAM 执行器配置 (第 129-148 行)
  - [ ] EntityCfg 定义 (第 167-265 行)

### 2.2 架构理解
- [ ] mjlab 的 Manager 架构: Event/Command/Action/Observation/Reward/Curriculum
- [ ] PPO 训练流程: Collection → Learning → Logging
- [ ] sim2real 方案: DR + BAM + ONNX 导出

### 2.3 关键概念
- [ ] 61 维观测布局: 48 本体感知 + 13 维命令
- [ ] 14 维动作空间: 14 个舵机关节
- [ ] 课程学习: 逐步增加难度
- [ ] 域随机化: 让 sim 策略迁移到 real

---

## 第三阶段: 其他任务理解 (第 3 周)

### 3.1 站立恢复任务
- [ ] 阅读 `microduck_velstand_env_cfg.py`
- [ ] 理解摔倒→恢复的奖励设计
- [ ] 理解 `_fallen_mask` 和 recovery 相关函数

### 3.2 站起任务
- [ ] 阅读 `microduck_standup_env_cfg.py`
- [ ] 理解从倒置到站立的奖励设计

### 3.3 坐站切换任务
- [ ] 阅读 `microduck_sitstand_env_cfg.py`
- [ ] 理解命令式二状态切换

### 3.4 Backlash 变体
- [ ] 阅读 `tasks/backlash.py`
- [ ] 理解齿轮间隙仿真
- [ ] 理解编码器反馈穿过的实现

---

## 第四阶段: 动手实践 (第 4 周)

### 4.1 修改奖励函数
- [ ] 修改一个奖励权重, 观察训练效果变化
- [ ] 添加一个新的惩罚项

### 4.2 添加新观测
- [ ] 向 actor obs 添加一个新维度
- [ ] 验证 ONNX 导出后维度正确

### 4.3 运行 Backlash 变体训练
- [ ] 对比 Backlash 和非 Backlash 的训练差异
- [ ] 理解 sim2real 中齿隙的影响

### 4.4 部署排练
- [ ] 用 `scripts/infer_policy.py` 在 CPU MuJoCo 中测试导出的 ONNX
- [ ] 理解命令槽的写入方式

---

## 里程碑

| 周次 | 里程碑 | 验证方式 |
|------|--------|----------|
| 1 | 能独立运行训练和 play | 训练 1000 步, play 看到机器人动 |
| 2 | 能解释 velocity 配置的每个区域 | 能画出数据流图 |
| 3 | 能解释任意任务的奖励设计 | 能预测修改某个权重的效果 |
| 4 | 能修改配置并训练 | 修改后训练, 效果符合预期 |
