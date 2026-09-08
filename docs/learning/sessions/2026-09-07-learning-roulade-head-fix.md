# 2026-09-07: Roulade 头颈姿态修复实战学习 (奖励设计 × 训练运维)

- 日期: 2026-09-07.
- 性质: 实战学习记录, 不占正式课程编号. 主线课程 (Session 2 / U1.3) 另有接续点.
- 状态: 本节工程目标已完成 (新模型已出, 待用户视频观感确认); 概念讲解部分已讲未验收.
- 入口: 用户请求分析刚训练完的 roulade (前滚翻) 结果, 过程中用户自己发现 "站起来后头歪" 并追问当前学习策略与修复方案; 最终由用户自己跑完全新重训.

## 本节实际讲了什么

### 1. 训练结果分析的方法 (导师示范)

- 用 headless rollout (无 viewer) 按起始状态分组评测 checkpoint: 站立桶 / 翻滚中途 50-180° / 180-260° / 260-340°, 每桶 4096 envs × 3 轮 ≈ 1.2 万 episode.
- 关键指标: 旋转前沿达到 ≥330° 的比例 (判定翻滚完成), 过头顶 latch 率, 完成耗时, 站定段 z/倾角.
- 结论: 基线模型站立桶完成率 ~94.9%, latch 100%, 站定质量极高 (z≈114 mm, 倾角≈2°).
- 方法要点: Episode_Reward 指标是"全 episode 步数上的 raw×weight 均值"; episode 恒满 250 步 (无早终止), 不能用 episode length 判断成功; 成功率必须 rollout 实测.

### 2. Roulade 的当前学习策略拆解 (概念讲解, 已讲未验收)

- 三层结构: 行为编码层 (把"翻滚"写成可学信号) / 课程层 (数据分布) / 算法层 (PPO+对称损失).
- 关键机制: 势函数型进度 reward(奖励) 只为"旋转前沿增量"付费; 落地后的站立 annuity(持续年金) 由状态闸门 (旋转前沿 ≥260-330° + 头顶 latch) 控制; 反向课程 (从翻滚中途生成, 累加器预置到生成角) 解决后半程无数据; 风格税 (action_rate, arrival_damping, |a_z|) 在技能发现后才引入.
- 用户追问过的两个理解点:
  a. "收税型 vs 引导型" 奖励信号的差别: 税 = 偏离正位每步扣钱 (L1/L2), 精确但会压制过程中的必要运动; 引导 = 宽高斯接近分, 只买方向不买精度, 远端可能无梯度. 本问题没有唯一正确, 取决于想要的姿态容差和过程是否必需.
  b. 术语澄清: "battery" (一组测试) 被错误直译成"电池", 已向用户道歉并约定不再使用怪直译.

### 3. 问题诊断与修复 (导师设计并实施, 用户执行重训)

- 诊断 (实测): 站定后头颈 4 关节相对 HOME 停在极端姿态 — neck_pitch −69°, head_pitch −95°, head_yaw +128°, 100% 复现且零抖动. 原因: 落地奖励只量躯干高度/直立/腿关节, 头颈是无主自由度; 头占体重 38%, 策略把翻滚出口的头姿当配重冻结最省事.
- 修复方案: 新增 `roulade_home_pose_l1` — 全身 14 关节对 HOME 的 L1 税, 三重闸门 (翻滚完成 260-330° + 头顶 latch; 高度窗 0.095→0.112; 直立窗 20°→45°) 保证翻滚抱头与起身过渡期零税.
- 训练路径对比 (重要经验):
  - resume 路径 (w=1.0 → 2.5): 头基本回正但站立桶完成率掉了 ~4.7 点 (90.2%), 因为"站定后拧头"会扰动躯干, 策略选择少转最后一点.
  - 从头重训 (w=2.5): 完成率 95.7% (不输基线) 且头回正 — 从零学时"翻滚出口顺带回头"被学进落地动作, resume 继承的妥协没有出现.
- 结果: head_pitch/yaw/roll 回到 HOME ±1°, neck_pitch 残留 −10° (两条独立训练路径收敛到同一值 → reward 平衡点, 非训练不足).

## 源码与证据

改动文件 (均已测试, 未提交):

- `src/mjlab_microduck/tasks/mdp.py`: 末尾新增 `roulade_home_pose_l1` (自否定 → cfg 用正权重; 复用 `_roulade_completion_gate` 与高度/直立窗写法).
- `src/mjlab_microduck/tasks/microduck_roulade_env_cfg.py`: 常量 `_BODY_JOINTS` (14 servo 全集), reward 项 `roulade_home_pose_l1` weight=2.5, docstring 记录变更.
- `tests/test_roulade_cfg.py`: 新建 11 个回归测试 (符号/闸门/关节覆盖/61D obs/核心权重), pytest 全过.
- `AGENTS.md`: Chat style 一节补禁令 — 禁止把英文术语硬翻成怪中文 (battery → "一组测试", 不译"电池").

实验记录 (wandb 项目 mjlab_microduck):

- 基线 (歪头): run `cx8x7hnr`, 目录 `logs/rsl_rl/microduck_roulade/2026-09-06_00-47-10_.../model_9999.pt`, 站立桶完成 94.9%, 头颈偏差 −69/−95/+128°.
- resume w=1.0: run `yt8i7lif` (9999→10880, 导师执行), 头颈 → −11/+8/−4/−1°.
- resume w=2.5: run `ly6nbl1g` (10750→11549, 导师执行), 完成率 90.2%.
- 完全重训 w=2.5: run `4pkjvk56`, 目录 `logs/rsl_rl/microduck_roulade/2026-09-07_07-00-52_.../`, 8192 envs × 10000 iters, **由用户自己启动执行**, 完成后目录内有 ONNX 导出. 站立桶完成率 95.7%, 头颈偏差 −10/+0.6/−0.5/−0.7°, 结束站立率 100%.

评测脚本目前只在临时目录 `/tmp/opencode/` (eval_roulade.py, diag5_neck.py 等), 关键数字已写在本记录; 整理进 `scripts/` 是待办.

## 练习与理解检查

- 用户独立执行: 完全重训命令 (`uv run train Mjlab-Roulade-Flat-MicroDuck` 8192 envs, 无 resume 参数) + ONNX 导出. 命令与解释由导师提供, 用户自己选择环境数并跑通 → 记"已实践 (用户执行)", 不视为可独立设计实验.
- 用户提问: (a) 两种奖励信号差别 — 已再解释, 无验收; (b) "行为完成率电池" 术语质疑 — 已澄清并道歉. 两者均不代表掌握证据.
- 未完成理解检查: 无正式小练习布置; 主线 Session 2 的 U1.3 副本隔离验证题仍待用户回答 (状态未变).

## 卡点与变化

- 2026-09-05 曾搁置"play 中头部姿势不自然"; 今天用户主动回到同类问题 (roulade 站定头歪) 并推进到修复. 是否与旧搁置项同源未确认, 旧项状态保留.
- 剩余小问题: neck_pitch 恒留 −10° (qpos ≈ +10.8° vs HOME +20°). 提出过的后续手段 (未实施): 只对直流偏差收税 (1 s 滑动平均 L1, 让回正瞬态不计数). 是否继续由用户看视频后决定.
- 改动未提交 git; smoke 测试在 64 envs 通过, 全量 pytest 唯一失败与本改动无关 (本机 pip 走清华镜像 vs 测试断言 pypi.org).

## 停在哪里, 下次做什么

- Roulade 收尾: 用户用 `uv run play` 观看 `logs/rsl_rl/microduck_roulade/2026-09-07_07-00-52_microduck_roulade/model_9999.pt` 判断观感; 决定是否接受 neck −10° 或试 EMA-DC 方案; 若满意则谈导出/发布.
- 工程收尾 (可选): 把评测脚本整理进 `scripts/`, 提交本轮改动.
- 主线课程: 仍从 Session 2 / U1.3 的副本隔离验证题接续, 入口 `mjlab.tasks.registry.load_env_cfg` + `ManagerBasedRlEnvCfg`; 用户回答后转 U2.1 模型词汇. (注: 此条 2026-09-08 已过时 — U1 已通过复测验收, 双种子验证与行为评测见下方 2026-09-08 两节; 当前状态以 [progress.md](../progress.md) 为准.)

## 2026-09-08 接续: 第二种子 seed2025 对照 (已跑完)

用户按建议于 2026-09-07 23:47 启动同配置第二种子 (8192 envs x 10000 iters, seed 2025), 2026-09-08 12:28 完成. 目录: `logs/rsl_rl/microduck_roulade/2026-09-07_23-47-51_microduck_roulade/` (含 model_9999.pt 与 ONNX).

- 配置对照: 与 4pkjvk56 的 params diff 仅 seed 42→2025 (agent/env 各一处), 其余逐字段一致, 双种子对照成立.
- 曲线对照 (本机 tfevents 解析, 导师执行; 无 GPU 未做 rollout): 0-9000 迭代两 run 几乎重合; 新 run 于 iter 9440-9537 出现约 26 个迭代的瞬时 critic 发散 (Loss/value 峰值 4.05e7 @9522, Loss/symmetry 峰值 425), 期间 mean_reward 一度回落至 ~33-35, action_rate_l2 税放大; 之后自愈, 末 60 迭代 value loss 回到 0.013-0.019, mean_reward 37. 末 100 迭代关键项与 seed42 逐项一致: mean_reward 36.67 vs 36.77, roulade_progress 1.150/1.146, landing_composite 3.17/3.17, roulade_home_pose_l1 -0.095/-0.095, action_rate_l2 -0.41/-0.40. seed42 自身在 iter 9283 也有小尖峰 (value loss 5.2) 但未放大; dump 的最终 cfg 中未定位到对应阶段事件, 疑为种子相关的稀有数值事件.
- 判定: 双种子终点策略在全部可记录量上复现; 新 run 最终 model_9999.pt/ONNX 为自愈后状态导出, 可用. 站立桶完成率 95.7% 与 neck_pitch -10° 残差是否复现仍需 GPU rollout 实测, 尚未执行.
- 待办未变: 评测脚本整理进 `scripts/` (评估路径由用户选择).

## 2026-09-08 行为评测结果 (双种子, 新脚本 scripts/roulade_eval.py)

用户在新 GPU 环境用重建的 `scripts/roulade_eval.py` 分别评测两 run 的 model_9999.pt (stand/50-180/180-260/260-340 四桶, 12288 episode/桶):

| 桶 | seed42 完成率 | seed2025 完成率 | Δ |
|---|---|---|---|
| stand | 94.3% | 93.6% | -0.7pt |
| 50-180 | 92.4% | 92.1% | -0.3pt |
| 180-260 | 95.1% | 94.9% | -0.2pt |
| 260-340 | 96.4% | 95.9% | -0.5pt |

- 两 run 的 latch = completion (各桶逐项相等); mean_steps 完全一致 (stand 48, 50-180 51, 180-260 47, 260-340 44); stand_of_comp 均 100%; 站定 z 均 115.6-115.7 mm, tilt 0.9-1.1 度.
- 头颈残差 (绝对量, stand 桶): seed42 neck_pitch 10.0 度, head_pitch 0.8/yaw 0.8/roll 1.0; seed2025 neck_pitch 12.6 度 (其他桶 12.5), head_pitch 1.0/yaw 0.7/roll 0.8. 旧 diag 记录 4pkjvk56 为 -10/+0.6/-0.5/-0.7 (带符号), 与新脚本的 10.0/0.8/0.8/1.0 量级一致 → 工具口径互通.
- 结论: 修复效果双种子复现. 完成率桶间差 <=0.7pt (采样噪声约 ±0.3-0.6pt, stand 桶略超噪声边缘), 行为模式 (完成耗时/结束站立率/站定质量) 完全一致; neck_pitch 静态前倾残差稳定复现于 10-12.6 度, 具体平衡角有约 2.5 度种子间漂移 → 坐实"reward 塑形平衡点"解释, 残差是系统性产物而非欠训练或单种子运气. 旧评测记录的 95.7% 与新脚本 94.3% 差 1.4pt, 属工具口径小差异 (脚本按记录重建), 双模型同口径对比不受影响.
- 工具落地: 评测脚本从 /tmp 临时文件重建为 `scripts/roulade_eval.py` (中文注释, 半角标点; head 关节按名称解析; 复用 env._roulade_max/latch 与 default joint pos), "评测脚本整理"待办销项. 对比 CSV 在 /tmp/roulade_compare.csv (临时, 关键数字已抄录于本记录).
- 待用户决策: 视频观感确认后, 接受当前模型收尾 (发布排演), 或做可选 EMA-DC 直流税实验再压 neck 残差.
