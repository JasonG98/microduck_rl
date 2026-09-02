# agent\_cn.md

> 本文件是 [AGENTS.md](AGENTS.md) 的中文镜像, 最后同步日期: 2026-09-02.
> 若 `AGENTS.md` 有实质性变更, 请手动同步更新本文件.

Microduck 的 RL 训练环境 — 一台约 800 g, 约 25 cm 高的双足机器人, 含 14
个 Dynamixel XL330 舵机 — 基于 [mjlab](https://github.com/mujocolab/mjlab)
(MuJoCo Warp) 和 PPO (rsl\_rl) 构建. 策略在此处以 50 Hz 训练, 导出为 ONNX,
然后由 `pollen-robotics/microduck` 仓库的 runtime 部署到真实机器人上.
Sim2real 迁移是整个仓库的核心目标: 下面每条约定之所以存在, 都是因为打破它
曾产出"在 viewer 里正常, 一上硬件就挂"的策略.

## 本地化约定

- **代码注释, docstring, 以及终端输出 (print/log 消息, raise 异常消息,
  argparse help/description) 必须使用中文.**

- **代码注释与 docstring 中的标点必须使用英文半角标点**
  (`,`, `.`, `:`, `;`, `?`, `!`, `"`, `'`, `(`, `)`), 绝不使用中文全角标点
  (`，`, `。`, `：`, `；`, `？`, `！`, `""`, `''`).

- 本文件的英文源文件 `AGENTS.md` 是权威的工程参考, 刻意保留为英文.
  中文镜像单独维护在 `agent_cn.md` (当 `AGENTS.md` 有实质性变更时按需同步).

- `docs/superpowers/plans/` 和 `docs/superpowers/specs/` 下的历史设计存档
  冻结为原始语言, 不做翻译.

- 标识符, wandb 指标 key (如 `Episode_Reward/...`, `Curriculum/...`),
  env/term 名, regex 模式, 文件路径, 以及其它内部字符串 key 保持英文;
  仅翻译用户可见文本.

## 常用命令

```bash
uv run list-envs                                    # 实时任务注册表
uv run train <TASK_ID> --env.scene.num-envs 4096    # 训练 (加 --hf-jobs 用 Hugging Face Jobs)
uv run train <TASK_ID> --env.scene.num-envs 64 --agent.max_iterations 5   # 冒烟测试 — 必须先跑
uv run play <TASK_ID> --wandb-run-path <entity/project/run_id>
uv run scripts/export.py <TASK_ID> --wandb-run-path <...>   # → ONNX (烘焙 obs 归一化器 — 必经之路)
uv run scripts/infer_policy.py --walking out.onnx   # CPU MuJoCo 部署演练
uv run --with pytest pytest tests/
```

64 环境 × 5 轮的冒烟测试能以极低成本抓住约 95% 的配置错误.
千万不要不做冒烟测试就启动长训练.

## 仓库地图

- `src/mjlab_microduck/tasks/mdp.py` — 全部自定义 MDP 函数
  (奖励, 事件, 观测, 指令, 课程). 新增函数统一放这里, 按任务分组.

- `src/mjlab_microduck/tasks/microduck_*_env_cfg.py` — 每个任务族一个 cfg 模块.
  `microduck_velocity_env_cfg.py` 是主行走配方, 同时也是其它 env 建立或镜像时的
  共享基底 (robot, DR, obs, commands).

- `src/mjlab_microduck/tasks/__init__.py` — 任务注册 (基础任务 + `-Backlash-` 变体).

- `src/mjlab_microduck/tasks/backlash.py` — 将任意 env cfg 包装成其 backlash 孪生版.

- `src/mjlab_microduck/robot/microduck_constants.py` — 机器人 cfg, HOME frame,
  BAM 执行器 cfg.

- `src/mjlab_microduck/robot/microduck/` — 来自 Onshape 的 MJCF 导出
  (onshape-to-robot, 每个模型一个 `config_mjcf_*.json`) + 场景 +
  `add_backlash.py`.

- `src/mjlab_microduck/actuator/friction_dr_bam.py` — BAM 执行器 + 摩擦 DR

  - backlash 编码器.

- `scripts/` — 导出, 推理, sim2real 对比, wandb 辅助脚本.

- `tests/` — 跨 cfg 不变量测试与 mdp 函数回归测试 (CPU 即可, 不需要 GPU).

## 不变量 — 绝不要破坏这些

- **Obs 布局 (actor) 为 61 维且全策略家族共享**, 以便 runtime 中热插拔策略:
  48 维基础本体感觉 + 13 维指令块 `[twist(3), head_pose(4), body_pose(6)]`,
  顺序固定. 不使用某指令槽的 env 必须对其**零填充**
  (保留 obs 项, 采样极小范围) — 绝不要删除任何槽位.

- **关节布局** (14 个舵机, walk/allcollisions 模型上 ctrl idx = joint idx):
  0-4 左腿 (hip\_yaw, hip\_roll, hip\_pitch, knee, ankle),
  5-8 颈部/头部 (neck\_pitch, head\_pitch, head\_yaw, head\_roll),
  9-13 右腿. 在 roller/backlash 模型上, 被动关节**交错插入** —
  绝不要在 mdp 函数里硬编码关节索引; 用 mdp.py 中的
  `_servo_joint_ids` / `_servo_joint_pos` 辅助函数
  (普通模型上为恒等映射, 其它模型上全部正确).

- **未驱动关节全部命名为** **`passive_*`** (轮子, backlash 铰链).
  每个执行器/obs/奖励选择器都使用 `^(?!passive_).*` — 添加关节时务必保留
  前缀约定, 新写的 `passive_` regex 绝不能意外匹配 backlash 关节
  (用 `^passive_.*wheel`, 不要用 `^passive_.*`).

- **执行器为 BAM** (电压控制的 XL330 模型, 摩擦由执行器自行计算).
  两个后果: 任何独立 env cfg 必须注册 `expand_bam_friction_fields` startup
  事件; 关节摩擦 DR 必须缩放执行器的 `friction_scale` — 在 BAM 下
  `dof_frictionloss` 被清零, 所以随机化它是一个静默的空操作.

- **Obs 归一化是开启的** → 归一化器必须被烘焙进 ONNX.
  `scripts/export.py` 会处理; in-sim play 会掩盖这个 bug (它反正会应用
  归一化器), 所以绝不要手动转换 checkpoint.

- **策略是未滤波的** (训练中不做 action 低通). 不要在没有匹配的 runtime
  flag 和迁移测试的情况下加入 EMA 滤波 — 训练时带/部署时不带
  (任一方向) 都会破坏迁移.

- **Domain randomization 不得跨 reset 累积.** mjlab 1.3.0 的 `dr.*` 操作
  在 `operation="add"/"scale"` 下天然不累积 (它们每次重编译时默认值) ;
  自定义 DR 函数必须"先还原再应用". 曾经有一个累积式 CoM 随机化器,
  让所有长训练退化达数月.

- 如果某个 obs 被重新映射到 sensor 视图 (backlash 编码器, bias), 则针对
  同一量的跟踪奖励也必须测量同一视图 — 否则策略会因"纠正它看到的东西"
  而被惩罚.

- `-Backlash-` 任务变体必须镜像其基础任务的机器人模型
  (walk / allcollisions / rollers), 这样 backlash A/B 对比才没有混淆变量.

## 构建新 env — 工作流程

1. **挑最接近的模板** 然后在它之上构建, 不要从零开始:
   运动 → velocity 配方; 以姿态结束的 episodic 特技 → standup;
   指令式二态切换 → sitstand; 动态机动 → roulade
   (读它的 cfg docstring — 它编码了一段 5-run 的教学弧线).
   在 `make_mjlab_velocity*_env_cfg` 上构建可以免费保持
   DR / obs / noise / delays 同步; 如果从 mjlab 基础模板独立构建,
   就必须自己移植整套 DR + obs-noise + NaN-guard 栈
   (grep velocity 接了哪些线: `_safe` critic obs 项, 带 sensor\_names 的
   `nan_state` 终止, `expand_bam_friction_fields`, encoder bias,
   IMU 失准).
2. **训练之前先在 sim 中验证物理假设** — 这是省时间最多的一步:

   - 目标/静止姿态必须是稳定平衡点: 从带噪声的初始状态保持 ctrl 3 s,
     然后检查**倾斜**, 而不仅是高度
     (只记录 z 的 settle 测试会把倒下状态也报告为"静止良好").

   - 在 sim 中从真实机器人上测量目标高度
     (例如站立策略下 trunk 的 z 值), 绝不要跨模型版本搬运.
     曾经有一个 5 mm 偏差的 STAND\_Z, 把目标变成了几天都达不到的不可能值.
3. **配置约定**: cfg 文件顶部放 `ENABLE_*` 开关 + 调谐过的常量;
   工厂函数 `make_..._env_cfg(play: bool, rough: bool)`;
   在 `tasks/__init__.py` 注册 (+ 如有必要加入 `_BACKLASH_TASKS` 表);
   拥有独立 `RslRl...RunnerCfg`, 使用不同的 `experiment_name`.
   提供了对称性镜像损失 (symmetry.py 中的 61D 表) — 默认关闭,
   绝不要用于不对称任务.
4. **写 cfg 测试** (见 `tests/test_*_cfg.py`): 关节索引能在实际模型上解析,
   奖励权重具有预期的符号, gate 在预期位置开/闭. 这些测试跑在 CPU 上,
   锁定不变量.
5. **冒烟测试** (64 envs, 5 iters): 构建通过, step 无 NaN, obs 为 61 维,
   每个奖励项都能计算, ONNX 可导出.
6. 训练, 观察日志 (见下), 并预期会有 2-5 轮"奖励漏洞打地鼠" — 这是正常的,
   下面的经验教训能绕过大部分.

## 奖励设计 — 每条都是踩过坑后总结的硬规则

- **符号约定 (四个 env 的坑):** mdp.py 里有两种惩罚风格. mjlab 原生的 cost
  函数返回 ≥ 0 → 权重取负. microduck 自带反号的函数
  (`*_penalty`, `*_l1` 返回 ≤ 0) → 权重取正. 对自反号惩罚用负权重就是
  双重否定, 变成"奖励违规", 策略会去刷它
  (屁股跳, 摔倒坐姿等). **绝对可靠的检查: 每次运行中 wandb 里每个**
  **`Episode_Reward/<penalty>`** **都必须 ≤ 0.**

- **RL 优化的是奖励的字面含义.** 每个指定不足的自由度都会被利用
  (用弹道甩代替滚, 用肩滚代替矢状面滚, 用头三脚架代替站立).
  要用严格的基于状态的 gate (支撑接触, 朝向轴检查, latch) 编码
  "什么才算这个动作", 而不是用小惩罚去"提醒".

- **不要 jackpot:** 任何"到达 X"的奖励必须限速率或做 slew 跟踪.
  早早到达一个每步持续给分的目标状态就是 jackpot, 它会购买任意暴力.
  对指令式过渡, 跟踪一个 slew 的内部目标 (恒速混合) — 超过斜坡不算分,
  所以"慢"才是 argmax. 单独的速度上限惩罚只能积出一个有界成本, 会输掉.

- **绝不要把正向奖励 gate 在坏状态上** (倒下, 低高度) — 策略会停在
  成本最低的合格姿势里刷分. 改用基于势的 shaping 代替
  (付 Δprogress, 例如 Δcos(tilt): 上升得分, 保持得零, 无法刷).
  对静止任务, 针对每个稳定的跌倒姿态
  (仰摔/脸摔/侧摔) 审计每个正向项: 如果跌倒还能拿到大部分堆叠,
  策略就会选择跌倒.

- **Episodic 姿态着陆任务:** 从 t=0 单一固定目标
  (关节和高度的 Gaussian + L1, std 放宽) + |a\_z| 冲击惩罚 + 两层 upright —
  **不要**用关键帧/航点轨迹 (策略会扎在航点上不走). 路径本身才是 RL
  应该发现的东西.

- **正则化分为两类.** Motion-blocker (body\_ang\_vel, angular\_momentum,
  pose std) 惩罚的是动态运动物理上本就需要的东西 — 动态任务把它们设**低**.
  Smoothness (action\_rate, joint\_torque\_rate) 抑制抖动而不阻碍慢的大动作
  — 权重可以放心加, 但要在技能发现**之后**引入 (课程从 \~0 开始):
  在难技能还在探索时就有尝试税, 会让"什么都不做"胜出.
  慢而精细的任务 (到达) 比行走需要更大的 smoothness 权重.

- **跨 env 复制正则化时比较奖励质量, 而非权重.**
  PPO 看到的是相对优势: 同一个 action\_rate 权重在 4 倍大的正向任务栈下
  会弱 4 倍.

- **跟踪 Gaussian 的 std:** 设为"你仍在意的误差", 不是最大误差 —
  太松会在小误差处没梯度. 但在收紧前先问问: 这个误差是策略能消除的,
  还是你要的行为里固有的
  (占体重 38% 的头部走路时必然振荡; 过紧的瞬时头跟踪 std 曾把行走
  压得策略干脆站着不动). 只为"可消除的部分"标价 — 例如 1 s EMA 的 L1
  罚的是直流偏置, 振荡会相互抵消.

- **在目标状态处乘法组合优于加法和:** 当加法和有一个"折中盆地"
  (靠一个 lean 姿势拿到每项 80%) 时, Gaussian 的乘积会在任一项不合格时
  整体崩塌 — 但要把 std 选得足够宽, 让当前策略能得分, 否则梯度不可见
  就什么都不会变.

- **关节停在硬限位上:** 用 qpos 侧的限位近距惩罚作用于涉事关节;
  原生的 `dof_pos_limits` 只在最后 \~7.5% 范围内触发, 而指令侧惩罚无效
  (宽 ctrlrange 是故意的 — 低 kp 舵机需要过冲).

## 指令, 观测, 死权重

- **永远非零的指令输入会永远死权重.** 每个指令槽从 step 0 起就保留一个
  小的非零采样范围 (即使奖励权重为 0), 这样它的输入神经元才能为后续
  课程保持活性.

- **零指令行为必须被显式训练** (`zero_command_prob` 风格的精确零采样):
  均匀采样本质上永远不会产生全零指令, 而那正是部署时的空闲状态.

- 稀少但重要的指令区域需要显式分桶 — 例如 turn-in-place
  (`rel_turn_in_place_envs`): 独立均匀采样让旋转只占 \~2% 经验,
  永远学不会.

## 课程 (Curricula)

- 步数指 env step: `iteration × 24` (`NUM_STEPS_PER_ENV = 24`).

- 用经验证的拆分: `microduck_mdp.reward_weight` 用于权重调度,
  一个独立的 params-curriculum 用于指令/事件范围.
  `mdp.reward_weight` 是阶梯函数, 不是插值 — 把斜坡离散化为阶段.

- 通过管理器变更 term cfg (`env.event_manager.get_term_cfg(...)`),
  不要直接写 `env.cfg.events[...]` — 管理器在 init 时会 deepcopy 它们的 cfg,
  所以写 `env.cfg` 是静默的空操作 (这也坑过强行生成状态的 eval 脚本).

- **每个阶段都要与策略实际学到的阶段对齐:** 在当前切片还没巩固之前不要
  收紧 spawn 混合; 在技能不存在之前不要引入税. 当一个 wandb 指标恰好在
  课程阶段边界下降时, 说明节奏错了 — 拉长阶段或推迟引入, 绝不要提前.

- 反向课程 spawn (在机动的中途开始 episode, 包括接近完成处) 是
  "学会了开头, 永远学不完最后一英里"的可靠解法 — 否则前沿永远
  得不到 on-policy 数据.

## 训练运维与读 run

- wandb project: `mjlab_microduck`; 日志: `logs/<experiment_name>/`;
  续训: `--agent.load-checkpoint model_XXXX.pt --agent.resume True`.

- 每轮观察: 平均奖励上升 **且** episode 长度符合任务需求;
  每个惩罚项 ≤ 0; **主任务项在实际增长**
  (总奖励可能完全靠正则化上升而特技根本没发生).
  `Episode_Reward/<term>` 记录的是**乘过权重的值** — 权重为 0 的项
  无论行为如何都是 0, 所以要结合权重计划来解读.

- 预算: 简单的 episodic 特技约 4096 envs × 1000 iters;
  步态和课程密集的恢复任务需要 4000-6000.

- **先测量再理论化.** 当一个 run"失败"时, 在改奖励之前先对实际 checkpoint
  跑一次无头 eval (按 spawn 类型的压力测试, 终态聚类, 角速率剖面):
  过去的"失败"最终证实分别是: checkpoint 太早, 成功标准一刀切开了
  一个行为簇, 以及奖励上限与实测物理冲突. sim 指标可能通过, 视频却
  通不过人眼 — 要**同时看视频并检查哪个 geom/轴在接触**.

- 汇报 rollout 实际表现 ("能滚但 1/3 概率脸砸地"), 不要说"成功了!".
  什么时候够好由用户决定.

## Sim2real 坑 (每个都耗过现实中的数周调试)

- 一次全新的 `uv sync` 是地面真值 (HF Jobs 会跑一次):
  任何只靠手动装包才能工作的东西到了远程都会死. 保持
  `pyproject.toml` 的诚实.

- **Wheel 是按架构区分的.** 在 linux-`aarch64` (DGX Spark / GB10) 上
  PyPI 的 torch wheel 是纯 CPU 的 (`2.9.1+cpu`,
  `torch.version.cuda is None`), 所以 `torch.cuda.device_count() == 0`,
  mjlab 的 `select_gpus()` 会索引空列表 → 在第 0 轮之前就 `IndexError`.
  `[tool.uv.sources]` 只为 `aarch64` 把 torch 路由到 cu129 索引
  (cu129 与 warp 带的 CUDA toolkit 匹配; x86\_64/HF Jobs 保持 PyPI).
  两个静默断点, 都由 `tests/test_aarch64_cuda_torch.py` 锁死:
  torch 必须保留为**直接依赖**
  (uv 只对直接依赖应用 `[tool.uv.sources]` — 删掉看似多余的
  `torch==` pin 会让路由变成空操作); 而且 pin 必须保留 `==`,
  因为 CUDA 索引带的构建比 PyPI 更新 (用 `>=` 会静默地把 torch
  2.9.1 拖到 2.13.0).

- 物理对齐的速度限制: 25 cm 的机器人自然翻滚就在 3.5-5.5 rad/s —
  不要用上限强加人类尺度的速度直觉; 把反暴力压力放在冲击和抖动上
  (|a\_z|, action\_rate, support gate), 而不是转速.

- IMU DR 是零中心的 — 它训练的是对失准幅度的容忍度, **不能**补偿
  系统性的安装偏置 (那是 runtime 标定的工作).

- 真实部署通过共享 obs 契约热插拔 ONNX 策略 (walk / stand / trick) —
  在碰机器人之前先用 `scripts/infer_policy.py` 演练, 写入正确的指令槽
  (姿态 flag 位于 twist vx 槽; 喂全零意味着"站立", 看起来就像
  "策略忽略了按钮").

