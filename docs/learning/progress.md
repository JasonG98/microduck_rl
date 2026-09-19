# 学习进度与接续状态

> 最后更新: 2026-09-15. 当前学习状态的唯一索引; 课程范围见 [plan.md](plan.md), 详细证据见各次记录.

## 学习者与教学偏好

- 背景: CV 从业者, 简单学过 RL 原理, 缺少 RL 项目实操, 对本项目整体结构还不熟悉.
- 目标: 系统理解工程结构, 调用链和开发流程, 最终自己写一套动作训练方案并训练出一个动作 (2026-09-05 用户明确). 导师建议复用现有 PPO/仿真基础设施, 独立完成任务定义, 奖励与终止, 训练和评估.
- 结课动作偏好: 用户要求有难度的新动作及开源简历价值, 已保留三个候选: 月球漫步/滑步舞、走梅花桩/精准踩点、推箱子转弯并停进目标区 (Push-T). 尚未选定唯一项目或实施顺序.
- 教学: 中文, 从职责和数据流讲起, 使用适量 CV 类比并指出边界. 不一次展开全部 Manager 或整份 `mdp.py`.
- 术语节奏: 每个新的源码变量或控制术语, 先说明它属于物理状态、传感器测量、控制目标还是软件中间变量, 说明来源和用途后再写公式; 不把一组未定义英文变量直接堆给用户 (2026-09-15 用户明确要求).
- 图文偏好: 真实模型标注图配简短解释, 图片/视频附入学习笔记并随仓库保存; 运动关系尽量配短动画. 树辅助说明包含关系, 不能替代模型图例. 2026-09-08 已按用户要求内化到 [导师 skill](../../.agents/skills/microduck-mentor/SKILL.md) 的“图文并茂的讲解”, 并同步媒体记录规则.
- XML 阅读偏好: 用树结构展示包含关系, 节点旁用短句解释作用; 不用 Python 示例或代码重写表达, 用户认为复杂且不直观 (2026-09-08 明确要求). 保留对象包含描述项的理解视角, 再按需解释物理含义.
- 节奏: 每次一个主问题, 小范围源码 + 小实践 + 简短理解检查. 允许追问, 跳转和暂停; 不强制考试或按固定周数推进.
- 持久化: 每次实质学习后更新项目文件, 跨电脑靠同步仓库接续, 不依赖聊天历史或导师的个人记忆. 用户要求只讨论时不抢先写记录或固化选题.

## 当前接续点

- 状态: 用户于 2026-09-15 明确开始 U3.1. 已按当前 velocity factory 核对 actor 61D: `base_ang_vel(3) + projected_gravity(3) + joint_pos(14) + joint_vel(14) + last_raw_action(14) + command(13)`, 其中 command 严格为 `[twist(3), head_pose(4), body_pose(6)]`. actor 不含 `base_lin_vel`; critic 另有真实线速度及足部接触类特权信息. 用户正确复述真机 actor 不应依赖 `base_lin_vel`, critic 为训练期价值估计使用它且不部署; 也正确说明未使用的 `body_pose` 应保留6D零槽位以维持统一输入合同. 两点均为用户复述级证据. 详见 [Session 3 U3.1](sessions/2026-09-15-session-03-u3-observation.md).
- 当前小节: U3.2 已讲解, 并应用户要求扩展为一次完整20 ms step 的六种量、时间线和 reset 边界. 随后已重新从“物理状态、传感器测量、控制目标、软件中间变量”四类解释 `qpos`、encoder、encoder bias、HOME/default、raw action、scale、processed action、position target、延迟 target 和 torque; 用户正确定义 position target. 还解释了 actor 并不天然记住上一轮输出, `actions` 14D 是为执行器延迟提供的一步 raw action history, 并回答了为何不盲目加入多帧 history: 当前61D接口、15--30 ms delay量级、学习负担和训练部署一致性共同限制该选择. 用户已正确概括它补充历史信息, 并经澄清达到复述级: 它具体补的是上一帧的控制意图, 而非完整历史状态. 已核对 raw action 每20 ms保存一次、`scale=1.0`、default joint position offset、encoder bias 校正、每20 ms的四次 target 写入及 `actions` observation 返回 raw action. 用户尚未进行 rollout 或训练. 详见 [Session 3 U3.2](sessions/2026-09-15-session-03-u3-observation.md#u32-14d-raw-action-如何成为关节目标).
- 当前小节: U3.3 已讲解. 用户对“完全匹配+直立+action差平方和1”的20 ms reward 初答 `0.06`, 随即说明是把动作变化惩罚误看为 `-0.02`; 已讲评正确单步值为 `+0.078`, 因为 `-0.1 * 0.02 = -0.002`. 用户正确将20秒时间上限分类为 `truncated`, 达到结束类型的用户作答级证据. 奖励算术的十倍差异留待读 W&B 分项时复习. 已从原始分数、weight、20 ms单步贡献、episode、termination/reset 等基本词汇讲解速度跟踪、直立与 action-rate 三项, 并核对当前 `terminated`/`truncated` 条件和 `step()` 的执行顺序. 未运行训练或 rollout. 详见 [Session 3 U3.3](sessions/2026-09-15-session-03-u3-observation.md#u33-这一-step-得多少分-为什么结束).
- 当前小节: U3.4 已讲解. 用户对并行 reset 和课程后 CoM 抽样的前两问作答正确: reset 只传 env 17, env 18 history/CoM 不会被连带重置, env 17 在刚跨过12,000共享步后使用±5 mm范围。对 interval push 将 `0.01 s` 误作100 ms; 已讲评它是10 ms, 因此紧接的20 ms step中 env 18 会过零并受推, env 17 则已重抽3--6 s计时器。用户追问 CoM, 已从真实 walk 模型图解、质量加权定义、静态支撑投影、动态行走限制和当前 `dr.body_ipos(trunk_base)` 的实际语义详细讲解。已从当前 EventManager 和 `_reset_idx(env_ids)` 源码核对: startup、reset、interval 是三种软件触发时机; reset event 只触及结束环境的索引行; push 默认每环境独立计时; CoM 和 BAM 摩擦 DR 均非累积; curriculum 用共享 `common_step_counter` 改写 live manager 配置而非旧的 `env.cfg`. 未启动训练或 rollout. 详见 [Session 3 U3.4](sessions/2026-09-15-session-03-u3-observation.md#u34-eventcurriculum-谁在什么时候改变哪一台机器人) 和 [CoM 图解](assets/u3-4/README.md).
- 当前小节: U3.5 已讲解. 用户对100条 forward episode 的三问表示暂不清楚, 已由导师示范: timeout 不能自行等于成功、只统计存活轨迹会产生幸存者偏差、checkpoint 对比需固定测试条件/随机性协议/统计协议。该示范不构成用户理解证据。已建立评估条件卡: checkpoint/配置、完整13D command、初态、DR/push/seed 协议和记录规则; 已区分 training reward、评估指标与成功判据; 已说明当前 auto-reset 会令 returned observation 变为下一回合初态, 终止状态须借助 done/terminal 记录保存。当前 velocity 指令范围、现存 `model_4998.pt` 与 `scripts/infer_policy.py` 的部署排演边界均已核对。详见 [Session 3 U3.5](sessions/2026-09-15-session-03-u3-observation.md#u35-训练前评估-先规定怎么判-再看策略表现).
- U3 复习: 第一轮4题用户作答 `B C B C`, 4/4正确, 覆盖 actor/critic 边界、terminal transition、action history、评估幸存者偏差, 为识别级证据。用户明确的出题偏好: 保留选择题选项, 但出题阶段不公布正确选项; 用户作答后再讲评。答题记录见 [U3 复习第一轮](quiz/records/2026-09-15-u3-review-round-1.md).
- U3 复习: 第二轮用户作答 `A A C B`, 2/4正确。action/encoder bias及并行reset/interval正确; reward的20 ms时间缩放与 `body_ipos`/整机CoM区别错误, 已定位为两个独立卡点, 待换数值/换场景复测。答题记录见 [U3 复习第二轮](quiz/records/2026-09-15-u3-review-round-2.md).
- U3 复习: 针对性复测用户作答 `B B`, 2/2正确。reward 的 `raw*weight*dt` 与局部 `body_ipos` 对整机CoM的质量加权影响均通过换场景复测, 第二轮两个卡点销项。结合第一轮4/4和第二轮其余2题正确, U3.1--U3.5 本轮抽查范围达到识别级通过。答题记录见 [U3 针对性复测](quiz/records/2026-09-15-u3-review-remedial.md).
- 下一步动作: U4.1 一次 PPO iteration 如何收集4096个环境各24步, 写入 rollout buffer, 再进行优势估计与网络更新。U3 的实际 rollout/基线评估仍待用户动手, 不因本轮答题通过而标为已实践。
- 下一步动作: 用户先计算给定 `left_knee` raw action/default/bias 时的 `actions` observation 与写入 position target, 并说明二者为何不同. 之后进入 U3.3 的 reward/termination 计算时机; 这能把 U2 的执行链与 U3 的数据链闭合, 且不需要启动训练.

### 2026-09-14 历史接续信息

- 状态: 2026-09-14 先做 U1/U2.1 主线复习. 初次复述检查 1 对 / 1 错 / 1 未答, 补答后两个卡点均完成用户复述级确认: CLI 覆盖发生在环境构造之前; 61D 是 actor observation 输入合同, 不是整个环境所有维度. U2.1 首批4/5、维度复测2/2, 错题已销项 (识别级). U2 综合高难测验首轮 5 对 / 1 未答, `nq/nv` 针对性换场景复测 2/2, 主题销项; 题库与答题记录见 [quiz 目录](quiz/README.md). SitStand run 尚未确认.
- 最近记录: [Session 2](sessions/2026-09-05-session-02-entry.md) 文末“2026-09-08 收课总结”; 更早含 U1 复习/测验/复测与收工总结; [2026-09-07 roulade 实战学习](sessions/2026-09-07-learning-roulade-head-fix.md); quiz 相关见 [quiz 目录](quiz/README.md); 选题仍见 [新动作项目选题](sessions/2026-09-05-learning-project-options.md).
- 当前正式课程: Session 3; Session 1 补充与 2026-09-07 实战记录不算正式课程编号.
- 当前小节: U2.3 BAM/控制频率阅读与导师排演已完成. 已核对 velocity 的 5 ms physics timestep、`decimation=4`、20 ms environment step, 以及 XL330 m6 BAM 的 3--6 物理步命令延迟; 已澄清 BAM 是电机/舵机模型而 MuJoCo 负责物理积分. 用户正确复述“20 ms 内 BAM 计算4次, action 不直接成为 torque”, 并能解释固定 target 下 `qpos` 靠近目标使 torque 下降. 已完成 delay buffer 源码与 3/6 步边界排演, 用户正确判断 lag=3 在第4子步读新 target、lag=6 的前四步仍读旧 target. 用户正确识别 action history 为 raw `a_t`, 并能以 command delay 解释 `qpos` 短时不变, 达到 U2.3 当前讲解范围的复述级证据. 图解/CSV/复现脚本见 `assets/u2-3/README.md`; 用户尚未操作实验. U2 综合测验首轮 5 对 / 1 未答, `nq/nv` 换场景复测2/2, 本轮识别级主题均已销项.
- 下一步动作: 先回答“一次 policy action 在一个 20 ms environment step 内发生什么”的一句复述, 再以单个关节小幅目标变化记录 target、qpos、torque、倾角/接触. 先做这一最小动态验证, 因为它能把源码中的目标、执行器响应和物理状态分开观察; 不启动训练.
- 下一步动作: 定位用户今晚实际 run (建议后缀 sitstand_wandb_lesson, project=mjlab_microduck), 先核对4096环境/6000迭代/seed42及是否结束, 再结合真实曲线讲总奖励、episode长度、坐站任务奖励、平滑与上升/下降惩罚分项、课程权重、loss, 用实际视频交叉判断. 若没运行成功先查状态, 不假设已有数据.
- 为什么: 用户明确要求“明天教我看 wandb 的数据”; 用户已跑过 velocity, 改用 SitStand 双向坐站任务; 有可核对的分项与500–2500迭代权重阶段, 适合从真实训练曲线理解日志. 训练命令及限制见最新 session 文末, 导师未启动训练.
- 理解检查: U1 识别级复测 5/5 全对 (见 quiz 记录); 2026-09-14 复习中 qpos 与 actuator 目标区分正确, 经补答确认 CLI 覆盖发生在环境构造之前, 61D 是 actor observation 输入合同而非整个环境所有维度, 两点达到用户复述级证据. U2.1 用户主动指出间接连接中可以有 joint, 记局部推理证据; 原题直接父子限定不清, 不判错; 删除 freejoint 题用户表示不知道, 已讲解; 用户自主用对象/属性类比准确表达 freejoint 属于 trunk_base 的描述项, 有结构复述证据; site 已有用户 ROS/TF 类比复述证据; 首批选择题对连接关系、geom碰撞、gyro角速度、目标/实际状态已有识别级证据; nq/nv换场景复测2/2全对, 本次抽查范围通过 (识别级), 独立建模/操作仍未验收.
- 暂停点: 用户已去休息, 明日回来再开 W&B 读图课; 不后台监控或自动开课. U2.2 延后保留. 2026-09-05 的 play 头部观感问题仍搁置; roulade 行为级评测已复现, 4pkjvk56 视频观感确认与可选 neck 残差微调由用户决定何时继续, 不自动推进.

## 能力状态

“已讲解”不表示已掌握; “已实践”需说明谁执行; “可独立完成”需用户复述/操作证据. 未验收不自动阻止用户进入下一节.

| 范围 | 当前状态 | 证据与限制 |
|---|---|---|
| smoke test | 已实践 (历史记录) | [Session 1](session-01.md) 记录 64 环境/5 迭代已跑通; 不据此判定掌握全部配置 |
| 长训练与 play | 已实践 (用户报告 + 本地核对) | 用户提供 model_4998.pt 和 play 命令; 见最近记录; 独立复现能力未验收 |
| PPO/GAE/损失/课程概念 | 已讲解, 待验证理解 | [Session 1 补充](session-01-supplement.md); 不默认从零重讲或已可独立解释 |
| U1 项目地图与入口 | U1.1/U1.2/U1.3 已讲解; 首测 6/10 + 复测 5/5 (识别级) | [Session 2](sessions/2026-09-05-session-02-entry.md) 已由导师核对启动链并实测配置副本隔离 (2026-09-06); 2026-09-07 换方式复习 + 两轮测验: 复测覆盖注册求值时机/CLI 跨进程隔离/分段权重表/采样单位算术/任务族词汇, 全对; 尚无文字复述级证据 |
| U2 机器人与仿真 | U2.1/U2.2/U2.3 已讲解; U2 综合题与针对性复测均通过 (识别级) | U2.2 有用户复述级证据: backlash 观测为 servo 与 passive backlash 的输出侧合成值, action 只控制 servo; first-match 在综合题中答对. U2.3 已有时间链、delay 与 action history 的用户复述级证据; 综合题相关项答对. `nq/nv` 经换场景复测2/2销项; 用户尚未完成动态实验. 图解见 [U2.1](assets/u2-1/README.md)、[U2.2](assets/u2-2/README.md)、[U2.3](assets/u2-3/README.md). |
| U3 环境交互与基础评估 | U3.1 已讲解, 用户复述级; 待实践 | 已核对 velocity actor 的 61D 分块、command 固定顺序、critic 特权项及 actor 噪声/延迟与归一化的区别; 用户能说明 critic-only `base_lin_vel` 与未使用槽位零填充的部署原因. 见 [Session 3 U3.1](sessions/2026-09-15-session-03-u3-observation.md). 尚无用户跟踪一步交互或完成基础评估的证据 |
| U4 短训练闭环与受控实验 | 部分已讲解/实践 | 2026-09-07 用户独立执行完全重训 (roulade, 8192 envs × 10000 iters, run 4pkjvk56) 并导出 ONNX — 命令与解释由导师提供, 记"已实践 (用户执行)", 独立设计实验未验收; 见 [2026-09-07 记录](sessions/2026-09-07-learning-roulade-head-fix.md) |
| U5 独立任务设计与项目实现 | 部分概念已讲解, 未系统开始 | 2026-09-07 讲解奖励设计原则 (势函数进度、状态闸门、自否定符号约定、税 vs 引导、resume vs 从头重训), 未验收; 三个候选已保留, 尚无独立任务定义或新环境实现 |
| U6 部署排演与开源交付 | 未系统开始 | run 目录有 ONNX 不等于用户掌握导出; 尚无用户排演验收 |

## 当前问题与复习队列

| 项目 | 状态 | 后续处理 |
|---|---|---|
| 项目各层职责与入口关系 | U1 已通过识别级复测 | U2.2 以配置到模型的映射接续, 不重启 U1 |
| U2.1 模型词汇 | 首批4/5 + 维度复测2/2, 错题已销项 | 本次抽查范围识别级通过, 保留历史答案, 不重发已通过题; 独立建模/操作未验收 |
| `nq/nv` 结构维度 | 综合测验未答, 换场景复测2/2通过 (识别级) | 已能在不同 freejoint/hinge 规模中按结构计数; 独立建模/操作仍未验收 |
| 独立动作项目的选题 | 三个候选已由用户保留, 最终项目未定 | 月球漫步/滑步舞、精准踩点、Push-T; 下次比较最小成功版本. 尚未开展专项实验或训练 |
| iteration / env step / physics step / episode | 已通过选择复测 (识别级) | 复测 rq4 全对 (12288 transition/迭代 vs 49152 物理子步); U4 再以真实配置举例巩固 |
| 任务 ID/任务族命名 (velocity, roulade, Flat/Rough, MicroDuck) | 已讲解并复测通过 (识别级) | 复测 rq5 全对; U2.2 项目映射时顺带确认命名与模型对应 |
| 课程完成与行为质量的区别 | 已说明, 未验收 | U4/U5 回顾, 不自动追加训练 |
| 历史笔记的指标解释 | 待按当前源码核对 | 小 value loss 不足以证明 critic 准确; surrogate 为负不单独证明训练健康; termination 日志需核对聚合方式, 不能直接当百分比 |
| roulade 站定后头颈歪 | 已修复且双种子行为评测复现 | 修复 run 4pkjvk56 (seed 42) 与新 run (seed 2025) 的 model_9999 经 scripts/roulade_eval.py 同口径评测: 各桶完成率差 <=0.7pt (stand 94.3 vs 93.6), mean_steps 完全一致, 结束站立率均 100%, 站定 z 115.6 mm / tilt ~1 度; neck_pitch 残差 10.0 度 vs 12.6 度 (head 其余 <=1 度) → 残差为 reward 塑形平衡点 (有约 2.5 度种子漂移), 非欠训练. 待用户视频观感确认与收尾决策 (见 [2026-09-07 记录](sessions/2026-09-07-learning-roulade-head-fix.md)) |
| play 中头部姿势不自然 (2026-09-05 用户搁置) | 搁置 | 与 roulade 问题是否同源未确认; 用户主动恢复时才处理 |
| 评测脚本整理 | 已完成 (2026-09-08) | roulade 分桶评测重建为 `scripts/roulade_eval.py` (headless, GPU, 复用训练记账口径); 双种子对比 CSV 为临时文件, 关键数字已抄入 [2026-09-07 记录](sessions/2026-09-07-learning-roulade-head-fix.md) |

## 历史与产物索引

- 2026-09-02: [Session 1 项目概览与 smoke](session-01.md).
- 2026-09-03: [Session 1 补充: PPO, 损失, 课程, GAE](session-01-supplement.md).
- 2026-09-05: [学习体系重整与导师工作流初始化](sessions/2026-09-05-learning-setup.md), 属于学习管理记录, 未消耗新课程编号.
- 2026-09-05: [Session 2 训练命令与任务入口](sessions/2026-09-05-session-02-entry.md), U1.1/U1.2, 理解检查待回答.
- 2026-09-05: [新动作项目选题](sessions/2026-09-05-learning-project-options.md), 用户确认保留三个候选并授权记录, 未确定实施顺序.
- 2026-09-07: [roulade 头颈修复实战学习](sessions/2026-09-07-learning-roulade-head-fix.md) — 奖励设计概念讲解 + 头颈姿态诊断与修复 (未占正式课号). 相关 run: 基线 `cx8x7hnr`, resume `yt8i7lif`/`ly6nbl1g`, 用户全量重训 `4pkjvk56`; 改动 (mdp/cfg/tests/AGENTS.md) 未提交.
- 2026-09-07: 面试拷问模式落地 (导师 skill 内化 + 文档规整): [U1 题库](quiz/questions/u1-bank.md) (15 题, 不记个人成绩) 与 [两轮测验答题记录](quiz/records/2026-09-07-u1-quiz-rounds.md) (首轮 6/10 + 复测 5/5); 约定与索引见 [quiz/README.md](quiz/README.md). 题库/记录与 session 记录分离存放, 逐题细节只在 quiz 目录.
- 2026-09-08: [Session 2 U2.1 模型词汇](sessions/2026-09-05-session-02-entry.md) (从头重讲已覆盖全部词汇, 已收课; 局部用户证据与未答项见文末总结).
- 2026-09-08: [U2.1 真实模型图解与短视频](assets/u2-1/README.md), 六张标注图 (含 site、sensor、actuator、qpos/qvel)、12 秒 MP4/动图、[生成器](tools/render_u21.py); 导师制作并检查, 运动学示意而非策略/动力学实验.
- 2026-09-15: [Session 3 U3.1: policy 的 61D 输入](sessions/2026-09-15-session-03-u3-observation.md), 从当前 velocity 配置核对 actor/critic observation 与 command; 用户对 critic-only `base_lin_vel` 和 6D 零填充完成复述级确认, rollout 尚待完成.
- 本地 run: `logs/rsl_rl/velocity/2026-09-03_23-57-23_velocity/`, checkpoint: `model_4998.pt`, wandb run ID: `vcpcqllk`.
- 上述训练大文件受 `.gitignore` 忽略, 同步 Git 不保证它们存在. 接续只依赖本文和笔记; 复现时再检查本机产物或可访问的远端.

## 机器与执行限制

不绑定固定用户名, 绝对路径, GPU 型号或 `.venv` 位置. 换电脑后按当前目录定位仓库; 需要执行练习时再检查 Python/uv/依赖和 GPU. 2026-09-05 导师工具环境曾检测到 CUDA 不可用, 这是当次观察, 不代表用户电脑永久无 GPU.

## 课程安排

2026-09-05 按用户要求更新课程: 保留 U1-U6 编号与已有学习证据, 每单元加入动手成果; U3 提前建立评估, U4 完成短训练和受控实验, U5 加入独立任务设计与三个选题分支, U6 补齐开源交付. 详见 [plan.md](plan.md). 本次仅修改学习文档, 不新增正式课编号、不提升能力状态. 文档聚焦有效安排, 不重复保存废弃方案.

## 更新约定

先记录当次事实, 再更新接续点和能力状态. 新掌握结论要链接会话证据; 用户自述和导师实测分开写. 只查进度且无新信息时不创建空记录. 若索引与较新记录不一致, 依据可核对记录修正并说明, 不凭空补课.
