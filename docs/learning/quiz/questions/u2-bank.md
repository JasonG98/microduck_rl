# U2 机器人与仿真题库

> 仅存题面、答案和考点, 个人作答见 `../records/`. 范围限 U2.1--U2.3 已讲内容; 首次使用前已按当前源码、模型和导师排演记录核对. 本文件不记录成绩.

## chain-1: 下一帧 observation 的两种"历史"

第 1 题｜某个未 reset 的环境在时刻 t 输出 raw policy action `a_t`. 该 action 经默认姿态偏移后写成 latest position target, 但 BAM 此刻由于 command delay 仍从缓冲区读旧 target. 四个 physics substep 后形成 `o_{t+1}`. `o_{t+1}` 的 `actions` 项和 `joint_pos` 项分别最准确地是什么?

- A. 延迟后的旧 target; 四子步后的实际 `qpos`
- B. raw `a_t`; 四子步后的实际 `qpos`
- C. raw `a_t`; latest position target
- D. BAM 本步平均 torque; 延迟后的旧 target
- E. 不知道

答案: B.
考点: `actions` observation 保存 ActionManager 的 raw action, 不是 position target、延迟 buffer 输出或 torque; `joint_pos` 是物理推进后的实际状态. 题目限定未 reset, 排除 reset 后 action history 被清零的分支.
核对来源: `docs/learning/sessions/2026-09-05-session-02-entry.md` 的 U2.3 action history 追踪; `microduck_mdp.last_action()`.

## order-1: HOME_FRAME 的 first-match 陷阱

第 2 题｜`resolve_expr()` 按 joint 名称逐个处理, 并按字典声明顺序采用**第一个**匹配规则. 若 `joint_pos` 顺序如下:

1. `r".*hip_pitch.*": 0.10`
2. `r".*_backlash$": 0.00`
3. `r"^left_hip_pitch$": -0.40`

在带 backlash 的模型上, `left_hip_pitch` 与 `passive_left_hip_pitch_backlash` 会分别得到什么初始值?

- A. `-0.40`, `0.00`
- B. `0.10`, `0.00`
- C. `0.10`, `0.10`
- D. `-0.40`, `0.10`
- E. 不知道

答案: C.
考点: 规则具体程度不改变优先级. 两个名字都先匹配规则1; 若要先把 backlash 关节置零, 规则2 必须排在宽泛的 `.*hip_pitch.*` 之前.
核对来源: `Entity._add_initial_state_keyframe()` 使用的 `resolve_expr()` 行为; `BACKLASH_HOME_FRAME` 的配置意图.

## backlash-1: 输出侧测量, 不是第 15 个 action

第 3 题｜backlash 变体中, 某一 servo 的电机侧角度是 `qpos[servo] = 0.30 rad`, 串联的 `passive_<joint>_backlash` 角度是 `-0.01 rad`. 忽略 default position、encoder bias 和归一化. 策略通过 output-side encoder observation 看到的该关节位置以及它能直接输出的控制量分别是?

- A. 看到 `0.30 rad`; 分别给 servo 和 passive joint 各一个 action
- B. 看到 `0.29 rad`; 只给 servo 一个 action
- C. 看到 `-0.01 rad`; 只给 passive joint 一个 action
- D. 看到 `0.29 rad`; 给 15D action, 最后一个专管 backlash
- E. 不知道

答案: B.
考点: output-side 位置由 servo 与 passive backlash 的和形成; passive joint 是未驱动的物理间隙, 不增加独立 action. 14D policy/servo 合同保持不变.
核对来源: `src/mjlab_microduck/actuator/friction_dr_bam.py` 的 `BacklashEncoderBamActuator`; `src/mjlab_microduck/tasks/mdp.py` 的 `joint_pos_rel_backlash`.

## delay-1: 20 ms 内不一定已经执行新命令

第 4 题｜一个 policy interval 为 20 ms, 含四个 5 ms physics substep. 刚写入新的 latest position target 后, 为解释边界暂时固定 BAM command delay 为 6 个 physics step. 到这个 interval 结束时, 哪项一定成立?

- A. BAM 已在第 4 子步读到新 target, 因而 `qpos` 必然已明显变化
- B. BAM 的四次控制律都仍读旧 target; 但 `qpos` 仍可能因旧 target、接触和惯性而变化
- C. ActionManager 没有保存 `a_t`, 因为 BAM 尚未读到它
- D. BAM 完全不运行, 要等第 7 个子步才第一次运行
- E. 不知道

答案: B.
考点: lag=6 时新 target 首次被 BAM 读取在第7个子步, 即经过30 ms; BAM 仍在每个 5 ms 子步运行, 只是控制律读历史 target. `qpos` 从不单由最新 action 决定.
核对来源: `docs/learning/tools/probe_u23_bam_substeps.py` 的固定 lag 3/6 边界实验; velocity XL330 BAM 延迟配置.

## torque-1: 固定 target 不代表固定 torque

第 5 题｜无 command delay 的单关节演示中, 四个 5 ms 子步的 position target 都保持不变. 实际 `qpos` 每步更接近该 target, `qvel` 不为零, 也没有饱和或限幅切换. 对 BAM torque 的最合理预测是?

- A. 必须严格不变, 因为 target 没变
- B. 通常会随位置误差缩小而下降, 但仍由当步状态和执行器模型共同决定
- C. 必须在第 4 子步变为 0, 因为刚好过了一个 policy interval
- D. 一定上升, 因为关节速度变大
- E. 不知道

答案: B.
考点: BAM 每个 physics substep 根据 target 与当前 `qpos`/`qvel` 重算控制、电机与摩擦. 固定 target 只固定输入之一, 不冻结 torque 或物理状态.
核对来源: `docs/learning/assets/u2-3/u2-3-substeps.csv` 的导师单关节实测; BAM/MuJoCo 分工追踪.

## structure-1: 静止不会移除自由度

第 6 题｜一个模型有 1 个 freejoint 和 3 个 hinge. 此刻机器人被接触约束卡住, 所有 `qvel` 数值都为零; 之后只是在数据里把这 3 个 hinge 的目标写成当前角度, 没有删除或新增任何 joint. 此时 `nq/nv` 是多少?

- A. `3 / 3`, 因为自由基座没有运动
- B. `10 / 9`, 因为 freejoint 贡献 `7 / 6`, 三个 hinge 各贡献 `1 / 1`
- C. `10 / 10`, 因为所有 joint 都以角度和角速度成对保存
- D. `0 / 0`, 因为当前完全静止
- E. 不知道

答案: B.
考点: `nq/nv` 是模型结构维度, 与当前速度值、位置目标或接触是否暂时卡住无关. freejoint 贡献位置3+四元数4和线速度3+角速度3.
核对来源: U2.1 的 freejoint 维度实测与 `robot_walk.xml` 的 1 个 freejoint/14 个 hinge.

## structure-2: 换规模的自由基座计数 (复测 R1)

第 7 题｜一个新的练习模型有 1 个 freejoint、2 个 hinge, 没有其他 joint. 它正在静止地悬在空中, 因此所有 `qvel` 的数值都是0. 此时 `nq/nv` 是多少?

- A. `2 / 2`
- B. `8 / 8`
- C. `9 / 8`
- D. `9 / 9`
- E. 不知道

答案: C.
考点: freejoint 仍贡献 `7 / 6`, 两个 hinge 各贡献 `1 / 1`; 当前静止只改变 `qvel` 的数值, 不改变数组长度. 因而为 `(7+2) / (6+2) = 9 / 8`.
核对来源: MuJoCo freejoint 的 qpos/qvel 布局; U2.1 已核对的 `robot_walk.xml` 结构规律.

## structure-3: 固定基座不是“零维度” (复测 R2)

第 8 题｜另一个练习模型固定在世界中, 只有 4 个 hinge, 没有 freejoint. 四个 hinge 的位置目标都恰好等于当前角度, 所有 `qvel` 的数值为0. 此时 `nq/nv` 是多少?

- A. `0 / 0`
- B. `4 / 0`
- C. `4 / 4`
- D. `11 / 10`
- E. 不知道

答案: C.
考点: 每个 hinge 始终贡献一个广义位置与一个广义速度分量. position target 等于当前值、静止或固定基座都不会移除它们; 只有实际删除 joint 才会改变维度.
核对来源: U2.1 的 hinge/freejoint 维度规律; 此题为换规模场景, 非项目模型改动.
