# U3 环境交互与评估题库

题库只存题面、正确答案和考点, 不记录个人作答或成绩。范围限于已讲的 U3.1--U3.5; 数值和执行顺序按 2026-09-15 当前 velocity 配置与安装的 mjlab 源码核对。

## U3-R1: 真机 actor 的观测边界

一台真机运行 walking ONNX 时, runtime 可读取 IMU、关节编码器、上一帧 policy raw action 和13D command, 但没有可靠的机身线速度真值。下面哪项处理符合当前项目的 actor/critic 分工?

- A. 将模拟器 `base_lin_vel` 直接添加到 actor, 因为 critic 训练时也使用它。
- B. actor 保持不含 `base_lin_vel`, critic 可在训练期使用该特权真值。
- C. 删除 actor 的 `body_pose` 六维槽, 因为当前 walking 中它的 reward weight 为0。
- D. 用零填充 actor 的所有 proprioception, 只留 command, 避免真机传感器噪声。

正确答案: B。

考点: actor 必须只依赖部署可获得的信息; critic 特权观测不导出; 61D 共享合同中的未重点使用 command 槽位保留而非删除。

## U3-R2: 一条 terminal transition 的时间边界

某个环境完成一个20 ms控制步后刚刚摔倒。当前环境开启 auto-reset。下列哪个顺序最接近实际 `step()` 数据流?

- A. reset该环境 → 计算reward → 判定termination → 返回新observation。
- B. 计算新observation → 判定termination → reset所有4096个环境 → 计算reward。
- C. 判定termination → 计算终止这一步的reward → 只reset done环境 → 返回其新回合observation。
- D. 计算reward → 将摔倒环境标为 `truncated` → 不reset, 等下一次policy调用。

正确答案: C。

考点: terminal reward 属于旧 episode; auto-reset 后返回 observation 可属于新 episode; 摔倒是 `terminated` 而非 timeout `truncated`。

## U3-R3: action history 补充的不是完整过去

两个环境当前观测到的关节位置与速度相同, 但一个环境上一控制步发出了“膝盖增加”的 raw action, 另一个发出了“膝盖减少”的 raw action; BAM 的延迟队列尚可能执行旧命令。`actions` 14D 最直接补上的信息是什么?

- A. 完整历史 qpos/qvel 序列, 因此能够精确重建所有过去状态。
- B. 上一帧 policy 的控制意图, 帮助区分短时未来响应可能不同的控制情形。
- C. 当前 BAM 实际 torque, 因此 policy 可以直接控制电流。
- D. critic 的 `base_lin_vel`, 因此 actor 获得真实世界速度。

正确答案: B。

考点: 前馈 actor 不天然记住自己上次输出; 14D history 是 raw action hint, 不泄露精确延迟队列也不是完整历史状态。

## U3-R4: 评估中为何不能只看 timeout

两个 checkpoint 都在100个固定前进测试中得到90条 `truncated`、10条 `fell_over`。checkpoint A 的存活轨迹速度误差更小, 但它的10条跌倒都发生在第2秒; checkpoint B 的10条跌倒都发生在第19秒。最合适的结论是?

- A. 两者成功率相同, 因而行为质量完全相同。
- B. A一定更好, 因为存活轨迹速度误差更小。
- C. 不能只凭 timeout 数量或幸存轨迹误差下结论; 应同时报告终止类型、首次跌倒时间、全体轨迹的误差定义及预先规定的成功判据。
- D. 10条跌倒都应从统计中删除, 以免干扰平均值。

正确答案: C。

考点: timeout 不自动等于成功; 幸存者偏差; 固定条件卡与行为级指标。

## U3-R5: raw action 与编码器偏差校正

当前 velocity 的 `left_knee` 默认位置是 `-0.0049 rad`, action scale 是1.0。某一步 policy 对该关节输出 raw action `0.0`, 当前模拟 encoder bias 是 `+0.0100 rad`。下列哪项同时正确描述下一帧 `actions` observation 与写入的位置目标?

- A. `actions=0.0`, target=`-0.0149 rad`。
- B. `actions=-0.0049 rad`, target=`-0.0049 rad`。
- C. `actions=+0.0100 rad`, target=`+0.0051 rad`。
- D. `actions=-0.0149 rad`, target=`0.0 rad`。

正确答案: A。

考点: actor history 保存 raw action; processed action 为 raw*scale+default; 位置目标还会扣除 encoder bias; target 不是 qpos。

## U3-R6: reward 的20 ms时间缩放

某个20 ms environment step中, 一个速度跟踪项的 raw value 为 `0.5`, weight 为 `+2.0`; 同时 action-rate 项的 raw value 为 `2.0`, weight 为 `-0.1`。忽略所有其他项, 两项合计的 step reward 是多少?

- A. `+0.80`
- B. `+0.016`
- C. `+0.04`
- D. `-0.004`

正确答案: B。

考点: 每项贡献均为 raw value * weight * dt, dt=0.02 s; 正向跟踪和负向平滑惩罚必须分别计算后相加。

## U3-R7: reset 与独立 interval timer

env 9 在本步结束时摔倒, env 10 没有摔倒且它自己的 `push_robot` 倒计时还剩 `0.015 s`。下一步仍是20 ms。当前 `push_robot` 使用默认独立 timer。哪项正确?

- A. env 9 的 reset 会让 env 10 的 timer 也重新抽样, 所以两台都不会受推。
- B. env 9 与 env 10 都会立刻受推, 因为 interval 对所有环境同步。
- C. env 9 被 reset 并重抽自己的 interval timer; env 10 继续运行且倒计时过零, 因而 env 10 受推。
- D. env 10 到计时器过零时会被 reset, 不会受到推力。

正确答案: C。

考点: reset event 只作用于 done `env_ids`; interval 默认 per-env timer; 0.015 s 小于下一步20 ms。

## U3-R8: 当前 CoM DR 实际改了什么

当前 `randomize_com` 选择 `trunk_base` 并调用 `dr.body_ipos(... operation="add")`。下列哪个描述正确?

- A. 直接给整机 `qpos` 加位移, 所有外观 mesh 一起平移。
- B. 直接将整机 CoM 精确平移所采样的同样距离。
- C. 改变躯干刚体内部的局部惯性中心参数; 整机 CoM 会受质量比例与姿态影响, 不必同量平移。
- D. 给 actor observation 增加一个可直接测得的 CoM 三维传感器。

正确答案: C。

考点: `body_ipos` 是物理模型参数, 非 qpos/mesh/传感器; 整机 CoM 是质量加权导出量。

## U3-R9: 换数值复测 reward 时间缩放

仍是20 ms environment step。tracking term 的 raw value 为 `0.25`, weight 为 `+2.0`; smoothness penalty 的 raw value 为 `1.5`, weight 为 `-0.2`。忽略其他项, 总 step reward 为多少?

- A. `+0.20`
- B. `+0.004`
- C. `+0.010`
- D. `-0.006`

正确答案: B。

考点: 两项先分别乘 raw * weight * 0.02, 再相加; A、C、D 分别对应遗漏dt、只算正项、只算负项的常见错误。

## U3-R10: 换场景复测 local CoM 与整机 CoM

假设一台简化机器人总质量 `0.80 kg`, 其中躯干质量 `0.20 kg`。姿态不变时, 只把躯干的 local `body_ipos` 沿一个轴移动 `+5 mm`, 其他部件的质量和位置均不变。整机 CoM 沿该轴大约移动多少?

- A. `+5 mm`, 因为整机 CoM 跟随躯干偏移一模一样的距离。
- B. `+1.25 mm`, 因为躯干占整机质量的1/4。
- C. `0 mm`, 因为 `body_ipos` 只影响渲染, 不影响物理。
- D. 无法变化, 因为 qpos 没变。

正确答案: B。

考点: 在其他项不变时, 整机CoM的变化为该部件质量占比乘以该部件CoM变化; `body_ipos` 改物理惯性中心而非 qpos/mesh。
