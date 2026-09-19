# U2.1 模型词汇题库

> 仅存题面、答案和考点, 个人作答见 ../records/. 使用前核对当前源码. 本批为 2026-09-08 授课后换场景检查, 不是旧题原样重发.

## tree-1

第 1 题｜连接链为：世界 → A（A 有 freejoint）→ B（B 有 hinge）→ C（C 没有 joint）。除此之外没有其他约束；B 的 hinge 正在转动。关于 C 的运动，哪项一定正确？

- A. C 没有 joint，所以相对世界的位置和方向固定
- B. B 的 hinge 会让 C 相对 B 转动
- C. C 相对 B 固定，但可以随 B 相对 A 运动
- D. C 只能平移，不能随 B 改变朝向
- E. 不知道

答案: C.
考点: 关节描述所属 body 相对直接父 body 的运动. C 无 joint 只表示相对 B 固定, 不表示相对 A 或世界固定.
核对来源: robot_walk.xml 的 body 嵌套与 joint 归属; 此题 A/B/C 为替换场景.

## geom-1

第 2 题｜在 walk 模型副本中，只删除 left_foot_collision，保留左脚的全部 visual geom、body 惯性和 site。visual 的碰撞仍关闭，且没有其他几何体替代这只脚底接触地面。哪项正确？

- A. 脚仍看得见，但原来这块脚底提供的地面碰撞支撑消失
- B. 外观仍在，所以脚底碰撞支撑不变
- C. 只要 left_foot site 还在，它就会替脚底提供碰撞支撑
- D. 删除碰撞 geom 会自动删除整个左脚 body 的质量
- E. 不知道

答案: A.
考点: visual 不自动接管碰撞, site 不参与碰撞, body 的显式惯性仍在. 只判断该脚底的接触, 不推断整机必然以何种轨迹倒下.
核对来源: robot_walk.xml visual 默认 contype/conaffinity=0; ankle_left inertial, left_foot_collision 与 left_foot.

## gyro-1

第 3 题｜机器人原地转向 90° 后已经完全停稳。imu site 仍固定在躯干上；只看理想、无噪声的 gyro 传感器 imu_ang_vel。此时它应该输出什么？

- A. [0, 0, 90]，记录累计转过的角度
- B. [0, 0, 1.57]，用弧度记录最终朝向
- C. 转向过程中最后一次非零读数，停稳后仍保持
- D. [0, 0, 0]，因为此刻没有角速度
- E. 不知道

答案: D.
考点: gyro 测量的是当前角速度而非朝向或累计角度. site 可改变世界朝向, 但停稳时三轴角速度均为零.
核对来源: robot_walk.xml gyro imu_ang_vel site=imu; 已核对官方 gyro 局部三轴角速度定义.

## state-1

第 4 题｜某一瞬间，左膝实际角度为 −0.10 rad，实际角速度为 +0.40 rad/s。执行器刚收到 +0.35 rad 的位置目标，物理仿真尚未推进。此刻左膝对应的 qpos、qvel 应分别是？

- A. +0.35 rad、+0.40 rad/s
- B. −0.10 rad、+0.40 rad/s
- C. +0.35 rad、0 rad/s
- D. −0.10 rad、+0.35 rad/s
- E. 不知道

答案: B.
考点: 位置目标单独存储, 收到目标不会立即覆盖实际位置或速度. 题目限定物理仿真尚未推进, qpos/qvel 保持该瞬间给定实际值.
核对来源: mjlab JointPositionAction.apply_actions 调用 set_joint_position_target; BAM compute 使用目标与实际状态.

## dims-1

第 5 题｜当前 walk 模型有 1 个 freejoint 和 14 个 hinge，qpos/qvel 长度为 21/20。在副本中只删除 freejoint，不增加任何新 joint，14 个 hinge 全部保留。新模型的 nq、nv 分别是多少？

- A. 21、20
- B. 15、14
- C. 14、14
- D. 20、20
- E. 不知道

答案: C.
考点: 自由基座贡献 7 个位置数和 6 个速度数; 删除后只剩 14 个 hinge, 每个贡献一个角度与一个角速度.
核对来源: 导师本轮用 MjSpec 副本删除 trunk_base_freejoint 后 compile 实测 nq=14,nv=14; 未改源 XML.



## dims-2

复测 R1: 保留 walk 模型的 freejoint 和全部 14 个 hinge, 不改变模型结构. 某一帧机器人完全静止, qvel 所有元素都是 0. 此刻 nq/nv 是多少?

- A. 14 / 14
- B. 21 / 20
- C. 21 / 0
- D. 0 / 0
- E. 不知道

答案: B.
考点: 速度值为零不删除运动自由度, 数组长度由模型结构决定. 要区分保留但不动与删除 joint.
核对来源: 当前 robot_walk.xml 编译为 nq21/nv20; 本轮新建 MjData 并将 qvel 全置零, 模型维度不变.

## dims-3

复测 R2: 一个用于练习的机器人原本固定在世界中, 只有 6 个 hinge, 没有其他 joint. 现在给根 body 增加一个 freejoint, 6 个 hinge 保留. 新的 nq/nv 是多少?

- A. 6 / 6
- B. 12 / 12
- C. 13 / 13
- D. 13 / 12
- E. 不知道

答案: D.
考点: 从固定基座到自由基座, 增加位置3+四元数4及线速度3+角速度3, 即 7/6; 原六个 hinge 各贡献 1/1.
核对来源: 本轮构造仅含六个 hinge 的内存 MJCF, 加 freejoint 前实测6/6, 加后13/12. 是练习模型, 非项目更改.
