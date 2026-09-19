# 2026-09-15 U3 综合复习, 第二轮

- 范围: action到位置目标、reward时间缩放、并行reset与interval、CoM DR语义。
- 形式: 四道换场景单选题, 题面与答案见 [U3 题库](../questions/u3-bank.md)。
- 用户作答: `A A C B`。
- 判定: 2/4 正确。U3-R5 action/encoder bias 正确, U3-R7 reset与独立interval timer正确。U3-R6错误: 选择 `+0.80`, 漏掉每个 reward term 都乘 `dt=0.02 s`; 正确计算应在讲评后以换数值场景复测。U3-R8错误: 选择“整机 CoM 精确平移相同距离”, 混淆躯干的 local `body_ipos` 与整机质量加权 CoM; 该点在换场景复测。
