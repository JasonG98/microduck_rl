# 2026-09-14 U2 综合复习测验

- 范围: U2.1 模型词汇, U2.2 名字映射与 backlash, U2.3 动作执行时序.
- 形式: 6 题单选, 每题一个主题. 作答为识别级证据, 不替代用户复述或独立操作.
- 题库: [u2-bank.md](../questions/u2-bank.md).
- 状态: 首轮已作答. 5 对 / 0 错 / 1 未答; `structure-1` 进入换场景复测队列.

| 展示题号 | 题库编号 | 用户选择 | 判定 |
|---|---|---|---|
| Q1 | chain-1 | B | 对 |
| Q2 | order-1 | C | 对 |
| Q3 | backlash-1 | B | 对 |
| Q4 | delay-1 | B | 对 |
| Q5 | torque-1 | B | 对 |
| Q6 | structure-1 | E (不知道) | 未答 |

## 首轮讲评摘要

- `chain-1`--`torque-1`: 用户连续答对, 说明能在换场景中区分 raw action、latest/delayed target、实际状态和 BAM torque, 也能应用 first-match 规则与 backlash 的 output-side 测量.
- `structure-1`: 用户明确表示忘记 `nq/nv` 的含义, 按约定判为未答而非错误. 已简短复习: `nq` 是广义位置向量长度, `nv` 是广义速度向量长度; 它们由 joint 结构而非当前静止状态决定. freejoint 贡献 `7 / 6`, 每个 hinge 贡献 `1 / 1`.

## 换场景复测: 2/2 对

| 展示题号 | 题库编号 | 用户选择 | 判定 |
|---|---|---|---|
| R1 | structure-2 | C | 对 |
| R2 | structure-3 | C | 对 |

用户提交 `C、C`, 依序对应 R1/R2, 两题均正确. `nq/nv` 主题本轮销项 (识别级): 能在换规模、静止与固定基座场景下按结构计数, 不再把当前 `qvel=0` 误当成没有速度维度. 这不替代独立建模或操作证据.
