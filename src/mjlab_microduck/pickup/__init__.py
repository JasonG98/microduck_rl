"""Pick-up 检测器: 一个小型分类器, 告诉 runtime 鸭子正被人拿着 (暂停 policy) 还是已回到地面 (恢复).

完全在 sim 中训练: 部署的 policy 运行时, virtual hand (``hand.py``) 把机器人抓起, oracle 以真实的
检测延迟执行暂停/恢复, 分类器 (``model.py``) 则从 runtime 每个 50 Hz tick 所读信号的短窗口中
学习 p(held) (``features.py``).
"""
