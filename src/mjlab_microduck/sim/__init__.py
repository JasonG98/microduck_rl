"""把一个模拟的 microduck 机体提供给真实的 `robotd`。

这是 `microduck` 的 `robotd --sim` 的另一半: MuJoCo 持有机体, daemon 持有其余一切, 两者之间的
一个 socket 替代了伺服总线. 关于协议以及孪生体是什么和不是什么, 参见 microduck 仓库中的
`docs/design/simulation.md`.
"""
