# Microduck 项目学习入口

这里保存学习路线, 当前进度和课程记录. 导师工作流程保存在项目 skill 中, 学习状态保存在 Markdown 中, 不依赖某个聊天或某台电脑.

## 平时怎么用

在本仓库的 Codex 会话中输入:

```text
$microduck-mentor 继续学习
$microduck-mentor 看看我的学习进度, 先不要开新课
$microduck-mentor 复习上次没理解的部分
$microduck-mentor 今天先到这里, 记录本次内容和下次起点
```

也可直接说“继续学习这个项目”或“记录今天的学习”. 根目录 `AGENTS.md` 为学习请求提供 skill 入口. 日常修 bug/训练任务不会因此自动变成一节课.

导师会先读进度, 结合源码授课, 在实质学习回合后保存内容, 实践证据, 未解决问题和下一步. “已讲过”和“能独立完成”分别记录, 不写入尚未执行的练习结果.

## 文件在哪里

| 文件 | 用途 |
|---|---|
| [plan.md](plan.md) | 六单元课程、逐步实践、候选专项和交付验收 |
| [progress.md](progress.md) | 当前能力, 复习队列, 最近记录和下次起点 |
| [session-01.md](session-01.md) / [补充](session-01-supplement.md) | 保留的历史学习内容 |
| `sessions/` | 新课程及学习管理记录; 正式课编号写在记录内 |
| `diagrams/` | 随仓库迁移的图示, 不只放聊天临时目录 |
| [导师 skill](../../.agents/skills/microduck-mentor/SKILL.md) | 接续, 教学, 记录和建议流程 |

## 换电脑如何继续

1. 将学习文档, `.agents/skills/microduck-mentor/`, 根目录 `AGENTS.md` 和 `agent_cn.md` 随仓库提交并同步到你可访问的 Git 远端, 或完整复制这些文件. 创建文件不等于已提交或推送.
2. 新电脑获取同一份仓库, 从项目目录打开 Codex, 输入 `$microduck-mentor 继续学习`.
3. 若当前会话尚未识别新 skill, 重启 Codex 后再试. 也可直接说“读取 `.agents/skills/microduck-mentor/SKILL.md` 和 `docs/learning/progress.md`, 按导师流程继续”.
4. 纯阅读无需先重装仿真环境. 到实际运行练习时再按项目说明安装依赖并检查平台/GPU.

Codex 的仓库级 skill 发现目录是 `.agents/skills`, 详见 [官方说明](https://learn.chatgpt.com/docs/build-skills). 无需另行复制到个人全局目录.

`logs/`, `wandb/`, `.pt`, `.onnx` 等大文件被本仓库忽略, 需要另行同步才能重放训练. 笔记保存相对路径, run/checkpoint 标识和必要摘要, 模型暂时缺失时仍可继续读源码.

多台电脑交替学习时, 开始前同步最新记录, 结束后同步本次变更; 合并冲突时保留双方会话证据, 再确定接续点. 不覆盖另一台机器上的记录.

“能力跟着项目走”指教学约定和已记录状态可迁移; 不复制完整聊天历史, 账号权限或保证不同模型逐字一致. 若环境不能写文件, 导师应说明记录未保存, 给出可手动保存的内容.
