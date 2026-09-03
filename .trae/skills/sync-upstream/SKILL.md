---
name: "sync-upstream"
description: "Syncs this fork with the upstream source repo (pollen-robotics/microduck_rl): fetches upstream branches, merges into local, and pushes to origin (the user's learning fork). Invoke when the user asks to sync upstream, pull upstream changes, update the fork from upstream, or merge upstream into local."
---

# Sync Upstream (fork 同步)

本仓库是一个出于学习目的的 fork:

- `upstream` = `https://github.com/pollen-robotics/microduck_rl.git` (源仓库)
- `origin` = `https://github.com/JasonG98/microduck_rl.git` (用户的 fork)

本 skill 将 upstream 的最新代码合并到本地分支并推送到 origin.

## 触发场景

- 用户说"同步 upstream / 拉取上游 / 更新 fork / 合并上游代码"等.
- 用户想让 fork 跟上 pollen-robotics 源仓库的进展.

## 工作流程

按顺序执行以下步骤. 每一步失败都要停下来向用户报告, 不要强行继续.

### 1. 前置检查

```bash
git remote -v          # 确认 upstream 和 origin 都存在且指向正确
git status             # 工作区必须干净, 否则先请用户处理(提交或 stash)
git branch --show-current
```

- 如果存在未提交的更改, 停止并询问用户是 stash 还是提交, 不要自行丢弃更改.
- 如果 `upstream` remote 缺失, 提示用户先执行:
  `git remote add upstream https://github.com/pollen-robotics/microduck_rl.git`

### 2. 拉取 upstream

```bash
git fetch upstream --prune
```

### 3. 确定目标分支

默认同步当前分支 (通常是 `develop`). 如果用户指定了分支名 (如 `main`), 以用户指定为准.

查看 upstream 有哪些分支可供参考:

```bash
git branch -r | grep upstream
```

### 4. 合并

```bash
git merge upstream/<branch> --no-edit
```

- 如果出现冲突: 列出冲突文件, 询问用户想保留哪边 (ours/theirs) 或逐个协助解决.
  冲突中涉及本仓库的自有改动 (如 `.trae/`, `AGENTS.md` 本地定制) 默认倾向保留本地版本.
- **不要**使用 `git rebase`(会改写已推送到 origin 的历史), 也不要 `--force` 推送.

### 5. 推送到 origin

```bash
git push origin <branch>
```

### 6. 汇报

完成后向用户简要汇报:

- 合并了哪些新提交 (可用 `git log --oneline origin/<branch>..upstream/<branch>` 在合并前先预览),
  或合并后用 `git log --oneline -5` 展示最新几条.
- 是否有冲突以及如何处理的.

## 可选: 同步前预览差异

用户只想看看上游有什么新东西, 还没决定合并时:

```bash
git fetch upstream --prune
git log --oneline HEAD..upstream/<branch>      # 上游新增的提交
git diff --stat HEAD...upstream/<branch>       # 文件级变更概览
```

只展示, 不做任何合并/推送动作.
