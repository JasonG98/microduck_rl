***

name: "sync-upstream"
description: "Syncs this fork with the upstream source repo (pollen-robotics/microduck\_rl): fetches upstream branches, merges into local, and pushes to origin (the user's learning fork). Invoke when the user asks to sync upstream, pull upstream changes, update the fork from upstream, or merge upstream into local."
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Sync Upstream (fork 同步)

本仓库是一个出于学习目的的 fork:

- `upstream` = `https://github.com/pollen-robotics/microduck_rl.git` (源仓库)

- `origin` = `https://github.com/JasonG98/microduck_rl.git` (用户的 fork)

本 skill 将 upstream 的最新代码合并到本地分支并推送到 origin.

> **本 fork 的关键事实 (决定合并策略):** 本地分支带着一个 `仓库中文化` 的提交历史
> (注释/docstring/print/raise 消息都被改成了中文, 见 `AGENTS.md` 本地化约定). 因此每次
> 同步 upstream, 冲突的来源几乎总是: **上游的真实代码改动** 撞上 **本地的中文注释改动**.
> 处理原则见下文第 4 步.

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
git log --oneline HEAD..upstream/develop   # 预览上游新增提交 (fetch 后才有, 也可放到第 2 步后)
```

- 如果存在未提交的更改, 停止并询问用户是 stash 还是提交, 不要自行丢弃更改.

- 如果 `upstream` remote 缺失, 提示用户先执行:
  `git remote add upstream https://github.com/pollen-robotics/microduck_rl.git`

### 2. 拉取 upstream

```bash
git fetch upstream --prune
git log --oneline HEAD..upstream/<branch>      # 上游新增的提交
git diff --stat HEAD...upstream/<branch>       # 文件级变更概览
```

### 3. 确定目标分支

默认同步当前分支 (通常是 `develop`). 如果用户指定了分支名 (如 `main`), 以用户指定为准.

查看 upstream 有哪些分支可供参考:

```bash
git branch -r | grep upstream
```

### 4. 合并 (核心: 中文化冲突处理)

```bash
git merge upstream/<branch> --no-edit
```

- **不要**使用 `git rebase`(会改写已推送到 origin 的历史), 也不要 `--force` 推送.

**若出现冲突** (几乎必然, 且集中在 `src/` 下的 .py、`scripts/`、`README.md`、`pyproject.toml`):

1. 先向用户说明冲突的**性质**: 上游真实代码改动 vs 本地中文注释改动, 并让用户在"全部用
   upstream 代码 / 逐个手动解决并重新中文 / 中止" 之间做选择. 默认推荐逐个解决并重新中文,
   因为它同时保留上游功能与本地的中文成果.
2. 若选择逐个解决: 对每个冲突块

   - **代码/逻辑/重命名/新增功能一律采用 upstream** (例如 `allcollisions`→`groundcontact`
     模型族重命名、新增 `publish`/`sim` 模块、`export.py` 重构).

   - **注释、docstring、print/log/raise/argparse help 写成中文**, 标点全用半角英文标点
     (`, . : ; ? ! " ' ( )`), 禁用全角标点.

   - 标识符、变量/函数/类名、键名、正则、文件路径、内部字符串键保持英文不动.

   - 不要丢失任何代码逻辑, 也不要凭空加功能.
3. 合并可能还会带来**冲突未覆盖、但同样是上游新增的 .py 文件** (如 `publish/`、`sim/`、
   `export.py`、配套 `tests/`). 这些虽无冲突标记, 但仍是英文注释, 需要同样中文化 (份量不小,
   可并行分给子代理: 每个子代理负责一组文件, 用 `python3 -m py_compile` 校验).
4. 中文化**不要动测试断言的业务逻辑**, 但要注意: 若本地化了代码里的 `raise`/报错消息, 测试里
   `pytest.raises(..., match=...)` 的正则若仍匹配英文串会失败, 需同步把正则改为匹配新中文串
   (例如 `match="16 actions"` → `match="16 个动作"`).

**合并后的暂存与自检:**

```bash
git add <所有冲突文件>            # 若整个仓库就绪可直接 git add -A
git status --short | grep -E "^(UU|AA|DD)" || echo "NO_UNMERGED"
git grep -n "<<<<<<<\|>>>>>>>" -- '*.py' '*.toml' '*.md' '*.xml' '*.json' || echo "NONE"
```

确认无未合并文件、无冲突标记残留, 再进入下一步.

### 5. 合并后验证 (代码级, 不必跑长训练)

```bash
uv run list-envs                  # 包能 import, 任务注册表正常 (快速)
uv run --with pytest pytest tests/ -q    # CPU 回归, 锁住配置不变量
```

- `.py` 新文件/改动文件可先用 `python3 -m py_compile <file>` 快速查语法.

- 若上有测试正则对齐问题, 一并修掉后 `git add`, 再重新跑对应测试确认.

### 6. 处理 origin 分歧

`push` 之前先看 origin 是否已经领先:

```bash
git fetch origin
git log --oneline develop..origin/develop
```

- 若 `origin/develop` 有本地没有的提交 (例如之前推过的文档修正), 先合并它:

```bash
git merge origin/develop --no-edit
```

- 冲突同样按第 4 步处理 (通常是注释/标点的细微修正, 冲突较轻).

### 7. 提交与推送

```bash
git commit -m "merge: sync upstream/develop 到 develop ..."   # 描述合并了什么 + 中文化处理
git push origin <branch>
```

- 提交信息里简要写清: 合并了哪些上游改动, 冲突如何解决, 是否中文化了新增文件.

- `push` 若遇 SSL/网络错误先重试一次; 若被拒 (fetch first) 说明 origin 又领先了, 回到第 6 步
  合并后再 push.

### 8. 汇报

完成后向用户简要汇报:

- 合并了哪些新提交 (`git log --oneline -5`).

- 冲突数量与处理方式 (保留上游代码 + 中文化注释; 中文化了哪些新增文件).

- 验证结果 (测试通过情况, 剩余失败及其原因).

- 是否有 `origin` 领先并被一并合并.

### 9. (按需) 把合并结果同步到学习笔记分支 docs/learning-notes

本仓库另有一个学习笔记分支 `docs/learning-notes`, 记录学习计划与训练笔记. 完成 develop 的
同步后, 若想让学习笔记分支也跟上 develop (含上游改动与被中文化的注释), 可做一次合并:

```bash
git checkout docs/learning-notes        # 切分支前必须保证工作区干净
git pull origin docs/learning-notes     # 先同步远端, 避免落后
git merge develop --no-edit             # 把 develop 的最新合并进来 (含 SKILL.md 改动)
# 若冲突: 处理原则同第 4 步, 这里的冲突通常较轻
git push origin docs/learning-notes
```

- 切分支前工作区若有未提交改动, 会阻改变或携带过去 — 先提交或 stash.

- 合并会把 develop 上的 `.trae/skills/sync-upstream/SKILL.md` 等改动一并带入学习笔记分支.

- `push` 被拒 (fetch first) 时先 `git pull origin docs/learning-notes` 再 push.

## 可选: 同步前预览差异

用户只想看看上游有什么新东西, 还没决定合并时:

```bash
git fetch upstream --prune
git log --oneline HEAD..upstream/<branch>      # 上游新增的提交
git diff --stat HEAD...upstream/<branch>       # 文件级变更概览
```

只展示, 不做任何合并/推送动作.
