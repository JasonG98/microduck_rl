---
name: sync-upstream
description: 把上游源仓库 pollen-robotics/microduck_rl 的最新提交合并进本 fork 并推送到 origin. 用户要求同步 upstream, 拉取上游, 更新 fork, 合并上游代码或预览上游有什么新东西时使用. 覆盖冲突处理, 中文化修复, 合并后验证, origin 分歧处理与 docs/learning-notes 分支同步.
---

# Sync Upstream (fork 同步)

把 upstream 的新提交合并进本 fork, 解决"上游代码改动 vs 本地中文注释"的冲突, 验证后推送到 origin.

## 仓库拓扑

| 名称 | 地址 | 角色 |
|---|---|---|
| `upstream` | `https://github.com/pollen-robotics/microduck_rl.git` | 源仓库, 只读, 永不 push |
| `origin` | `https://github.com/JasonG98/microduck_rl.git` | 用户的 fork, 同步结果推这里 |
| `develop` | 本地与远端同名分支 | 集成分支, 同步默认落在这里 |
| `docs/learning-notes` | 本地与远端同名分支 | 学习笔记分支, 由 develop 合并而来, 不直接合 upstream |

关键事实, 它们决定了整个流程:

- upstream 的默认分支是 `develop`, 不是 `main`.
- 本 fork 带一条中文化提交历史 (注释 / docstring / print / log / raise / argparse help 均为中文, 见 `AGENTS.md` 的本地化约定). 所以冲突几乎总是同一类: **上游真实代码改动** 撞上 **本地中文注释改动**, 而不是两边逻辑分歧.
- 仓库里可能只有 `origin` 一个 remote, upstream 缺失是常态而不是异常, 第 2 步补上即可.
- 工作区未提交的改动不参与合并, 且很容易被覆盖或误删; 当前 checkout 的改动也不等于已提交或已推送.

## 红线

- 不 `git rebase` 已推送的分支, 不 `git push --force` / `--force-with-lease`.
- 不向 upstream 推送任何东西; 只推 `origin`.
- 不丢弃用户未提交的改动: 不用 `git checkout .`, `git reset --hard`, `git clean -fd`, `git stash drop`. 有改动先停下来问用户.
- 不用 `-X ours` / `-X theirs` 整体策略合并: 它会静默丢掉一整侧的改动, 后续无法审计.
- 冲突未解决 (存在未合并文件或冲突标记) 不 commit, 不 push.
- 不为了让测试变绿而改测试断言的业务逻辑.
- 不把未验证的结果说成通过. 测试失败要给出失败项与归因.

## 工作流程

每一步失败都停下来向用户报告, 不要强行继续.

### 1. 确认意图与前置状态

先分清用户要哪一种: **只预览** → 跳到"只预览"一节, 不做任何写操作; **完整同步** → 继续往下.

```bash
git remote -v
git status --short
git branch --show-current
git rev-parse --short HEAD        # 记下合并前 HEAD, 后面自检与回退都要用
```

- 工作区必须干净. 有未提交改动时, 让用户选择提交还是 stash, 不要代选.
- 目标分支默认 `develop`. 若当前在 `docs/learning-notes`, 先 `git switch develop` (切换前同样要干净), 不要直接把 upstream 合进学习笔记分支.
- 用户指定了别的分支名 (如 `main`) 时以用户为准, 但先说明该分支与 `develop` 的关系.

### 2. 配置 upstream 并抓取

```bash
# 仅当 git remote -v 里没有 upstream 时执行
git remote add upstream https://github.com/pollen-robotics/microduck_rl.git

git fetch upstream --prune
git fetch origin --prune
```

- `fetch` 不改工作区, 失败 (网络 / SSL) 可重试一次; 仍失败则报告并停止.

### 3. 预览差异

```bash
git log --oneline --no-merges develop..upstream/develop
git rev-list --count develop..upstream/develop
git diff --stat develop...upstream/develop
```

- 计数为 0 → 汇报"已是最新", 结束, 不产生空提交.
- 用户只要预览 → 展示上面输出并停止.
- 差异很大时 (几十个提交, 或上游新增整块模块与大量 `.py`) 先说明中文化工作量, 问用户本次全量合并还是分批.

### 4. 合并

```bash
git config merge.conflictstyle zdiff3     # 冲突块额外给出共同祖先, 判断"谁改了什么"更准
git merge upstream/develop --no-edit
```

- 记录 `git rev-parse --short ORIG_HEAD` (即合并前的 HEAD), 后续列新增文件, 回退都用它.
- 出现冲突属正常, 进入下一步; 冲突文件列表用 `git diff --name-only --diff-filter=U`.
- 合并中途想放弃: `git merge --abort` 回到合并前, 然后向用户确认.

### 5. 解决冲突 (核心: 中文化)

在 merge 状态下 `--ours` 是本地分支 (带中文注释), `--theirs` 是 upstream, 别搞反.

| 冲突内容 | 采用哪边 |
|---|---|
| 代码逻辑, 重命名, 新增功能, 依赖与 pin, MJCF / XML 模型 | upstream |
| 注释, docstring, print / log / raise / argparse help | 本地中文重写, 内容跟上 upstream 的新语义 |
| 标识符, 函数 / 变量 / 类名, 键名 (如 `Episode_Reward/...`), 正则, 文件路径 | 英文, 不动 |
| 两边都做过的同一处修复 | 两边都保留并逐个说明, 不默认丢本地 |
| `docs/superpowers/plans/`, `docs/superpowers/specs/` 历史档案 | 冻结, 保持原语言, 不翻译 |

操作建议:

1. 先给冲突文件分类: 纯翻译冲突 / 上游代码改动 / 两边都有实质改动.
2. 纯翻译冲突: `git checkout --theirs -- <file>` 取 upstream 版本, 再只对上游改动的 hunk 补中文注释; 不要重译整文件.
3. 两边都有实质改动: 手动编辑, 逻辑取 upstream, 注释用中文重述.
4. 中文化规范: 中文, 标点用半角 `, . : ; ? ! " ' ( )`; 标识符 / 键名 / 正则 / 路径保持英文; 不改测试断言的业务逻辑.
5. `raise` / 报错文案改成中文后, 搜同批改动里的 `pytest.raises(..., match=...)`, 同步正则 (例如 `match="16 actions"` → `match="16 个动作"`).
6. 上游若改了 `AGENTS.md`, 保留英文原文并按其内容行事; `agent_cn.md` 是中文镜像, 按需同步, 不要求逐次同译.

### 6. 中文化合并带入的新文件

冲突文件之外, 上游还会带进无冲突但仍是英文注释的新 `.py`:

```bash
git diff --name-only --diff-filter=A ORIG_HEAD -- '*.py'
git diff --name-only ORIG_HEAD -- '*.py' '*.toml'   # 全部被本次合并改动的文件
```

- 只中文化本次合并带入 / 改动的文件, 不顺手翻译未改动文件: 否则 diff 巨大且无法审计.
- 文件多时可分组并行: 每个子代理负责互斥的一组文件, 返回完整文件内容; 分组不得重叠, 否则互相覆盖.
- 每组完成后 `python3 -m py_compile <file>` 校验语法, 再 `git add`.

### 7. 暂存与自检

```bash
git add -A
git status --short | grep -E "^(UU|AA|DD|AU|UA|DU|UD)" || echo NO_UNMERGED
git grep --cached -nE '^(<{7}|>{7}|\|{7})' -- '*.py' '*.toml' '*.xml' '*.json' '*.md' || echo NO_MARKERS
git diff --cached --check          # 空白错误
python3 -m compileall -q src scripts tests
```

- 冲突标记检查只看行首的 `<<<<<<<` / `>>>>>>>` / `|||||||`, 不要搜 `=======`, markdown 分隔线会误报.
- 确认无未合并文件, 无标记残留, 语法通过, 再进入验证.

### 8. 合并后验证 (代码级, 不必跑长训练)

```bash
uv run list-envs                            # 包可导入, 任务注册表完整 (快)
uv run --with pytest pytest tests/ -q       # CPU 回归, 锁住配置不变量与 mdp 函数
```

- `pyproject.toml` / `uv.lock` 有改动时先 `uv sync`.
- 失败归因: 看失败文件是否本次合并新增 / 改动; 用 `ORIG_HEAD` 上的旧代码单独复现一次, 区分"上游带进来的既有失败"与"本地合并引入的失败". 存疑时按测得的结论说明, 不猜.
- 可选: 大改动再由用户决定是否跑 smoke (`uv run train <TASK_ID> --env.scene.num-envs 64 --agent.max_iterations 5`, 需 GPU).

### 9. 处理 origin 分歧

```bash
git fetch origin --prune
git log --oneline HEAD..origin/develop
```

- 有本地没有的提交 (例如之前在别处推过的文档修正) → `git merge origin/develop --no-edit`, 冲突按第 5 步处理, 然后重跑第 7-8 步自检.
- 没有 → 直接推送.

### 10. 提交与推送

```bash
git commit -F - <<'EOF'
merge: sync upstream/develop (<N> commits) 到 develop

- 上游要点: <模块 / 行为变化>
- 冲突: <文件数与解决方式>
- 中文化: <新增或改动的文件>
- 验证: uv run list-envs 通过; pytest tests/ <结果>
EOF

git push origin develop
```

- 提交信息写清: 合并了哪些上游改动, 冲突如何解决, 中文化了哪些文件, 验证结果.
- push 被拒 (non-fast-forward) 说明 origin 又领先了 → 回第 9 步合并后重推.
- SSL / 网络错误先重试一次; 仍失败就报告, 不 force.

### 11. (按需) 同步到 docs/learning-notes

只在用户要学习笔记分支也跟上时做. 先保证工作区干净.

```bash
git switch docs/learning-notes
git fetch origin --prune
git merge origin/docs/learning-notes --no-edit   # 先接远端, 避免落后
git merge develop --no-edit                      # 再合 develop (含 upstream 改动与中文化)
git push origin docs/learning-notes
```

- 首次推送该分支时用 `git push -u origin docs/learning-notes`; 用 `git ls-remote --heads origin docs/learning-notes` 确认远端是否存在.
- 冲突按第 5 步; 学习分支的冲突多为同一文件两边的中文措辞, 取内容更完整的一边, 并保留双方各自独有的记录.
- 学习笔记分支的独有内容 (`docs/learning/**` 笔记, quiz 题库与答题记录) 不能被 develop 覆盖.
- push 被拒时先 `git merge origin/docs/learning-notes` 再推.
- 本 skill 的正本是 `.agents/skills/sync-upstream/SKILL.md`. develop 上可能还留着旧位置的 `.trae/skills/sync-upstream/SKILL.md`, 若确认 `.trae/` 下没有其它在用技能, 顺手删除以免两份说明漂移.

### 12. 汇报

向用户简要汇报, 每项都要有实测依据:

- 目标分支与合并前后 HEAD (`<旧> -> <新>`).
- 上游新增提交数与要点 (哪些模块 / 行为变了).
- 冲突文件数, 每个文件怎么解决 (取了谁的代码, 注释怎么处理).
- 中文化了哪些新增 / 改动文件.
- 验证命令与实际结果; 失败项及其归因.
- 是否一并合并了 origin 领先的提交.
- push 结果 (远端分支与提交号).
- 遗留风险: 未验证项, 上游既有失败, 需要用户决定的事项.

## 只预览 (不改任何东西)

用户只想看上游有什么新东西时:

```bash
git remote add upstream https://github.com/pollen-robotics/microduck_rl.git   # 缺 upstream 时
git fetch upstream --prune
git log --oneline --no-merges HEAD..upstream/develop
git diff --stat HEAD...upstream/develop
```

只展示输出, 不做 merge / commit / push / add, 也不切分支.

## 回退

- 合并冲突中想放弃: `git merge --abort` (回到合并前 HEAD).
- 已提交未推送, 且用户明确同意丢弃这次合并: `git reset --hard ORIG_HEAD` (先确认工作区干净); 有疑虑就改用 `git revert -m 1 <merge-commit>` 留痕.
- 已推送: 不改写历史, 用一个新的 revert 或修复提交解决.
