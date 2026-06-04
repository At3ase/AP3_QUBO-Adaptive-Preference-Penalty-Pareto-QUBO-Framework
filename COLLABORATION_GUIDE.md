# AP3 QUBO 项目协作指南

> 本文档面向所有项目共创者。请务必在参与开发前完整阅读。
> 仓库地址：https://github.com/At3ase/AP3_QUBO-Adaptive-Preference-Penalty-Pareto-QUBO-Framework

---

## 一、项目简介与协作理念

本项目是 **AP3 QUBO（Adaptive Preference-Penalty Pareto QUBO Framework）** 高熵合金优化框架。

**核心协作原则：**
- **所有代码变更必须经过 Pull Request（PR）审查**，禁止直接推送到 `main` 分支
- **主仓库维护者 @At3ase 拥有最终审查权和合并权**
- 任何人都可以 fork 仓库、创建分支、提交 PR

---

## 二、准备工作

### 2.1 安装 Git

**Windows 用户：**
1. 下载安装包：https://git-scm.com/download/win
2. 安装时一路 Next 即可，推荐勾选：
   - ☑️ "Git from the command line and also from 3rd-party software"
   - ☑️ "Use Windows' default console window"

**验证安装：**
```bash
git --version
# 输出类似：git version 2.42.0
```

### 2.2 配置 Git 身份

```bash
git config --global user.name "你的GitHub用户名"
git config --global user.email "你注册GitHub时用的邮箱"
```

> ⚠️ 邮箱必须与 GitHub 账户一致，否则提交记录无法关联到你的 GitHub 账号。

### 2.3 配置 GitHub 身份验证

推荐方式：使用 **Personal Access Token (PAT)**

1. 登录 GitHub → 点击右上角头像 → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. 点击 "Generate new token (classic)"
3. 勾选权限：`repo`（全部子权限）
4. 生成后复制 Token（只显示一次，务必保存）
5. 首次 `git push` 时输入用户名和该 Token 作为密码

> 💡 如果不想每次输入 Token，可以配置 Git Credential Manager：
> ```bash
> git config --global credential.helper manager
> ```
> Windows 下安装 Git 时已经自带，首次 push 会自动弹出窗口让你保存凭据。

---

## 三、获取项目代码

### 3.1 克隆仓库

```bash
git clone https://github.com/At3ase/AP3_QUBO-Adaptive-Preference-Penalty-Pareto-QUBO-Framework.git
cd AP3_QUBO-Adaptive-Preference-Penalty-Pareto-QUBO-Framework
```

### 3.2 验证远程连接

```bash
git remote -v
```

应显示：
```
origin  https://github.com/At3ase/AP3_QUBO-Adaptive-Preference-Penalty-Pareto-QUBO-Framework.git (fetch)
origin  https://github.com/At3ase/AP3_QUBO-Adaptive-Preference-Penalty-Pareto-QUBO-Framework.git (push)
```

---

## 四、分支策略（非常重要）

本项目采用 **GitHub Flow** 简化工作流：

```
main 分支（受保护）←── PR 审查 ←── feature/xxx 分支（个人开发）
```

| 分支 | 用途 | 权限 |
|------|------|------|
| `main` | 稳定代码，生产环境 | ⚠️ **受保护，禁止直接推送** |
| `feature/xxx` | 个人功能开发 | ✅ 自由创建和推送 |

### 分支命名规范

| 前缀 | 用途 | 示例 |
|------|------|------|
| `feature/` | 新功能开发 | `feature/qubo-solver-v2` |
| `fix/` | Bug 修复 | `fix/encoding-error` |
| `docs/` | 文档更新 | `docs/api-reference` |
| `refactor/` | 代码重构 | `refactor/optimize-loop` |
| `test/` | 测试相关 | `test/unit-coverage` |

---

## 五、日常开发完整流程

### 5.1 开始新工作：创建并切换分支

```bash
# 先确保你在 main 分支上，且本地是最新的
git checkout main
git pull origin main

# 创建新分支（用你自己的功能名替换）
git checkout -b feature/你的功能名称

# 示例：
# git checkout -b feature/optimize-hea-encoding
```

> 💡 `checkout -b` = 创建分支 + 立即切换。这是一个原子操作。

### 5.2 编写代码

在分支上自由修改代码。使用你习惯的编辑器（VS Code / PyCharm / 其他）。

### 5.3 查看改动状态

```bash
git status
```

输出解读：
- `Changes not staged for commit`（红色）→ 已修改但未暂存
- `Changes to be committed`（绿色）→ 已暂存，待提交
- `Untracked files`（红色）→ 新增文件，Git 尚未跟踪

### 5.4 选择要提交的文件（暂存）

```bash
# 暂存所有修改和新增文件
git add .

# 或只暂存特定文件
git add 文件名.py

# 或暂存整个目录
git add src/
```

### 5.5 提交代码

```bash
# 提交暂存的文件，写清楚做了什么
git commit -m "feat: 简短描述本次改动"

# 示例：
git commit -m "feat: add adaptive penalty weight calculation for QUBO solver"
```

**提交信息规范（Commit Message Convention）：**

| 前缀 | 含义 | 示例 |
|------|------|------|
| `feat:` | 新功能 | `feat: implement Pareto front selection` |
| `fix:` | 修复 Bug | `fix: resolve encoding error in HEA scheme` |
| `docs:` | 文档更新 | `docs: add API usage examples` |
| `refactor:` | 代码重构 | `refactor: simplify QUBO matrix generation` |
| `test:` | 测试相关 | `test: add unit tests for fitness function` |
| `chore:` | 杂项维护 | `chore: update .gitignore` |
| `perf:` | 性能优化 | `perf: optimize crossover operation` |

> 💡 提交信息应该能回答 "这次改动做了什么" 和 "为什么"。
> 好例子：`fix: resolve integer overflow in fitness calculation (#42)`
> 坏例子：`update` 或 `fix bug`

### 5.6 推送分支到 GitHub

```bash
# 第一次推送新分支（-u 设置上游关联）
git push -u origin feature/你的功能名称

# 后续更新同一分支（直接 push 即可）
git push
```

> ⚠️ 如果提示 "Failed to connect to github.com"，说明网络问题，稍后再试。

### 5.7 在 GitHub 上发起 Pull Request

**方式 A：GitHub 网页（推荐）**

1. 打开仓库主页：https://github.com/At3ase/AP3_QUBO-Adaptive-Preference-Penalty-Pareto-QUBO-Framework
2. 点击 `Pull requests` 标签 → 绿色按钮 `New pull request`
3. 配置对比：
   - **base**：`main`（要合并到的目标分支）
   - **compare**：`feature/你的功能名称`（你的开发分支）
4. 点击 `Create pull request`
5. 填写 PR 信息（见下方模板）

**方式 B：命令行快捷方式（GitHub CLI）**

如果你安装了 `gh`：
```bash
gh pr create --base main --title "feat: 你的功能描述" --body "详细说明"
```

### 5.8 PR 描述模板

创建 PR 时，请复制以下模板填写：

```markdown
## 改动概述
一句话描述本次 PR 的目的。

## 改动详情
- 新增/修改了哪些文件
- 具体改动了什么逻辑
- 如果有算法改动，说明思路

## 测试情况
- [ ] 已运行相关测试
- [ ] 已通过本地验证
- 测试命令和结果

## 关联 Issue
如有相关 Issue，请填写：Fixes #123

## 备注
任何需要审查者注意的事项
```

### 5.9 等待审查

PR 创建后：
- 系统会自动通知 @At3ase 进行审查
- 你可以在 PR 页面看到审查状态
- 审查者可能会提出修改意见（Request changes），你需要：
  1. 在本地继续修改代码
  2. `git add .` → `git commit -m "fix: 根据审查意见修改"` → `git push`
  3. 修改会自动同步到同一 PR，无需重新创建
- 审查通过后，审查者会点击 **Merge pull request**

---

## 六、保持本地代码同步（重要）

其他人合并了 PR 后，你的本地 `main` 分支会过时。在创建新分支前，务必同步：

```bash
# 切换到 main 分支
git checkout main

# 拉取最新代码
git pull origin main

# 现在基于最新的 main 创建新分支
git checkout -b feature/下一个功能
```

> ⚠️ 如果不执行 `git pull`，你的新分支可能基于旧代码，导致后续冲突。

---

## 七、冲突解决

当多人修改同一文件时，合并会产生冲突。

### 7.1 冲突场景

你在 `feature/xxx` 分支开发时，`main` 分支已更新，且修改了同一个文件。当你尝试合并或创建 PR 时，GitHub 会提示 "This branch has conflicts"。

### 7.2 解决步骤

```bash
# 1. 确保在你的功能分支上
git checkout feature/你的功能名称

# 2. 拉取 main 的最新代码到本地
git fetch origin main

# 3. 将 main 的改动合并到你的分支（这时可能会冲突）
git merge origin/main

# 4. 如果冲突，Git 会提示哪些文件有冲突
# 打开冲突文件，搜索 `<<<<<<< HEAD` 标记
# 格式如下：
# <<<<<<< HEAD
# 你的代码
# =======
# main 上的代码
# >>>>>>> origin/main
# 
# 保留你需要的部分，删除所有标记行（<<<<<<<, =======, >>>>>>>）

# 5. 解决后暂存并提交
git add .
git commit -m "resolve: merge conflict with main"

# 6. 推送
git push
```

> 💡 如果冲突太复杂，可以放弃合并，用 rebase 代替：
> ```bash
> git fetch origin main
> git rebase origin/main
> # 解决冲突后：git add . && git rebase --continue
> ```

---

## 八、常见问题 FAQ

### Q1: 我直接 `git push origin main` 为什么被拒绝了？

**A:** 这是正常的。`main` 分支已启用保护规则，**禁止任何人直接推送**。你必须通过 PR 流程提交改动。请创建 `feature/xxx` 分支并推送该分支。

### Q2: 我修改了代码但还没提交，想放弃所有改动？

```bash
# 放弃工作区的所有修改（未暂存的）
git checkout -- .

# 如果已暂存但未提交：
git reset HEAD .
git checkout -- .
```

> ⚠️ 此操作不可恢复，请确认你真的不需要这些改动。

### Q3: 我想修改上一次提交信息？

```bash
git commit --amend -m "新的提交信息"
git push --force-with-lease  # 如果已经推送过，需要强制推送覆盖
```

> ⚠️ 如果已经推送到远程且被其他人看到，建议不要 amend，而是提交一个新的修复 commit。

### Q4: 我想查看提交历史？

```bash
# 简洁版
git log --oneline -10

# 图形版（带分支图）
git log --oneline --graph --all

# 查看某个文件的修改历史
git log -p -- 文件名.py
```

### Q5: 我 push 时提示 "Failed to connect to github.com"？

**A:** 这是网络问题，常见原因：
1. 检查网络连接是否正常
2. 尝试关闭 VPN/代理后再 push
3. 等待几分钟后重试：`git push`
4. 如果持续失败，检查 GitHub 服务状态：https://www.githubstatus.com

### Q6: 如何查看当前所在分支？

```bash
git branch
# 带 * 号的即为当前分支
```

### Q7: 我创建了错误的分支，想删除？

```bash
# 删除本地分支（必须先切到其他分支）
git checkout main
git branch -d feature/错误分支名

# 如果分支有未合并的提交，用 -D 强制删除
git branch -D feature/错误分支名

# 删除远程分支
git push origin --delete feature/错误分支名
```

### Q8: 我的 PR 被 Request changes 了，怎么修改？

1. 在你的本地功能分支上继续修改代码
2. 提交新的 commit：`git add . && git commit -m "fix: 根据审查意见修改"`
3. 推送：`git push`
4. 回到 GitHub PR 页面，修改会自动同步，审查者会收到通知

> 不需要重新创建 PR，所有修改会自动附加到同一个 PR 中。

### Q9: 如何给代码添加 .gitignore 忽略某些文件？

项目根目录已有 `.gitignore` 文件。如果需要新增规则：

```bash
# 编辑 .gitignore
echo "*.pyc" >> .gitignore       # 忽略 Python 编译文件
echo "__pycache__/" >> .gitignore # 忽略缓存目录
echo "*.log" >> .gitignore       # 忽略日志文件

# 提交 .gitignore 的更新
git add .gitignore
git commit -m "chore: update .gitignore"
```

### Q10: 我不小心把密码/密钥提交到仓库了？

**立即处理：**
1. 不要 panic，这在 Git 历史中无法真正删除，但可以使其不可用
2. 立即在相应平台撤销/更换该密码/密钥
3. 使用 Git 历史重写工具（BFG Repo-Cleaner 或 git-filter-repo）清理历史
4. 联系 @At3ase 协助处理

> ⚠️ 提交到 GitHub 的代码一旦公开，即使删除 commit，历史也可能被缓存。立即更换密钥是最安全的做法。

---

## 九、VS Code 用户快捷操作

如果你使用 VS Code，推荐安装插件：
- **GitLens** — 增强 Git 可视化，显示每行代码的提交信息
- **GitHub Pull Requests** — 直接在 VS Code 中创建和审查 PR

**VS Code 常用 Git 快捷键：**
| 操作 | 快捷键 |
|------|--------|
| 查看 Git 面板 | `Ctrl+Shift+G` |
| 暂存（Stage）文件 | 点击文件旁的 `+` |
| 提交 | 填写消息 → `Ctrl+Enter` |
| 推送 | 点击面板右上角的 `...` → Push |
| 切换分支 | 点击左下角分支名 → 选择分支 |

---

## 十、Git 命令速查表

```bash
# 基础操作
git clone <url>          # 克隆仓库
git status               # 查看当前状态
git add <file>           # 暂存文件
git add .                # 暂存所有改动
git commit -m "msg"      # 提交
git push                 # 推送到远程
git pull                 # 拉取远程更新

# 分支操作
git branch               # 查看本地分支
git branch -a            # 查看所有分支（含远程）
git checkout <branch>    # 切换分支
git checkout -b <branch> # 创建并切换分支
git branch -d <branch>   # 删除本地分支
git merge <branch>       # 合并分支

# 远程操作
git remote -v            # 查看远程仓库
git fetch origin         # 拉取远程信息（不合并）
git pull origin main     # 拉取 main 并合并

# 查看历史
git log --oneline        # 简洁日志
git log --graph --all    # 图形化日志
git diff                 # 查看未暂存的改动
git diff --cached        # 查看已暂存的改动

# 撤销操作
git checkout -- <file>   # 放弃文件修改
git reset HEAD <file>    # 取消暂存
git reset --soft HEAD~1  # 撤销上一次提交（保留改动）
git reset --hard HEAD~1  # 撤销上一次提交（丢弃改动）
```

---

## 十一、紧急联系

如果遇到指南无法解决的问题，请：
1. 在 GitHub 上创建 Issue 描述问题
2. 或联系 @At3ase

---

> 📌 请记住：本项目的核心规则是 **所有改动走 PR → 审查通过 → 合并**。这是为了保证代码质量，避免意外破坏主分支。感谢配合！
