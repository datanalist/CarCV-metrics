# Git Commit + Push

## Overview

Create a focused commit and push current branch to origin. Combines commit workflow with push/sync.

## Steps

### 1. Commit

1. **Review changes**
    - `git diff --cached` (staged) or `git diff` (unstaged)
2. **Issue key (optional)**
    - Check branch name for issue key; optionally ask user if not in context
3. **Stage if needed**
    - `git add -A`
4. **Commit**
    - `git commit -m "<issue-key>: <type>(<scope>): <short summary>"` (or without issue key)
    - Rules: ≤72 chars, imperative mood, capitalize first letter, no period, describe why

### 2. Push

5. **Fetch and rebase (recommended)**
    - `git fetch origin`
    - `git rebase origin/main || git rebase --abort`
6. **Push**
    - `git push -u origin HEAD`
7. **If rejected**
    - `git pull --rebase && git push`
8. **Force push** — ask user first: `git push --force-with-lease`

## Template

- `git commit -m "<type>(<scope>): <short summary>"`
- With issue: `git commit -m "<issue-key>: <type>(<scope>): <short summary>"`

## Rules

- Commit length ≤72 chars, imperative mood, no trailing period
- Prefer rebase over merge for linear history
- Ask before force push
