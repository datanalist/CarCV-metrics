# Merge File/Dir from Branch

## Overview

Перенести конкретный файл или каталог из одной ветки в другую (без full merge). Использует `git checkout <branch> -- <path>` — состояние path из source подхватывается в destination и попадает в staging.

## Аргументы

| # | Аргумент | Описание |
|---|----------|----------|
| 1 | **source_branch** | Ветка, откуда берём файл/каталог |
| 2 | **dest_branch** | Ветка назначения (куда вносим изменения) |
| 3 | **path** | Путь к файлу или каталогу (от корня репо) |

## Steps

### 1. Проверить аргументы

- Убедиться, что переданы все три аргумента
- Если передан только path — спросить source и dest

### 2. Проверить наличие веток и path

```bash
git branch -a | grep -E "<source_branch>|<dest_branch>"
git show <source_branch>:<path> 2>/dev/null || echo "path не существует в source"
```

- Если ветки нет — сообщить пользователю
- Если path не существует в source — предупредить до выполнения

### 3. Сохранить незакоммиченные изменения (если есть)

```bash
git status
```

- При наличии changes в working dir/index — спросить: stash или отменить? Для `git checkout --` текущие изменения path будут перезаписаны

### 4. Выполнить перенос

```bash
git checkout <dest_branch>
git checkout <source_branch> -- <path>
```

### 5. Проверить результат

```bash
git status
git diff --cached <path>  # если нужно показать diff
```

## Example

```
/merge-file-from-branch main develop scripts/vehiclemakenet_finetune/dataset.py
```

→ Переключаемся на `develop`, берём `scripts/vehiclemakenet_finetune/dataset.py` из `main`, изменения в staging.

## Rules

- Не выполнять без явных аргументов (source_branch, dest_branch, path)
- Путь — относительно корня репозитория
- При конфликтах с working dir — предупредить до перезаписи
- После выполнения: показать `git status` и кратко описать, что было перенесено
