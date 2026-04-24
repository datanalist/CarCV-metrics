---
name: rule-creator
description: Create high-quality Cursor rules (.mdc files) with industry best practices. Use proactively when the user wants to create a rule, add coding standards, set up project conventions, configure AI behavior for specific files, or asks about .cursor/rules/ directory.
---

# Rule Creator

Create Cursor rules (`.mdc` files in `.cursor/rules/`) that provide persistent, reusable instructions for AI agent behavior.

## What Are Cursor Rules

Rules are `.mdc` files with YAML frontmatter that control AI behavior in Cursor:
- Loaded automatically based on activation mode
- Provide consistent context across conversations
- Version-controlled in the project repository
- Support glob patterns for file-specific application

## File Format

```
---
description: "Clear description of what this rule enforces"
globs: "*.py,*.ts"
alwaysApply: false
---

# Rule Title

## Context
Why this rule exists (1 sentence).

## Rules
- Concise, actionable instructions
- Imperative mood: "Use X", "Avoid Y"

## Examples

### Good
` ` `python
# concrete positive example
` ` `

### Bad
` ` `python
# concrete anti-pattern
` ` `
```

## Frontmatter Fields

| Field | Type | Description |
|-------|------|-------------|
| `description` | string | What the rule enforces. Agent reads this to decide relevance. |
| `globs` | string/array | Glob patterns for auto-attachment: `"*.py"`, `"src/**/*.tsx"` |
| `alwaysApply` | boolean | `true` = included in every conversation |

## Activation Modes

Choose based on scope:

| Mode | Frontmatter | When to Use |
|------|-------------|-------------|
| **Always Apply** | `alwaysApply: true` | Universal project standards |
| **Auto Attached** | `globs: "*.py"`, `alwaysApply: false` | Language/directory-specific rules |
| **Agent Requested** | `description: "..."`, `alwaysApply: false` | AI decides from description |
| **Manual** | all empty/false | On-demand via `@rule-name` |

Decision tree:
1. Should this ALWAYS apply? → `alwaysApply: true`
2. Applies to specific file types? → set `globs`
3. AI can detect when it's relevant? → write clear `description`
4. Only needed occasionally? → leave all empty (manual `@`-mention)

## Workflow

### 1. Understand Intent

Determine:
- **What** behavior/standard the rule enforces
- **Why** it matters
- **Where** it applies (all files, specific types, specific directories)

If ambiguous — ask with `AskQuestion`:

**Scope** (always ask if not specified):
- "Всегда применять" (`alwaysApply: true`)
- "Только для определённых файлов" (set `globs`)

**File patterns** (if file-specific):
- `**/*.py`, `**/*.ts`, `scripts/**`, `notebooks/**/*.ipynb`, etc.

If the user's prompt is detailed enough — proceed directly.

### 2. Research Context

Before writing:
1. Read existing rules in `.cursor/rules/` — avoid conflicts and duplication
2. Scan the codebase (if relevant) — ground the rule in actual patterns
3. Check for existing conventions the rule should codify

### 3. Design the Rule

Follow these principles:

| Principle | Details |
|-----------|---------|
| **Concise** | Under 50 lines of body. Every line must earn its place. |
| **Actionable** | Instructions AI can act on immediately |
| **Concrete** | Include good/bad examples wherever possible |
| **One concern** | One rule per file. Split broad topics. |
| **No fluff** | No "Introduction", "Overview", "Background" sections |
| **Imperative** | "Use X" not "You should consider using X" |
| **Specific** | "Wrap async in try/catch" not "Write good error handling" |

### 4. Write the Rule

Create at `.cursor/rules/{rule-name}.mdc`.

**Naming**: kebab-case, descriptive: `python-style.mdc`, `error-handling.mdc`, `api-design.mdc`

**Structure**:

```
---
description: "Brief, clear description"
globs: "<pattern>"           # omit if alwaysApply: true
alwaysApply: <bool>
---

# Rule Title

## Context
One sentence: why this rule exists.

## Rules
- Actionable bullet points
- Imperative mood
- Specific, not vague

## Examples

### Good
[concrete positive example with code]

### Bad
[concrete anti-pattern with code]
```

**Tips**:
- Reference canonical files in codebase instead of copying code
- Group related file extensions: `"*.ts,*.tsx"` or `"*.{ts,tsx}"`
- For glob patterns, use recursive wildcards: `"src/**/*.ts"` not `"src/*.ts"`

### 5. Validate

After creating, verify:

1. YAML frontmatter parses correctly (`---` delimiters)
2. `description` is present and meaningful
3. Scope is correct — either `alwaysApply: true` or valid `globs`
4. Body is under 50 lines (excluding frontmatter)
5. Contains concrete examples (good/bad patterns)
6. No conflicts with existing rules in `.cursor/rules/`
7. File is at `.cursor/rules/{name}.mdc`

Read back the file and verify all checks pass.

### 6. Report

Tell the user:
- **Rule name**: `{name}`
- **File**: `.cursor/rules/{name}.mdc`
- **Scope**: Always / Auto Attached (pattern) / Agent Requested / Manual
- **Summary**: what the rule enforces (1-2 sentences)
- **Validation**: PASSED / FAILED

## Common Rule Patterns

### Universal Standards (Always Apply)

```
---
description: ""
alwaysApply: true
---
# Coding Standards

## Rules
- PEP 8 for Python, type hints on all functions
- Descriptive names, no abbreviations
- Keep functions under 30 lines
```

### Language-Specific (Auto Attached)

```
---
description: "Python code conventions"
globs: "**/*.py"
alwaysApply: false
---
# Python Conventions

## Rules
- Type hints on all function signatures
- Use `pathlib.Path` instead of `os.path`
- Prefer f-strings over `.format()`
```

### Situational (Agent Requested)

```
---
description: "Database migration patterns. Apply when creating or modifying migrations, schemas, or seed files."
globs: ""
alwaysApply: false
---
# Database Migrations

## Rules
- Always create both `up` and `down` operations
- Add indexes for new foreign key columns
- Test migrations in both directions
```

## Anti-patterns

- **Too long** — rule > 50 lines = split into focused files
- **Too vague** — "write good code" → be specific
- **Duplicates linter** — don't repeat what ruff/eslint already enforces
- **Stale examples** — reference canonical code files, don't copy them
- **Wrong mode** — using Always Apply for rules only relevant to 1 file type
- **Conflicting rules** — two rules giving opposite instructions

## Rules

- Always start by reading `.cursor/skills/rule-creator/SKILL.md`.
- Always respond in the same language as the user.
- Never create a rule without understanding the user's intent.
- Never omit `description` from frontmatter.
- Never create rules longer than 50 lines of body.
- If a rule would exceed 50 lines, split into multiple focused rules.
- Always check existing rules before creating new ones.
- Prefer concrete good/bad examples over lengthy explanations.
- Use the project's actual code patterns in examples when possible.
- After writing, always read back and validate the file.
- Do not commit without user permission.
