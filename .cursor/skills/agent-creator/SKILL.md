---
name: agent-creator
description: Create custom Cursor subagents with YAML-frontmatter, prompts, and validation. Use proactively when the user needs a new subagent, wants to create an agent file, define a custom agent workflow, or asks about .cursor/agents/ directory.
---

# Agent Creator

Create fully functional Cursor agent files (`.cursor/agents/agent-{kebab-id}.md`, e.g. `agent-cv-engineer.md`) with proper YAML frontmatter and effective prompts.

## Agent File Format

```markdown
---
name: kebab-case-name
description: "What the agent does. Use proactively when {trigger condition}."
---

# Agent Title

{Role statement — one sentence.}

**Language**: Always respond in the same language as the user.

## First Action

{What the agent must do first — usually read a skill or gather context.}

## Workflow

{Numbered steps the agent follows.}

## Rules

- {Explicit constraints and boundaries.}
- Do not commit without user permission.
```

### Frontmatter

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | kebab-case, matches filename stem (`agent-<id>.md` → `name: agent-<id>`) |
| `description` | Yes | What + when. Must contain "Use proactively" with trigger |
| `model` | No | Override model if needed |

## Workflow

### 1. Gather Requirements

Determine before writing:

- **Purpose** — what task the agent handles
- **Name** — concise, kebab-case (e.g., `code-reviewer`, `test-writer`)
- **Triggers** — when the agent should activate
- **Linked skill** — does the agent need a skill for domain knowledge, or is it self-contained?
- **Constraints** — scope limits, language, tools

If the user provides a complete specification, skip questions and proceed.
If unclear — ask with `AskQuestion`. Do not guess critical details.

### 2. Check for Linked Skill

If the agent's domain requires specialized knowledge:

1. Check if a skill exists in `.cursor/skills/`
2. If yes — agent's first action must read that skill
3. If no — create skill first (use skill-creator)

Self-contained agents (no skill): all instructions in the prompt body.

### 3. Design the Prompt

Build from these blocks:

- **Role** (1 sentence) — who the agent is
- **First Action** — read linked skill, or gather initial context
- **Workflow** — numbered steps, one clear action each. Reference skill procedures, don't duplicate
- **Rules** — explicit DO/DON'T constraints. Always include:
  - "Always start by reading `.cursor/skills/{name}/SKILL.md`" (if linked)
  - "Do not commit without user permission"
- **Output** — what the agent returns to the user

### 4. Write the Agent File

Create at `.cursor/agents/agent-{kebab-id}.md` (YAML `name` must match the filename stem).

**Key principle: agent .md is a thin wrapper when linked to a skill.**

- Role + skill reference + brief workflow step names + rules
- All domain knowledge and detailed procedures live in the skill
- Never duplicate content from the linked skill
- Imperative mood: "Read the file", not "You should consider reading"
- Self-contained: agent has no memory of creation conversation
- With linked skill: target **< 50 lines**

### 5. Validate

After creating, verify ALL of the following:

1. File exists at `.cursor/agents/agent-{kebab-id}.md`
2. YAML frontmatter has `name` and `description`
3. `description` contains "Use proactively" with clear trigger
4. First action reads the linked skill (if any)
5. Has: role statement, workflow, rules
6. No duplicate instructions between agent and skill
7. With linked skill: agent .md < 50 lines
8. Role statement is specific (not "you are a helpful assistant")
9. Workflow steps are numbered and actionable
10. No vague language ("consider", "might want to", "if possible")

### 6. Report

Tell the user:

- **Agent name**: `{name}`
- **File path**: `.cursor/agents/agent-{kebab-id}.md`
- **Linked skill**: path or "none"
- **Proactive trigger**: when it activates
- **Validation**: PASSED / FAILED with details

## Patterns

### Agent with Skill (recommended)

Agent is a thin wrapper; skill holds domain knowledge:

```
## First Action

Read the skill before any work:

`.cursor/skills/{name}/SKILL.md`

This is mandatory. Do not proceed without reading the skill.
```

### Standalone Agent (simple tasks)

All instructions in the prompt body. Use when:

- Task is simple (< 50 lines)
- No reusable domain knowledge
- One-off or utility agent

### Agent with Context Gathering

First action reads project state:

```
## First Action

1. Read `docs/rules/task.md` for current task
2. Read `docs/architecture.md` for system context
```

## Anti-patterns

| Anti-pattern | Problem | Fix |
|---|---|---|
| Bloated agent | Duplicates skill content (details, examples, checklists) | Agent = role + skill ref + step names + rules |
| Verbose workflow | Agent describes sub-steps already in skill | "Follow the skill's workflow: 1. Step 2. Step" |
| Missing triggers | Description doesn't explain WHEN to use | "Use proactively when..." in description |
| Vague workflow | "Analyze the code and do what's needed" | Specific numbered steps with clear actions |
| No validation | Created but never verified | Always read back and check against step 5 |
| Hardcoded paths | Absolute paths break portability | Relative paths from project root |

## Rules

- Always respond in the same language as the user.
- Never create an agent without understanding its purpose first.
- Never omit `name` or `description` from YAML frontmatter.
- Never omit "Use proactively" from `description`.
- Agent .md with linked skill: thin wrapper, < 50 lines, zero content duplication with skill.
- Do not commit without user permission.
