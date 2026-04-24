# Примеры команд

Готовые примеры для каждого паттерна. Использовать как референс при создании новых команд.

---

## Пошаговый паттерн

### Code Review

```markdown
# Code Review

## Overview

Perform a thorough code review that verifies functionality, maintainability, and
security before approving a change.

## Steps

1. **Understand the change**
   - Read the PR description and related issues for context
   - Identify the scope of files and features impacted
2. **Validate functionality**
   - Confirm the code delivers the intended behavior
   - Exercise edge cases mentally or by running locally
   - Check error handling paths and logging
3. **Assess quality**
   - Ensure functions are focused, names are descriptive
   - Watch for duplication, dead code, or missing tests
4. **Review security and risk**
   - Look for injection points, insecure defaults
   - Confirm secrets or credentials are not exposed

## Checklist

- [ ] Intended behavior works and matches requirements
- [ ] Edge cases handled gracefully
- [ ] No unnecessary duplication or dead code
- [ ] No obvious security vulnerabilities introduced
- [ ] Tests/documentation updated as needed
```

### Debug Issue

```markdown
# Debug Issue

## Overview

Systematically debug the current issue and provide actionable solutions.

## Steps

1. **Problem Analysis**
   - Identify the specific problem or error
   - Understand the expected vs actual behavior
   - Trace the execution flow to find the root cause
2. **Debugging Strategy**
   - Add appropriate logging statements
   - Identify key variables and states to monitor
   - Recommend breakpoint locations
3. **Solution Approach**
   - Propose potential fixes with explanations
   - Evaluate trade-offs of different approaches
   - Provide step-by-step resolution plan
4. **Prevention**
   - Suggest ways to prevent similar issues
   - Recommend additional tests or checks

## Checklist

- [ ] Identified the specific problem
- [ ] Traced execution flow to root cause
- [ ] Proposed fixes with explanations
- [ ] Suggested prevention measures
```

### Write Unit Tests

```markdown
# Write Unit Tests

## Overview

Create comprehensive unit tests for the current code using the project's testing conventions.

## Steps

1. **Test Coverage**
   - Test all public methods and functions
   - Cover edge cases and error conditions
   - Test both positive and negative scenarios
2. **Test Structure**
   - Use the project's testing framework conventions
   - Write clear, descriptive test names
   - Follow the Arrange-Act-Assert pattern
3. **Test Cases to Include**
   - Happy path scenarios
   - Edge cases and boundary conditions
   - Error handling and exception cases
   - Mock external dependencies appropriately

## Checklist

- [ ] Tested all public methods and functions
- [ ] Covered edge cases and error conditions
- [ ] Used the project's testing framework
- [ ] Written clear, descriptive test names
- [ ] Followed the Arrange-Act-Assert pattern
- [ ] Mocked external dependencies
- [ ] Tests are independent and deterministic
```

---

## Директивный паттерн

### Deslop (удалить AI-мусор)

```markdown
# Remove AI code slop

Check the diff against main, and remove all AI generated slop introduced in this branch.

This includes:

- Extra comments that a human wouldn't add or is inconsistent with the rest of the file
- Extra defensive checks or try/catch blocks that are abnormal for that area of the codebase
- Casts to any to get around type issues
- Any other style that is inconsistent with the file

Report at the end with only a 1-3 sentence summary of what you changed.
```

### Clarify Task

```markdown
# Clarify Task

Before doing ANY coding work on the task I describe:

1. **Ask clarifying questions** - Use 2-4 multiple choice questions to clarify:
   - Data flow and architecture
   - APIs and integrations
   - Edge cases and error handling
   - UI/UX expectations (if applicable)

2. **Restate requirements** - After I answer, restate the final requirements to confirm understanding.

3. **Confirm before proceeding** - ONLY then ask if I want to proceed.

Keep asking questions until you have enough context to give an accurate & confident answer.
```

---

## Шаблонный паттерн

### Git Commit

```markdown
# Git Create Commit

## Overview

Create a short, focused commit message and commit staged changes.

## Steps

1. **Review changes**
   - Check the diff: `git diff --cached` (staged) or `git diff` (unstaged)
   - Understand what changed and why
2. **Stage changes (if not already staged)**
   - `git add -A`
3. **Create short commit message**
   - Base the message on the actual changes in the diff
   - Example: `git commit -m "fix(auth): handle expired token refresh"`

## Template

`git commit -m "<type>(<scope>): <summary>"`

## Rules

- **Length:** <= 72 characters
- **Imperative mood:** "fix", "add", "update" (not "fixed", "added")
- **Capitalize:** First letter of summary
- **No period:** Don't end the subject line with a period
- **Describe why:** Not just what
```

### Create PR

```markdown
# Create PR

## Overview

Create a well-structured pull request with proper description.

## Steps

1. **Prepare branch**
   - Ensure all changes are committed
   - Push branch to remote
   - Verify branch is up to date with main
2. **Write PR description**
   - Summarize changes clearly
   - Include context and motivation
   - List any breaking changes
3. **Set up PR**
   - Create PR with descriptive title
   - Add appropriate labels
   - Link related issues

## PR Template

- [ ] Feature/bug fix implemented
- [ ] Tests pass
- [ ] Manual testing completed
```

---

## Визуальный паттерн

### Diagrams

```markdown
# Generate Mermaid Diagram

## Overview

Analyze the provided code or concept and generate a Mermaid diagram.

## Instructions

1. **Analyze the input** - Understand what to visualize
2. **Choose diagram type**:
   - `flowchart` - Process flows, decision trees
   - `sequenceDiagram` - API calls, request/response
   - `classDiagram` - Class structures, inheritance
   - `erDiagram` - Database schemas
   - `stateDiagram-v2` - State machines
3. **Generate the diagram** with:
   - Clear, descriptive node labels
   - Logical grouping with subgraphs
   - Meaningful relationship labels
   - Max ~15-20 nodes per diagram

## Style Guidelines

- Descriptive IDs: `userService` not `a1`
- Arrow styles: `-->` solid, `-.->` dotted, `==>` thick
- Use subgraphs to group related components
```

### Overview

```markdown
# Overview: Visual Architecture Diagram

Generate two Mermaid diagrams to overview the product.

## Diagram 1: User Journey

- 5-7 nodes max, action verbs
- `flowchart LR` with subgraphs

## Diagram 2: Architecture Flow

- `sequenceDiagram` showing temporal flow
- 4-6 participants max

## Output

Render directly in chat:
1. 2-paragraph product description
2. User journey diagram
3. Architecture diagram
```

---

## Другие полезные паттерны

### Lint and Fix

```markdown
# Lint and Fix Code

## Overview

Analyze the current file for linting issues and fix them.

## Steps

1. **Identify issues**
   - Formatting and style consistency
   - Unused imports and variables
   - Best practice violations
   - Type safety issues
2. **Apply fixes**
   - Fix all identified issues
   - Explain what changed

## Checklist

- [ ] Fixed formatting and style issues
- [ ] Removed unused imports/variables
- [ ] Applied best practice corrections
- [ ] Fixed type safety issues
- [ ] Explained changes
```

### Refactor Code

```markdown
# Refactor Code

## Overview

Refactor the selected code to improve quality while maintaining functionality.

## Steps

1. **Code Quality**
   - Extract reusable functions
   - Eliminate duplication
   - Improve naming
   - Simplify complex logic
2. **Performance**
   - Identify bottlenecks
   - Optimize algorithms and data structures
3. **Maintainability**
   - Make code more readable
   - Follow SOLID principles
   - Improve error handling

## Checklist

- [ ] Extracted reusable functions
- [ ] Eliminated duplication
- [ ] Improved naming
- [ ] Simplified complex logic
- [ ] Made code more readable
- [ ] Improved error handling
```

---

## Источник

Примеры адаптированы из [cursor-commands](https://github.com/hamzafer/cursor-commands).
