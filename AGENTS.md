# AI Agent Instructions

## Skills

All AI agents (including Claude, Copilot, Codex, Cursor, etc.) **must** read and apply the skills located in:

```
MySkills/skills/
```

### How to load skills

Before starting any task, scan the `MySkills/skills/` directory and load all relevant skill files:

- Each subdirectory under `MySkills/skills/` represents a skill category.
- Each `.md` file within a skill directory contains instructions, patterns, and workflows for that skill.
- If the current task matches a skill's domain, read the corresponding skill file and follow its instructions.

### Current available skills

| Skill | Path | Description |
|-------|------|-------------|
| debug-pytorch-xpu-issues | `MySkills/skills/debug-pytorch-xpu-issues/pytorch-xpu-debug.md` | Debugging PyTorch XPU issues and common patterns |

### Rules

1. Always check the `MySkills/skills/` directory at the start of a session.
2. If a task relates to a loaded skill, follow the skill's instructions and workflows.
3. When new skill files are added to `MySkills/skills/`, automatically incorporate them.
