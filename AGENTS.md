# AI Agent Instructions

## Workspace Layout

```
/path/to/workspace/              # User workspace root (this is where AGENTS.md lives)
├── AGENTS.md                    # This file
├── MySkills/                    # Skills repository (read-only reference)
│   └── skills/                  # Skill files for agents to load
├── temp/                        # AI agent temporary working directory
├── tools/                       # Utility tools
│   └── sysmon                   # GPU information monitor
├── pytorch/                     # User code repositories (examples)
├── xpu_kernel_optimizer/        # ...
└── ...                          # Other user-cloned repos
```

- The workspace (`/path/to/workspace/`) contains various user-cloned code repositories. Agents should **not** create temporary files or scratch directories in the workspace root or inside user repos.
- For any temporary work (scratch files, test scripts, build artifacts, intermediate outputs, etc.), use the **temp directory**: `/path/to/workspace/temp/`.
- Agents may create new subdirectories under the temp directory organized by task, e.g. `temp/debug_sdpa_issue/`, `temp/build_test/`.
- Keep the workspace clean — only modify files in user repos when explicitly asked to.

# Env setup

Use `workspace/env.sh` to setup basic conda/oneapi environment.

## Tools

The following utility tools are available under `tools/`:

| Tool | Path | Description |
|------|------|-------------|
| sysmon | `tools/sysmon` | Monitor and display GPU information |

Agents may use these tools when relevant to the current task (e.g. run `tools/sysmon` to check GPU status).

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
| hardware-projection | `MySkills/skills/hardware-projection/SKILL.md` | Roofline model performance projection for operators/models on target hardware |
| hardware-property-query | `MySkills/skills/hardware-property-query/` | Query hardware device properties (GPU specs, memory, compute units) |
| pytorch-xpu-knowledge-base | `MySkills/skills/pytorch-xpu-knowledge-base.md` | PyTorch XPU knowledge base and reference |

### Rules

1. Always check the `MySkills/skills/` directory at the start of a session.
2. If a task relates to a loaded skill, follow the skill's instructions and workflows.
3. When new skill files are added to `MySkills/skills/`, automatically incorporate them.
4. Use the temp directory for all scratch/temporary work — do not pollute the workspace root or user repos with temp files.
