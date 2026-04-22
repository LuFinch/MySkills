# MySkills

A collection of skills for AI agents to read and apply during tasks.

## Prerequisites

- Your workspace is located at `/path/to/workspace/`
- This repository is cloned to `/path/to/workspace/MySkills/`

## Setup

Before running any AI agent, copy `AGENTS.md` from `MySkills/` to your workspace root (`/path/to/workspace/`) and run the agent from there:

```bash
# Copy AGENTS.md to the workspace root
cp /path/to/workspace/MySkills/AGENTS.md /path/to/workspace/

# Run your AI agent from the workspace root
cd /path/to/workspace
# launch your AI agent here
```

> **Important:** The AI agent must be launched from `/path/to/workspace/` (not from inside `MySkills/`), so that it can automatically discover the `AGENTS.md` instruction file and load the skills from `/path/to/workspace/MySkills/`.

## How it works

Once the agent is running in `/path/to/workspace/`, it reads `AGENTS.md` which instructs it to load skill files from `MySkills/skills/`. Skills are matched to the current task and applied automatically.

## Available skills

| Skill | Path | Description |
|-------|------|-------------|
| debug-pytorch-xpu-issues | `MySkills/skills/debug-pytorch-xpu-issues/SKILL.md` | Debugging PyTorch XPU issues and common patterns |
