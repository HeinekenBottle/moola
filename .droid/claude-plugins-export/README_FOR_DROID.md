# Claude Code Plugins Export for Droid

This directory contains exported Claude Code plugins and templates that you can use as reference or adapt for Droid by Factory AI.

## 📂 Directory Structure

```
.droid/claude-plugins-export/
├── README_FOR_DROID.md (this file)
├── commit-commands/          # Git workflow automation
├── feature-dev/               # Feature development toolkit
├── agent-sdk-dev/            # Claude Agent SDK helpers
├── pr-review-toolkit/        # PR review automation
├── security-guidance/        # Security analysis
└── templates/                # Additional agent templates
    └── components/
        ├── agents/           # Pre-built agent specifications
        └── commands/         # Slash command templates
```

## 🎯 What's Inside

### 1. **Plugins** (Root Level)

Each plugin directory contains:
- **`commands/`** - Slash command definitions (`.md` files)
- **`agents/`** - Specialized agent prompts (`.md` files)
- **`README.md`** - Plugin documentation

#### Available Plugins:

1. **`commit-commands/`** - Git workflow automation
   - `/commit` - Smart commit message generation
   - `/commit-push-pr` - Commit, push, and open PR in one command
   - `/clean_gone` - Clean up stale git branches

2. **`feature-dev/`** - Feature development toolkit
   - `/feature-dev` - Guided feature implementation
   - Agents: code-reviewer, code-explorer, code-architect

3. **`agent-sdk-dev/`** - Claude Agent SDK development
   - `/new-sdk-app` - Bootstrap new SDK applications
   - Agents: TypeScript and Python SDK verifiers

4. **`pr-review-toolkit/`** - PR review automation
   - `/review-pr` - Comprehensive PR analysis
   - Agents: code-reviewer, silent-failure-hunter, code-simplifier, etc.

5. **`security-guidance/`** - Security analysis tools

### 2. **Templates** (`templates/components/`)

Additional specialized agents organized by category:

- **`performance-testing/`** - Performance optimization agents
  - `performance-engineer.md`
  - `load-testing-specialist.md`
  - `web-vitals-optimizer.md`
  - `test-automator.md`

- **`ai-ml-toolkit/`** - ML/AI specialized agents
  - `ai-engineer.md` - LLM/RAG systems
  - `ml-engineer.md` - ML production systems
  - `nlp-engineer.md` - NLP specialists
  - `computer-vision-engineer.md` - CV specialists
  - `mlops-engineer.md` - MLOps infrastructure

- **`project-management-suite/`** - Project management agents
  - `product-strategist.md`
  - `business-analyst.md`

## 🔧 How to Use with Droid

### Option 1: Direct Copy (Easiest)
```bash
# Copy a specific plugin's command to your Droid droids folder
cp .droid/claude-plugins-export/commit-commands/commands/commit.md ~/path/to/droid/droids/
```

### Option 2: Adapt the Structure

Claude Code uses this format:
```markdown
# Command Name

Command description here.

## Behavior

Detailed instructions for Claude Code...
```

For Droid, you might need to adapt to their format. Check Droid's documentation for their prompt structure.

### Option 3: Use as Reference

Read through the `.md` files to understand:
1. How the prompts are structured
2. What instructions work well
3. How to chain commands together
4. Best practices for agent definitions

## 📝 File Format

All commands and agents are stored as **Markdown files** (`.md`).

Structure:
- **Title** - Agent/command name
- **Description** - What it does
- **Instructions** - Detailed prompt for the AI
- **Examples** (sometimes) - Usage examples

## 🎨 Example: Commit Command

Location: `commit-commands/commands/commit.md`

This is one of the most useful - it:
1. Analyzes `git status` and `git diff`
2. Reads recent commit messages for style
3. Generates a contextual commit message
4. Stages files and creates the commit
5. Handles pre-commit hooks

You can adapt this pattern for any git workflow automation.

## 🚀 Popular Use Cases for Droid

Based on what you were working on last night, here are the most relevant:

### 1. **ML/AI Development**
Files: `templates/components/agents/ai-ml-toolkit/*`
- Use `ai-engineer.md` for LLM/RAG work
- Use `ml-engineer.md` for model training/deployment
- Use `mlops-engineer.md` for ML pipelines

### 2. **Performance Optimization**
Files: `templates/components/agents/performance-testing/*`
- Use `performance-engineer.md` for bottleneck analysis
- Use `load-testing-specialist.md` for stress testing

### 3. **Git Workflows**
Files: `commit-commands/commands/*`
- Use `commit.md` for smart commits
- Use `clean_gone.md` for branch cleanup

### 4. **Feature Development**
Files: `feature-dev/commands/feature-dev.md`
- Guided feature implementation with codebase analysis

## 🔍 Quick Reference: Key Files

| Purpose | File Path |
|---------|-----------|
| Smart commits | `commit-commands/commands/commit.md` |
| PR creation | `commit-commands/commands/commit-push-pr.md` |
| Feature dev | `feature-dev/commands/feature-dev.md` |
| Code review | `pr-review-toolkit/commands/review-pr.md` |
| ML engineering | `templates/components/agents/ai-ml-toolkit/ml-engineer.md` |
| Performance audit | `templates/components/agents/performance-testing/performance-engineer.md` |

## 💡 Tips for Adapting to Droid

1. **Check Droid's prompt format** - They might use a different structure
2. **Simplify if needed** - Some Claude Code commands are very complex
3. **Remove Claude-specific features** - Like tool calls that Droid doesn't have
4. **Test incrementally** - Start with simple commands first
5. **Combine prompts** - You can merge multiple agent prompts into one

## 📚 Original Sources

- **Plugins:** `~/.claude/plugins/marketplaces/claude-code-plugins/`
- **Templates:** `~/.claude/plugins/marketplaces/claude-code-templates/`

These are clones of the official repos:
- https://github.com/anthropics/claude-code-plugins
- https://github.com/anthropics/claude-code-templates

## 🆘 Need Help?

1. Read the original plugin's `README.md` in each directory
2. Look at the examples in the official repos
3. Check Droid's documentation for their format requirements
4. Test with simple prompts first before complex ones

---

**Note:** These are exported from Claude Code for reference. You'll need to adapt them to work with Droid's specific format and capabilities.

Last exported: 2025-10-12
