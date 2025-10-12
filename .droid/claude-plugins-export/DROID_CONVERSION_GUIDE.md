# Converting Claude Code Plugins to Droid

This guide shows you how to adapt Claude Code plugins for use with Droid by Factory AI.

## 📋 Quick Comparison

| Feature | Claude Code | Droid |
|---------|-------------|-------|
| Command format | Slash commands (`.md` files) | Droids (custom format - check docs) |
| Agent system | Built-in with Task tool | Custom droid definitions |
| Tools | Rich tool ecosystem | Different tool set |
| Dynamic content | `!`command`` syntax | TBD - check Droid docs |
| Metadata | YAML frontmatter | TBD - check Droid docs |

## 🔧 Conversion Process

### Step 1: Understand Claude Code Format

Claude Code commands use this structure:

```markdown
---
allowed-tools: Bash(git add:*), Bash(git status:*)
description: Create a git commit
---

## Context

- Current git status: !`git status`
- Current git diff: !`git diff HEAD`

## Your task

Based on the above changes, create a single git commit.

Instructions for AI...
```

**Key features:**
- **YAML frontmatter** - Metadata about the command
- **Dynamic shell execution** - `!`command`` runs before prompt
- **Tool restrictions** - Can limit which tools the AI uses
- **Structured sections** - Context, Task, Examples, etc.

### Step 2: Identify What to Keep/Adapt/Remove

#### ✅ **Keep (Universal)**
- Core prompt logic
- Problem-solving approach
- Best practices
- Examples and patterns

#### 🔄 **Adapt (Platform-Specific)**
- Command invocation format
- Tool calls (Claude's tools vs Droid's)
- Dynamic content generation
- Metadata structure

#### ❌ **Remove (Claude-Specific)**
- Claude Code-specific tool references
- Task tool calls (for sub-agents)
- SlashCommand tool usage
- TodoWrite tool calls

### Step 3: Example Conversion

#### Original Claude Code Command
```markdown
---
allowed-tools: Bash(git add:*), Bash(git status:*)
description: Create a git commit
---

## Context
- Current git status: !`git status`
- Current git diff: !`git diff HEAD`

## Your task
Create a single git commit based on the changes above.
```

#### Converted for Droid (Hypothetical - check Droid docs!)
```markdown
# Smart Commit

Create intelligent git commits with contextual messages.

## Instructions

1. First, analyze the current state:
   - Run: git status
   - Run: git diff HEAD
   - Run: git log --oneline -5

2. Based on the changes:
   - Identify the type (feat/fix/refactor/docs)
   - Write a concise summary
   - Stage relevant files
   - Create the commit

3. Follow conventional commits format:
   - type(scope): description
   - Include body if needed
   - Reference issues if applicable

## Example

Input: Modified login.js and added tests
Output:
```
git add login.js login.test.js
git commit -m "feat(auth): add two-factor authentication

- Implement TOTP verification
- Add comprehensive test coverage
- Update user model schema"
```
```

**Changes made:**
1. ❌ Removed YAML frontmatter (adapt to Droid's format)
2. ❌ Removed `!`command`` (run commands explicitly in prompt)
3. ✅ Kept core logic and instructions
4. ✅ Added explicit steps
5. ✅ Kept example pattern

## 🎯 Plugin-Specific Conversion Tips

### 1. **Commit Commands** (`commit-commands/`)

**Core value:** Smart git workflow automation

**What to adapt:**
- Remove `allowed-tools` restrictions
- Convert dynamic git commands to explicit instructions
- Keep the commit message generation logic
- Keep the conventional commits format

**Droid implementation:**
```bash
# Save as: droids/smart-commit.md
Create smart git commits by:
1. Analyzing git status and diff
2. Reading recent commits for style
3. Generating contextual message
4. Staging and committing files
```

### 2. **Feature Dev** (`feature-dev/`)

**Core value:** Guided feature implementation with codebase analysis

**What to adapt:**
- Remove Task tool calls for sub-agents
- Keep the analysis framework
- Keep the step-by-step approach
- Adapt file search to Droid's capabilities

**Droid implementation:**
```bash
# Save as: droids/feature-dev.md
Implement features with:
1. Codebase exploration phase
2. Architecture planning
3. Implementation with tests
4. Documentation updates
```

### 3. **PR Review** (`pr-review-toolkit/`)

**Core value:** Comprehensive PR analysis

**What to adapt:**
- Remove `gh` CLI commands if Droid doesn't support
- Keep the review checklist
- Keep the analysis categories
- Adapt to Droid's file reading capabilities

**Droid implementation:**
```bash
# Save as: droids/review-pr.md
Review PRs by analyzing:
1. Code quality and patterns
2. Test coverage
3. Security concerns
4. Performance implications
5. Documentation completeness
```

### 4. **ML/AI Agents** (`templates/components/agents/data-ai/`)

**Core value:** Specialized ML engineering knowledge

**What to adapt:**
- These are mostly pure prompts (easiest to convert!)
- Remove tool-specific references
- Keep the domain expertise
- Keep the problem-solving approaches

**Droid implementation:**
```bash
# Save as: droids/ml-engineer.md
ML engineering specialist for:
- Model training optimization
- Production deployment
- Feature engineering
- A/B testing frameworks
[Copy most of the original prompt]
```

### 5. **Performance Testing** (`templates/components/agents/performance-testing/`)

**Core value:** Performance optimization expertise

**What to adapt:**
- Remove Claude-specific profiling tools
- Keep optimization strategies
- Keep benchmarking approaches
- Adapt to tools Droid has access to

## 🛠️ Step-by-Step Conversion Template

Use this template for any plugin:

### 1. Read the Original
```bash
cat .droid/claude-plugins-export/[plugin-name]/commands/[command].md
```

### 2. Extract Core Logic
- What problem does it solve?
- What steps does it follow?
- What knowledge does it contain?

### 3. Remove Claude-Specific Features
- `---` YAML frontmatter
- `!`command`` dynamic execution
- `Task()` tool calls
- `TodoWrite()` calls
- `SlashCommand()` calls

### 4. Adapt to Droid Format
Check Droid's documentation for:
- How to define droids
- What tools are available
- How to structure prompts
- How to pass context

### 5. Test and Iterate
- Start with simple commands
- Test with real scenarios
- Refine based on results

## 📚 Recommended Conversion Order

Start with these (easiest → hardest):

1. ✅ **ML/AI Agents** - Pure prompts, minimal tool usage
   - `templates/components/agents/data-ai/ml-engineer.md`
   - `templates/components/agents/data-ai/ai-engineer.md`

2. ✅ **Performance Agents** - Mostly advisory
   - `templates/components/agents/performance-testing/performance-engineer.md`

3. 🔄 **Commit Commands** - Simple bash commands
   - `commit-commands/commands/commit.md`

4. 🔄 **Feature Dev** - More complex, uses file ops
   - `feature-dev/commands/feature-dev.md`

5. ⚠️ **PR Review** - Complex, uses GitHub CLI
   - `pr-review-toolkit/commands/review-pr.md`

## 💡 Pro Tips

### Tip 1: Start with Agents, Not Commands
Agents (pure prompts) are easier to convert than commands (which use tools).

### Tip 2: Simplify First
Don't try to replicate all features. Start with core functionality.

### Tip 3: Use Composition
Break complex commands into smaller droids, then chain them.

### Tip 4: Test with Real Code
Use your actual projects to test - like this moola repo!

### Tip 5: Document Your Changes
Keep notes on what works and what doesn't for future reference.

## 🔍 Example: Full Conversion

Let's convert the **ML Engineer** agent:

### Original (`ml-engineer.md`)
```markdown
---
name: ml-engineer
description: ML production systems and model deployment specialist
proactive: true
---

You are an expert ML engineer specializing in...
[500+ lines of detailed prompt]
```

### For Droid
```markdown
# ML Engineer Droid

Expert in ML production systems and deployment.

## Specializations
- Model training and optimization
- Production deployment (Docker, K8s)
- Feature engineering pipelines
- A/B testing frameworks
- Model monitoring and debugging

## When to Use
- Deploying models to production
- Optimizing training pipelines
- Setting up feature stores
- Implementing model serving
- Building ML infrastructure

## Approach
1. **Understand context** - Current stack and constraints
2. **Plan architecture** - Design scalable solutions
3. **Implement** - Write production-ready code
4. **Monitor** - Set up logging and metrics
5. **Iterate** - Optimize based on results

[Copy relevant sections from original 500 lines]
```

**What changed:**
- ✅ Kept core expertise
- ✅ Kept problem-solving approach
- ✅ Kept best practices
- ❌ Removed proactive flag (Droid-specific)
- ❌ Removed tool references
- 🔄 Adapted structure to be more readable

## 📖 Resources

- **Claude Code Plugins Repo:** https://github.com/anthropics/claude-code-plugins
- **Claude Code Templates Repo:** https://github.com/anthropics/claude-code-templates
- **This Export:** `.droid/claude-plugins-export/`
- **Droid Docs:** Check Factory AI documentation

## 🆘 Troubleshooting

### Issue: "Droid doesn't understand the format"
→ Check Droid's docs for their expected format

### Issue: "Dynamic commands don't work"
→ Droid probably doesn't support `!`command`` syntax. Run commands explicitly in the prompt.

### Issue: "Tool calls fail"
→ Droid has different tools than Claude Code. Check what's available.

### Issue: "Prompt too long"
→ Some Claude prompts are 500+ lines. Trim to essentials for Droid.

### Issue: "Agent doesn't activate"
→ Check Droid's trigger system - might be different from Claude's proactive flag.

## 🎉 Success Stories (Add Yours!)

Document your successful conversions here:

```
[Date] - Converted [plugin] - Works great for [use case]
```

---

**Last Updated:** 2025-10-12
**Author:** Claude Code (exported for Jack's Droid experiments)

Good luck with your Droid conversions! 🚀
