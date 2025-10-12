# Claude Code Plugin Paths Reference

Quick reference for where everything is stored.

## 📍 Original Locations (On Your Mac)

### Global Claude Directory
```bash
~/.claude/
```

### Plugins Source
```bash
~/.claude/plugins/marketplaces/claude-code-plugins/
~/.claude/plugins/marketplaces/claude-code-templates/
```

### Full Paths
```bash
# Plugins
~/.claude/plugins/marketplaces/claude-code-plugins/plugins/
├── commit-commands/
├── feature-dev/
├── agent-sdk-dev/
├── pr-review-toolkit/
└── security-guidance/

# Templates
~/.claude/plugins/marketplaces/claude-code-templates/cli-tool/components/
├── agents/
│   ├── ai-ml-toolkit/
│   ├── performance-testing/
│   ├── project-management-suite/
│   └── [many more...]
└── commands/
```

## 📂 Exported Location (For Droid)

### This Export Directory
```bash
~/projects/moola/.droid/claude-plugins-export/
```

### Structure
```bash
.droid/claude-plugins-export/
├── README_FOR_DROID.md          # Main documentation
├── DROID_CONVERSION_GUIDE.md    # How to adapt for Droid
├── PATHS_REFERENCE.md           # This file
│
├── commit-commands/             # Git workflow automation
│   └── commands/
│       ├── commit.md
│       ├── commit-push-pr.md
│       └── clean_gone.md
│
├── feature-dev/                 # Feature development toolkit
│   ├── commands/
│   │   └── feature-dev.md
│   └── agents/
│       ├── code-reviewer.md
│       ├── code-explorer.md
│       └── code-architect.md
│
├── agent-sdk-dev/              # Claude Agent SDK helpers
│   ├── commands/
│   │   └── new-sdk-app.md
│   └── agents/
│       ├── agent-sdk-verifier-ts.md
│       └── agent-sdk-verifier-py.md
│
├── pr-review-toolkit/          # PR review automation
│   ├── commands/
│   │   └── review-pr.md
│   └── agents/
│       ├── code-reviewer.md
│       ├── silent-failure-hunter.md
│       ├── code-simplifier.md
│       └── [more...]
│
├── security-guidance/          # Security analysis
│
└── templates/                  # Additional specialized agents
    └── components/
        └── agents/
            ├── data-ai/                    # ML/AI specialists
            │   ├── ai-engineer.md
            │   ├── ml-engineer.md
            │   ├── mlops-engineer.md
            │   ├── nlp-engineer.md
            │   ├── computer-vision-engineer.md
            │   ├── data-engineer.md
            │   ├── data-scientist.md
            │   └── quant-analyst.md
            │
            ├── performance-testing/        # Performance experts
            │   ├── performance-engineer.md
            │   ├── load-testing-specialist.md
            │   ├── web-vitals-optimizer.md
            │   └── test-automator.md
            │
            ├── project-management-suite/   # PM/Business
            │   ├── product-strategist.md
            │   └── business-analyst.md
            │
            └── [many more categories...]
```

## 🔍 Finding Specific Files

### By Category

#### Git/DevOps
```bash
.droid/claude-plugins-export/commit-commands/commands/
```

#### ML/AI Development
```bash
.droid/claude-plugins-export/templates/components/agents/data-ai/
```

#### Performance Optimization
```bash
.droid/claude-plugins-export/templates/components/agents/performance-testing/
```

#### Code Review
```bash
.droid/claude-plugins-export/pr-review-toolkit/
```

#### Feature Development
```bash
.droid/claude-plugins-export/feature-dev/
```

### By File Type

#### All Commands
```bash
find .droid/claude-plugins-export -type d -name "commands"
```

#### All Agents
```bash
find .droid/claude-plugins-export -type d -name "agents"
```

#### All Markdown Files
```bash
find .droid/claude-plugins-export -name "*.md"
```

## 🚀 Quick Access Commands

### View a Specific Plugin
```bash
# Commit automation
cat .droid/claude-plugins-export/commit-commands/commands/commit.md

# ML Engineer agent
cat .droid/claude-plugins-export/templates/components/agents/data-ai/ml-engineer.md

# Performance audit
cat .droid/claude-plugins-export/templates/components/agents/performance-testing/performance-engineer.md
```

### List All Available
```bash
# List all commands
find .droid/claude-plugins-export -name "*.md" -path "*/commands/*"

# List all agents
find .droid/claude-plugins-export -name "*.md" -path "*/agents/*"
```

### Search for Keywords
```bash
# Find all ML-related agents
find .droid/claude-plugins-export -name "*.md" | xargs grep -l "machine learning"

# Find performance-related content
find .droid/claude-plugins-export -name "*.md" | xargs grep -l "performance"
```

## 📋 Copy Commands for Droid

### Copy Specific Files
```bash
# Copy commit command to Droid
cp .droid/claude-plugins-export/commit-commands/commands/commit.md ~/path/to/droid/droids/

# Copy ML engineer agent
cp .droid/claude-plugins-export/templates/components/agents/data-ai/ml-engineer.md ~/path/to/droid/droids/

# Copy performance engineer
cp .droid/claude-plugins-export/templates/components/agents/performance-testing/performance-engineer.md ~/path/to/droid/droids/
```

### Copy Entire Categories
```bash
# Copy all commit commands
cp -r .droid/claude-plugins-export/commit-commands ~/path/to/droid/

# Copy all ML/AI agents
cp -r .droid/claude-plugins-export/templates/components/agents/data-ai ~/path/to/droid/
```

## 🎯 Most Useful Files for ML Work

Based on your project (moola - ML trading system):

### Essential
```bash
# 1. ML Engineering
.droid/claude-plugins-export/templates/components/agents/data-ai/ml-engineer.md

# 2. Performance Optimization
.droid/claude-plugins-export/templates/components/agents/performance-testing/performance-engineer.md

# 3. Smart Commits
.droid/claude-plugins-export/commit-commands/commands/commit.md
```

### Highly Recommended
```bash
# 4. Data Science
.droid/claude-plugins-export/templates/components/agents/data-ai/data-scientist.md

# 5. MLOps
.droid/claude-plugins-export/templates/components/agents/data-ai/mlops-engineer.md

# 6. Quant Analysis (for trading!)
.droid/claude-plugins-export/templates/components/agents/data-ai/quant-analyst.md

# 7. Feature Development
.droid/claude-plugins-export/feature-dev/commands/feature-dev.md
```

## 🔄 Updating This Export

If you want to refresh from the latest Claude Code plugins:

```bash
# Update plugins
cd ~/.claude/plugins/marketplaces/claude-code-plugins/
git pull

# Update templates
cd ~/.claude/plugins/marketplaces/claude-code-templates/
git pull

# Re-export to moola
rm -rf ~/projects/moola/.droid/claude-plugins-export
mkdir -p ~/projects/moola/.droid/claude-plugins-export

cp -r ~/.claude/plugins/marketplaces/claude-code-plugins/plugins/* \
      ~/projects/moola/.droid/claude-plugins-export/

cp -r ~/.claude/plugins/marketplaces/claude-code-templates/cli-tool/components \
      ~/projects/moola/.droid/claude-plugins-export/templates/
```

## 📱 Access from Droid

When running Droid from your project directory:

```bash
# You're here
cd ~/projects/moola

# Plugins are here
ls .droid/claude-plugins-export/

# So Droid can access like:
# ./.droid/claude-plugins-export/[plugin]/[command].md
```

---

**Tip:** Add `.droid/` to your `.gitignore` if you don't want to commit this export to your repo.

**Note:** These are local copies. Original repos:
- https://github.com/anthropics/claude-code-plugins
- https://github.com/anthropics/claude-code-templates

**Last Updated:** 2025-10-12
