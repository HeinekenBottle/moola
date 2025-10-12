# Claude Code Plugins Export - Index

**Location:** `~/projects/moola/.droid/claude-plugins-export/`
**Total Files:** 395+ markdown files
**Exported:** 2025-10-12

## 📖 Start Here

1. **`QUICKSTART_FOR_MOOLA.md`** ← START HERE! ⭐
   - Tailored for your ML trading project
   - Top 5 recommended plugins
   - Real usage examples

2. **`README_FOR_DROID.md`**
   - Overview of what's included
   - Directory structure
   - General usage guide

3. **`DROID_CONVERSION_GUIDE.md`**
   - How to adapt Claude plugins for Droid
   - Step-by-step conversion process
   - Examples and templates

4. **`PATHS_REFERENCE.md`**
   - Where everything is located
   - Quick access commands
   - Search tips

5. **`INDEX.md`** (this file)
   - Quick navigation

## 🎯 Quick Access by Category

### Git & DevOps
```
commit-commands/
├── commands/
│   ├── commit.md              ⭐ Smart commit messages
│   ├── commit-push-pr.md      Push + PR in one command
│   └── clean_gone.md          Clean stale branches
```

### ML & AI Development
```
templates/components/agents/data-ai/
├── ml-engineer.md             ⭐⭐⭐ ML production systems
├── ai-engineer.md             ⭐⭐⭐ LLM/RAG systems
├── mlops-engineer.md          ⭐⭐ MLOps infrastructure
├── data-scientist.md          ⭐⭐ Exploratory analysis
├── data-engineer.md           Data pipelines
├── nlp-engineer.md            NLP specialists
├── computer-vision-engineer.md CV specialists
└── quant-analyst.md           ⭐ Quantitative analysis
```

### Performance Optimization
```
templates/components/agents/performance-testing/
├── performance-engineer.md    ⭐⭐⭐ Bottleneck analysis
├── load-testing-specialist.md Load/stress testing
├── web-vitals-optimizer.md    Web performance
└── test-automator.md          Test automation
```

### Feature Development
```
feature-dev/
├── commands/
│   └── feature-dev.md         ⭐⭐ Guided feature implementation
└── agents/
    ├── code-reviewer.md       Code quality review
    ├── code-explorer.md       Codebase navigation
    └── code-architect.md      Architecture design
```

### PR Review & Code Quality
```
pr-review-toolkit/
├── commands/
│   └── review-pr.md           Comprehensive PR analysis
└── agents/
    ├── code-reviewer.md
    ├── silent-failure-hunter.md
    ├── code-simplifier.md
    └── [more...]
```

### SDK Development
```
agent-sdk-dev/
├── commands/
│   └── new-sdk-app.md         Bootstrap SDK apps
└── agents/
    ├── agent-sdk-verifier-ts.md
    └── agent-sdk-verifier-py.md
```

### Security
```
security-guidance/
└── [security analysis tools]
```

## ⭐ Top 10 Most Useful for Moola

Based on your ML trading project:

| Rank | File | Use Case |
|------|------|----------|
| 1 | `data-ai/ml-engineer.md` | Model training & optimization |
| 2 | `performance-testing/performance-engineer.md` | Performance audits |
| 3 | `commit-commands/commands/commit.md` | Smart git commits |
| 4 | `data-ai/mlops-engineer.md` | RunPod deployment |
| 5 | `data-ai/data-scientist.md` | Feature exploration |
| 6 | `data-ai/quant-analyst.md` | Trading-specific analysis |
| 7 | `feature-dev/commands/feature-dev.md` | New feature implementation |
| 8 | `data-ai/data-engineer.md` | Data pipeline work |
| 9 | `pr-review-toolkit/commands/review-pr.md` | PR reviews |
| 10 | `commit-commands/commands/commit-push-pr.md` | Quick PR creation |

## 🔍 Finding What You Need

### By Keyword
```bash
# Search all files
grep -r "keyword" .droid/claude-plugins-export/

# Search just agents
grep -r "keyword" .droid/claude-plugins-export/templates/components/agents/

# Search just commands
find .droid/claude-plugins-export -name "*.md" -path "*/commands/*" | xargs grep -l "keyword"
```

### By Category
```bash
# List all ML agents
ls .droid/claude-plugins-export/templates/components/agents/data-ai/

# List all commands
find .droid/claude-plugins-export -type d -name "commands"

# List everything
find .droid/claude-plugins-export -name "*.md" | sort
```

## 📊 Statistics

- **Total Files:** 395 markdown files
- **Plugins:** 5 plugin collections
- **Template Agents:** 100+ specialized agents
- **Commands:** 20+ slash commands
- **Categories:** 15+ agent categories

## 🚦 Status

✅ **Exported:** All plugins and templates copied
✅ **Documentation:** 5 guide documents created
✅ **Ready:** For Droid experimentation

## 🔄 Updating

To refresh this export with latest Claude Code plugins:

```bash
# Update source repos
cd ~/.claude/plugins/marketplaces/claude-code-plugins && git pull
cd ~/.claude/plugins/marketplaces/claude-code-templates && git pull

# Re-export (warning: overwrites current export)
rm -rf ~/projects/moola/.droid/claude-plugins-export
mkdir -p ~/projects/moola/.droid/claude-plugins-export

cp -r ~/.claude/plugins/marketplaces/claude-code-plugins/plugins/* \
      ~/projects/moola/.droid/claude-plugins-export/

cp -r ~/.claude/plugins/marketplaces/claude-code-templates/cli-tool/components \
      ~/projects/moola/.droid/claude-plugins-export/templates/
```

## 📞 Support

- **Claude Code Repo:** https://github.com/anthropics/claude-code-plugins
- **Templates Repo:** https://github.com/anthropics/claude-code-templates
- **Droid Docs:** Check Factory AI documentation
- **Ask Claude:** I can help find/adapt specific plugins!

## 📝 Notes

- These are copies, not live links
- Adapt format for Droid (see DROID_CONVERSION_GUIDE.md)
- Test with simple prompts first
- Not all features will work in Droid (tool calls, etc.)
- Focus on the prompt logic, not the tooling

---

**Quick Start:** Open `QUICKSTART_FOR_MOOLA.md` for your tailored guide!

**Last Updated:** 2025-10-12
