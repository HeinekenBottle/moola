# Quick Start: Using Claude Plugins with Droid for Moola

Tailored guide for using these exported plugins with Droid on your ML trading project.

## 🎯 Your Project Context

**Project:** Moola - ML-based trading system
**Tech Stack:** Python, PyTorch, XGBoost, sklearn
**Focus:** Model training, stacking ensembles, GPU optimization
**Current Work:** Just fixed GPU utilization issues, optimizing performance

## 🚀 Top 5 Plugins to Try First

Based on what you're working on, here are the most relevant plugins to convert for Droid:

### 1. **ML Engineer** ⭐⭐⭐⭐⭐
**Why:** Your bread and butter for ML development

```bash
# Location
.droid/claude-plugins-export/templates/components/agents/data-ai/ml-engineer.md

# Use for
- Model training optimization
- Deployment planning
- Feature engineering
- A/B testing setup
```

**How to use with Droid:**
1. Copy the file to your Droid droids folder
2. Adapt the format (remove YAML frontmatter if needed)
3. Call it when working on model improvements

**Example Droid prompt:**
```
Using ml-engineer: Help me optimize the CNN-Transformer training pipeline
```

---

### 2. **Performance Engineer** ⭐⭐⭐⭐⭐
**Why:** You JUST did a performance audit - this agent specializes in that!

```bash
# Location
.droid/claude-plugins-export/templates/components/agents/performance-testing/performance-engineer.md

# Use for
- Profiling bottlenecks
- Optimization recommendations
- Load testing
- GPU/CPU utilization analysis
```

**Perfect for:**
- Following up on your recent GPU fixes
- Continuous performance monitoring
- Benchmarking improvements

**Example Droid prompt:**
```
Using performance-engineer: Profile the OOF generation pipeline and suggest optimizations
```

---

### 3. **Smart Commit** ⭐⭐⭐⭐
**Why:** You commit a lot during ML experiments

```bash
# Location
.droid/claude-plugins-export/commit-commands/commands/commit.md

# Use for
- Automated commit messages
- Conventional commit format
- Context-aware staging
```

**What it does:**
1. Reads git status/diff
2. Analyzes recent commits for style
3. Generates contextual message
4. Stages and commits

**Example Droid usage:**
```
Using smart-commit: Create a commit for the GPU optimization changes
```

---

### 4. **MLOps Engineer** ⭐⭐⭐⭐
**Why:** For when you deploy to RunPod or productionize models

```bash
# Location
.droid/claude-plugins-export/templates/components/agents/data-ai/mlops-engineer.md

# Use for
- Docker/container setup
- RunPod deployment
- Model versioning
- Pipeline automation
```

**Perfect for:**
- Setting up automated training runs
- CI/CD for ML models
- Model registry setup

**Example Droid prompt:**
```
Using mlops-engineer: Help me containerize the training pipeline for RunPod
```

---

### 5. **Data Scientist** ⭐⭐⭐
**Why:** For exploratory work and new model ideas

```bash
# Location
.droid/claude-plugins-export/templates/components/agents/data-ai/data-scientist.md

# Use for
- Feature exploration
- Model selection
- Statistical analysis
- Visualization
```

**Example Droid prompt:**
```
Using data-scientist: Analyze the feature importance from our stacking ensemble
```

---

## 📋 Conversion Checklist

For each plugin you want to use:

### Step 1: Copy the File
```bash
# Example: ML Engineer
cp .droid/claude-plugins-export/templates/components/agents/data-ai/ml-engineer.md \
   ~/path/to/droid/droids/ml-engineer.md
```

### Step 2: Check Droid Format
Open the file and verify:
- [ ] Remove YAML frontmatter if Droid doesn't support it
- [ ] Check if tool calls need adaptation
- [ ] Remove Claude-specific features (Task, TodoWrite, etc.)

### Step 3: Test
```bash
# In your moola project
droid ml-engineer "Help me optimize batch size for GPU training"
```

### Step 4: Refine
- Adjust based on what works
- Simplify if too complex
- Add project-specific context

## 🎓 Learning Path

Start simple, get complex:

### Week 1: Pure Prompts (Easiest)
- ✅ ML Engineer agent
- ✅ Data Scientist agent
- ✅ Performance Engineer agent

These are mostly advice/guidance - no complex tool usage.

### Week 2: Simple Commands
- ✅ Smart Commit
- ✅ Feature Dev (simplified)

These use basic bash commands.

### Week 3: Advanced
- ⚠️ PR Review (uses GitHub CLI)
- ⚠️ Full Feature Dev workflow

These are more complex, adapt as needed.

## 💡 Real Usage Examples for Moola

Based on your actual work:

### Example 1: After Training a Model
```bash
# Use ML Engineer to analyze results
droid ml-engineer "I just trained a CNN-Transformer model.
Validation accuracy is 0.75 but training is 0.92.
What's wrong and how do I fix it?"
```

### Example 2: Before Deploying to RunPod
```bash
# Use MLOps Engineer for deployment
droid mlops-engineer "I need to deploy this training pipeline to RunPod.
What's the best way to containerize it and handle GPU allocation?"
```

### Example 3: Performance Optimization
```bash
# Use Performance Engineer
droid performance-engineer "My OOF generation takes 60 minutes on CPU.
I have an A100 GPU. How should I optimize the code?"
```

### Example 4: Smart Commits
```bash
# After making changes
git add .
droid smart-commit "Create a commit for these changes"
```

### Example 5: Feature Exploration
```bash
# Use Data Scientist
droid data-scientist "I have OHLC data with 420 features.
What feature engineering techniques should I try for time series classification?"
```

## 🔍 Finding the Right Plugin

Quick reference by task:

| Task | Plugin | Priority |
|------|--------|----------|
| Model training issues | ml-engineer.md | ⭐⭐⭐⭐⭐ |
| Performance problems | performance-engineer.md | ⭐⭐⭐⭐⭐ |
| Git commits | commit.md | ⭐⭐⭐⭐ |
| RunPod deployment | mlops-engineer.md | ⭐⭐⭐⭐ |
| Feature exploration | data-scientist.md | ⭐⭐⭐ |
| Code review | code-reviewer.md | ⭐⭐⭐ |
| New features | feature-dev.md | ⭐⭐⭐ |
| Time series specific | quant-analyst.md | ⭐⭐⭐ |

## 🛠️ Droid-Specific Tips

### Tip 1: Add Project Context
When calling Droid, include:
```bash
droid ml-engineer "Context: Moola is a PyTorch-based trading system.
[Your question here]"
```

### Tip 2: Chain Droids
```bash
# First analyze
droid data-scientist "What's wrong with these metrics?"

# Then fix
droid ml-engineer "Based on that analysis, implement the fix"

# Then commit
droid smart-commit "Commit the changes"
```

### Tip 3: Keep Sessions Contextual
Droid (like me) benefits from context. Reference previous work:
```bash
droid ml-engineer "Following up on the GPU optimization we discussed,
now I need to optimize the DataLoader..."
```

## 📝 Template for New Droids

When creating a custom Droid based on these plugins:

```markdown
# [Droid Name]

## Purpose
What this droid does.

## When to Use
- Scenario 1
- Scenario 2
- Scenario 3

## Context Needed
- Project structure
- Current state
- Goals

## Instructions
Step-by-step what to do.

## Examples
Real examples from Moola project.
```

## 🎯 Success Metrics

You'll know it's working when:
- ✅ Droid gives ML-specific advice (not generic)
- ✅ Commits are contextual and well-formatted
- ✅ Performance recommendations are actionable
- ✅ You're saving time on repetitive tasks
- ✅ Code quality improves

## 🆘 Troubleshooting

### Issue: "Droid doesn't understand the plugin"
→ Check Droid's format requirements in their docs
→ Simplify the prompt
→ Remove Claude-specific features

### Issue: "Plugin is too generic"
→ Add moola-specific context to the prompt
→ Reference your actual code/structure
→ Give examples from your project

### Issue: "Can't find the right plugin"
→ Use the search: `grep -r "keyword" .droid/claude-plugins-export/`
→ Check PATHS_REFERENCE.md for categories
→ Ask me to help find it!

## 🚀 Next Steps

1. **Today:** Try ML Engineer and Performance Engineer
2. **This Week:** Set up Smart Commit
3. **Next Week:** Customize for your workflow
4. **Ongoing:** Share what works (and what doesn't)

## 📚 Additional Resources

- **Main README:** `README_FOR_DROID.md`
- **Conversion Guide:** `DROID_CONVERSION_GUIDE.md`
- **Path Reference:** `PATHS_REFERENCE.md`
- **All Files:** 395 markdown files to explore!

---

**Remember:** These are starting points. Adapt them to work with Droid's capabilities and your specific needs.

**Have fun experimenting!** 🚀

Last updated: 2025-10-12
