# ML Workflow Orchestrator

## Description
Specialized workflow orchestration system for ML operations with sophisticated command chaining, parallel coordination, and conditional logic patterns similar to Claude Code plugins. Integrates with moola's existing SSH/SCP RunPod workflow.

## When to Use
- Training pipeline orchestration with decision branches
- Parallel hyperparameter experiments with result aggregation
- RunPod deployment with monitoring and rollback capabilities
- Conditional ML workflows based on model performance or data quality

## Core Patterns

### 1. Command Syntax
```
/command [args...]                    # Execute command
$variable                             # State variable access
${expression}                        # Dynamic evaluation
> result                              # Store result in variable

# Chain operators
;                                    # Sequential execution
&&                                   # Conditional AND (stop on failure)
||                                   # Conditional OR (continue on failure)
|                                    # Pipeline (pass output)
&                                    # Parallel execution background
```

### 2. Conditional Logic
```
if:condition → then_command → else_command
if:$model_accuracy > 0.8 → /deploy → /retry

# Multi-branch conditionals
case:$resource_type
  ↳ gpu → /train_gpu
  ↳ cpu → /train_cpu  
  ↳ * → /error

# Retry patterns
retry:3 delay:60 → /unstable_training
```

### 3. Parallel Coordination
```
# Fork pattern
fork:4 → /experiment_1 & /experiment_2 & /experiment_3 & /experiment_4

# Droid delegation
droid:feature-developer → /implement_new_features
droid:performance-engineer → /optimize_inference

# Barrier synchronization
barrier → $all_results_ready → /analysis
```

### 4. State Management
```
# Variable assignment
$best_model = /select_best_model > winner.json
$phase1_results = /run_phase1 > phase1_results.json

# State persistence
state:save → checkpoint.json
state:load → checkpoint.json

# Context passing
context:phase1_winner → $best_sigma → /phase2_experiments
```

## Integration Points
- SSH/SCP RunPod workflow integration
- DVC pipeline orchestration
- Results logger integration
- Pre-commit hook coordination
- MLflow tracking (optional)

## Error Handling
```
try: /risky_operation → catch:/fallback_strategy
timeout:300 → /long_running_task
on_failure → /cleanup && /notify_admin
```

## Example Workflows
See specific workflow implementations for detailed patterns and usage examples.
