"""Quick test to verify PCGrad implementation works correctly."""

import torch
import sys
sys.path.insert(0, 'src')

# Import the PCGrad function from finetune_jade
import importlib.util
spec = importlib.util.spec_from_file_location("finetune_jade", "scripts/finetune_jade.py")
finetune_jade = importlib.util.module_from_spec(spec)
spec.loader.exec_module(finetune_jade)

project_conflicting_gradients = finetune_jade.project_conflicting_gradients
GradientConflictMonitor = finetune_jade.GradientConflictMonitor

print("Testing PCGrad implementation...")
print()

# Test 1: Non-conflicting gradients (should not project)
print("Test 1: Non-conflicting gradients (cos > 0)")
grads1 = [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0])]
grads2 = [torch.tensor([1.0, 1.5, 2.0]), torch.tensor([2.0, 2.5])]

proj1, proj2, was_projected, cos_sim, mag1, mag2 = project_conflicting_gradients(grads1, grads2)

print(f"  Cosine similarity: {cos_sim:.4f}")
print(f"  Was projected: {was_projected}")
print(f"  Gradient magnitudes: task1={mag1:.4f}, task2={mag2:.4f}")
assert not was_projected, "Should not project non-conflicting gradients"
assert cos_sim > 0, "Non-conflicting gradients should have positive cosine"
print("  ✓ PASS\n")

# Test 2: Conflicting gradients (should project)
print("Test 2: Conflicting gradients (cos < 0)")
grads1 = [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0])]
grads2 = [torch.tensor([-1.0, -2.0, -3.0]), torch.tensor([-4.0, -5.0])]

proj1, proj2, was_projected, cos_sim, mag1, mag2 = project_conflicting_gradients(grads1, grads2)

print(f"  Cosine similarity: {cos_sim:.4f}")
print(f"  Was projected: {was_projected}")
print(f"  Gradient magnitudes: task1={mag1:.4f}, task2={mag2:.4f}")
assert was_projected, "Should project conflicting gradients"
assert cos_sim < 0, "Conflicting gradients should have negative cosine"

# Verify orthogonality after projection
flat_proj1 = torch.cat([g.flatten() for g in proj1])
flat_proj2 = torch.cat([g.flatten() for g in proj2])
dot_after = torch.dot(flat_proj1, flat_proj2).item()
print(f"  Dot product after projection: {dot_after:.6f}")
assert abs(dot_after) < 1e-4, "Projected gradients should be orthogonal"
print("  ✓ PASS\n")

# Test 3: GradientConflictMonitor
print("Test 3: GradientConflictMonitor")
monitor = GradientConflictMonitor(log_interval=10)

# Record some conflicts
for i in range(15):
    is_conflict = i % 3 == 0  # Every 3rd step is a conflict
    was_projected = i % 5 == 0  # Every 5th step is projected
    cos_sim = -0.5 if is_conflict else 0.5
    monitor.record(cos_sim, 1.0, 1.0, was_projected, is_conflict)

summary = monitor.get_summary()
print(f"  Total steps: {summary['total_steps']}")
print(f"  Conflict rate: {summary['conflict_rate']:.2%}")
print(f"  Projection rate: {summary['projection_rate']:.2%}")
assert summary['total_steps'] == 15
assert summary['conflict_rate'] == 5/15  # 5 conflicts out of 15
assert summary['projection_rate'] == 3/15  # 3 projections out of 15
print("  ✓ PASS\n")

print("=" * 60)
print("All PCGrad tests passed! ✓")
print("=" * 60)
