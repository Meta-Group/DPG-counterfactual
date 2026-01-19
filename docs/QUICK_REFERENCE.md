# 🚀 Quick Reference Card

## Essential Commands

### Run Experiment
```bash
# Default config
python3 scripts/run_experiment.py --config configs/experiment_config.yaml

# Quick test (2 samples, fast)
python3 scripts/run_experiment.py --config configs/quick_test.yaml

# With overrides
python3 scripts/run_experiment.py --config configs/experiment_config.yaml \
  --set counterfactual.population_size=100 \
  --set experiment_params.num_samples=10

# Offline mode
python3 scripts/run_experiment.py --config configs/quick_test.yaml --offline
```

### View Results
```bash
# List all runs
python3 scripts/query_results.py list

# Compare runs
python3 scripts/query_results.py compare --runs abc123 def456

# Best run
python3 scripts/query_results.py best --metric experiment/overall_success_rate

# Export CSV
python3 scripts/query_results.py export --output my_results.csv
```

### Hyperparameter Sweep
```bash
# Start sweep
wandb sweep configs/sweep_config.yaml

# Run agent (copy sweep_id from above)
wandb agent <your-entity>/CounterFactualDPG/<sweep_id>
```

## Key Configuration Parameters

### Genetic Algorithm Weights
```yaml
counterfactual:
  diversity_weight: 0.5      # Diversity bonus
  repulsion_weight: 4.0      # Repulsion bonus  
  boundary_weight: 15.0      # Decision boundary
  distance_factor: 2.0       # Distance penalty
  sparsity_factor: 1.0       # Sparsity penalty
  constraints_factor: 3.0    # Constraint violation
```

### Runtime Parameters
```yaml
counterfactual:
  population_size: 20        # GA population
  max_generations: 60        # Max iterations
  mutation_rate: 0.8         # Mutation probability
```

### Experiment Setup
```yaml
experiment_params:
  seed: 42                   # Random seed
  num_samples: 5             # Samples to process
  num_replications: 3        # Replications per combo
```

## Important Metrics

| Metric | Description |
|--------|-------------|
| `experiment/overall_success_rate` | Overall CF generation success |
| `sample/success_rate` | Success rate per sample |
| `replication/final_fitness` | Final fitness achieved |
| `metrics/distance_euclidean` | Distance to original |
| `metrics/num_feature_changes` | Features modified |

## File Locations

- **Configs**: `configs/experiment_config.yaml`
- **Scripts**: `scripts/run_experiment.py`
- **Results**: `experiment_results/<sample_id>/`
- **Docs**: `EXPERIMENT_TRACKING.md`

## WandB URLs

- Dashboard: https://wandb.ai/
- Login: https://wandb.ai/authorize

## Troubleshooting

```bash
# Not logged in?
wandb login

# Check WandB status
wandb verify

# Test config loading
python3 -c "import yaml; yaml.safe_load(open('configs/experiment_config.yaml'))"

# Run in offline mode
python3 scripts/run_experiment.py --config configs/quick_test.yaml --offline
```

## Tips

1. **Start small**: Use `configs/quick_test.yaml` first
2. **Tag experiments**: Add meaningful tags for organization
3. **Compare often**: Use `query_results.py compare` to track progress
4. **Save good configs**: Copy successful configs to `configs/experiments/`
5. **Use sweeps**: Let WandB optimize hyperparameters automatically
