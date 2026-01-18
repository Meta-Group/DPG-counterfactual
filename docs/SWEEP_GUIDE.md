# WandB Sweep Guide for DPG Hyperparameter Optimization

This guide explains how to use the WandB sweep functionality to find optimal hyperparameters for the DPG genetic algorithm counterfactual generator.

## Quick Start

```bash
# 1. Initialize a sweep (choose your target metric)
python scripts/run_sweep.py --init-sweep --dataset iris --target-metric plausibility_sum

# 2. Run the sweep agent (replace <sweep_id> with the ID from step 1)
python scripts/run_sweep.py --run-agent --sweep-id <sweep_id> --count 20

# Or run multiple agents in parallel for faster exploration:
# Terminal 1:
python scripts/run_sweep.py --run-agent --sweep-id <sweep_id>
# Terminal 2:
python scripts/run_sweep.py --run-agent --sweep-id <sweep_id>
```

## Available Target Metrics

| Metric | Goal | Description |
|--------|------|-------------|
| `plausibility_sum` ★ | minimize | Distance to nearest training sample |
| `perc_valid_cf` | maximize | Percentage of valid counterfactuals |
| `distance_l2` | minimize | Euclidean distance from original |
| `distance_mad` | minimize | MAD-normalized distance |
| `avg_nbr_changes_per_cf` | minimize | Feature sparsity (changes per CF) |
| `diversity_l2` | maximize | Pairwise L2 diversity among CFs |
| `perc_valid_actionable_cf` | maximize | Valid AND actionable CF percentage |
| `accuracy_knn_sklearn` | maximize | KNN fidelity score |
| `delta` | maximize | Mean prediction probability change |

★ = Default metric

To list all metrics: `python scripts/run_sweep.py --list-metrics`

## Hyperparameters Being Optimized

The sweep explores these 5 most impactful GA parameters:

| Parameter | Range | Description |
|-----------|-------|-------------|
| `population_size` | 20-100 | Number of individuals per generation |
| `max_generations` | 30-200 | Maximum GA iterations |
| `mutation_rate` | 0.05-0.5 | Per-feature mutation probability |
| `diversity_weight` | 0.1-3.0 | Weight for diversity bonus |
| `repulsion_weight` | 1.0-10.0 | Weight for minimum distance to neighbors |

## Usage Examples

### Example 1: Optimize for validity
```bash
python scripts/run_sweep.py --init-sweep --dataset iris --target-metric perc_valid_cf
```

### Example 2: Optimize for sparsity (fewer changes)
```bash
python scripts/run_sweep.py --init-sweep --dataset iris --target-metric avg_nbr_changes_per_cf
```

### Example 3: Optimize for diversity
```bash
python scripts/run_sweep.py --init-sweep --dataset iris --target-metric diversity_l2
```

### Example 4: Run with specific entity/project
```bash
python scripts/run_sweep.py --init-sweep \
    --dataset iris \
    --target-metric plausibility_sum \
    --entity mllab-ts-universit-di-trieste \
    --project CounterFactualDPG
```

### Example 5: Run limited number of experiments
```bash
python scripts/run_sweep.py --run-agent --sweep-id <id> --count 10
```

## Alternative: Using wandb CLI Directly

You can also use the sweep config file directly:

```bash
# Create sweep from config file
wandb sweep configs/iris/sweep_config.yaml

# Run agent
wandb agent <entity>/<project>/<sweep_id>
```

## Viewing Results

1. Go to [wandb.ai](https://wandb.ai)
2. Navigate to your project
3. Click on "Sweeps" in the left sidebar
4. View parallel coordinates, parameter importance, and run comparisons

## Tips

- **Start small**: Use `--count 10` to run a few experiments first
- **Run parallel agents**: Open multiple terminals with `--run-agent` for faster exploration
- **Check early**: Look at results after 10-20 runs to see if the search space is reasonable
- **Use hyperband**: The sweep uses early termination to stop poor runs early
