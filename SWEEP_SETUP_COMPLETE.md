# WandB Sweep Implementation - Quick Reference

## ✅ Implementation Complete

The WandB sweep system is now fully functional and configured to use the `.venv` environment.

## Files Created

1. **configs/iris/sweep_config.yaml** - Sweep configuration file
2. **scripts/run_sweep.py** - Main sweep runner script
3. **scripts/quick_sweep.sh** - Helper bash script for easy sweep management
4. **docs/SWEEP_GUIDE.md** - Complete usage guide

## Quick Start

```bash
# Activate venv (if not already active)
source .venv/bin/activate

# Option 1: Using helper script (recommended)
./scripts/quick_sweep.sh init plausibility_sum      # Initialize sweep
./scripts/quick_sweep.sh run <sweep_id> 20          # Run 20 experiments

# Option 2: Using Python script directly
.venv/bin/python scripts/run_sweep.py --init-sweep \
    --dataset iris \
    --target-metric plausibility_sum \
    --entity mllab-ts-universit-di-trieste

wandb agent mllab-ts-universit-di-trieste/CounterFactualDPG/<sweep_id>
```

## Current Sweep

- **Sweep ID**: eicwwczo
- **Dataset**: iris
- **Target Metric**: plausibility_sum (minimize)
- **URL**: https://wandb.ai/mllab-ts-universit-di-trieste/CounterFactualDPG/sweeps/eicwwczo

Run it with:
```bash
wandb agent mllab-ts-universit-di-trieste/CounterFactualDPG/eicwwczo
```

## 9 Recommended Target Metrics

1. **plausibility_sum** (minimize) - Distance to nearest training sample ⭐ DEFAULT
2. **perc_valid_cf** (maximize) - Percentage of valid CFs
3. **distance_l2** (minimize) - Euclidean distance
4. **distance_mad** (minimize) - MAD-normalized distance
5. **avg_nbr_changes_per_cf** (minimize) - Feature sparsity
6. **diversity_l2** (maximize) - Pairwise L2 diversity
7. **perc_valid_actionable_cf** (maximize) - Valid & actionable percentage
8. **accuracy_knn_sklearn** (maximize) - KNN fidelity
9. **delta** (maximize) - Mean prediction change

## 5 Hyperparameters Being Tuned

| Parameter | Range | Description |
|-----------|-------|-------------|
| population_size | 20-100 | GA population size |
| max_generations | 30-200 | Maximum iterations |
| mutation_rate | 0.05-0.5 | Per-feature mutation probability |
| diversity_weight | 0.1-3.0 | Diversity bonus weight |
| repulsion_weight | 1.0-10.0 | Repulsion weight |

## Test Result

✅ Successfully ran one sweep experiment:
- Population: 35, Generations: 38, Mutation: 0.336
- Generated 5/5 valid counterfactuals (100% success)
- Properly logged to WandB with all metrics and visualizations

## Next Steps

1. Run the sweep: `wandb agent mllab-ts-universit-di-trieste/CounterFactualDPG/eicwwczo`
2. Monitor results at: https://wandb.ai/mllab-ts-universit-di-trieste/CounterFactualDPG/sweeps/eicwwczo
3. Once optimal parameters are found, update configs/iris/config.yaml
4. Optionally run sweeps for other target metrics or datasets
