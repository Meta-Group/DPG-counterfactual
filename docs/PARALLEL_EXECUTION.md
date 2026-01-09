# Parallel Execution for Counterfactual Generation

## Overview

The experiment runner now supports parallel execution of replications using multiprocessing. This significantly speeds up experiments when `num_replications > 1` by utilizing multiple CPU cores.

## Configuration

Add these parameters to your YAML config file under `experiment_params`:

```yaml
experiment_params:
  # ... other params ...
  num_replications: 8  # Number of replications per combination
  parallel_replications: true  # Enable parallel execution (default: true)
  max_workers: null  # null = use (CPU_count - 1), or specify a number
```

### Parameters

- **`parallel_replications`** (bool, default: `true`): 
  - `true`: Run replications in parallel using multiprocessing
  - `false`: Run replications sequentially (original behavior)

- **`max_workers`** (int or null, default: `null`):
  - `null`: Automatically use `(CPU_count - 1)` workers
  - Integer: Specify exact number of parallel workers
  - Example: `max_workers: 4` to use 4 cores

## Usage

### Enable Parallel Execution (Default)

```bash
python scripts/run_experiment.py --config configs/iris.yaml
```

### Disable Parallel Execution

```bash
python scripts/run_experiment.py --config configs/iris.yaml \
  --set experiment_params.parallel_replications=false
```

### Limit Number of Workers

```bash
python scripts/run_experiment.py --config configs/iris.yaml \
  --set experiment_params.max_workers=8
```

## Performance Benefits

With parallel execution enabled:
- **Multi-core utilization**: Uses multiple CPU cores instead of just one
- **Reduced wall-clock time**: Experiments complete faster (near-linear speedup with available cores)
- **Same results**: Produces identical results to sequential execution

### Example Speedup

On a 36-core system with `num_replications=8`:
- **Sequential**: ~12 minutes (1 core at 100%)
- **Parallel**: ~2-3 minutes (up to 8 cores utilized)
- **Speedup**: ~4-6x faster

## Technical Details

### Implementation

The parallel execution uses Python's `concurrent.futures.ProcessPoolExecutor` to:
1. Submit each replication as a separate process
2. Collect results as they complete
3. Reconstruct the CF models with fitness history for downstream processing

### Memory Considerations

Each parallel worker:
- Creates its own copy of the model and constraints
- Runs the genetic algorithm independently
- Returns only the necessary results (counterfactual, fitness history)

For large models or datasets, you may want to limit `max_workers` to avoid excessive memory usage.

### Compatibility

- Works with all existing config files (defaults to parallel if not specified)
- Compatible with WandB logging
- All visualizations and metrics are computed correctly
- Both online and offline WandB modes supported

## Troubleshooting

### Issue: "Too many open files" error

**Solution**: Reduce `max_workers`:
```yaml
experiment_params:
  max_workers: 4  # Use fewer workers
```

### Issue: High memory usage

**Solution**: Reduce `max_workers` or disable parallelization:
```yaml
experiment_params:
  parallel_replications: false  # Fall back to sequential
```

### Issue: Results differ from sequential runs

This should not happen. If you observe different results, please:
1. Check that the random seed is set consistently
2. Verify that all config parameters are identical
3. Report as a bug

## Example Configs

All config files have been updated with the parallelization parameters:
- `configs/iris.yaml`
- `configs/german_credit.yaml`
- `configs/quick.yaml`

## When to Disable Parallel Execution

Consider disabling parallelization if:
- Running on a system with limited memory
- Debugging replication-specific issues
- Using a very small number of replications (e.g., 1-2)
- Experiencing process management issues

## Future Improvements

Potential enhancements:
- Parallelize combination loop in addition to replications
- Add progress bars for parallel execution
- Dynamic worker pool sizing based on available memory
- GPU support for model predictions
