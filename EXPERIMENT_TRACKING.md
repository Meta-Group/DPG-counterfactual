# CounterFactual DPG - Experiment Tracking Setup

This project now includes a comprehensive experiment tracking system using **Weights & Biases (wandb.ai)**.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Login to WandB

```bash
wandb login
```

### 3. Run Your First Experiment

```bash
python scripts/run_experiment.py --config configs/experiment_config.yaml
```

## Configuration System

All experiments are configured via YAML files in the `configs/` directory.

### Default Configuration (`configs/experiment_config.yaml`)

The default config includes:
- **Data parameters**: dataset selection, train/test split
- **Model parameters**: classifier type and hyperparameters  
- **Counterfactual parameters**: GA hyperparameters (diversity_weight, repulsion_weight, etc.)
- **Experiment parameters**: seeds, sample selection, replications
- **Output parameters**: where to save results

### Override Parameters

Override any config parameter via CLI:

```bash
python scripts/run_experiment.py --config configs/experiment_config.yaml \
  --set counterfactual.population_size=50 \
  --set experiment_params.seed=123 \
  --set experiment.tags='["large_pop","test"]'
```

## Running Experiments

### Single Experiment

```bash
# Run with default config
python scripts/run_experiment.py --config configs/experiment_config.yaml

# Run in offline mode (no internet sync)
python scripts/run_experiment.py --config configs/experiment_config.yaml --offline

# Enable verbose logging
python scripts/run_experiment.py --config configs/experiment_config.yaml --verbose
```

### Resume Interrupted Experiment

```bash
python scripts/run_experiment.py --resume <wandb_run_id>
```

### Hyperparameter Sweep

Run automated hyperparameter optimization:

```bash
# Initialize sweep
wandb sweep configs/sweep_config.yaml

# Run sweep agents (can run multiple in parallel)
wandb agent <sweep_id>
```

## Querying Results

Use the `query_results.py` utility to analyze experiment results:

### List All Runs

```bash
python scripts/query_results.py list --project counterfactual-dpg
```

### Compare Specific Runs

```bash
python scripts/query_results.py compare \
  --runs run_id_1 run_id_2 run_id_3 \
  --metrics sample/success_rate experiment/overall_success_rate
```

### Export to CSV

```bash
python scripts/query_results.py export \
  --project counterfactual-dpg \
  --output results.csv
```

### Find Best Run

```bash
python scripts/query_results.py best \
  --project counterfactual-dpg \
  --metric experiment/overall_success_rate
```

## Logged Metrics

The system automatically logs:

### Sample-Level Metrics
- `sample/sample_id`: Unique sample identifier
- `sample/original_class`: Original predicted class
- `sample/target_class`: Target class for counterfactual
- `sample/num_valid_counterfactuals`: Number of successful CFs generated
- `sample/success_rate`: Success rate for this sample

### Replication-Level Metrics
- `replication/final_fitness`: Final fitness value achieved
- `replication/generations_to_converge`: Number of GA generations
- `replication/num_feature_changes`: Features modified in CF
- `replication/constraint_violations`: Constraint penalty
- `metrics/distance_euclidean`: Euclidean distance to original
- `metrics/distance_manhattan`: Manhattan distance to original
- `metrics/sparsity`: Sparsity score

### Fitness Evolution
- `fitness/generation`: Generation number
- `fitness/best`: Best fitness in generation
- `fitness/average`: Average fitness in generation

### Experiment-Level Summary
- `experiment/total_samples`: Total samples processed
- `experiment/total_valid_counterfactuals`: Total successful CFs
- `experiment/overall_success_rate`: Overall success rate

## Artifacts

The system automatically logs:
- Raw counterfactual data (`.pkl` files)
- Visualization data
- Fitness curves (as images)

## File Structure

```
configs/
├── experiment_config.yaml      # Default experiment configuration
├── sweep_config.yaml           # Hyperparameter sweep configuration
└── experiments/                # Custom experiment configs

scripts/
├── experiment_generation.py    # Legacy script (backwards compatible)
├── run_experiment.py          # New main entry point with WandB
└── query_results.py           # Query and analyze results

experiment_results/             # Local results storage
└── <sample_id>/
    ├── metadata.pkl
    ├── raw_counterfactuals.pkl
    └── after_viz_generation.pkl
```

## WandB Dashboard

Access your experiments at: https://wandb.ai/

The dashboard provides:
- Real-time metric tracking
- Interactive visualizations
- Run comparison tools
- Hyperparameter importance analysis
- Artifact versioning

## Tips for Iterative Research

1. **Tag your experiments**: Use meaningful tags to organize runs
   ```bash
   --set experiment.tags='["baseline","iris","v1"]'
   ```

2. **Use descriptive names**: Set experiment names that describe what you're testing
   ```bash
   --set experiment.name='high_diversity_test'
   ```

3. **Create config snapshots**: Save successful configs in `configs/experiments/`
   ```bash
   cp configs/experiment_config.yaml configs/experiments/best_so_far.yaml
   ```

4. **Compare regularly**: Use `query_results.py compare` to track improvements

5. **Export for analysis**: Export to CSV for custom analysis in pandas/R
   ```bash
   python scripts/query_results.py export --output my_results.csv
   ```

## Troubleshooting

### WandB Not Available
If wandb is not installed, the system will run without experiment tracking:
```
Warning: wandb not available. Install with: pip install wandb
```

### Offline Mode
If you don't have internet access, run in offline mode:
```bash
python scripts/run_experiment.py --config configs/experiment_config.yaml --offline
```

Sync later with:
```bash
wandb sync <run_directory>
```

### Resume Failed Runs
If a run crashes, resume with the run ID:
```bash
python scripts/run_experiment.py --resume <run_id>
```
