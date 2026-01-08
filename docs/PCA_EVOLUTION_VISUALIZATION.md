# PCA Evolution Visualization Enhancement

## Overview
The PCA visualization now shows the complete **evolutionary trajectory** of the Genetic Algorithm (GA) used for counterfactual generation, not just the final results.

## What Changed

### 1. Evolution History Tracking
**File:** `CounterFactualModel.py`

- Added `evolution_history` attribute to store the best individual from each generation
- Each generation's best solution is captured during the GA evolution loop
- History is preserved for post-hoc visualization

```python
self.evolution_history = []  # List of dicts, one per generation
```

### 2. Data Structure Enhancement
**File:** `scripts/run_experiment.py`

- Modified `run_single_sample()` to capture and store evolution history
- Evolution data is saved in the replication visualization structure
- Passed to PCA plotting function for rendering

```python
replication_viz = {
    'counterfactual': counterfactual,
    'cf_model': cf_model,
    'evolution_history': evolution_history,  # NEW: Full GA trajectory
    'visualizations': [],
    'explanations': {}
}
```

### 3. Enhanced PCA Visualization
**File:** `CounterFactualVisualizer.py`

Updated `plot_pca_with_counterfactuals()` function with new parameter:

```python
def plot_pca_with_counterfactuals(model, dataset, target, sample, 
                                   counterfactuals_df, evolution_histories=None):
```

**Visual Features:**
- **Opacity Gradient**: 0.1 (initial generation) → 1.0 (final generation)
- **Size Variation**: Intermediate crosses (60px) vs final crosses (100-150px)
- **Class-Based Coloring**: Each cross colored by its predicted class
- **Multi-Replication Support**: Shows evolution trails for all GA replications

## Visualization Example

### Before
- Only final counterfactuals shown as crosses
- No information about GA convergence path
- Static single-frame view

### After
- **Full evolution trail visible**: 20 generations × 3 replications = 60 evolution points
- **Opacity progression**: Shows how solution evolved from initial (faint) to final (solid)
- **Size emphasis**: Final solutions stand out with larger markers
- **Class dynamics**: Color shows if/when the solution crosses into target class region

## Output Files (Sample 73)

```
experiment_results/73/
├── pca_with_evolution_0.png       # Main PCA plot with evolution trails
├── pca_evolution_annotated.png    # Detailed annotated version
├── pca_coords.csv                 # Numeric PC coordinates
├── pca_loadings.csv               # Feature loadings for PC1/PC2
├── after_viz_generation.pkl       # Full visualization data with evolution
└── raw_counterfactuals.pkl        # Raw counterfactual data
```

## Interpretation Guide

### Reading the Evolution Trail
1. **Faint crosses**: Early generations exploring search space
2. **Progressive darkening**: GA converging toward optimal solution
3. **Solid crosses**: Final converged counterfactuals
4. **Color transitions**: May show intermediate class predictions during evolution
5. **Spatial clustering**: Multiple replications converging to similar regions

### What to Look For
- **Convergence speed**: How quickly opacity increases (fewer generations = faster)
- **Exploration breadth**: How far initial crosses spread from original sample
- **Class boundary crossing**: When crosses change color (enter target class region)
- **Replication consistency**: Whether multiple trails converge to same area

## Technical Details

### Evolution Capture Mechanism
```python
# In genetic_algorithm() loop:
for generation in range(generations):
    # ... evaluate population ...
    hof.update(population)
    
    # Capture best individual this generation
    if hof[0].fitness.values[0] != np.inf:
        self.evolution_history.append(dict(hof[0]))
```

### Opacity Formula
```python
alpha = 0.1 + (0.9 * gen_idx / max(1, num_generations - 1))
```
- Generation 0: alpha = 0.1 (10% opacity - very faint)
- Generation 19: alpha = 1.0 (100% opacity - fully solid)

### Size Formula
```python
size = 60 if gen_idx < num_generations - 1 else 100
```
- Intermediate: 60px
- Final generation: 100px+

## Usage in Experiments

The enhancement is **automatic** - no config changes needed:

```bash
python scripts/run_experiment.py --config configs/quick.yaml
```

Evolution trails will appear in all generated PCA plots.

## Performance Impact

- **Memory**: ~20 dicts per replication (negligible for standard configs)
- **Computation**: Minimal overhead (just dict copying per generation)
- **Visualization**: Slight increase in render time (60 vs 3 scatter points)

## Future Enhancements

Potential additions:
- [ ] Connecting lines between generations (trajectory paths)
- [ ] Animation of evolution over time (GIF/video)
- [ ] Fitness value annotations on hover
- [ ] Interactive 3D PCA with evolution trails
- [ ] Side-by-side comparison of multiple samples' evolution

## Code Changes Summary

| File | Lines Changed | Purpose |
|------|--------------|---------|
| `CounterFactualModel.py` | +2, +3 | Add evolution_history tracking |
| `scripts/run_experiment.py` | +4, +4 | Capture & pass evolution data |
| `CounterFactualVisualizer.py` | +60 | Enhanced PCA plotting with trails |

**Total**: ~70 lines added/modified

## Examples

### Quick Test (1 sample, 3 replications)
```bash
python scripts/run_experiment.py --config configs/quick.yaml
```
- Sample 73: Original class 1 → Target class 0
- 3 evolution trails of 20 generations each
- Clear convergence visible in opacity gradient

### Full Experiment
```bash
python scripts/run_experiment.py --config configs/iris.yaml
```
- Multiple samples with evolution trails
- Compare convergence patterns across different starting points

## Related Documentation
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Running experiments
- [EXPERIMENT_TRACKING.md](EXPERIMENT_TRACKING.md) - WandB integration
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Overall architecture

---

**Date**: January 8, 2026  
**Author**: Enhanced by GitHub Copilot  
**Status**: ✅ Implemented and tested
