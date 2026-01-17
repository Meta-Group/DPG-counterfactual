#!/usr/bin/env python3
"""Run experiments for all datasets with all methods.

This script iterates through all dataset directories in configs/ and runs
experiments with each specified method (dpg, dice).

Usage:
  # Run all datasets with all methods
  python scripts/run_all_experiments.py
  
  # Run specific datasets only
  python scripts/run_all_experiments.py --datasets iris german_credit
  
  # Run specific methods only
  python scripts/run_all_experiments.py --methods dpg
  
  # Skip datasets that have already been processed
  python scripts/run_all_experiments.py --skip-existing
  
  # Dry run (show what would be executed)
  python scripts/run_all_experiments.py --dry-run
  
  # Run in offline mode (no wandb sync)
  python scripts/run_all_experiments.py --offline
  
  # Limit number of datasets (useful for testing)
  python scripts/run_all_experiments.py --limit 3
  
  # Run multiple experiments in parallel
  python scripts/run_all_experiments.py --parallel 4
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import List, Optional, Dict, Any

# Ensure repo root is on sys.path
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Default methods to run
DEFAULT_METHODS = ['dpg', 'dice']

# Files to exclude from dataset detection
EXCLUDED_FILES = {'config.yaml', 'sweep_config.yaml'}


def get_all_datasets(configs_dir: pathlib.Path) -> List[str]:
    """Get all dataset names from the configs directory.
    
    Args:
        configs_dir: Path to the configs directory
        
    Returns:
        Sorted list of dataset names (directory names)
    """
    datasets = []
    for item in configs_dir.iterdir():
        if item.is_dir() and item.name not in EXCLUDED_FILES:
            # Check if it has a config.yaml file
            config_file = item / 'config.yaml'
            if config_file.exists():
                datasets.append(item.name)
    return sorted(datasets)


def check_existing_output(dataset: str, method: str, output_dir: pathlib.Path) -> bool:
    """Check if output already exists for a dataset/method combination.
    
    Args:
        dataset: Dataset name
        method: Method name
        output_dir: Path to outputs directory
        
    Returns:
        True if output exists, False otherwise
    """
    # Check for output directory with the naming pattern
    expected_output = output_dir / f"{dataset}_{method}"
    return expected_output.exists() and any(expected_output.iterdir())


def run_experiment(
    dataset: str,
    method: str,
    verbose: bool = False,
    offline: bool = False,
    overrides: Optional[List[str]] = None,
    dry_run: bool = False
) -> Dict[str, Any]:
    """Run a single experiment.
    
    Args:
        dataset: Dataset name
        method: Method name (dpg, dice)
        verbose: Enable verbose output
        offline: Run WandB in offline mode
        overrides: List of config overrides
        dry_run: If True, just print the command without executing
        
    Returns:
        Dict with keys: success (bool), message (str), dataset, method, elapsed_time
    """
    experiment_key = f"{dataset}/{method}"
    
    # Build command
    cmd = [
        sys.executable,
        str(REPO_ROOT / 'scripts' / 'run_experiment.py'),
        '--dataset', dataset,
        '--method', method,
    ]
    
    if verbose:
        cmd.append('--verbose')
    
    if offline:
        cmd.append('--offline')
    
    if overrides:
        for override in overrides:
            cmd.extend(['--set', override])
    
    cmd_str = ' '.join(cmd)
    
    if dry_run:
        return {
            'success': True,
            'message': "Dry run - command not executed",
            'dataset': dataset,
            'method': method,
            'experiment_key': experiment_key,
            'elapsed_time': 0,
            'cmd': cmd_str
        }
    
    try:
        start_time = time.time()
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True
        )
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            return {
                'success': True,
                'message': f"Completed in {elapsed_time:.1f}s",
                'dataset': dataset,
                'method': method,
                'experiment_key': experiment_key,
                'elapsed_time': elapsed_time,
                'cmd': cmd_str
            }
        else:
            error_msg = result.stderr if result.stderr else f"Exit code: {result.returncode}"
            return {
                'success': False,
                'message': f"Failed: {error_msg[:200]}",
                'dataset': dataset,
                'method': method,
                'experiment_key': experiment_key,
                'elapsed_time': elapsed_time,
                'cmd': cmd_str
            }
            
    except Exception as e:
        return {
            'success': False,
            'message': f"Exception: {str(e)}",
            'dataset': dataset,
            'method': method,
            'experiment_key': experiment_key,
            'elapsed_time': 0,
            'cmd': cmd_str
        }


def run_experiments_sequential(
    experiments: List[Dict[str, Any]],
    args,
    output_dir: pathlib.Path,
    results: Dict[str, List[str]]
) -> None:
    """Run experiments sequentially (original behavior).
    
    Args:
        experiments: List of experiment dicts with 'dataset' and 'method' keys
        args: Parsed command line arguments
        output_dir: Path to outputs directory
        results: Dict to track success/failed/skipped experiments
    """
    for i, exp in enumerate(experiments, 1):
        dataset = exp['dataset']
        method = exp['method']
        experiment_key = f"{dataset}/{method}"
        
        print(f"\n[{i}/{len(experiments)}] {experiment_key}")
        print("-" * 40)
        
        # Check if should skip
        if args.skip_existing:
            if check_existing_output(dataset, method, output_dir):
                print(f"  Skipping (output exists)")
                results['skipped'].append(experiment_key)
                continue
        
        print(f"  Running...")
        
        result = run_experiment(
            dataset=dataset,
            method=method,
            verbose=args.verbose,
            offline=args.offline,
            overrides=args.overrides,
            dry_run=args.dry_run
        )
        
        if result['success']:
            print(f"  ✓ {result['message']}")
            results['success'].append(experiment_key)
        else:
            print(f"  ✗ {result['message']}")
            results['failed'].append(experiment_key)
            
            if not args.continue_on_error and not args.dry_run:
                print("\nERROR: Stopping due to failure. Use --continue-on-error to keep going.")
                break


def run_experiments_parallel(
    experiments: List[Dict[str, Any]],
    args,
    output_dir: pathlib.Path,
    results: Dict[str, List[str]],
    max_workers: int
) -> None:
    """Run experiments in parallel using ProcessPoolExecutor.
    
    Args:
        experiments: List of experiment dicts with 'dataset' and 'method' keys
        args: Parsed command line arguments
        output_dir: Path to outputs directory
        results: Dict to track success/failed/skipped experiments
        max_workers: Maximum number of parallel workers
    """
    # Filter out experiments to skip
    experiments_to_run = []
    for exp in experiments:
        dataset = exp['dataset']
        method = exp['method']
        experiment_key = f"{dataset}/{method}"
        
        if args.skip_existing and check_existing_output(dataset, method, output_dir):
            print(f"  [{experiment_key}] Skipping (output exists)")
            results['skipped'].append(experiment_key)
        else:
            experiments_to_run.append(exp)
    
    if not experiments_to_run:
        print("No experiments to run (all skipped).")
        return
    
    print(f"\nRunning {len(experiments_to_run)} experiments with {max_workers} parallel workers...")
    print("-" * 60)
    
    if args.dry_run:
        # In dry run mode, just show what would be executed
        for exp in experiments_to_run:
            result = run_experiment(
                dataset=exp['dataset'],
                method=exp['method'],
                verbose=args.verbose,
                offline=args.offline,
                overrides=args.overrides,
                dry_run=True
            )
            print(f"  [DRY RUN] {result['experiment_key']}: {result['cmd']}")
            results['success'].append(result['experiment_key'])
        return
    
    # Submit all experiments to the pool
    completed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_exp = {
            executor.submit(
                run_experiment,
                exp['dataset'],
                exp['method'],
                False,  # verbose=False for parallel (output would be interleaved)
                args.offline,
                args.overrides,
                False  # dry_run=False
            ): exp for exp in experiments_to_run
        }
        
        # Process completed tasks as they finish
        for future in as_completed(future_to_exp):
            completed += 1
            result = future.result()
            experiment_key = result['experiment_key']
            
            if result['success']:
                print(f"  [{completed}/{len(experiments_to_run)}] ✓ {experiment_key} - {result['message']}")
                results['success'].append(experiment_key)
            else:
                print(f"  [{completed}/{len(experiments_to_run)}] ✗ {experiment_key} - {result['message']}")
                results['failed'].append(experiment_key)


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments for all datasets with all methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=None,
        help='Specific datasets to run (default: all datasets in configs/)'
    )
    parser.add_argument(
        '--methods',
        nargs='+',
        default=DEFAULT_METHODS,
        help=f'Methods to run (default: {DEFAULT_METHODS})'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip dataset/method combinations that already have output'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print commands without executing them'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output for each experiment'
    )
    parser.add_argument(
        '--offline',
        action='store_true',
        help='Run WandB in offline mode'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit the number of datasets to process (useful for testing)'
    )
    parser.add_argument(
        '--set',
        action='append',
        default=[],
        dest='overrides',
        help='Override config values for all experiments (e.g., --set experiment_params.num_samples=3)'
    )
    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='Continue running other experiments even if one fails'
    )
    parser.add_argument(
        '--parallel',
        type=int,
        default=None,
        metavar='N',
        help='Run N experiments in parallel (default: sequential execution)'
    )
    
    args = parser.parse_args()
    
    configs_dir = REPO_ROOT / 'configs'
    output_dir = REPO_ROOT / 'outputs'
    
    # Get datasets
    if args.datasets:
        datasets = args.datasets
        # Validate that specified datasets exist
        all_datasets = set(get_all_datasets(configs_dir))
        invalid_datasets = set(datasets) - all_datasets
        if invalid_datasets:
            print(f"ERROR: Invalid datasets: {invalid_datasets}")
            print(f"Available datasets: {sorted(all_datasets)}")
            return 1
    else:
        datasets = get_all_datasets(configs_dir)
    
    if args.limit:
        datasets = datasets[:args.limit]
    
    methods = args.methods
    
    # Print summary
    print("=" * 60)
    print("COUNTERFACTUAL EXPERIMENTS BATCH RUNNER")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Datasets: {len(datasets)}")
    print(f"Methods: {methods}")
    print(f"Total experiments: {len(datasets) * len(methods)}")
    print(f"Skip existing: {args.skip_existing}")
    print(f"Dry run: {args.dry_run}")
    print(f"Continue on error: {args.continue_on_error}")
    print(f"Parallel workers: {args.parallel if args.parallel else 'sequential'}")
    if args.overrides:
        print(f"Config overrides: {args.overrides}")
    print("=" * 60)
    print()
    
    # Track results
    results = {
        'success': [],
        'failed': [],
        'skipped': []
    }
    
    total_start_time = time.time()
    
    # Build list of all experiments
    experiments = [
        {'dataset': dataset, 'method': method}
        for dataset in datasets
        for method in methods
    ]
    
    # Run experiments (parallel or sequential)
    if args.parallel and args.parallel > 1:
        run_experiments_parallel(
            experiments=experiments,
            args=args,
            output_dir=output_dir,
            results=results,
            max_workers=args.parallel
        )
    else:
        run_experiments_sequential(
            experiments=experiments,
            args=args,
            output_dir=output_dir,
            results=results
        )
    
    total_elapsed = time.time() - total_start_time
    
    # Print summary
    print("\n")
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print()
    print(f"✓ Successful: {len(results['success'])}")
    print(f"✗ Failed: {len(results['failed'])}")
    print(f"⊘ Skipped: {len(results['skipped'])}")
    
    if results['failed']:
        print("\nFailed experiments:")
        for exp in results['failed']:
            print(f"  - {exp}")
    
    if results['skipped']:
        print("\nSkipped experiments:")
        for exp in results['skipped']:
            print(f"  - {exp}")
    
    print("=" * 60)
    
    # Return exit code based on failures
    return 1 if results['failed'] else 0


if __name__ == '__main__':
    sys.exit(main())
