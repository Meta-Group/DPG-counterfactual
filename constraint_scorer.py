"""Constraint Score Metric for evaluating DPG constraint quality.

This module provides a metric to evaluate how well DPG constraints separate
different classes, which is crucial for counterfactual generation. Better
separation means the genetic algorithm can more easily find samples that
are outside the original class bounds and inside the target class bounds.

Score range: [0, 1]
  - 0: Complete overlap (all classes have identical constraints)
  - 1: Perfect separation (no overlap between any classes)
"""

from __future__ import annotations

import json
import numpy as np
from itertools import combinations
from typing import Dict, List, Optional, Tuple, Union


def load_constraints_from_json(path: str) -> Dict:
    """Load constraints from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def _get_interval_bounds(
    constraint: Dict[str, Optional[float]],
    feature_min: float,
    feature_max: float,
) -> Tuple[float, float]:
    """Extract interval bounds, using feature range for unbounded values.
    
    Args:
        constraint: Dict with 'min' and 'max' keys (can be None/null)
        feature_min: Minimum value observed for this feature across all classes
        feature_max: Maximum value observed for this feature across all classes
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    lower = constraint.get("min")
    upper = constraint.get("max")
    
    # Handle null/None bounds by using feature extremes
    if lower is None:
        lower = feature_min
    if upper is None:
        upper = feature_max
    
    return (float(lower), float(upper))


def _compute_interval_overlap(
    interval1: Tuple[float, float],
    interval2: Tuple[float, float],
) -> float:
    """Compute the overlap between two intervals.
    
    Returns the length of the overlapping region.
    """
    lower1, upper1 = interval1
    lower2, upper2 = interval2
    
    overlap_start = max(lower1, lower2)
    overlap_end = min(upper1, upper2)
    
    return max(0.0, overlap_end - overlap_start)


def _compute_interval_union_length(
    interval1: Tuple[float, float],
    interval2: Tuple[float, float],
) -> float:
    """Compute the length of the union of two intervals.
    
    Note: This is not the true union for non-overlapping intervals,
    but the span from min to max (convex hull).
    """
    lower1, upper1 = interval1
    lower2, upper2 = interval2
    
    union_start = min(lower1, lower2)
    union_end = max(upper1, upper2)
    
    return max(0.0, union_end - union_start)


def _compute_separation_score_for_pair(
    interval1: Tuple[float, float],
    interval2: Tuple[float, float],
    feature_range: float,
) -> float:
    """Compute separation score between two intervals.
    
    The score is based on the "non-overlap ratio" normalized by feature range.
    
    Returns:
        Score in [0, 1] where:
        - 0 = complete overlap (one interval contains the other or identical)
        - 1 = no overlap (completely separated)
    """
    if feature_range <= 0:
        return 0.0  # Degenerate case: constant feature
    
    overlap = _compute_interval_overlap(interval1, interval2)
    
    # Compute lengths
    len1 = interval1[1] - interval1[0]
    len2 = interval2[1] - interval2[0]
    
    # Handle degenerate intervals (point intervals or inverted)
    if len1 <= 0 and len2 <= 0:
        # Both are points - check if same point
        return 0.0 if abs(interval1[0] - interval2[0]) < 1e-9 else 1.0
    
    min_len = min(len1, len2)
    if min_len <= 0:
        min_len = max(len1, len2)
    
    if min_len <= 0:
        return 0.0
    
    # Non-overlap ratio: 1 - (overlap / min_length)
    # Using min length because if a small interval is fully inside a large one,
    # the overlap ratio should reflect that the small one is completely overlapped
    overlap_ratio = overlap / min_len
    separation = 1.0 - min(1.0, overlap_ratio)
    
    return separation


def _get_feature_range_across_classes(
    constraints: Dict[str, Dict],
    feature: str,
) -> Tuple[float, float, float]:
    """Get the min, max, and range for a feature across all classes.
    
    Returns:
        Tuple of (min_value, max_value, range)
    """
    min_vals = []
    max_vals = []
    
    for class_label, features in constraints.items():
        if feature not in features:
            continue
        
        feat_constraint = features[feature]
        if feat_constraint.get("min") is not None:
            min_vals.append(feat_constraint["min"])
        if feat_constraint.get("max") is not None:
            max_vals.append(feat_constraint["max"])
    
    if not min_vals and not max_vals:
        return (0.0, 1.0, 1.0)  # Default range
    
    # Use the observed bounds
    feature_min = min(min_vals) if min_vals else (min(max_vals) - 1.0)
    feature_max = max(max_vals) if max_vals else (max(min_vals) + 1.0)
    
    # Ensure we have a valid range
    if feature_max <= feature_min:
        feature_max = feature_min + 1.0
    
    return (feature_min, feature_max, feature_max - feature_min)


def compute_constraint_score(
    constraints: Dict[str, Dict],
    verbose: bool = False,
) -> Dict[str, Union[float, Dict]]:
    """Compute the constraint separation score.
    
    This metric measures how well the constraints separate different classes,
    which is important for counterfactual generation. Higher scores indicate
    better separation (less overlap between class intervals).
    
    Args:
        constraints: Dictionary mapping class labels to feature constraints.
            Format: {
                "Class 0": {
                    "feature1": {"min": value, "max": value},
                    ...
                },
                ...
            }
        verbose: If True, return detailed breakdown by feature and class pair.
    
    Returns:
        Dictionary containing:
            - "score": Overall constraint score in [0, 1]
            - "n_classes": Number of classes
            - "n_features": Number of features with constraints
            - "per_feature_scores": (if verbose) Dict of scores per feature
            - "per_pair_scores": (if verbose) Dict of scores per class pair
    """
    class_labels = list(constraints.keys())
    n_classes = len(class_labels)
    
    if n_classes < 2:
        return {
            "score": 1.0,  # Single class = trivially separated
            "n_classes": n_classes,
            "n_features": 0,
            "message": "Need at least 2 classes to compute separation score",
        }
    
    # Collect all features across all classes
    all_features = set()
    for class_label, features in constraints.items():
        all_features.update(features.keys())
    
    all_features = sorted(all_features)
    n_features = len(all_features)
    
    if n_features == 0:
        return {
            "score": 0.0,
            "n_classes": n_classes,
            "n_features": 0,
            "message": "No features with constraints found",
        }
    
    # Compute feature ranges
    feature_ranges = {}
    for feature in all_features:
        feature_ranges[feature] = _get_feature_range_across_classes(
            constraints, feature
        )
    
    # Compute pairwise separation scores for each feature
    class_pairs = list(combinations(class_labels, 2))
    
    per_feature_scores = {}
    per_pair_scores = {f"{c1} vs {c2}": {} for c1, c2 in class_pairs}
    
    for feature in all_features:
        feat_min, feat_max, feat_range = feature_ranges[feature]
        pair_scores = []
        
        for class1, class2 in class_pairs:
            # Get intervals for both classes
            feat_constraints1 = constraints[class1].get(feature, {})
            feat_constraints2 = constraints[class2].get(feature, {})
            
            # Skip if feature not constrained in either class
            if not feat_constraints1 or not feat_constraints2:
                continue
            
            interval1 = _get_interval_bounds(feat_constraints1, feat_min, feat_max)
            interval2 = _get_interval_bounds(feat_constraints2, feat_min, feat_max)
            
            score = _compute_separation_score_for_pair(
                interval1, interval2, feat_range
            )
            pair_scores.append(score)
            per_pair_scores[f"{class1} vs {class2}"][feature] = score
        
        if pair_scores:
            per_feature_scores[feature] = np.mean(pair_scores)
    
    # Compute overall score
    if per_feature_scores:
        # Weight by number of class pairs each feature participates in
        all_scores = []
        for feature, score in per_feature_scores.items():
            all_scores.append(score)
        overall_score = np.mean(all_scores)
    else:
        overall_score = 0.0
    
    # Compute per-pair average scores
    per_pair_avg = {}
    for pair_key, feature_scores in per_pair_scores.items():
        if feature_scores:
            per_pair_avg[pair_key] = np.mean(list(feature_scores.values()))
        else:
            per_pair_avg[pair_key] = 0.0
    
    result = {
        "score": float(overall_score),
        "n_classes": n_classes,
        "n_features": n_features,
        "n_class_pairs": len(class_pairs),
    }
    
    if verbose:
        result["per_feature_scores"] = per_feature_scores
        result["per_pair_scores"] = per_pair_scores
        result["per_pair_average"] = per_pair_avg
    
    return result


def compute_constraint_score_from_file(
    json_path: str,
    verbose: bool = False,
) -> Dict[str, Union[float, Dict]]:
    """Compute constraint score from a JSON file.
    
    Args:
        json_path: Path to JSON file with constraints
        verbose: If True, return detailed breakdown
    
    Returns:
        Dictionary with score and metadata
    """
    constraints = load_constraints_from_json(json_path)
    return compute_constraint_score(constraints, verbose=verbose)


def compare_constraints(
    constraints1: Dict[str, Dict],
    constraints2: Dict[str, Dict],
    name1: str = "Constraints 1",
    name2: str = "Constraints 2",
) -> Dict:
    """Compare two constraint sets and return their scores.
    
    Args:
        constraints1: First constraint dictionary
        constraints2: Second constraint dictionary
        name1: Name for first constraints
        name2: Name for second constraints
    
    Returns:
        Dictionary with comparison results
    """
    result1 = compute_constraint_score(constraints1, verbose=True)
    result2 = compute_constraint_score(constraints2, verbose=True)
    
    return {
        name1: result1,
        name2: result2,
        "score_difference": result1["score"] - result2["score"],
        "better": name1 if result1["score"] > result2["score"] else name2,
    }


# CLI interface
if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="Compute constraint separation score for DPG constraints"
    )
    parser.add_argument(
        "json_file",
        type=str,
        nargs="?",
        help="Path to constraints JSON file",
    )
    parser.add_argument(
        "--compare",
        type=str,
        nargs=2,
        metavar=("FILE1", "FILE2"),
        help="Compare two constraint files",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed breakdown",
    )
    
    args = parser.parse_args()
    
    if args.compare:
        file1, file2 = args.compare
        constraints1 = load_constraints_from_json(file1)
        constraints2 = load_constraints_from_json(file2)
        
        result = compare_constraints(
            constraints1, constraints2,
            name1=file1, name2=file2,
        )
        
        print(f"\n{'='*60}")
        print("CONSTRAINT SCORE COMPARISON")
        print(f"{'='*60}\n")
        
        for name in [file1, file2]:
            r = result[name]
            print(f"{name}:")
            print(f"  Score: {r['score']:.4f}")
            print(f"  Classes: {r['n_classes']}, Features: {r['n_features']}")
            if args.verbose and "per_feature_scores" in r:
                print("  Per-feature scores:")
                for feat, score in sorted(r["per_feature_scores"].items()):
                    print(f"    {feat}: {score:.4f}")
            print()
        
        print(f"{'='*60}")
        print(f"Score difference: {result['score_difference']:+.4f}")
        print(f"Better constraints: {result['better']}")
        print(f"{'='*60}\n")
        
    elif args.json_file:
        result = compute_constraint_score_from_file(args.json_file, verbose=args.verbose)
        
        print(f"\n{'='*60}")
        print("CONSTRAINT SEPARATION SCORE")
        print(f"{'='*60}\n")
        print(f"File: {args.json_file}")
        print(f"Score: {result['score']:.4f}")
        print(f"Classes: {result['n_classes']}")
        print(f"Features: {result['n_features']}")
        
        if args.verbose and "per_feature_scores" in result:
            print(f"\nPer-feature separation scores:")
            for feat, score in sorted(result["per_feature_scores"].items()):
                print(f"  {feat}: {score:.4f}")
            
            print(f"\nPer-class-pair average scores:")
            for pair, score in sorted(result["per_pair_average"].items()):
                print(f"  {pair}: {score:.4f}")
        
        print(f"\n{'='*60}\n")
    else:
        parser.print_help()
        sys.exit(1)
