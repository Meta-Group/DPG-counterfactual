"""
Feature utility functions for counterfactual generation.

Provides shared feature name normalization and matching functions
used across CounterFactualModel, BoundaryAnalyzer, and ConstraintValidator.
"""

import re


def normalize_feature_name(feature: str) -> str:
    """
    Normalize feature name by stripping whitespace, removing units in parentheses,
    converting to lowercase, and replacing underscores with spaces. This helps match
    features that may have slight variations in naming (e.g., "sepal width" vs
    "sepal_width" vs "sepal width (cm)").

    Args:
        feature (str): The feature name to normalize.

    Returns:
        str: Normalized feature name.
    """
    # Remove anything in parentheses (like units)
    feature = re.sub(r"\s*\([^)]*\)", "", feature)
    # Replace underscores with spaces
    feature = feature.replace("_", " ")
    # Normalize multiple spaces to single space
    feature = re.sub(r"\s+", " ", feature)
    # Strip whitespace and convert to lowercase
    return feature.strip().lower()


def features_match(feature1: str, feature2: str) -> bool:
    """
    Check if two feature names match, using normalized comparison.

    Args:
        feature1 (str): First feature name.
        feature2 (str): Second feature name.

    Returns:
        bool: True if features match, False otherwise.
    """
    return normalize_feature_name(feature1) == normalize_feature_name(feature2)


def correct_escape_direction(
    escape_dir: str, original_value: float, raw_target_min, raw_target_max
) -> str:
    """
    Correct escape direction based on actual sample position relative to target bounds.

    If the original value is already below the target minimum, the direction should
    be 'increase' regardless of what boundary analysis suggests. Likewise for above
    the target maximum.

    Args:
        escape_dir (str): Current escape direction ('increase' or 'decrease').
        original_value (float): The original sample's value for this feature.
        raw_target_min: The target class lower bound (or None).
        raw_target_max: The target class upper bound (or None).

    Returns:
        str: Corrected escape direction.
    """
    if raw_target_min is not None and raw_target_max is not None:
        if original_value < raw_target_min and escape_dir == "decrease":
            return "increase"
        elif original_value > raw_target_max and escape_dir == "increase":
            return "decrease"
    elif raw_target_min is not None and original_value < raw_target_min and escape_dir == "decrease":
        return "increase"
    elif raw_target_max is not None and original_value > raw_target_max and escape_dir == "increase":
        return "decrease"
    return escape_dir
