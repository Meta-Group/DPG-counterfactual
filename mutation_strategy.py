"""
MutationStrategy: Provides individual creation for candidate generation.

Simple utility class for creating candidate individuals for counterfactual generation.
"""


class MutationStrategy:
    """
    Provides individual creation utilities for candidate generation.
    """

    def __init__(
        self,
        constraints,
        dict_non_actionable=None,
        escape_pressure=0.5,
        prioritize_non_overlapping=True,
        boundary_analyzer=None,
    ):
        """
        Initialize the MutationStrategy.

        Args:
            constraints (dict): Feature constraints per class.
            dict_non_actionable (dict): Dictionary mapping features to non-actionable constraints.
            escape_pressure (float): Unused, kept for initialization compatibility.
            prioritize_non_overlapping (bool): Unused, kept for initialization compatibility.
            boundary_analyzer: Optional BoundaryAnalyzer instance.
        """
        self.constraints = constraints
        self.dict_non_actionable = dict_non_actionable
        self.escape_pressure = escape_pressure
        self.prioritize_non_overlapping = prioritize_non_overlapping
        self.boundary_analyzer = boundary_analyzer

    def create_deap_individual(self, sample_dict, feature_names):
        """Create an individual from a dictionary.
        
        Args:
            sample_dict (dict): Sample feature values.
            feature_names: Unused, kept for API compatibility.
        
        Returns:
            dict: A copy of the sample dictionary.
        """
        return dict(sample_dict)
