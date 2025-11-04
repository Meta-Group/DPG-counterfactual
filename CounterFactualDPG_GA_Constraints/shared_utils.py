"""
Shared utilities for CounterFactual DPG notebooks.

This module contains common functions and classes used across multiple notebooks
in the CounterFactualDPG_GA_Constraints directory.
"""

import os
import ast
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict, List, Tuple, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean, cityblock, cosine
from scipy.spatial.distance import cdist


# ============================================================================
# Path Setup
# ============================================================================

def setup_paths():
    """
    Set up paths for accessing constraints directory.
    
    Returns:
        tuple: (notebook_dir, constraints_dir, PATH)
            - notebook_dir: Current working directory
            - constraints_dir: Absolute path to constraints directory
            - PATH: Path to constraints directory with trailing slash
    """
    notebook_dir = os.getcwd()
    constraints_dir = os.path.abspath(os.path.join(notebook_dir, '..', 'constraints'))
    PATH = constraints_dir + '/'
    return notebook_dir, constraints_dir, PATH


# ============================================================================
# Data Loading and Model Training
# ============================================================================

def load_and_train_model():
    """
    Load the Iris dataset, split it, and train a RandomForestClassifier.
    
    Returns:
        tuple: (iris, X, y, X_train, X_test, y_train, y_test, model)
            - iris: The Iris dataset object
            - X: Feature data
            - y: Target labels
            - X_train, X_test: Training and test feature sets
            - y_train, y_test: Training and test labels
            - model: Trained RandomForestClassifier
    """
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train a RandomForestClassifier with 3 base learners
    model = RandomForestClassifier(n_estimators=3, random_state=42)
    model.fit(X_train, y_train)
    
    return iris, X, y, X_train, X_test, y_train, y_test, model


# ============================================================================
# Constraint Loading
# ============================================================================

def load_constraints_source(PATH, filename="iris_l3_pv0.001_t2_dpg_metrics.txt"):
    """
    Load constraints source from a file.
    
    Args:
        PATH: Path to the constraints directory with trailing slash
        filename: Name of the constraints file to load
        
    Returns:
        str: The constraints source string (line 1 from the file)
    """
    with open(PATH + filename, 'r') as file:
        lines = file.readlines()
    
    constraints_source = lines[1]
    return constraints_source


# ============================================================================
# Constraint Parsing Functions
# ============================================================================

def parse_condition(condition):
    """Parse a single condition string into a list of dictionaries with feature, operator, and value."""
    # Split by logical operators while ignoring the first part if it's a standalone number
    parts = re.split(r" (<=|>=|<|>|==) ", condition.strip())

    if len(parts) == 3:
        # Simple case: single comparison
        feature, operator, value = parts
        return [{"feature": feature.strip(), "operator": operator, "value": float(value.strip())}]
    elif len(parts) == 5:
        # Complex case: range comparison
        value1, operator1, feature, operator2, value2 = parts
        return [
            {"feature": feature.strip(), "operator": operator1, "value": float(value1.strip())},
            {"feature": feature.strip(), "operator": operator2, "value": float(value2.strip())}
        ]
    else:
        # In case of an unexpected format, return None
        return None


def constraints_v1_to_dict(raw_string):
    """
    Convert raw constraint string to a nested dictionary.
    
    Args:
        raw_string: Raw string containing class bounds
        
    Returns:
        dict: Nested dictionary with class names as keys and constraint conditions as values
    """
    # Remove the prefix and newline character from the string
    stripped_string = raw_string.replace("Class Bounds: ", "").strip()

    # Use ast.literal_eval to safely parse the string into a dictionary
    parsed_dict = ast.literal_eval(stripped_string)

    # Create the nested dictionary with parsed conditions
    nested_dict = {}
    for class_name, conditions in parsed_dict.items():
        nested_conditions = []
        for condition in conditions:
            # Parse each condition into a list of dictionaries of feature, operator, and value
            parsed_conditions = parse_condition(condition)
            if parsed_conditions:
                nested_conditions.extend(parsed_conditions)
        nested_dict[class_name] = nested_conditions

    return nested_dict


def transform_by_feature(nested_dict):
    """
    Transform constraint dictionary to be organized by feature instead of by class.
    
    Args:
        nested_dict: Dictionary with class-based constraints
        
    Returns:
        dict: Dictionary with feature-based constraints
    """
    feature_dict = {}

    # Iterate through each class and its conditions
    for class_name, conditions in nested_dict.items():
        for condition in conditions:
            feature = condition["feature"]
            if feature not in feature_dict:
                feature_dict[feature] = []
            # Append the condition along with the class it belongs to
            feature_dict[feature].append({"class": class_name, "operator": condition["operator"], "value": condition["value"]})

    return feature_dict


def get_intervals_by_feature(feature_based_dict):
    """
    Get min/max intervals for each feature based on constraints.
    
    Args:
        feature_based_dict: Dictionary with feature-based constraints
        
    Returns:
        dict: Dictionary mapping features to (lower_bound, upper_bound) tuples
    """
    feature_intervals = {}

    for feature, conditions in feature_based_dict.items():
        # Initialize intervals
        lower_bound = float('-inf')
        upper_bound = float('inf')

        # Check each condition and update the bounds
        for condition in conditions:
            operator = condition["operator"]
            value = condition["value"]

            if operator == "<":
                upper_bound = min(upper_bound, value)
            elif operator == "<=":
                upper_bound = min(upper_bound, value)
            elif operator == ">":
                lower_bound = max(lower_bound, value)
            elif operator == ">=":
                lower_bound = max(lower_bound, value)

        # Store the interval for this feature
        feature_intervals[feature] = (lower_bound, upper_bound)

    return feature_intervals


def is_value_valid_for_class(class_name, feature, value, nested_dict):
    """
    Check if a value is valid for a specific class and feature.
    
    Args:
        class_name: Name of the class to check against
        feature: Feature name
        value: Value to validate
        nested_dict: Nested dictionary containing constraints
        
    Returns:
        bool: True if value satisfies all conditions, False otherwise
    """
    # Get the conditions for the given class
    conditions = nested_dict.get(class_name, [])

    # Iterate through each condition for the specified feature
    for condition in conditions:
        if condition["feature"] == feature:
            operator = condition["operator"]
            comparison_value = condition["value"]

            # Check if the value satisfies the condition
            if operator == "<" and not (value < comparison_value):
                return False
            elif operator == "<=" and not (value <= comparison_value):
                return False
            elif operator == ">" and not (value > comparison_value):
                return False
            elif operator == ">=" and not (value >= comparison_value):
                return False

    # If all conditions are satisfied, return True
    return True


def read_constraints_from_file(filename):
    """
    Read constraints from a file in JSON format.
    
    Args:
        filename: Path to the constraints file
        
    Returns:
        dict: Dictionary containing parsed constraints
    """
    constraints_dict = {}
    with open(filename, 'r') as file:
        for line in file:
            # Remove new line characters and any leading/trailing whitespace
            line = line.strip()
            if not line:
                continue

            # Split the line at the first colon to separate the class label from the JSON data
            class_label, json_string = line.split(":", 1)

            # Clean up json_string by replacing single quotes with double quotes to make it valid JSON
            json_string = json_string.strip().replace("'", '"').replace("None", "null")

            try:
                # Convert the JSON string into a Python dictionary
                constraints_dict[class_label.strip()] = json.loads(json_string)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON for {class_label}: {e}")

    return constraints_dict


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_pca_with_counterfactual(model, dataset, target, sample, counterfactual):
    """
    Plot a PCA visualization of the dataset with the original sample and counterfactual.

    Args:
        model: Trained scikit-learn model used for predicting the class of the counterfactual.
        dataset: The original dataset (features) used for PCA.
        target: The target labels for the dataset.
        sample: The original sample as a dictionary of feature values.
        counterfactual: The counterfactual sample as a dictionary of feature values.
    """

    # Perform PCA on the scaled dataset
    pca = PCA(n_components=2)
    iris_pca = pca.fit_transform(dataset)

    # Transform the original sample and counterfactual using the same PCA
    original_sample_pca = pca.transform(pd.DataFrame([sample]))
    counterfactual_pca = pca.transform(pd.DataFrame([counterfactual]))

    # Predict the class of the counterfactual
    counterfactual_class = model.predict(pd.DataFrame([counterfactual]))[0]

    # Plot the PCA results with class colors and 'x' marker for the counterfactual
    plt.figure(figsize=(10, 6))
    colors = ['purple', 'green', 'orange']  # Colors for the classes

    for class_value in np.unique(target):
        plt.scatter(
            iris_pca[target == class_value, 0],
            iris_pca[target == class_value, 1],
            label=f"Class {class_value}",
            color=colors[class_value],
            alpha=0.6
        )

    plt.scatter(
        original_sample_pca[:, 0], original_sample_pca[:, 1],
        color='red', label='Original Sample', edgecolor='black'
    )
    plt.scatter(
        counterfactual_pca[:, 0], counterfactual_pca[:, 1],
        color=colors[counterfactual_class], marker='x', s=100, label='Counterfactual', edgecolor='black'
    )

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA Plot with Original Sample and Counterfactual')
    plt.legend()
    plt.show()


def plot_pairwise_with_counterfactual(model, dataset, target, sample, counterfactual):
    """
    Plot a Seaborn pairplot of the dataset, highlighting the original sample and counterfactual.

    Args:
        model: Trained scikit-learn model used for predicting the class of the counterfactual.
        dataset: The original dataset (features) used for the plot.
        target: The target labels for the dataset.
        sample: The original sample as a dictionary of feature values.
        counterfactual: The counterfactual sample as a dictionary of feature values.
    """
    # Convert the dataset into a DataFrame and add the target labels
    data_df = pd.DataFrame(dataset, columns=pd.DataFrame([sample]).columns)
    data_df['label'] = 'Dataset'

    # Convert the original sample and counterfactual to DataFrames
    original_sample_df = pd.DataFrame([sample])
    counterfactual_df = pd.DataFrame([counterfactual])

    # Add labels to distinguish the original sample and counterfactual in the plot
    original_sample_df['label'] = 'Original Sample'
    counterfactual_df['label'] = 'Counterfactual'

    # Combine the original sample and counterfactual with the dataset for plotting
    combined_df = pd.concat([data_df, original_sample_df, counterfactual_df], ignore_index=True)

    # Plot the pairplot with Seaborn
    sns.pairplot(combined_df, hue='label', palette={'Dataset': 'gray', 'Original Sample': 'red', 'Counterfactual': 'blue'})
    plt.suptitle('Pairwise Plot with Original Sample and Counterfactual', y=1.02)
    plt.show()


def plot_sample_and_counterfactual_heatmap(sample, class_sample, counterfactual, class_counterfactual, restrictions):
    """
    Plot the original sample, the differences, and the counterfactual as a heatmap,
    and indicate restrictions using icons.

    Args:
        sample (dict): Original sample values.
        class_sample: Class of the original sample.
        counterfactual (dict): Counterfactual sample values.
        class_counterfactual: Class of the counterfactual sample.
        restrictions (dict): Restrictions applied to each feature.
    """
    # Set larger font sizes globally
    plt.rcParams['font.size'] = 12  # Adjusts the default font size
    plt.rcParams['axes.labelsize'] = 16  # Font size for x and y labels
    plt.rcParams['axes.titlesize'] = 16  # Font size for the plot title
    plt.rcParams['xtick.labelsize'] = 12  # Font size for x-axis tick labels
    plt.rcParams['ytick.labelsize'] = 12  # Font size for y-axis tick labels
    
    # Create DataFrame from the samples
    sample_df = pd.DataFrame([sample], index=['Original'])
    cf_df = pd.DataFrame([counterfactual], index=['Counterfactual'])

    # Calculate differences
    differences = (cf_df.loc['Counterfactual'] - sample_df.loc['Original']).to_frame('Difference').T

    # Combine all data
    full_df = pd.concat([sample_df, differences, cf_df])

    # Map restrictions to symbols
    symbol_map = {
        'no_change': '⊝',  # Locked symbol for no change
        'non_increasing': '⬇️',  # Down arrow for non-increasing
        'non_decreasing': '⬆️'  # Up arrow for non-decreasing
    }
    restrictions_ser = pd.Series(restrictions).replace(symbol_map)

    mask = np.full_like(full_df, False, dtype=bool)  # Start with no masking
    mask[[0, -1], :] = True  # Only mask the first and last rows

    vmax = np.max(np.abs(full_df.values))
    vmin = -vmax

    # Plotting the heatmap for numeric data
    plt.figure(figsize=(10, 5))
    ax = sns.heatmap(full_df, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, linewidths=1.2, linecolor='k',
                     vmin=vmin, vmax=vmax, mask=mask)

    # Annotate with restrictions
    for i, (feat, restr) in enumerate(restrictions_ser.items()):
        ax.text(i + 0.5, 3.5, restr, ha='center', va='center', color='black', fontweight='bold', fontsize=14)

    annotations = full_df.round(2).copy().astype(str)
    for col in full_df.columns:
        annotations.loc['Difference', col] = f"Δ {full_df.loc['Difference', col]:.2f}"

    for (i, j), val in np.ndenumerate(full_df):
        if i == 1:
            continue
        ax.text(j + 0.5, i + 0.5, annotations.iloc[i, j],
                horizontalalignment='center', verticalalignment='center', color='black')

    plt.title(f'Original (Class {class_sample}), Counterfactual (Class {class_counterfactual}) with Restrictions')
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0, va="center")
    plt.show()



class CounterFactualDPG:
    def __init__(self, model, constraints, dict_non_actionable=None):
        """
        Initialize the CounterFactualDPG object.

        Args:
            model: The machine learning model used for predictions.
            constraints (dict): Nested dictionary containing constraints for features.
            dict_non_actionable (dict): Dictionary mapping features to non-actionable constraints.
              non_decreasing: feature cannot decrease
              non_increasing: feature cannot increase
              no_change: feature cannot change
        """
        self.model = model
        self.constraints = constraints
        self.dict_non_actionable = dict_non_actionable #non_decreasing, non_increasing, no_change
        self.average_fitness_list = []
        self.best_fitness_list = []

    def is_actionable_change(self, counterfactual_sample, original_sample):
      """
      Check if changes in features are actionable based on constraints.

      Args:
          counterfactual_sample (dict): The modified sample with new feature values.
          original_sample (dict): The original sample with feature values.

      Returns:
          bool: True if all changes are actionable, False otherwise.
      """
      if not self.dict_non_actionable:
          return True

      for feature, new_value in counterfactual_sample.items():
          if feature not in self.dict_non_actionable:
              continue

          original_value = original_sample.get(feature)
          constraint = self.dict_non_actionable[feature]

          if constraint == "non_decreasing" and new_value < original_value:
              return False
          if constraint == "non_increasing" and new_value > original_value:
              return False
          if constraint == "no_change" and new_value != original_value:
              return False

      return True


    def check_validity(self, counterfactual_sample, original_sample, desired_class):
        """
        Checks the validity of a counterfactual sample.

        Parameters:
        - counterfactual_sample: Array-like, shape (n_features,), the counterfactual sample.
        - original_sample: Array-like, shape (n_features,), the original input sample.
        - desired_class: The desired class label.

        Returns:
        - 0 if the predicted class matches the desired class and the sample is different from the original.
        - np.inf if the predicted class does not match the desired class or the sample is identical to the original.
        """
        # Ensure the input samples are numpy arrays
        counterfactual_sample = np.array(counterfactual_sample).reshape(1, -1)
        original_sample = np.array(original_sample).reshape(1, -1)

        # Check if the counterfactual sample is different from the original sample
        if np.array_equal(counterfactual_sample, original_sample):
            return False  # Return np.inf if the samples are identical

        # Predict the class for the counterfactual sample
        #print('self.model.predict(counterfactual_sample)[0]', self.model.predict(counterfactual_sample)[0])
        predicted_class = self.model.predict(counterfactual_sample)[0]

        # Check if the predicted class matches the desired class
        if predicted_class == desired_class:
            return True
        else:
            return False

    def plot_fitness(self):
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))  # Create two subplots vertically

        # Plot best fitness
        axs[0].plot(self.best_fitness_list, label='Best Fitness', color='blue')
        axs[0].set_title('Best Fitness Over Generations')
        axs[0].set_xlabel('Generation')
        axs[0].set_ylabel('Best Fitness')
        axs[0].legend()

        # Plot average fitness
        axs[1].plot(self.average_fitness_list, label='Average Fitness', color='green')
        axs[1].set_title('Average Fitness Over Generations')
        axs[1].set_xlabel('Generation')
        axs[1].set_ylabel('Average Fitness')
        axs[1].legend()

        plt.tight_layout()
        plt.show()



    def calculate_distance(self,original_sample, counterfactual_sample, metric="euclidean"):
        """
        Calculates the distance between the original sample and the counterfactual sample.

        Parameters:
        - original_sample: Array-like, shape (n_features,), the original input sample.
        - counterfactual_sample: Array-like, shape (n_features,), the counterfactual sample.
        - metric: String, the distance metric to use. Options are "euclidean", "manhattan", or "cosine".

        Returns:
        - Distance between the original sample and the counterfactual sample.
        """
        # Ensure inputs are numpy arrays
        original_sample = np.array(original_sample)
        counterfactual_sample = np.array(counterfactual_sample)

        # Validate metric and compute distance
        if metric == "euclidean":
            distance = euclidean(original_sample, counterfactual_sample)
        elif metric == "manhattan":
            distance = cityblock(original_sample, counterfactual_sample)
        elif metric == "cosine":
            # Avoid division by zero in cosine similarity
            if np.all(original_sample == 0) or np.all(counterfactual_sample == 0):
                distance = 1  # Max cosine distance if one vector is zero
            else:
                distance = cosine(original_sample, counterfactual_sample)
        else:
            raise ValueError("Invalid metric. Choose from 'euclidean', 'manhattan', or 'cosine'.")

        return distance

    def validate_constraints(self, S_prime, sample, target_class):
        """
        Validate if the modified sample S_prime meets all constraints for the specified target class.

        Args:
            S_prime (dict): Modified sample with feature values.
            sample (dict): The original sample with feature values.
            target_class (int): The target class for filtering constraints.

        Returns:
            (bool, float): Tuple of whether the changes are valid and a penalty score.
        """
        penalty = 0.0
        valid_change = True

        # Filter the constraints for the specified target class
        class_constraints = self.constraints.get(str("Class "+str(target_class)), [])

        for feature, new_value in S_prime.items():
            original_value = sample.get(feature)

            # Check if the feature value has changed
            if new_value != original_value:
                # Validate numerical constraints specific to the target class
                for condition in class_constraints:
                    if condition["feature"] == feature:
                        operator = condition["operator"]
                        constraint_value = condition["value"]

                        #print("Feature:", feature)
                        #print("Operator:", operator)
                        #print("Constraint Value:", constraint_value)
                        #print("New Value:", new_value)

                        # Check if the new value violates any constraints
                        if operator == "<" and not (new_value < constraint_value):
                            valid_change = False
                            penalty += constraint_value
                        elif operator == "<=" and not (new_value <= constraint_value):
                            valid_change = False
                            penalty += constraint_value
                        elif operator == ">" and not (new_value > constraint_value):
                            valid_change = False
                            penalty += constraint_value
                        elif operator == ">=" and not (new_value >= constraint_value):
                            valid_change = False
                            penalty += constraint_value

        # Collect all constraints that are NOT related to the target class
        non_target_class_constraints = [
            condition
            for class_name, conditions in self.constraints.items()
            if class_name != "Class " + str(target_class)  # Exclude the target class constraints
            for condition in conditions
        ]

        for feature, new_value in S_prime.items():
            original_value = sample.get(feature)

            # Check if the feature value has changed
            if new_value != original_value:
                # Validate numerical constraints NOT related to the target class
                for condition in non_target_class_constraints:
                    if condition["feature"] == feature:
                        operator = condition["operator"]
                        constraint_value = condition["value"]

                        # Check if the new value violates any constraints
                        if operator == "<" and (new_value < constraint_value):
                            valid_change = False
                            penalty += constraint_value
                        elif operator == "<=" and (new_value <= constraint_value):
                            valid_change = False
                            penalty += constraint_value
                        elif operator == ">" and (new_value > constraint_value):
                            valid_change = False
                            penalty += constraint_value
                        elif operator == ">=" and (new_value >= constraint_value):
                            valid_change = False
                            penalty += constraint_value


        #print('Total Penalty:', penalty)
        return valid_change, penalty


    def get_valid_sample(self, sample, target_class):
        """
        Generate a valid sample that meets all constraints for the specified target class
        while respecting actionable changes.

        Args:
            sample (dict): The sample with feature values.
            target_class (int): The target class for filtering constraints.

        Returns:
            dict: A valid sample that meets all constraints for the target class
                  and respects actionable changes.
        """
        adjusted_sample = sample.copy()  # Start with the original values

        for feature, original_value in sample.items():
            min_value = -np.inf
            max_value = np.inf

            # Filter the constraints for the specified target class
            class_constraints = self.constraints.get(f"Class {target_class}", [])

            # Find the constraints for this feature
            for condition in class_constraints:
                if condition["feature"] == feature:
                    operator = condition["operator"]
                    constraint_value = condition["value"]

                    # Update the min and max values based on the constraints
                    if operator == "<":
                        max_value = min(max_value, constraint_value - 1e-5)
                    elif operator == "<=":
                        max_value = min(max_value, constraint_value)
                    elif operator == ">":
                        min_value = max(min_value, constraint_value + 1e-5)
                    elif operator == ">=":
                        min_value = max(min_value, constraint_value)

            # Incorporate non-actionable constraints
            if self.dict_non_actionable and feature in self.dict_non_actionable:
                actionability = self.dict_non_actionable[feature]
                if actionability == "non_decreasing":
                    min_value = max(min_value, original_value)
                elif actionability == "non_increasing":
                    max_value = min(max_value, original_value)
                elif actionability == "no_change":
                    adjusted_sample[feature] = original_value
                    continue

            # Generate a random value within the valid range
            if min_value == -np.inf:
                min_value = 0  # Default lower bound if no constraint is specified
            if max_value == np.inf:
                max_value = min_value + 10  # Default upper bound if no constraint is specified

            adjusted_sample[feature] = np.random.uniform(min_value, max_value)

        return adjusted_sample

    def calculate_sparsity(self, original_sample, counterfactual_sample):
        total_features = len(original_sample)
        unchanged_features = sum(
            original_sample[feature]*3 for feature in original_sample if original_sample[feature] != counterfactual_sample[feature]
        )
        sparsity = unchanged_features / total_features
        return sparsity

    def calculate_fitness(self, individual, original_features, sample, target_class, metric="cosine"):
            """
            Calculate the fitness score for an individual sample.

            Args:
                individual (dict): The individual sample with feature values.
                original_features (np.array): The original feature values.
                sample (dict): The original sample with feature values.
                target_class (int): The desired class for the counterfactual.
                metric (str): The distance metric to use for calculating distance.

            Returns:
                float: The fitness score for the individual.
            """
            #print('individual', individual)

            # Convert individual feature values to a numpy array
            features = np.array([individual[feature] for feature in sample.keys()]).reshape(1, -1)

            # Calculate validity score based on class
            is_valid_class = self.check_validity(features.flatten(), original_features.flatten(), target_class)
            #print('is_valid_class', is_valid_class)

            # Calculate distance score
            distance_score = self.calculate_distance(original_features, features.flatten(), metric)

            #Calculate sparcity (number of features modified)
            sparsity_score = self.calculate_sparsity(sample, individual)

            # Calculate_manufold_distance
            #manifold_distance = self.calculate_manifold_distance(self.X, individual)
            #print('calculate_manifold_distance', manifold_distance)

            # Check the constraints
            is_valid_constraint, penalty_constraints = self.validate_constraints(individual, sample, target_class)

            # Check if the change is actionable
            if not self.is_actionable_change(individual, sample) or not is_valid_class:
                fitness = +np.inf
                return fitness

            if is_valid_class :
                fitness = (2*distance_score) + penalty_constraints + sparsity_score
            elif is_valid_constraint:
                fitness = 5 * ((2*distance_score) + penalty_constraints + sparsity_score)  # High penalty for invalid samples
            else:
                fitness = 10 * ((2*distance_score) + penalty_constraints + sparsity_score)  # High penalty for invalid samples

            return fitness


    def genetic_algorithm(self, sample, target_class, population_size=100, generations=100, mutation_rate=0.8, metric="euclidean", delta_threshold=0.01, patience=20):
      # Initialize population with random values within a reasonable range
      population = []
      feature_names = list(sample.keys())
      previous_best_fitness = float('inf')
      stable_generations = 0  # Counter for generations with minimal fitness improvement

      for _ in range(population_size):
          individual = self.get_valid_sample(sample, target_class)
          population.append(individual)

      original_features = np.array([sample[feature] for feature in feature_names])

      self.best_fitness_list = []
      self.average_fitness_list = []
      # Main loop for generations
      for generation in range(generations):
          fitness_scores = []

          # Calculate fitness for each individual
          for individual in population:
              fitness = self.calculate_fitness(individual, original_features, sample, target_class, metric)
              fitness_scores.append(fitness)

          # Find the best candidate and its fitness score
          best_index = np.argmin(fitness_scores)
          best_candidate = population[best_index]
          best_fitness = fitness_scores[best_index]

          # Check for convergence based on the fitness delta threshold
          fitness_improvement = previous_best_fitness - best_fitness
          if fitness_improvement < delta_threshold:
              stable_generations += 1
          else:
              stable_generations = 0  # Reset if there's sufficient improvement

          # Print the average fitness and the best candidate
          #print(f"****** Generation {generation + 1}: Average Fitness = {np.mean(fitness_scores):.4f}, Best Fitness = {best_fitness:.4f}, fitness improvement = {fitness_improvement:.4f}")

          previous_best_fitness = best_fitness
          self.best_fitness_list.append(best_fitness)
          self.average_fitness_list.append(np.mean(fitness_scores))

          # Stop if improvement is less than the threshold for a consecutive number of generations
          if stable_generations >= patience:
              print(f"Convergence reached at generation {generation + 1}")
              break

            # Use tournament selection to choose parents
          selected_parents = []
          for _ in range(population_size):
              tournament = np.random.choice(population, size=4, replace=False)
              tournament_fitness = [fitness_scores[population.index(ind)] for ind in tournament]
              selected_parents.append(tournament[np.argmin(tournament_fitness)])

          # Generate new population using crossover and mutation
          new_population = []
          for parent in selected_parents:
              offspring = parent.copy()
              for feature in feature_names:
                  if np.random.rand() < mutation_rate:
                      # Apply mutation only if the feature is actionable
                      if self.dict_non_actionable and feature in self.dict_non_actionable:
                          actionability = self.dict_non_actionable[feature]
                          original_value = parent[feature]
                          if actionability == "non_decreasing":
                              mutation_value = np.random.uniform(0, 0.5)  # Only allow increase
                              offspring[feature] += mutation_value
                          elif actionability == "non_increasing":
                              mutation_value = np.random.uniform(-0.5, 0)  # Only allow decrease
                              offspring[feature] += mutation_value
                          elif actionability == "no_change":
                              offspring[feature] = original_value  # Do not change
                          else:
                              # If no specific actionability rule, apply normal mutation
                              offspring[feature] += np.random.uniform(-0.5, 0.5)
                      else:
                          # If the feature is not in the non-actionable list, apply normal mutation
                          offspring[feature] += np.random.uniform(-0.5, 0.5)

                      # Ensure offspring values stay within valid domain constraints
                      offspring[feature] = np.round(max(0, offspring[feature]), 2)  # Adjust this based on domain-specific constraints
              new_population.append(offspring)


          # Reduce mutation rate over generations (adaptive mutation)
          mutation_rate *= 0.99

          # Update population
          population = new_population

      if best_fitness == np.inf:
          return None

      # Return the best individual based on the lowest fitness score
      best_index = np.argmin(fitness_scores)
      return population[best_index]




    def generate_counterfactual(self, sample, target_class):
        """
        Generate a counterfactual for the given sample and target class using a genetic algorithm.

        Args:
            sample (dict): The original sample with feature values.
            target_class (int): The desired class for the counterfactual.

        Returns:
            dict: A modified sample representing the counterfactual or None if not found.
        """
        sample_class = self.model.predict(pd.DataFrame([sample]))[0]
        # Check if the predicted class matches the desired class
        if sample_class == target_class:
            raise ValueError("Target class need to be different from the predicted class label.")
        #counterfactual = None
        #while counterfactual is None:
        counterfactual = self.genetic_algorithm(sample, target_class, population_size=100,  generations=100)
        return counterfactual
