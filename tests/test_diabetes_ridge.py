#!/usr/bin/env python3
"""Quick test for diabetes ridge plot with debug output."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from CounterFactualVisualizer import plot_ridge_comparison
from utils.config_manager import load_config
from utils.dataset_loader import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import json

# Load dataset
config_path = 'configs/diabetes/config.yaml'
config = load_config(config_path)

# Set random seed
seed = getattr(config.experiment_params, 'seed', 42)
np.random.seed(seed)

# Load dataset
dataset_info = load_dataset(config)
features_df = dataset_info["features_df"]
labels = dataset_info["labels"]

# Split and train model
test_size = getattr(config.data, 'test_size', 0.3)
random_state = getattr(config.data, 'random_state', 42)

train_features, test_features, train_labels, test_labels = train_test_split(
    features_df, labels, test_size=test_size, random_state=random_state
)

model_config = config.model.to_dict()
model_params = {k: v for k, v in model_config.items() if k != 'type' and v is not None}
model = RandomForestClassifier(**model_params)
model.fit(train_features, train_labels)

# Load constraints
with open('outputs/_comparison_results/dpg_constraints/diabetes_dpg_constraints.json', 'r') as f:
    constraints = json.load(f)

# Create a test sample (from class 0)
sample = {
    'Pregnancies': 2,
    'Glucose': 120,
    'BloodPressure': 70,
    'SkinThickness': 30,
    'Insulin': 150,
    'BMI': 30.0,
    'DiabetesPedigreeFunction': 0.3,
    'Age': 40
}

# Create test CFs (for class 1)
cf_dpg = {
    'Pregnancies': 8,
    'Glucose': 130,
    'BloodPressure': 75,
    'SkinThickness': 35,
    'Insulin': 200,
    'BMI': 32.0,
    'DiabetesPedigreeFunction': 0.35,
    'Age': 30
}

cf_dice = {
    'Pregnancies': 9,
    'Glucose': 135,
    'BloodPressure': 76,
    'SkinThickness': 36,
    'Insulin': 210,
    'BMI': 33.0,
    'DiabetesPedigreeFunction': 0.36,
    'Age': 31
}

# Predict classes
sample_df = pd.DataFrame([sample])
original_class = model.predict(sample_df)[0]
cf_df = pd.DataFrame([cf_dpg])
target_class = model.predict(cf_df)[0]

print(f"Original class: {original_class}")
print(f"Target class: {target_class}")

# Create ridge plot
fig = plot_ridge_comparison(
    sample=sample,
    cf_list_1=[cf_dpg],
    cf_list_2=[cf_dice],
    technique_names=('DPG', 'DiCE'),
    dataset_df=features_df,
    constraints=constraints,
    target=labels,
    target_class=target_class,
    original_class=original_class,
    show_per_class_distribution=True,
    show_original_class_constraints=True
)

if fig:
    fig.savefig('test_diabetes_ridge.png', dpi=150, bbox_inches='tight')
    print("\nSaved to test_diabetes_ridge.png")
else:
    print("\nFailed to create plot")
