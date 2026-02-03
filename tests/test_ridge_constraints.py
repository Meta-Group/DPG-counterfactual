#!/usr/bin/env python3
"""Quick test to verify ridge plot constraint handling."""

import json

# Load the diabetes constraints
with open('outputs/_comparison_results/dpg_constraints/diabetes_dpg_constraints.json', 'r') as f:
    constraints = json.load(f)

print("Constraints loaded:")
print(json.dumps(constraints, indent=2))

print("\n\nClass 0 constraints:")
for feat, vals in constraints['Class 0'].items():
    print(f"  {feat}: min={vals['min']}, max={vals['max']}")

print("\n\nClass 1 constraints:")
for feat, vals in constraints['Class 1'].items():
    print(f"  {feat}: min={vals['min']}, max={vals['max']}")

print("\n\nDifferences (where Class 0 and Class 1 differ):")
for feat in constraints['Class 0'].keys():
    c0_min = constraints['Class 0'][feat]['min']
    c0_max = constraints['Class 0'][feat]['max']
    c1_min = constraints['Class 1'][feat]['min']
    c1_max = constraints['Class 1'][feat]['max']
    
    if c0_min != c1_min or c0_max != c1_max:
        print(f"  {feat}:")
        print(f"    Class 0: [{c0_min}, {c0_max}]")
        print(f"    Class 1: [{c1_min}, {c1_max}]")
