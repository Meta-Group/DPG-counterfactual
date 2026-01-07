# Fix: RandomForestClassifier Feature Names Warning

## Issue
When running the experiment config, the following warning was produced:
```
UserWarning: X has feature names, but RandomForestClassifier was fitted without feature names
  warnings.warn(
```

## Root Cause
The mismatch occurred in the training and prediction workflow:

1. **Training (Line 91-92 in `experiment_generation.py`):**
   - Model was trained with raw numpy arrays from `sklearn.datasets.load_iris()`:
   ```python
   TRAIN_FEATURES, TEST_FEATURES, TRAIN_LABELS, TEST_LABELS = train_test_split(
       IRIS_FEATURES,  # numpy array - NO feature names
       IRIS_LABELS, test_size=0.3, random_state=42
   )
   model.fit(TRAIN_FEATURES, TRAIN_LABELS)  # Trained without feature names
   ```

2. **Prediction (Line 104-105):**
   - Predictions were made with pandas DataFrames (which have feature names):
   ```python
   SAMPLE_DATAFRAME = pd.DataFrame([ORIGINAL_SAMPLE])  # Has feature names!
   model.predict(SAMPLE_DATAFRAME)[0]
   ```

3. **Additional locations:**
   - `CounterFactualModel.py` line 692: `model.predict(pd.DataFrame([sample]))`

sklearn's validation logic now tracks whether the model was fitted with feature names, and warns when there's a mismatch between training and prediction.

## Solution
Train the model with a pandas DataFrame that includes feature names from the start:

```python
# Create DataFrame with feature names for consistent handling
IRIS_FEATURES_DF = pd.DataFrame(IRIS_FEATURES, columns=IRIS.feature_names)

TRAIN_FEATURES, TEST_FEATURES, TRAIN_LABELS, TEST_LABELS = train_test_split(
    IRIS_FEATURES_DF,  # DataFrame - HAS feature names
    IRIS_LABELS, test_size=0.3, random_state=42
)

# Train model with DataFrame (preserves feature names)
model.fit(TRAIN_FEATURES, TRAIN_LABELS)  # Now fitted WITH feature names
```

This ensures that:
- The model stores feature names during fit: `model.feature_names_in_`
- All subsequent predictions with DataFrames match the training setup
- No more warnings about feature name mismatches

## Additional Notes
- The ConstraintParser receives `.values` (numpy array) to maintain compatibility with DPG's iteration logic
- All predictions in downstream code continue to use DataFrames as before
- The fix maintains backward compatibility with existing code

## Files Modified
- `/home/rafaelgoncalves/gitgud/CounterFactualDPG/scripts/experiment_generation.py` (Lines 83-99)
