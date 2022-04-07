# Radiotherapy-Prediction
Predict the probability of a patient requiring post-mastectomy radiotherapy

To visualize data:
`python3 src/visualize/visualizer.py`

To visualize all columns:
`python3 src/visualize/visualizer.py -vizall`

To run tests:
`python -m pytest tests/`

## Preprocessing Module
1. *Column Renaming*. Prefix "PRE_", "INT_", and "POS_" to indacate pre-operative, intra-operative, and post-operative columns.
2. *Dataset Cleansing*. Systematically replacing noisy values using predetermined rules.
3. *Feature Engineering*. Construct new interaction features guided by clinical insights.
4. *Missing Value Imputation*. Fill in missing values using imputation strategies and machine learning. For each column:
    1. Determine the type of column as real, ordinal, or categorical
    2. Apply K-Nearest Neighbor Imputer. Perform grid-search for optimal `K`.
    3. Apply Random Forest Imputer. Perform grid-search for optimal `max_depth` and `n_trees` using 5-fold cross-validation.
    4. Compare the KNN and RF imputers using accuracy, F1, and other metrics.
    5. ? use multiple imputation OR select the more accurate imputation strategy ?
5. *Visualize Dataset*. Generate a figure highlighting the changed cell values in the dataset.

