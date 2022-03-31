# Radiotherapy-Prediction
Predict the probability of a patient requiring post-mastectomy radiotherapy

To visualize data:
`python3 src/visualize/visualizer.py`

To visualize all columns:
`python3 src/visualize/visualizer.py -vizall`

To run tests:
`python -m pytest tests/`

## Preprocess Module:
1. Feature Engineering. Construct new columns from available information
2. Missing Value Imputation. Fill in missing cells using expert rules and ML inference.


## Visualize Module: