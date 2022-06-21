# Machine Learning for Radiotherapy Prediction
---
## Table of Contents
<!-- - [Machine Learning for Radiotherapy Prediction](#machine-learning-for-radiotherapy-prediction) -->
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Installation](#installation)
  - [Usage](#usage)
  - [An End-to-End ML Pipeline](#an-end-to-end-ml-pipeline)
    - [Pipeline Overview Diagram](#pipeline-overview-diagram)
    - [Data Preprocessing Module](#data-preprocessing-module)
    - [ML Training-Evaluation Module](#ml-training-evaluation-module)
  - [Results](#results)
    - [Figures](#figures)
      - [AUC Performance Metrics](#auc-performance-metrics)
      - [Feature Importance and Feature Selection](#feature-importance-and-feature-selection)
    - [Tables](#tables)
      - [Feature Scores by Model (Example)](#feature-scores-by-model-example)
  - [Miscellaneous Notes](#miscellaneous-notes)
    - [Feature Engineering Using A Sentinel LN Nomogram](#feature-engineering-using-a-sentinel-ln-nomogram)

## Introduction
---
This is an end-to-end machine learning pipeline for predicting the need of post-mastectomy radiotherapy before breast cancer pathology diagnosis.

Many breast cancer requires post-mastectomy reconstruction. However, if a patient also needs post-mastectomy radiotherapy, it's important for the surgeon and the patient to predict such need *before* the mastectomy surgery. Due to the difficulty in these predictions, and the high clinical value of an accurate prediction, we explore the applicability of Machine Learning (ML) methods. 

To build ML models, we curated a dataset containing 115 clinical features/variables spanning across demographic, preoperative, intraoperative, and postoperative information. ML models are trained on this dataset and are used to predict the need of post-mastectomy radiotherapy.


## Installation And Usage
---
The necessary libraries could be installed with the following command:

`pip install -r requirements.txt` 

To run the pipeline: `python3 main.py`

To run the pipeline in debug mode: `python3 main.py --debug`

To run the pipeline with previously-found best models: `python3 main.py -f `

To visualize data: `python3 src/visualize/visualizer.py`

To visualize all columns: `python3 src/visualize/visualizer.py -vizall`

To run tests: `python -m pytest tests/`

To display logs (with color-coded texts):
`cat data/preprocessed/<OUTPUT_TIMESTAMP>/log.txt`

To generate subsets of the original dataset using specific subsets of columns: `python scripts/generate_datasets.py`

To compare the two versions of the same dataset and obtain the differences: `python scripts/compare_datasets.py`

## An End-to-End ML Pipeline 
---
### Pipeline Overview Diagram
![Pipeline Overview Diagram](plots_tables/RTx_Pipeline_Diagram.png)
### Data Preprocessing Module
1. *Column Renaming*. Editing the column names of pre-operative, intra-operative, and post-operative features.
2. *Data     Cleansing*. Systematically replacing noisy values using predetermined rules.
3. *Feature Engineering*. Construct new interaction features guided by clinical insights.
4. *Missing Value Imputation*. Fill in missing values using imputation strategies and machine learning. For each column:
    1. Determine the type of column as real, ordinal, or categorical
    2. Using 5-Fold Cross-Validation:
        * Apply K-Nearest Neighbor Imputer. Perform grid-search for optimal `K`.
        * Apply Random Forest Imputer. Perform grid-search for optimal `max_depth` and `n_trees`
        * Compare the KNN and RF imputers using accuracy, F1, and other metrics.
    5. Perform use multiple imputation using the more accurate imputation strategy
5. *Visualize Dataset*. Generate a figure highlighting the changed cell values in the dataset.
6. *Dataset Generation*. Generate subsets of the original dataset using specific subsets of columns.

### ML Training-Evaluation Module
1. *Dataset Loading*. Load the dataset from the preprocessed folder.
2. *Dataset Splitting*. Split the dataset into training and testing sets using stratified sampling of target variable.
3. *Model Training*. Train a model using the training set, and evaluate the model using the validation set. Uses K-Fold cross-validation.
4. *Model Evaluation*. Evaluate the model using AUC, F1, Accuracy, and other metrics.
5. *Metrics Visualization*. Generate a figure highlighting the performance of the models.
6. *Feature Importance*. Generate a feature importance plot for the model to determine which features are most important.
7. *Feature Selection*. Select features using a variety of methods, with the help of clinical insights.
8. *Model Selection*. Select the best model using a variety of methods, including using the visualizations generated in previous steps.
9. *Model Prediction*. Predict the probability of a patient requiring post-mastectomy radiotherapy using the best model.


## Results
---
### Figures

#### AUC Performance Metrics
![AUC Performance Metrics](README_figures/AUC_full_axis.png)


#### Feature Importance and Feature Selection
![Feature Selection Plot](README_figures/7_PRE-1.0spars-expert-imputed86cols_top_features_metrics.png)

### Tables

#### Feature Scores by Model (Example)
| Features                          | RF\_rank | LR\_rank | XGB\_rank | RF\_score | LR\_score | XGB\_score | Rank\_Product^(1/3) | Rank\_Product\_Rank |
| --------------------------------- | -------- | -------- | --------- | --------- | --------- | ---------- | ------------------- | ------------------- |
| PRE\_axillary\_lymphadenopathy    | 2        | 4        | 1         | 0.23      | 0.48      | 0.36       | 2                   | 1                   |
| PRE\_int\_mammary\_lymphade\_pet  | 4        | 1        | 2         | 0.18      | 1.07      | 0.22       | 2                   | 2                   |
| PRE\_abnormal\_lymph              | 1        | 3        | 4         | 0.24      | 0.58      | 0.14       | 2.29                | 3                   |
| PRE\_axillary\_lymphadenopathy\_p | 3        | 2        | 3         | 0.22      | 0.59      | 0.15       | 2.62                | 4                   |
| PRE\_prominent\_axillary\_lymph   | 5        | 6        | 5         | 0.09      | 0.08      | 0.07       | 5.31                | 5                   |
| PRE\_internal\_mammary\_lymphaden | 6        | 5        | 6         | 0.04      | 0.32      | 0.05       | 5.65                | 6                   |


## Miscellaneous Notes


### Feature Engineering Using A Sentinel LN Nomogram
An additional feaature was created during the feature engineering step by using a peer-reviewed nomogram for predicting the probability of sentinel lymph node metastasis. The nomogram is available as a [browser-based calculator](https://nomograms.mskcc.org/Breast/BreastSLNodeMetastasisPage.aspx). Their publication can be found on PubMed at [this link](https://pubmed.ncbi.nlm.nih.gov/17664461/).

Below is an instruction on how to use our custom Python script to query the calculator autonomously at scale, to save the hassle of manually entering the values for hundreds of patients.

