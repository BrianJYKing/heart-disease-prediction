# DSCubed Heart Disease Prediction Competition

## Overview
This project was developed as part of the **DSCubed Heart Disease Prediction Competition** hosted on Kaggle. The challenge was to predict the severity of heart disease for patients based on a dataset collected from five hospitals across Melbourne. The dataset includes various features such as blood sugar levels, cholesterol, electrocardiogram (ECG) results, and other patient information.

## Problem Statement
The goal of this project was to build a machine learning model to accurately predict whether a patient is diagnosed with heart disease and assess its severity based on the provided features.

## Project Structure
```bash
├── data/
│   ├── train.csv               # Training dataset
│   ├── test.csv                # Test dataset
├── notebooks/
│   ├── DSCubed_Heart_Disease_Prediction.ipynb # Jupyter notebook with the full process
├── heart-disease-prediction-results.csv  # Final predictions
├── README.md                   # This README file
└── requirements.txt            # Python dependencies
```

--- 

## Approach

### 1. Data Loading and Exploration

- Load the datasets (`train.csv` and `test.csv`) and analyze the structure, missing values, and class distribution for the target variable.

### 2. Data Cleaning

- **Numerical Columns**: Impute missing values using the median.
- **Categorical Columns**: Impute missing values using the mode.

### 3. Feature Engineering

- **One-Hot Encoding**: Categorical variables were encoded using one-hot encoding.
- **Feature Alignment**: Train and test datasets were aligned to ensure consistency in feature sets.

### 4. Class Balancing Using SMOTE

- The dataset was highly imbalanced. We applied **SMOTE (Synthetic Minority Oversampling Technique)** to balance the class distribution.

### 5. Model Training and Cross-Validation

- **Random Forest Classifier** was trained using **cross-validation**.
- Evaluated the model using **accuracy** and **weighted F1 score** metrics to ensure good generalization across all patient categories.

### 6. Prediction and Submission

- Predictions were made on the test dataset, and a submission file (`heart-disease-prediction-results.csv`) was prepared for the competition.

## Evaluation Metrics

- **Cross-Validation Accuracy**: Assesses how well the model generalizes across different folds of the data.
- **Weighted F1 Score**: Used to handle class imbalances and ensure that the model performs well across all classes.

## Requirements

To run this project, install the following dependencies:

- `pandas`
- `scikit-learn`
- `seaborn`
- `matplotlib`
- `imblearn`

You can install all the required packages by running:

```bash
pip install -r requirements.txt
```

## Usage
1. Clone the repository and navigate to the project directory.
2. Run the Jupyter notebook DSCubed_Heart_Disease_Prediction.ipynb to follow the full process from data loading to prediction.
3. To generate predictions, you can run the model training code and generate the submission file.

---

## Results
- Cross-Validation Accuracy: The model achieved an accuracy score of around 0.8.
- Weighted F1 Score: The model achieved a weighted F1 score of around 0.79.

## Future Work
- Explore more advanced hyperparameter tuning to improve model performance.
- Experiment with other machine learning models such as XGBoost or Gradient Boosting for better accuracy.
- Further refine the feature engineering process to boost model performance.