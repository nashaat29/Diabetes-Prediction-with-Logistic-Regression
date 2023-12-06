# Diabetes Prediction with Logistic Regression

Welcome to the "Diabetes Prediction with Logistic Regression" project repository! This project focuses on utilizing machine learning techniques, specifically logistic regression, to predict the likelihood of diabetes in individuals based on essential health-related features.

![Cover](https://github.com/nashaat29/Diabetes-Prediction-with-Logistic-Regression/assets/138555343/7147df4a-79b7-4112-a664-6a6b125e82be)

## Project Overview

- **Objective**: Develop a reliable and interpretable machine learning model for early diabetes detection.
- **Features**: Pregnancies, Glucose levels, Blood Pressure, Skin Thickness, Insulin, BMI (Body Mass Index), Diabetes Pedigree Function, and Age.
- **Dataset**: Information on various individuals, including diabetes diagnosis (Outcome).
- **Methodology**: Extensive exploration, preprocessing, and feature engineering, followed by model training and evaluation.

## Key Steps

### 1. Data Exploration

In the data exploration phase, we delved into the dataset to gain a comprehensive understanding of its structure and content. This involved:

- **Dataset Overview**: Examining the size, structure, and basic statistics of the dataset.
- **Feature Analysis**: Investigating the distribution of each feature, identifying potential outliers or anomalies.
- **Target Variable Exploration**: Analyzing the 'Outcome' variable to understand the distribution of positive and negative cases.

### 2. Preprocessing

Effective preprocessing is crucial for preparing the dataset for accurate modeling. Key preprocessing steps included:

- **Handling Outliers**: Identifying and addressing outliers that could potentially distort model training.
- **Correlation Analysis**: Investigating feature correlations to identify redundant or highly correlated variables.
- **Optimizing Dataset**: Cleaning and transforming the data to ensure it aligns with the assumptions of the selected algorithms.

### 3. Modeling

In the modeling phase, we experimented with various machine learning algorithms to identify the most suitable one for our predictive task. The use of Stratified K-Fold cross-validation provided a robust assessment of each model's performance across different folds of the dataset.

- **Algorithm Selection**: Tried multiple algorithms such as Logistic Regression, SVM, Decision Tree, Random Forest, AdaBoost, XGBoost, LightGBM, and Naive Bayes.
- **Cross-Validation**: Employed Stratified K-Fold cross-validation to ensure an unbiased evaluation of each model's performance.

To ensure the robustness of our predictive model, we adopted a meticulous approach. Employing the Stratified K-Fold cross-validation technique, we trained and evaluated multiple machine learning algorithms, including Logistic Regression, SVM, Decision Tree, Random Forest, AdaBoost, XGBoost, LightGBM, and Naive Bayes. The table below summarizes the performance of each algorithm, showcasing the highest and lowest accuracies, overall accuracy, and standard deviation across folds:

| Algorithm           | Highest Accuracy | Lowest Accuracy | Overall Accuracy | Standard Deviation |
|---------------------|-------------------|------------------|-------------------|--------------------|
| Logistic Regression | 85.94             | 68.25            | 77.97             | 0.05               |
| SVM                 | 84.38             | 73.02            | 78.92             | 0.04               |
| Decision Tree       | 76.56             | 63.49            | 70.27             | 0.04               |
| Random Forest       | 82.81             | 66.67            | 77.50             | 0.05               |
| AdaBoost            | 79.69             | 70.31            | 76.42             | 0.03               |
| XGBoost             | 80.95             | 68.25            | 75.31             | 0.04               |
| LightGBM            | 82.54             | 69.84            | 75.31             | 0.04               |
| Naive Bayes         | 81.25             | 66.67            | 76.40             | 0.04               |

### 4. Evaluation

Model evaluation is crucial for assessing its effectiveness in predicting diabetes. Performance metrics were employed to comprehensively evaluate each model:

- **Accuracy**: Overall correctness of the predictions.
- **Precision**: Proportion of true positive predictions among all positive predictions.
- **Recall (Sensitivity)**: Proportion of actual positive instances correctly predicted.
- **F1 Score**: Harmonic mean of precision and recall, providing a balanced metric.

### 5. Feature Selection

Feature selection aimed to enhance the model's efficiency by focusing on the most relevant features. This involved:

- **Correlation Analysis**: Identifying and removing features with high correlations to reduce multicollinearity.
- **Selection Criteria**: Choosing features that demonstrated significant influence on predicting the target variable, 'Outcome.'

These detailed steps ensured a thorough exploration of the dataset, optimal preprocessing for modeling, and a robust evaluation process to select the most effective algorithm for diabetes prediction.

## Selected Model

After a comprehensive evaluation, the Logistic Regression model emerged as the top performer, achieving an impressive accuracy of 85.94%. This model has been selected for its accuracy and reliability in predicting the likelihood of diabetes.

Kaggle Notebook: https://www.kaggle.com/code/mohammednashat/diabetes-prediction-with-logistic-regression
