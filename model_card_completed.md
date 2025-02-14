# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- **Model Name**: Random Forest Classifier for Census Income Prediction
- **Algorithm**: RandomForestClassifier (from `sklearn.ensemble`)
- **Hyperparameters**:
  - Number of estimators: 100
  - Random state: 42
  - Max depth: None
  - Min samples split: 2
  - Min samples leaf: 1
- **Input Features**:
  - Categorical: `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`
  - Numerical: Other numerical features from the dataset.
- **Output**: Binary classification (`<=50K` or `>50K` income category)

## Intended Use
This model is designed to predict whether an individual’s income is greater than **$50K** based on census demographic and employment data. It is intended for educational and research purposes in machine learning pipeline deployment.

## Training Data
- **Dataset**: `census.csv`
- **Source**: U.S. Census data 
- **Preprocessing**:
  - One-hot encoding for categorical features
  - Label binarization for the target variable (`salary`)
  - Data split: 80% training, 20% testing

## Evaluation Data
- The test dataset was created using a **20% split** of the original dataset.
- The same preprocessing steps were applied as in training.

## Metrics
- **Precision**: `0.7353` (Replace with actual value)
- **Recall**: `0.6378` (Replace with actual value)
- **F1-score**: `0.6831` (Replace with actual value)

## Ethical Considerations
- **Bias & Fairness**: The dataset contains demographic attributes such as race and sex, which could introduce bias in model predictions. Model predictions should be carefully evaluated to ensure fairness across different demographic groups.
- **Privacy**: The model does not store any personally identifiable information (PII). However, census data might contain sensitive information, so it should be used responsibly.
- **Impact**: Predictions should not be used for critical decision-making, such as hiring or financial assessments, without human oversight.

## Caveats and Recommendations
- The model is trained on census data and may not generalize well to different demographics or regions.
- Future improvements could include:
  - Hyperparameter tuning to optimize model performance.
  - Addressing class imbalance if necessary.
  - Evaluating fairness metrics for potential biases in predictions.
