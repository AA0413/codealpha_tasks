# Task 1: Credit Scoring Model

## Objective
Predict an individual's creditworthiness (good vs. bad credit risk) using past financial data.

## Dataset
German Credit Data (UCI Statlog dataset) — 1,000 records with 20 financial and personal
attributes (checking account status, credit history, loan purpose, amount, savings,
employment duration, age, housing, etc.) and a binary target: 1 = good credit risk,
0 = bad credit risk.

Loaded automatically at runtime from a public GitHub mirror, so no manual download is
needed:
`https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv`

## Approach
Three classification algorithms are trained and compared:
- Logistic Regression
- Decision Tree
- Random Forest

## Pipeline
1. Load the dataset directly from the URL above.
2. Encode categorical columns (credit history, purpose, savings, etc.) with label encoding.
3. Scale all features with `StandardScaler`.
4. Split into train/test sets (80/20, stratified on the target).
5. Train all three models.
6. Evaluate each on Accuracy, Precision, Recall, F1-Score, and ROC-AUC, plus a confusion
   matrix.
7. Print the Random Forest's top 10 most important features.

## Requirements
```
pandas
scikit-learn
```
Install with:
```
pip install pandas scikit-learn
```

## How to Run
```
python credit_scoring_model.py
```

## Output
Console output showing, for each model: Accuracy, Precision, Recall, F1-Score, ROC-AUC,
and a confusion matrix — followed by the top 10 features driving the Random Forest's
predictions.
