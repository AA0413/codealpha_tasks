# TASK 1: Credit Scoring Model
# Objective: Predict an individual's creditworthiness using past financial data.
# Dataset: German Credit Data (UCI Statlog), loaded from a public GitHub mirror.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix

# 1. Load the dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv"
data = pd.read_csv(url)

print("Dataset shape:", data.shape)
print(data["credit_risk"].value_counts())
# credit_risk: 1 = good credit risk, 0 = bad credit risk

# 2. Feature engineering
# Separate target from features
target = data["credit_risk"]
features = data.drop(columns=["credit_risk"])

# Identify categorical columns (financial history like credit_history, purpose,
# savings, employment_duration, etc. are stored as text)
categorical_columns = features.select_dtypes(include=["object", "str"]).columns

# Label-encode each categorical column so models can use it
label_encoder = LabelEncoder()
for column in categorical_columns:
    features[column] = label_encoder.fit_transform(features[column])

# Scale numeric features (helps Logistic Regression converge properly)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    features_scaled, target, test_size=0.2, random_state=42, stratify=target
)

# 4. Train models
log_reg_model = LogisticRegression(max_iter=1000, random_state=42)
log_reg_model.fit(X_train, y_train)

decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train, y_train)

random_forest_model = RandomForestClassifier(n_estimators=200, random_state=42)
random_forest_model.fit(X_train, y_train)

# 5. Evaluate models
models = {
    "Logistic Regression": log_reg_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model,
}

print("\nModel Performance Comparison")
print("-" * 60)

for model_name, model in models.items():
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    print(f"\n{model_name}")
    print(f"  Accuracy : {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall   : {recall:.3f}")
    print(f"  F1-Score : {f1:.3f}")
    print(f"  ROC-AUC  : {roc_auc:.3f}")
    print(f"  Confusion Matrix:\n{confusion_matrix(y_test, predictions)}")

# 6. Feature importance (Random Forest)
importances = pd.Series(random_forest_model.feature_importances_, index=features.columns)
importances = importances.sort_values(ascending=False)

print("\nTop 10 most important features (Random Forest):")
print(importances.head(10))
