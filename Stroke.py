# Imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, roc_auc_score,
    classification_report, confusion_matrix, RocCurveDisplay
)
from imblearn.over_sampling import SMOTE

# Load & preprocess data
df = pd.read_csv('healthcare-dataset-stroke-data.csv')
df.drop('id', axis=1, inplace=True)
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())


# Encode categorical features
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Split
X = df.drop('stroke', axis=1)
y = df['stroke']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "MLP Neural Net": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
}

# Evaluation results
results = {}

# Plot setup
plt.figure(figsize=(10, 6))
plt.title("ROC Curves")

# Train & Evaluate
for name, model in models.items():
    model.fit(X_train_bal, y_train_bal)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results[name] = {
        "Accuracy": acc,
        "Recall": recall,
        "ROC AUC": auc
    }

    RocCurveDisplay.from_estimator(model, X_test, y_test, name=name)

# Show ROC plot
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal
plt.legend()
plt.grid()
plt.show()

# Print results
print("\nModel Comparison:\n")
for name, scores in results.items():
    print(f"{name}:")
    for metric, val in scores.items():
        print(f"  {metric}: {val:.4f}")
    print("-" * 30)

