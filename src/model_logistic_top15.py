import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import csv
import os

# Load top-15 feature dataset
df = pd.read_csv("top15_features_dataset.csv")

# Features and target
X = df.drop("heart_attack", axis=1)
y = df["heart_attack"]

# One-hot encode categorical variables
X = pd.get_dummies(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

# Timing
start = time.time()
pipeline.fit(X_train, y_train)
end = time.time()

# Predict and evaluate
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Output
print(f"Logistic Regression on top-15 features trained in {end - start:.2f} seconds")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# Save results
results_file = "model_results_top15.csv"
header = ["model", "accuracy", "precision", "recall", "f1"]

if not os.path.exists(results_file):
    with open(results_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

with open(results_file, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["LogisticRegression_top15", accuracy, precision, recall, f1])
