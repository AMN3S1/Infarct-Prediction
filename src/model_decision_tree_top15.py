import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
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

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", DecisionTreeClassifier(random_state=42))
])

# Training with time tracking
start = time.time()
pipeline.fit(X_train, y_train)
end = time.time()

# Evaluation
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Decision Tree (top-15 features) training time: {end - start:.2f} seconds")
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
    writer.writerow(["DecisionTree_top15", accuracy, precision, recall, f1])
