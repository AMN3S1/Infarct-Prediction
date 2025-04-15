import pandas as pd
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv("top15_features_dataset.csv")

# Features and target
X = df.drop("heart_attack", axis=1)
y = df["heart_attack"]

# One-hot encoding for categorical features (если есть категориальные признаки)
X = pd.get_dummies(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Start timing
start = time.time()

# Initialize and train model
model = GaussianNB()
model.fit(X_train, y_train)

# End timing
end = time.time()
train_time = end - start

# Predictions
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Training time: {train_time:.2f} sec")

# Save to CSV
results_df = pd.DataFrame([{
    "Model": "NaiveBayes (Gaussian)",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-score": f1,
    "Train time (s)": round(train_time, 2)
}])

file_path = "model_results_top15.csv"
write_header = not os.path.exists(file_path)
results_df.to_csv(file_path, mode="a", index=False, header=write_header)
